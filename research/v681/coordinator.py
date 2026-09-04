"""V681-native closed learning loop: collect, batch, evaluate, promote, resume."""
from __future__ import annotations

import json
import math
import os
import shutil
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .experience import ExperienceSource, ExperienceStore, attention_step_experience
from .importers import import_chat_record, import_worker_event
from .learners import AttentionDistillationLearner, JEPAAuxiliaryLearner
from .native_learning.engine import NativeLearningEngine
from .native_runtime.runtime import NativeRuntime

V681_VERSION = "v681.5-native-closed-loop-1"


@dataclass
class RuntimePolicy:
    min_sequential_episodes: int = 8
    new_sequential_episodes: int = 1
    bootstrap_samples: int = 20
    epochs: int = 8
    seed: int = 6815
    poll_seconds: float = 1.0
    minimum_overall_action_accuracy: float = .75
    minimum_abstain_accuracy: float = .80
    maximum_false_positive_traverse: float = .10
    maximum_premature_stop: float = .10
    maximum_premature_abstain: float = .10
    accuracy_regression_tolerance: float = .02
    safety_regression_tolerance: float = .01


class V681Coordinator:
    """The single owner of V681 runtime, store, learning cursor, and artifacts."""
    def __init__(self, root=None, output_dir=None, policy=None, engine=None, chat_runtime=None):
        self.root = Path(root or Path(__file__).resolve().parents[2]).resolve()
        self.output = Path(output_dir or self.root / "results" / "v681")
        self.policy, self.engine, self.chat = policy or RuntimePolicy(), engine, chat_runtime
        self.session_id = f"v681-{uuid.uuid4().hex}"
        self.session_dir = self.output / "sessions" / self.session_id
        self.output.mkdir(parents=True, exist_ok=True); self.session_dir.mkdir(parents=True, exist_ok=True)
        self.store = ExperienceStore(self.output / "experience" / "experience.sqlite")
        self.cycles, self.failures, self.chat_records, self.worker_events = [], [], 0, 0

    def run(self, once=False, dry_run=False, smoke=False):
        started, before, discovery = time.time(), self._inspection(), self._discover()
        if self.chat is not None:
            self._capture()
        if dry_run:
            return self._finish(started, before, discovery, "dry_run")
        self._learn_if_due("startup", bootstrap=True)
        if once:
            return self._finish(started, before, discovery, "once")
        if not discovery["chat_runtime"]["available"]:
            self.failures.append({"stage": "chat_runtime", "reason": discovery["chat_runtime"]["reason"]})
            return self._finish(started, before, discovery, "chat_unavailable")
        self.chat = self.chat or NativeRuntime(discovery["graph_database"]["path"], discovery["local_llm"]["path"],
                                               self.session_dir, self.session_id, mode="smoke" if smoke else "chat",
                                               attention_policy=self.output / "models" / "current_attention.pt"
                                               if (self.output / "models" / "current_attention.pt").is_file() else "")
        try:
            self.chat.start()
            while self.chat.running():
                self._capture(); self._learn_if_due("new_experience")
                time.sleep(self.policy.poll_seconds)
        except KeyboardInterrupt:
            pass
        finally:
            self.chat.stop(); self._capture(); self._learn_if_due("shutdown")
        return self._finish(started, before, discovery, "complete")

    def _discover(self):
        graph = _discover_graph(self.root / "data")
        configured = os.environ.get("GRAPH_TOPOLOGY_LLM_MODEL", "").strip()
        model = Path(configured).expanduser() if configured else self.root / "llm" / "SmolLM3-3B"
        ready = bool(graph and model.is_dir())
        return {
            "repository_root": str(self.root), "graph_database": {"available": bool(graph), "path": str(graph or "")},
            "local_llm": {"available": model.is_dir(), "path": str(model) if model.is_dir() else ""},
            "chat_runtime": {"available": ready, "reason": "" if ready else
                             f"native chat needs a focused graph in {self.root / 'data'} and model at {model}"},
            "native_learning": {"available": True, "source_learning_version": "v680.1"},
        }

    def _capture(self):
        if self.chat is None:
            return
        events = self.chat.poll()
        for event in events["chat"]:
            try:
                import_chat_record(self.store, event["value"], event["source_path"], event["line"])
                self.chat_records += 1
            except (KeyError, ValueError) as error:
                self.failures.append({"stage": "chat_capture", "reason": str(error), **event})
        for event in events["worker"]:
            if event.get("value", {}).get("event") != "analysis_batch":
                continue
            try:
                import_worker_event(self.store, event["value"], event["source_path"], event["line"])
                self.worker_events += 1
            except (KeyError, ValueError) as error:
                self.failures.append({"stage": "worker_capture", "reason": str(error), **event})

    def _learn_if_due(self, trigger, bootstrap=False):
        self._materialize_live_sequential()
        cursor = self.store.learning_cursor("attention")
        if cursor and not self._new_eligible_episodes(cursor["last_experience_id"]):
            self.cycles.append({"kind": "attention", "trigger": trigger, "status": "no_op",
                                "reason": "no new training experience"})
            return
        if self._inspection()["train_sequential_episodes"] < self.policy.min_sequential_episodes and bootstrap:
            self._bootstrap(trigger)
        if self._inspection()["train_sequential_episodes"] >= self.policy.min_sequential_episodes:
            self._train_candidate(trigger)

    def _bootstrap(self, trigger):
        try:
            path = self.session_dir / "bootstrap_teacher.jsonl"
            self._engine().generate_teacher_records(path, self.policy.bootstrap_samples)
            records = _read_jsonl(path)
            _append_records(self.store, records, ExperienceSource.ATTENTION_EVAL, "evaluation-", heldout_only=True)
            aggregate = self._engine().run_dagger(path, self.session_dir / "dagger", 1, self.policy.epochs, self.policy.seed)
            _append_records(self.store, _read_jsonl(aggregate), ExperienceSource.DAGGER, "dagger-")
            _append_records(self.store, records, ExperienceSource.SYNTHETIC_CHAT, "synthetic-")
            self.cycles.append({"kind": "bootstrap", "trigger": trigger, "status": "completed"})
        except Exception as error:
            self.failures.append({"stage": "bootstrap", "trigger": trigger, "reason": str(error)})

    def _train_candidate(self, trigger):
        try:
            items, learner = self.store.load(), AttentionDistillationLearner()
            sources = {ExperienceSource.DAGGER, ExperienceSource.SYNTHETIC_CHAT, ExperienceSource.CHAT_SEQUENTIAL}
            train, rejected = learner.prepare(items, sources)
            evaluation, rejected_eval = learner.prepare(items, {ExperienceSource.ATTENTION_EVAL}, allowed_splits=("heldout",))
            if not train or not evaluation:
                return
            cycle = len(self.cycles) + 1
            dataset, heldout = self.session_dir / f"attention-{cycle}.jsonl", self.session_dir / f"heldout-{cycle}.jsonl"
            _write_jsonl(dataset, train); _write_jsonl(heldout, evaluation)
            candidate = self.output / "models" / f"attention-{self.session_id}-{cycle}.pt"
            candidate.parent.mkdir(parents=True, exist_ok=True); (self.output / "evaluations").mkdir(parents=True, exist_ok=True)
            learner.train(self._engine(), dataset, candidate, self.policy.epochs, self.policy.seed)
            metrics = learner.evaluate(self._engine(), heldout, candidate, self.output / "evaluations" / f"attention-{cycle}.json")
            jepa = JEPAAuxiliaryLearner(); transitions, jepa_rejected = jepa.prepare(items, sources)
            jepa_data = self.session_dir / f"jepa-{cycle}.jsonl"; _write_jsonl(jepa_data, transitions)
            jepa_result = jepa.train(self._engine(), jepa_data, self.output / "models" / f"jepa-{self.session_id}-{cycle}.pt",
                                     self.output / "evaluations" / f"jepa-{cycle}.json", self.policy.epochs, self.policy.seed)
            current = self.output / "models" / "current_attention.pt"
            current_metrics = (learner.evaluate(self._engine(), heldout, current,
                                                self.output / "evaluations" / f"current-{cycle}.json")
                               if current.is_file() else None)
            promotion = _promotion(metrics, current_metrics, self.policy)
            artifact = {"v681_version": V681_VERSION, "candidate": str(candidate), "metrics": metrics,
                        "promotion": promotion, "sources": sorted(x.value for x in sources), "rejected": rejected,
                        "heldout_rejected": rejected_eval, "jepa": jepa_result, "jepa_rejected": jepa_rejected}
            provenance = candidate.with_suffix(".provenance.json"); provenance.write_text(json.dumps(artifact, indent=2, sort_keys=True))
            if promotion["promoted"]:
                staged = current.with_suffix(".staged")
                shutil.copy2(candidate, staged); os.replace(staged, current)
                artifact["promotion"]["current_model"] = str(current)
                if self.chat is not None and hasattr(self.chat, "set_attention_policy"):
                    self.chat.set_attention_policy(current)
            self.store.update_learning_cursor("attention", self.store.latest_experience_id(), str(dataset), str(provenance))
            self.cycles.append({"kind": "candidate", "trigger": trigger, "status": "evaluated", **artifact})
        except Exception as error:
            self.failures.append({"stage": "learning", "trigger": trigger, "reason": str(error)})

    def _materialize_live_sequential(self):
        """Create immutable train views once for eligible live trajectories."""
        items = self.store.load()
        existing = {item.provenance.get("materialized_from") for item in items}
        for item in items:
            if item.split != "live" or item.sequence_capability != "sequential" or item.experience_id in existing:
                continue
            record = item.as_dict()
            record["experience_id"], record["split"], record["timestamp"] = uuid.uuid4().hex, "train", time.time()
            record["provenance"] = {**record["provenance"], "materialized_from": item.experience_id,
                                    "materialization_version": V681_VERSION}
            self.store.append(record)

    def _new_eligible_episodes(self, last_id):
        items, seen = self.store.load(), False
        fresh = []
        for item in items:
            if seen: fresh.append(item)
            if item.experience_id == last_id: seen = True
        return len({item.episode_id for item in fresh if item.split == "train" and item.sequence_capability == "sequential"})

    def _engine(self):
        self.engine = self.engine or NativeLearningEngine()
        return self.engine

    def _inspection(self):
        items = self.store.load()
        train = [item for item in items if item.split == "train" and item.sequence_capability == "sequential"]
        return {"total_records": len(items), "chat_records": sum(item.source in {ExperienceSource.CHAT_DECISION_ONLY, ExperienceSource.CHAT_SEQUENTIAL} for item in items),
                "worker_records": sum(item.source is ExperienceSource.OFFLINE_WORKER for item in items),
                "train_sequential_episodes": len({item.episode_id for item in train}),
                "source_composition": {source.value: sum(item.source is source for item in items) for source in ExperienceSource}}

    def _finish(self, started, before, discovery, status):
        result = {"v681_version": V681_VERSION, "session_id": self.session_id, "status": status, "start_time": started,
                  "end_time": time.time(), "experience_before": before, "experience_after": self._inspection(),
                  "training_cycles": self.cycles, "models_created": [x["candidate"] for x in self.cycles if "candidate" in x],
                  "worker_events": self.worker_events, "chat_episodes": self.chat_records, "discovery": discovery,
                  "failures": self.failures}
        for name, value in (("runtime_results.json", result), ("experience_manifest.json", self.store.manifest()),
                            ("current_model_manifest.json", self.store.learning_cursor("attention") or {})):
            (self.output / name).write_text(json.dumps(value, indent=2, sort_keys=True))
        (self.output / "runtime_report.md").write_text("# V681 runtime report\n\n" + json.dumps(result, indent=2, sort_keys=True))
        (self.session_dir / "session_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True))
        self.store.close()
        return result


def _discover_graph(data):
    for path in sorted(Path(data).glob("v*_focused_semantic.sqlite"), reverse=True):
        try:
            with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
                connection.execute("SELECT 1 FROM nodes LIMIT 1"); connection.execute("SELECT 1 FROM edges LIMIT 1")
            return path.resolve()
        except sqlite3.Error: continue
    return None


def _read_jsonl(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def _write_jsonl(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in values))
def _append_records(store, records, source, prefix, heldout_only=False):
    for episode in records:
        for step in episode["trajectory"]:
            if heldout_only and not str(step["split"]).startswith("held_out"): continue
            item = attention_step_experience(step, source=source); item.episode_id = prefix + item.episode_id
            item.provenance["supervision_source"] = "native_frozen_teacher"; store.append(item)
def _promotion(candidate_metrics, current_metrics, policy):
    """Fail closed on absent rollout safety metrics and preserve current model on regressions."""
    candidate = _safety_metrics(candidate_metrics)
    current = _safety_metrics(current_metrics) if current_metrics is not None else None
    thresholds = {
        "minimum_overall_action_accuracy": policy.minimum_overall_action_accuracy,
        "minimum_abstain_accuracy": policy.minimum_abstain_accuracy,
        "maximum_false_positive_traverse": policy.maximum_false_positive_traverse,
        "maximum_premature_stop": policy.maximum_premature_stop,
        "maximum_premature_abstain": policy.maximum_premature_abstain,
        "accuracy_regression_tolerance": policy.accuracy_regression_tolerance,
        "safety_regression_tolerance": policy.safety_regression_tolerance,
    }
    required = ("overall_action_accuracy", "abstain_accuracy", "false_positive_traverse",
                "premature_stop", "premature_abstain")
    missing = [name for name in required if name not in candidate]
    checks = {
        "overall_accuracy": "overall_action_accuracy" in candidate and
                            candidate["overall_action_accuracy"] >= policy.minimum_overall_action_accuracy,
        "abstain_accuracy": "abstain_accuracy" in candidate and
                            candidate["abstain_accuracy"] >= policy.minimum_abstain_accuracy,
        "false_positive_traverse": "false_positive_traverse" in candidate and
                                   candidate["false_positive_traverse"] <= policy.maximum_false_positive_traverse,
        "premature_stop": "premature_stop" in candidate and
                          candidate["premature_stop"] <= policy.maximum_premature_stop,
        "premature_abstain": "premature_abstain" in candidate and
                             candidate["premature_abstain"] <= policy.maximum_premature_abstain,
    }
    checks["non_regression"] = _non_regression(candidate, current, policy) if current is not None else True
    promoted = not missing and all(checks.values())
    failed = [name for name, passed in checks.items() if not passed]
    reason = ("missing required rollout metrics: " + ", ".join(missing) if missing else
              "failed promotion checks: " + ", ".join(failed) if failed else
              "passed absolute safety and non-regression gates")
    return {"promoted": promoted, "candidate_metrics": candidate_metrics, "current_metrics": current_metrics,
            "safety_metrics": {"candidate": candidate, "current": current},
            "thresholds": thresholds, "checks": checks, "reason": reason}


def _safety_metrics(metrics):
    """Aggregate the existing evaluator's held-out rollout measurements by decision count."""
    if not metrics:
        return {}
    rollout = metrics.get("rollout", metrics)
    groups = [value for value in rollout.values()
              if isinstance(value, dict) and "overall_action_accuracy" in value]
    if not groups and "overall_action_accuracy" in rollout:
        groups = [rollout]
    if not groups:
        return {}
    weights = [max(1, group.get("decisions", 0)) for group in groups]
    return {metric: sum(group[metric] * weight for group, weight in zip(groups, weights)) / sum(weights)
            for metric in ("overall_action_accuracy", "abstain_accuracy", "false_positive_traverse",
                           "premature_stop", "premature_abstain")
            if all(isinstance(group.get(metric), (int, float)) and math.isfinite(group[metric])
                   for group in groups)}


def _non_regression(candidate, current, policy):
    required = ("overall_action_accuracy", "abstain_accuracy", "false_positive_traverse",
                "premature_stop", "premature_abstain")
    if any(name not in candidate or name not in current for name in required):
        return False
    return (candidate["overall_action_accuracy"] >= current["overall_action_accuracy"] - policy.accuracy_regression_tolerance
            and candidate["abstain_accuracy"] >= current["abstain_accuracy"] - policy.safety_regression_tolerance
            and candidate["false_positive_traverse"] <= current["false_positive_traverse"] + policy.safety_regression_tolerance
            and candidate["premature_stop"] <= current["premature_stop"] + policy.safety_regression_tolerance
            and candidate["premature_abstain"] <= current["premature_abstain"] + policy.safety_regression_tolerance)
