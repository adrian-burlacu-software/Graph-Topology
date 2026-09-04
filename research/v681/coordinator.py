"""Central V681 lifecycle: discovery, capture, learning, evaluation, reporting."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .experience import ExperienceSource, ExperienceStore, attention_step_experience
from .importers import import_chat_record, import_chat_traces, import_worker_event, import_worker_logs
from .learners import AttentionDistillationLearner, JEPAAuxiliaryLearner
from .runtime_adapters import RepositoryRuntime, V679ChatRuntimeAdapter
from .trajectory import AttentionTrajectoryAdapter
from .v680_adapter import V680EngineAdapter

V681_VERSION = "v681.4-runtime-1"


@dataclass
class RuntimePolicy:
    min_sequential_episodes: int = 8
    bootstrap_samples: int = 20
    epochs: int = 8
    seed: int = 6814
    poll_seconds: float = 1.0
    training_interval_seconds: int = 300


class V681Coordinator:
    """Owns orchestration; source adapters only expose runtime records."""
    def __init__(self, runtime=None, output_dir=None, policy=None, engine=None, chat_runtime=None):
        self.runtime = runtime or RepositoryRuntime.discover()
        self.output = Path(output_dir or self.runtime.results_root)
        self.policy = policy or RuntimePolicy()
        self.session_id = f"v681-{uuid.uuid4().hex}"
        self.session_dir = self.output / "sessions" / self.session_id
        self.output.mkdir(parents=True, exist_ok=True)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.store = ExperienceStore(self.output / "v681_experience.sqlite")
        self.engine = engine
        self.chat = chat_runtime or V679ChatRuntimeAdapter(self.runtime, self.session_dir, self.session_id)
        self.cycles, self.failures, self.chat_records, self.worker_events = [], [], 0, 0

    def run(self, once=False, dry_run=False):
        started = time.time()
        before = self._inspection()
        discovery = self.runtime.capabilities()
        self._ingest_discovered()
        self._capture_runtime()
        if dry_run:
            return self._finish(started, before, discovery, "dry_run")
        self._learn_if_due("startup")
        if once:
            return self._finish(started, before, discovery, "once")
        available, reason = self.chat.available()
        discovery["chat_runtime"] = {"available": available, **({"reason": reason} if reason else {})}
        if not available:
            self.failures.append({"stage": "chat_runtime", "reason": reason})
            return self._finish(started, before, discovery, "chat_unavailable")
        try:
            self.chat.start()
            last_cycle = time.monotonic()
            while self.chat.running():
                self._capture_runtime()
                if time.monotonic() - last_cycle >= self.policy.training_interval_seconds:
                    self._learn_if_due("interval")
                    last_cycle = time.monotonic()
                time.sleep(self.policy.poll_seconds)
        except KeyboardInterrupt:
            self.chat.stop()
        finally:
            self._capture_runtime()
            self._learn_if_due("shutdown")
        return self._finish(started, before, discovery, "complete")

    def _ingest_discovered(self):
        trace = self.runtime.root / "results" / "v679_chat_traces.jsonl"
        workers = self.runtime.root / "results" / "v679_workers"
        if trace.is_file():
            report = import_chat_traces(self.store, trace)
            self.chat_records += len(report["accepted"])
            self.failures.extend({"stage": "existing_chat", **item} for item in report["rejected"])
        if workers.is_dir():
            self.worker_events += import_worker_logs(self.store, workers)

    def _capture_runtime(self):
        events = self.chat.poll()
        for event in events["chat"]:
            if "invalid" in event:
                self.failures.append({"stage": "chat_capture", **event})
                continue
            try:
                import_chat_record(self.store, event["value"], event["source_path"], event["line"])
                self.chat_records += 1
            except ValueError as error:
                self.failures.append({"stage": "chat_capture", "line": event["line"], "reason": str(error)})
        for event in events["worker"]:
            if "invalid" in event:
                self.failures.append({"stage": "worker_capture", **event})
                continue
            if event["value"].get("event") == "analysis_batch":
                try:
                    import_worker_event(self.store, event["value"], event["source_path"], event["line"])
                    self.worker_events += 1
                except ValueError as error:
                    self.failures.append({"stage": "worker_capture", "line": event["line"], "reason": str(error)})

    def _learn_if_due(self, trigger):
        inspection = self._inspection()
        sequential = inspection["train_sequential_episodes"]
        if sequential < self.policy.min_sequential_episodes:
            self._bootstrap(trigger)
        inspection = self._inspection()
        if inspection["train_sequential_episodes"] >= self.policy.min_sequential_episodes:
            self._train_candidate(trigger)

    def _bootstrap(self, trigger):
        try:
            engine = self._engine()
            path = self.session_dir / "bootstrap_teacher.jsonl"
            engine.generate_teacher_records(path, self.policy.bootstrap_samples)
            records = _read_jsonl(path)
            _append_engine_records(self.store, records, ExperienceSource.ATTENTION_EVAL, "evaluation-", heldout_only=True)
            dagger = engine.run_dagger(path, self.session_dir / "dagger", 1, self.policy.epochs, self.policy.seed)
            _append_engine_records(self.store, _read_jsonl(dagger), ExperienceSource.DAGGER, "dagger-")
            _append_engine_records(self.store, records, ExperienceSource.SYNTHETIC_CHAT, "synthetic-")
            self.cycles.append({"kind": "bootstrap", "trigger": trigger, "status": "completed"})
        except Exception as error:
            self.failures.append({"stage": "bootstrap", "trigger": trigger, "reason": str(error)})

    def _train_candidate(self, trigger):
        try:
            engine, items = self._engine(), self.store.load()
            learner = AttentionDistillationLearner()
            sources = {ExperienceSource.DAGGER, ExperienceSource.SYNTHETIC_CHAT, ExperienceSource.CHAT_SEQUENTIAL}
            train, rejected = learner.prepare(items, sources)
            evaluation, rejected_eval = learner.prepare(items, {ExperienceSource.ATTENTION_EVAL}, allowed_splits=("heldout",))
            if not train or not evaluation:
                self.cycles.append({"kind": "attention", "trigger": trigger, "status": "deferred",
                                    "reason": "requires train and heldout sequential teacher-labelled experience"})
                return
            cycle_id = len(self.cycles) + 1
            dataset = self.session_dir / f"attention_{cycle_id}_train.jsonl"
            heldout = self.session_dir / f"attention_{cycle_id}_heldout.jsonl"
            _write_jsonl(dataset, train); _write_jsonl(heldout, evaluation)
            candidate = self.output / "models" / f"attention_{self.session_id}_{cycle_id}.pt"
            candidate.parent.mkdir(parents=True, exist_ok=True)
            learner.train(engine, dataset, candidate, self.policy.epochs, self.policy.seed)
            metrics_path = self.session_dir / f"attention_{cycle_id}_metrics.json"
            metrics = learner.evaluate(engine, heldout, candidate, metrics_path)
            jepa = JEPAAuxiliaryLearner()
            transitions, jepa_rejected = jepa.prepare(items, sources)
            jepa_data = self.session_dir / f"jepa_{cycle_id}.jsonl"
            _write_jsonl(jepa_data, transitions)
            jepa_result = jepa.train(engine, jepa_data, self.output / "models" / f"jepa_{self.session_id}_{cycle_id}.pt",
                                     self.session_dir / f"jepa_{cycle_id}_metrics.json", self.policy.epochs, self.policy.seed)
            provenance = {"v681_version": V681_VERSION, "session_id": self.session_id, "candidate": str(candidate),
                          "dataset": str(dataset), "sources": sorted(source.value for source in sources),
                          "metrics": metrics, "rejected": rejected, "jepa_rejected": jepa_rejected,
                          "promotion": {"promoted": False, "reason": "automatic promotion is disabled pending explicit safety criteria"}}
            provenance_path = candidate.with_suffix(".provenance.json")
            provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True))
            self.cycles.append({"kind": "candidate", "trigger": trigger, "status": "evaluated", "candidate": str(candidate),
                                "metrics": metrics, "jepa": jepa_result, "promotion": provenance["promotion"]})
        except Exception as error:
            self.failures.append({"stage": "learning", "trigger": trigger, "reason": str(error)})

    def _engine(self):
        if self.engine is None:
            self.engine = V680EngineAdapter(self.runtime.v680_root)
        return self.engine

    def _inspection(self):
        items = self.store.load()
        train = [item for item in items if item.split == "train" and item.sequence_capability == "sequential"]
        composition = {}
        for source in ExperienceSource:
            sourced = [item for item in items if item.source is source]
            composition[source.value] = {
                "records": len(sourced),
                "episodes": len({item.episode_id for item in sourced}),
                "qualities": {quality.value: sum(item.quality is quality for item in sourced)
                              for quality in type(items[0].quality)} if items else {},
                "actions": _action_counts(sourced),
            }
        return {"total_records": len(items), "chat_records": sum(item.source in {ExperienceSource.CHAT_DECISION_ONLY, ExperienceSource.CHAT_SEQUENTIAL} for item in items),
                "worker_records": sum(item.source is ExperienceSource.OFFLINE_WORKER for item in items),
                "decision_only_records": sum(item.sequence_capability == "decision_only" for item in items),
                "train_sequential_records": len(train), "train_sequential_episodes": len({item.episode_id for item in train}),
                "source_composition": composition}

    def _finish(self, started, before, discovery, status):
        result = {"v681_version": V681_VERSION, "session_id": self.session_id, "status": status,
                  "start_time": started, "end_time": time.time(), "experience_before": before,
                  "experience_after": self._inspection(), "training_cycles": self.cycles,
                  "models_created": [cycle["candidate"] for cycle in self.cycles if "candidate" in cycle],
                  "evaluations": [cycle for cycle in self.cycles if cycle["kind"] == "candidate"],
                  "worker_events": self.worker_events, "chat_episodes": self.chat_records,
                  "discovery": discovery, "failures": self.failures}
        (self.output / "v681_runtime_results.json").write_text(json.dumps(result, indent=2, sort_keys=True))
        (self.output / "v681_experience_manifest.json").write_text(json.dumps(
            {"v681_version": V681_VERSION, "inspection": result["experience_after"], "store": self.store.manifest()}, indent=2, sort_keys=True))
        (self.output / "v681_runtime_report.md").write_text(
            "# V681 runtime report\n\n"
            f"Session `{self.session_id}` finished with status `{status}`.\n\n"
            f"- chat episodes captured: {self.chat_records}\n- worker events captured: {self.worker_events}\n"
            f"- training cycles: {len(self.cycles)}\n- failures: {len(self.failures)}\n")
        (self.session_dir / "v681_session_manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True))
        (self.output / "v681_latest_results.json").write_text(json.dumps(result, indent=2, sort_keys=True))
        (self.output / "v681_latest_report.md").write_text(
            (self.output / "v681_runtime_report.md").read_text())
        self.store.close()
        return result


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path, records):
    Path(path).write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))


def _append_engine_records(store, records, source, prefix, heldout_only=False):
    for episode in records:
        for step in episode["trajectory"]:
            if heldout_only and not str(step["split"]).startswith("held_out"):
                continue
            item = attention_step_experience(step, source=source)
            item.episode_id = prefix + item.episode_id
            item.provenance["supervision_source"] = "frozen_v680_teacher"
            store.append(item)


def _action_counts(items):
    counts = {}
    for item in items:
        kind = item.model_view.get("selected_action", {}).get("kind", "none")
        counts[kind] = counts.get(kind, 0) + 1
    return counts
