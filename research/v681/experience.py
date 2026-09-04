"""Canonical V681 experience contract: model view, supervision, diagnostics."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

EXPERIENCE_VERSION = "v681.3-experience-1"


class ExperienceSource(str, Enum):
    CHAT = "chat"
    CHAT_SEQUENTIAL = "chat_sequential"
    CHAT_DECISION_ONLY = "chat_decision_only"
    SYNTHETIC_CHAT = "synthetic_chat"
    OFFLINE_WORKER = "offline_worker"
    DAGGER = "dagger"
    ATTENTION_EVAL = "attention_eval"
    JEPA = "jepa"
    OTHER = "other"


class ExperienceQuality(str, Enum):
    VERIFIED = "verified"
    TEACHER_LABELLED = "teacher_labelled"
    GROUNDED = "grounded"
    USER_CONFIRMED = "user_confirmed"
    UNVERIFIED = "unverified"


_SPLITS = {"train", "validation", "heldout", "live"}
_QUALITY_ORDER = {value: index for index, value in enumerate(ExperienceQuality)}
_FORBIDDEN_MODEL_FIELDS = {"teacher", "teacher_action", "oracle", "ground_truth", "reward",
                           "future_state", "future_reward", "terminal_outcome", "terminal_answer"}


@dataclass
class Experience:
    source: ExperienceSource
    episode_id: str
    model_view: dict = field(default_factory=dict)
    supervision: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    split: str = "live"
    task: str = ""
    quality: ExperienceQuality = ExperienceQuality.UNVERIFIED
    provenance: dict = field(default_factory=dict)
    experience_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    version: str = EXPERIENCE_VERSION

    def __post_init__(self):
        self.source = ExperienceSource(self.source)
        self.quality = ExperienceQuality(self.quality)
        if self.split not in _SPLITS or not self.episode_id:
            raise ValueError("experience requires a known split and episode_id")
        if self.source is ExperienceSource.ATTENTION_EVAL and self.split == "train":
            raise ValueError("evaluation experience cannot enter the training split")
        if not isinstance(self.model_view, dict) or not isinstance(self.supervision, dict):
            raise ValueError("model_view and supervision must be objects")
        _audit_model_view(self.model_view)

    def as_dict(self):
        item = asdict(self)
        item["source"], item["quality"] = self.source.value, self.quality.value
        return item

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value))

    @property
    def sequence_capability(self):
        return self.model_view.get("sequence_capability", "none")


class ExperienceStore:
    """Append-only full-record SQLite persistence; every replay reads canonical JSON."""
    def __init__(self, path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS experience_v681 (
               experience_id TEXT PRIMARY KEY, source TEXT NOT NULL, episode_id TEXT NOT NULL,
               timestamp REAL NOT NULL, split TEXT NOT NULL, task TEXT NOT NULL, quality TEXT NOT NULL,
               sequence_capability TEXT NOT NULL, outcome_kind TEXT, record TEXT NOT NULL)"""
        )
        self.connection.execute("CREATE INDEX IF NOT EXISTS experience_v681_filter ON experience_v681(source,split,task,quality)")
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS learning_cursor_v681 (
               learner TEXT PRIMARY KEY, last_experience_id TEXT NOT NULL,
               dataset_path TEXT NOT NULL, artifact_path TEXT NOT NULL, updated_at REAL NOT NULL)"""
        )
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS learning_failure_v681 (
               learner TEXT NOT NULL, dataset_version TEXT NOT NULL, failure TEXT NOT NULL,
               timestamp REAL NOT NULL, PRIMARY KEY(learner,dataset_version))"""
        )
        self.connection.commit()

    def close(self): self.connection.close()

    def append(self, experience):
        item = experience if isinstance(experience, Experience) else Experience.from_dict(experience)
        self.connection.execute(
            "INSERT OR IGNORE INTO experience_v681 VALUES(?,?,?,?,?,?,?,?,?,?)",
            (item.experience_id, item.source.value, item.episode_id, item.timestamp, item.split, item.task,
             item.quality.value, item.sequence_capability, item.supervision.get("outcome", {}).get("kind"),
             json.dumps(item.as_dict(), sort_keys=True)),
        )
        self.connection.commit()
        return item.experience_id

    def load(self, source=None, task=None, outcome=None, quality=None, training_only=False,
             allowed_splits=None, min_quality=None):
        clauses, values = [], []
        filters = (("source", source.value if isinstance(source, Enum) else source), ("task", task),
                   ("outcome_kind", outcome), ("quality", quality.value if isinstance(quality, Enum) else quality))
        for column, value in filters:
            if value is not None: clauses.append(f"{column}=?"); values.append(value)
        if training_only: clauses.append("split='train'")
        if allowed_splits:
            clauses.append("split IN (%s)" % ",".join("?" * len(allowed_splits))); values.extend(allowed_splits)
        rows = self.connection.execute("SELECT record FROM experience_v681" +
            (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY timestamp,experience_id", values)
        items = [Experience.from_dict(json.loads(row[0])) for row in rows]
        if min_quality is not None:
            items = [item for item in items if _QUALITY_ORDER[item.quality] <= _QUALITY_ORDER[ExperienceQuality(min_quality)]]
        return items

    def sample_episodes(self, **filters):
        groups = {}
        for item in self.load(**filters):
            groups.setdefault(item.episode_id, []).append(item)
        return [sorted(items, key=lambda item: item.timestamp) for _, items in sorted(groups.items())]

    def manifest(self):
        rows = self.connection.execute(
            "SELECT source,split,quality,sequence_capability,COUNT(*) FROM experience_v681 "
            "GROUP BY source,split,quality,sequence_capability ORDER BY source,split,quality"
        ).fetchall()
        return {"experience_version": EXPERIENCE_VERSION, "records": [
            {"source": row[0], "split": row[1], "quality": row[2], "sequence_capability": row[3], "count": row[4]}
            for row in rows]}

    def learning_cursor(self, learner):
        row = self.connection.execute(
            "SELECT last_experience_id,dataset_path,artifact_path,updated_at FROM learning_cursor_v681 WHERE learner=?",
            (learner,)).fetchone()
        return dict(zip(("last_experience_id", "dataset_path", "artifact_path", "updated_at"), row)) if row else None

    def latest_experience_id(self):
        row = self.connection.execute(
            "SELECT experience_id FROM experience_v681 ORDER BY timestamp DESC,experience_id DESC LIMIT 1").fetchone()
        return row[0] if row else ""

    def update_learning_cursor(self, learner, experience_id, dataset_path, artifact_path):
        self.connection.execute(
            """INSERT INTO learning_cursor_v681 VALUES(?,?,?,?,?)
               ON CONFLICT(learner) DO UPDATE SET last_experience_id=excluded.last_experience_id,
                 dataset_path=excluded.dataset_path,artifact_path=excluded.artifact_path,updated_at=excluded.updated_at""",
            (learner, experience_id, dataset_path, artifact_path, time.time()))
        self.connection.commit()

    def learning_failure(self, learner, dataset_version):
        row = self.connection.execute(
            "SELECT failure,timestamp FROM learning_failure_v681 WHERE learner=? AND dataset_version=?",
            (learner, dataset_version)).fetchone()
        return dict(zip(("failure", "timestamp"), row)) if row else None

    def record_learning_failure(self, learner, dataset_version, failure):
        self.connection.execute(
            """INSERT OR REPLACE INTO learning_failure_v681 VALUES(?,?,?,?)""",
            (learner, dataset_version, str(failure), time.time()))
        self.connection.commit()


def attention_step_experience(step, source=ExperienceSource.DAGGER, quality=ExperienceQuality.TEACHER_LABELLED):
    """Translate V680 records into canonical V681 sections; the raw step is diagnostics only."""
    state = step["state"]
    return Experience(
        source=source, episode_id=str(step["episode_id"]), split=_split(step["split"]), task="attention",
        model_view={"sequence_capability": "sequential", "state": state, "goal": {
            "relation": state["goal_relation"], "terms": state["goal_terms"]}, "candidate_actions": step["candidates"],
            "selected_action": step["action"], "available_actions": [*step["candidates"], {"kind": "stop"}, {"kind": "abstain"}],
            "graph_context": {"graph_version": step.get("graph_version", "unknown")},
            "evidence_acquired": [], "next_state": step["next_state"]},
        supervision={"teacher": dict(step["teacher"]), "outcome": {
            "kind": _attention_outcome(step["terminal_outcome"]), "verified": bool(step["oracle"].get("valid_proof_edge"))},
            "reward": float(step["reward"])},
        diagnostics={"raw_v680_step": step, "oracle": dict(step["oracle"])},
        quality=quality, provenance={"producer": "v680", "graph_version": step.get("graph_version", "unknown"),
                                      "teacher_version": step.get("teacher_version", "unknown"),
                                      "dataset_version": step.get("dataset_version", "unknown")},
    )


def chat_trace_experience(trace, source=ExperienceSource.CHAT_DECISION_ONLY):
    """Sanitize V679 traces: they are decision-only unless an explicit sequential trace exists."""
    route = dict(trace.get("route", {})); verified = bool(route.get("success"))
    return Experience(
        source=source, episode_id=f"chat-{int(trace.get('timestamp', time.time()) * 1_000_000)}", task="chat_attention",
        model_view={"sequence_capability": "decision_only", "goal": {
            "relation": route.get("relation"), "subject": route.get("subject"), "intent": route.get("intent")},
            "candidate_actions": list(trace.get("candidate_evidence", [])), "selected_action": {
                "kind": "traverse", "candidate_id": trace.get("semantic_decision", {}).get("selected_candidate_index")}
                if trace.get("semantic_decision", {}).get("selected_candidate_index") is not None else {"kind": "abstain"},
            "available_actions": [*trace.get("candidate_evidence", []), {"kind": "stop"}, {"kind": "abstain"}],
            "graph_context": {"graph_version": trace.get("graph_version", "unknown")}, "evidence_acquired": [
                {"path": route.get("path", []), "target": route.get("target"), "direct": route.get("direct_proof", False)}]},
        supervision={"outcome": {"kind": "verified_answer" if verified else "insufficient_evidence", "verified": verified},
                     "confidence": 1.0 if verified else 0.0},
        diagnostics={"raw_trace": trace}, quality=ExperienceQuality.VERIFIED if verified else ExperienceQuality.UNVERIFIED,
        provenance={"producer": "v679_trace_import", "graph_version": trace.get("graph_version", "unknown"),
                    "session_id": trace.get("v681_session_id", "unknown"),
                    "policy_model_version": trace.get("attention_controller", {}).get("policy_model_version", "fallback")},
    )


def worker_batch_experience(event):
    """Telemetry/knowledge evidence only: never an attention label."""
    return Experience(
        source=ExperienceSource.OFFLINE_WORKER,
        episode_id=f"worker-{event.get('worker_id', 'unknown')}-batch-{event.get('batch', 0)}",
        task="offline_graph_learning",
        model_view={"sequence_capability": "knowledge_only", "graph_context": {
            "graph_version": event.get("graph_version", "unknown"),
            "before_graph_version": event.get("before_graph_version", "unknown"),
            "after_graph_version": event.get("after_graph_version", "unknown")},
            "knowledge_event": {"lane": event.get("lane"), "batch": event.get("batch"),
                                "learned": event.get("learned"), "new_results": event.get("new_results"),
                                "telemetry_only": True},
            "knowledge_delta": None},
        supervision={}, diagnostics={"raw_worker_event": event},
        quality=ExperienceQuality.GROUNDED, provenance={"producer": "v679_worker_log",
            "worker_id": event.get("worker_id"), "worker_version": event.get("worker_version", "unknown"),
            "batch": event.get("batch"), "graph_version": event.get("graph_version", "unknown")},
    )


def _split(value):
    return "heldout" if str(value).startswith("held_out") else ("validation" if value == "validation" else "train")


def _attention_outcome(value):
    return "verified_answer" if value == "success" else ("verified_no_proof" if value == "abstain" else "unknown")


def _audit_model_view(value):
    if isinstance(value, dict):
        forbidden = _FORBIDDEN_MODEL_FIELDS & value.keys()
        if forbidden:
            raise ValueError(f"model_view contains supervision/diagnostic fields: {sorted(forbidden)}")
        for child in value.values():
            _audit_model_view(child)
    elif isinstance(value, list):
        for child in value:
            _audit_model_view(child)
