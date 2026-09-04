"""Versioned, replayable learning experience envelope shared across V679 and V680."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


EXPERIENCE_VERSION = "v681-experience-1"


class ExperienceSource(str, Enum):
    CHAT = "chat"
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


@dataclass
class Experience:
    source: ExperienceSource
    episode_id: str
    timestamp: float = field(default_factory=time.time)
    split: str = "live"
    task: str = ""
    state: dict | None = None
    goal: dict | None = None
    candidate_actions: list[dict] = field(default_factory=list)
    selected_action: dict | None = None
    available_actions: list[dict] = field(default_factory=list)
    graph_context: dict = field(default_factory=dict)
    evidence_acquired: list[dict] = field(default_factory=list)
    next_state: dict | None = None
    teacher_action: dict | None = None
    outcome: dict | None = None
    reward: float | None = None
    confidence: float | None = None
    provenance: dict = field(default_factory=dict)
    quality: ExperienceQuality = ExperienceQuality.UNVERIFIED
    payload: dict = field(default_factory=dict)
    experience_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    version: str = EXPERIENCE_VERSION

    def __post_init__(self):
        self.source = ExperienceSource(self.source)
        self.quality = ExperienceQuality(self.quality)
        if self.split not in _SPLITS:
            raise ValueError(f"unknown experience split {self.split!r}")
        if not self.episode_id:
            raise ValueError("experience requires episode_id")
        if self.source is ExperienceSource.ATTENTION_EVAL and self.split == "train":
            raise ValueError("evaluation experience cannot enter the training split")
        if self.teacher_action is not None and self.outcome is not None:
            # Their coexistence is intentional; separate fields prevent target conflation.
            pass

    def as_dict(self):
        value = asdict(self)
        value["source"] = self.source.value
        value["quality"] = self.quality.value
        return value

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value))


class ExperienceStore:
    """SQLite append-only envelope store; learning readers explicitly exclude evaluation."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS experience (
               experience_id TEXT PRIMARY KEY, source TEXT NOT NULL, episode_id TEXT NOT NULL,
               timestamp REAL NOT NULL, split TEXT NOT NULL, task TEXT NOT NULL, quality TEXT NOT NULL,
               outcome_kind TEXT, payload TEXT NOT NULL)"""
        )
        self.connection.execute("CREATE INDEX IF NOT EXISTS experience_filter ON experience(source,split,task,quality)")
        self.connection.commit()

    def close(self):
        self.connection.close()

    def append(self, experience):
        experience = experience if isinstance(experience, Experience) else Experience.from_dict(experience)
        value = experience.as_dict()
        self.connection.execute(
            "INSERT OR IGNORE INTO experience VALUES(?,?,?,?,?,?,?,?,?)",
            (experience.experience_id, experience.source.value, experience.episode_id, experience.timestamp,
             experience.split, experience.task, experience.quality.value,
             (experience.outcome or {}).get("kind"), json.dumps(value, sort_keys=True)),
        )
        self.connection.commit()
        return experience.experience_id

    def load(self, source=None, task=None, outcome=None, quality=None, training_only=False):
        clauses, values = [], []
        for column, value in (("source", source.value if isinstance(source, Enum) else source), ("task", task),
                              ("outcome_kind", outcome), ("quality", quality.value if isinstance(quality, Enum) else quality)):
            if value is not None:
                clauses.append(f"{column}=?"); values.append(value)
        if training_only:
            clauses.append("split='train'")
        query = "SELECT payload FROM experience" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY timestamp,experience_id"
        return [Experience.from_dict(json.loads(row[0])) for row in self.connection.execute(query, values)]

    def sample_episodes(self, source=None, training_only=True, limit=None):
        grouped = {}
        for item in self.load(source=source, training_only=training_only):
            grouped.setdefault(item.episode_id, []).append(item)
        episodes = [sorted(items, key=lambda item: item.timestamp) for _, items in sorted(grouped.items())]
        return episodes[:limit] if limit else episodes

    def manifest(self):
        rows = self.connection.execute(
            "SELECT source,split,quality,COUNT(*) FROM experience GROUP BY source,split,quality ORDER BY source,split,quality"
        ).fetchall()
        return {"experience_version": EXPERIENCE_VERSION,
                "records": [{"source": row[0], "split": row[1], "quality": row[2], "count": row[3]} for row in rows]}


def attention_step_experience(step, source=ExperienceSource.DAGGER, quality=ExperienceQuality.TEACHER_LABELLED):
    """Adapt V680 serialized trace without making teacher or oracle fields model-visible."""
    state = step["state"]
    action = step["action"]
    candidates = list(step["candidates"])
    return Experience(
        source=source, episode_id=str(step["episode_id"]), split=_split(step["split"]),
        task="attention", state=state, goal={"relation": state["goal_relation"], "terms": state["goal_terms"]},
        candidate_actions=candidates, selected_action=action,
        available_actions=[*candidates, {"kind": "stop"}, {"kind": "abstain"}],
        next_state=step["next_state"], teacher_action={"selected_action": step["teacher"]["selected_action"]},
        outcome={"kind": _attention_outcome(step["terminal_outcome"]), "verified": bool(step["oracle"].get("valid_proof_edge"))},
        reward=float(step["reward"]), confidence=None, quality=quality,
        provenance={"producer": "v680", "split": step["split"], "teacher_version": step.get("teacher_version", "")},
        payload={"attention_step": step},
    )


def chat_trace_experience(trace, source=ExperienceSource.CHAT):
    route = dict(trace.get("route", {}))
    verified = bool(route.get("success"))
    return Experience(
        source=source, episode_id=f"chat-{int(trace.get('timestamp', time.time()) * 1_000_000)}",
        split="live", task="chat_attention",
        goal={"relation": route.get("relation"), "subject": route.get("subject"), "intent": route.get("intent")},
        candidate_actions=list(trace.get("candidate_evidence", [])),
        selected_action={"kind": "traverse", "candidate_id": trace.get("semantic_decision", {}).get("selected_candidate_index")}
                        if trace.get("semantic_decision", {}).get("selected_candidate_index") is not None else {"kind": "abstain"},
        available_actions=[*trace.get("candidate_evidence", []), {"kind": "stop"}, {"kind": "abstain"}],
        graph_context={"route": route, "search": trace.get("search", {})},
        evidence_acquired=[{"path": route.get("path", []), "target": route.get("target"), "direct": route.get("direct_proof", False)}],
        outcome={"kind": "verified_answer" if verified else "insufficient_evidence", "verified": verified},
        confidence=1.0 if verified else 0.0, quality=ExperienceQuality.VERIFIED if verified else ExperienceQuality.UNVERIFIED,
        provenance={"producer": "v679_semantic_chat_gateway", "trace_timestamp": trace.get("timestamp")},
        payload={"chat_trace": trace},
    )


def worker_batch_experience(event):
    return Experience(
        source=ExperienceSource.OFFLINE_WORKER,
        episode_id=f"worker-{event.get('worker_id', 'unknown')}-batch-{event.get('batch', 0)}",
        task="offline_graph_learning", graph_context={"lane": event.get("lane"), "ram": event.get("ram", {})},
        evidence_acquired=[{"lane": event.get("lane"), "learned": event.get("learned", 0),
                            "new_results": event.get("new_results", 0), "provenance": event.get("provenance", {})}],
        confidence=None, quality=ExperienceQuality.GROUNDED,
        provenance={"producer": "v679_offline_learning", "worker_id": event.get("worker_id")},
        payload={"worker_event": event},
    )


def _split(value):
    return "heldout" if str(value).startswith("held_out") else ("validation" if value == "validation" else "train")


def _attention_outcome(value):
    return "verified_answer" if value == "success" else ("verified_no_proof" if value == "abstain" else "unknown")
