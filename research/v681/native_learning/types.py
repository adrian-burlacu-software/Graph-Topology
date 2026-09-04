# V681-owned learning implementation; derived from V680.
"""One validated, serializable schema for every V680 policy record."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class AttentionActionKind(str, Enum):
    TRAVERSE = "traverse"
    STOP = "stop"
    ABSTAIN = "abstain"


CANDIDATE_VECTOR_FIELDS = (
    "path_length", "goal_relation_match", "target_term_match", "specificity",
    "lexical_score", "relation_activation", "candidate_activation",
    "provenance", "verified", "contradiction", "direct_proof", "already_visited",
)
TEACHER_VERSION = "v679-frozen-attention-teacher-1"
DATASET_VERSION = "v681.7-stop-boundary-2"
STUDENT_VERSION = "v680.1-attention-policy-1"
JEPA_VERSION = "v680.1-action-conditioned-jepa-1"
FORBIDDEN_MODEL_FIELDS = {
    "teacher", "teacher_action", "proof_target", "proof_exists", "oracle", "valid_paths",
    "valid_proof", "ground_truth_answer", "future_state", "next_state", "future_reward",
    "reward", "terminal_outcome", "terminal_answer",
}


@dataclass(frozen=True)
class CandidateFeatures:
    relation: str
    target: str
    path_length: int = 1
    goal_relation_match: float = 0.0
    target_term_match: float = 0.0
    specificity: float = 0.0
    lexical_score: float = 0.0
    relation_activation: float = 0.0
    candidate_activation: float = 0.0
    provenance: float = 0.0
    verified: float = 0.0
    contradiction: float = 0.0
    direct_proof: float = 0.0
    already_visited: float = 0.0

    def vector(self):
        return [float(getattr(self, name)) for name in CANDIDATE_VECTOR_FIELDS]

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("candidate features must be an object")
        missing = {"relation", "target"} - value.keys()
        if missing:
            raise ValueError(f"candidate features missing {sorted(missing)}")
        leaked = FORBIDDEN_MODEL_FIELDS & value.keys()
        if leaked:
            raise ValueError(f"oracle fields must not appear in candidate features: {sorted(leaked)}")
        return cls(**{key: value[key] for key in cls.__dataclass_fields__ if key in value})


@dataclass(frozen=True)
class AttentionAction:
    kind: AttentionActionKind
    candidate_id: int | None = None

    def index(self, candidate_count):
        if self.kind is AttentionActionKind.TRAVERSE:
            if self.candidate_id is None or not 0 <= self.candidate_id < candidate_count:
                raise ValueError("traverse action requires an available candidate")
            return self.candidate_id
        return candidate_count + (0 if self.kind is AttentionActionKind.STOP else 1)

    def as_dict(self):
        return {"kind": self.kind.value, "candidate_id": self.candidate_id}

    @classmethod
    def from_dict(cls, value, candidate_count):
        if not isinstance(value, dict) or "kind" not in value:
            raise ValueError("action must contain kind")
        try:
            action = cls(AttentionActionKind(value["kind"]), value.get("candidate_id"))
        except ValueError as exc:
            raise ValueError(f"unknown attention action: {value.get('kind')!r}") from exc
        action.index(candidate_count)
        return action


@dataclass
class AttentionObservation:
    goal_relation: str
    goal_terms: list[str]
    current_focus: str
    current_node: str
    relation_features: dict[str, float]
    candidate_features: list[CandidateFeatures]
    relation_activation: dict[str, float]
    candidate_activation: dict[str, float]
    visited_nodes: list[str]
    visited_relations: list[str]
    attention_history: list[dict[str, Any]]
    step: int
    remaining_budget: int

    def state_vector(self, recurrent=True):
        return [
            float(self.step), float(self.remaining_budget),
            float(len(self.candidate_features)), float(len(self.visited_nodes)) if recurrent else 0.0,
            float(len(self.visited_relations)) if recurrent else 0.0,
            self.relation_activation.get(self.goal_relation, 0.0),
        ]

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("attention state must be an object")
        required = set(cls.__dataclass_fields__) - {"relation_features"}
        missing = required - value.keys()
        if missing:
            raise ValueError(f"attention state missing {sorted(missing)}")
        leaked = FORBIDDEN_MODEL_FIELDS & value.keys()
        if leaked:
            raise ValueError(f"oracle fields must not appear in observations: {sorted(leaked)}")
        return cls(
            **{key: value.get(key, {}) if key == "relation_features" else value[key]
               for key in cls.__dataclass_fields__ if key != "candidate_features"},
            candidate_features=[CandidateFeatures.from_dict(item) for item in value["candidate_features"]],
        )


def validate_step_record(record):
    required = {"episode_id", "split", "step", "state", "candidates", "teacher",
                "action", "next_state", "reward", "terminal_outcome", "oracle", "provenance"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"trajectory record missing {sorted(missing)}")
    state = AttentionObservation.from_dict(record["state"])
    action = AttentionAction.from_dict(record["action"], len(state.candidate_features))
    teacher = record["teacher"]
    action_count = len(state.candidate_features) + 2
    if len(teacher.get("logits", [])) != action_count or len(teacher.get("probabilities", [])) != action_count:
        raise ValueError("teacher logits and probabilities must match bounded action count")
    selected = teacher.get("selected_action")
    if not isinstance(selected, int) or not 0 <= selected < action_count:
        raise ValueError("teacher selected_action must be an available bounded action index")
    AttentionObservation.from_dict(record["next_state"])
    return record


def validate_jepa_transition_record(record):
    """Validate a JEPA-only transition without accepting teacher or oracle data."""
    required = {"episode_id", "step", "state", "action", "next_state", "provenance"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"JEPA transition missing {sorted(missing)}")
    forbidden = {"teacher", "oracle", "reward", "terminal_outcome", "terminal_answer"} & record.keys()
    if forbidden:
        raise ValueError(f"JEPA transition contains forbidden fields: {sorted(forbidden)}")
    state = AttentionObservation.from_dict(record["state"])
    AttentionAction.from_dict(record["action"], len(state.candidate_features))
    AttentionObservation.from_dict(record["next_state"])
    return record


def audit_model_input(value):
    """Fail closed if any recursively supplied model-visible data contains oracle metadata."""
    if isinstance(value, dict):
        leaked = FORBIDDEN_MODEL_FIELDS & value.keys()
        if leaked:
            raise ValueError(f"forbidden oracle/future fields in model input: {sorted(leaked)}")
        for item in value.values():
            audit_model_input(item)
    elif isinstance(value, list):
        for item in value:
            audit_model_input(item)
    return value
