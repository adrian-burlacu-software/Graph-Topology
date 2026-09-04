"""Serializable bounded-action representations for the V680 experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class AttentionActionKind(str, Enum):
    TRAVERSE = "traverse"
    STOP = "stop"
    ABSTAIN = "abstain"


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
        return [
            self.path_length, self.goal_relation_match, self.target_term_match,
            self.specificity, self.lexical_score, self.relation_activation,
            self.candidate_activation, self.provenance, self.verified,
            self.contradiction, self.direct_proof, self.already_visited,
        ]


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
    step: int
    remaining_budget: int

    def state_vector(self):
        return [
            float(self.step), float(self.remaining_budget),
            float(len(self.candidate_features)), float(len(self.visited_nodes)),
            float(len(self.visited_relations)),
            self.relation_activation.get(self.goal_relation, 0.0),
        ]

    def as_dict(self):
        return asdict(self)


@dataclass
class AttentionEpisode:
    episode_id: str
    split: str
    observations: list[AttentionObservation]
    actions: list[AttentionAction]
    rewards: list[float] = field(default_factory=list)
    terminal_outcome: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self):
        return {
            **asdict(self),
            "actions": [
                {"kind": action.kind.value, "candidate_id": action.candidate_id}
                for action in self.actions
            ],
        }
