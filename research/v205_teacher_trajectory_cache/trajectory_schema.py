from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


ACTIONS = (
    "NOOP",
    "REUSE",
    "CREATE",
    "BRANCH",
    "INHIBIT",
    "BIND",
    "COMMIT",
)

ACTION_TO_ID = {
    name: i
    for i, name in enumerate(ACTIONS)
}


@dataclass(frozen=True)
class CandidateAction:
    candidate_id: int
    action: str
    source: str | None = None
    target: str | None = None
    relation: str | None = None
    rationale: str = ""


@dataclass
class WorkingNode:
    concept: str
    activation: float
    role: int
    persistent: bool = False


@dataclass
class WorkingEdge:
    source: str
    relation: str
    target: str
    activation: float
    persistent: bool = False


@dataclass
class WorkingState:
    nodes: list[WorkingNode]
    edges: list[WorkingEdge]

    def clone(self) -> "WorkingState":
        return WorkingState(
            nodes=[
                WorkingNode(
                    concept=n.concept,
                    activation=n.activation,
                    role=n.role,
                    persistent=n.persistent,
                )
                for n in self.nodes
            ],
            edges=[
                WorkingEdge(
                    source=e.source,
                    relation=e.relation,
                    target=e.target,
                    activation=e.activation,
                    persistent=e.persistent,
                )
                for e in self.edges
            ],
        )

    def node(
        self,
        concept: str,
    ) -> WorkingNode | None:
        for node in self.nodes:
            if node.concept == concept:
                return node
        return None

    def has_edge(
        self,
        source: str,
        relation: str,
        target: str,
        *,
        active_only: bool = False,
    ) -> bool:
        for edge in self.edges:
            if (
                edge.source == source
                and edge.relation == relation
                and edge.target == target
                and (
                    not active_only
                    or edge.activation > 0.5
                )
            ):
                return True
        return False

    def add_node(
        self,
        concept: str,
        *,
        activation: float = 0.0,
        role: int = 0,
        persistent: bool = False,
    ) -> None:
        existing = self.node(concept)

        if existing is not None:
            existing.activation = max(
                existing.activation,
                activation,
            )
            existing.persistent |= persistent
            if role:
                existing.role = role
            return

        self.nodes.append(
            WorkingNode(
                concept=concept,
                activation=activation,
                role=role,
                persistent=persistent,
            )
        )

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        *,
        activation: float = 1.0,
        persistent: bool = False,
    ) -> None:
        for edge in self.edges:
            if (
                edge.source == source
                and edge.relation == relation
                and edge.target == target
            ):
                edge.activation = max(
                    edge.activation,
                    activation,
                )
                edge.persistent |= persistent
                return

        self.edges.append(
            WorkingEdge(
                source=source,
                relation=relation,
                target=target,
                activation=activation,
                persistent=persistent,
            )
        )

    def signature(self) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "concept": n.concept,
                    "activation": round(
                        n.activation,
                        4,
                    ),
                    "role": n.role,
                    "persistent": n.persistent,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "source": e.source,
                    "relation": e.relation,
                    "target": e.target,
                    "activation": round(
                        e.activation,
                        4,
                    ),
                    "persistent": e.persistent,
                }
                for e in self.edges
            ],
        }


def candidate_to_dict(
    candidate: CandidateAction,
) -> dict[str, Any]:
    return asdict(
        candidate
    )
