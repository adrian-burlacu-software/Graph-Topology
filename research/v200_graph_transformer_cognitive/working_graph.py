from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorkingNode:
    concept: str
    node_type: int = 0
    activation: float = 1.0
    role: int = 0
    persistent: bool = False
    source: str = "working"


@dataclass
class WorkingEdge:
    source: int
    relation_id: int
    target: int
    weight: float = 1.0
    activation: float = 1.0
    persistent: bool = False


@dataclass
class WorkingGraph:
    """Temporary cognitive graph manipulated by the designer."""

    nodes: list[WorkingNode] = field(default_factory=list)
    edges: list[WorkingEdge] = field(default_factory=list)

    def add_node(
        self,
        concept: str,
        *,
        node_type: int = 0,
        activation: float = 1.0,
        role: int = 0,
        persistent: bool = False,
        source: str = "working",
    ) -> int:
        for index, node in enumerate(self.nodes):
            if node.concept == concept:
                node.activation = max(
                    node.activation,
                    activation,
                )
                node.role = role if role else node.role
                return index

        self.nodes.append(
            WorkingNode(
                concept=concept,
                node_type=node_type,
                activation=activation,
                role=role,
                persistent=persistent,
                source=source,
            )
        )
        return len(self.nodes) - 1

    def add_edge(
        self,
        source: int,
        relation_id: int,
        target: int,
        *,
        weight: float = 1.0,
        activation: float = 1.0,
        persistent: bool = False,
    ) -> None:
        if any(
            edge.source == source
            and edge.relation_id == relation_id
            and edge.target == target
            for edge in self.edges
        ):
            return

        self.edges.append(
            WorkingEdge(
                source=source,
                relation_id=relation_id,
                target=target,
                weight=weight,
                activation=activation,
                persistent=persistent,
            )
        )

    def inhibit_node(self, node_id: int) -> None:
        if 0 <= node_id < len(self.nodes):
            self.nodes[node_id].activation = 0.0

    def clear_transient(self) -> None:
        keep_nodes = [
            node
            for node in self.nodes
            if node.persistent
        ]

        old_to_new: dict[int, int] = {}

        for old_index, node in enumerate(self.nodes):
            if node.persistent:
                old_to_new[old_index] = len(
                    old_to_new
                )

        keep_edges = []
        for edge in self.edges:
            if (
                edge.persistent
                and edge.source in old_to_new
                and edge.target in old_to_new
            ):
                keep_edges.append(
                    (
                        old_to_new[edge.source],
                        edge.relation_id,
                        old_to_new[edge.target],
                        edge.weight,
                        edge.activation,
                    )
                )

        self.nodes = keep_nodes
        self.edges = [
            WorkingEdge(
                source=source,
                relation_id=relation_id,
                target=target,
                weight=weight,
                activation=activation,
                persistent=True,
            )
            for (
                source,
                relation_id,
                target,
                weight,
                activation,
            ) in keep_edges
        ]

    def edge_index(self) -> list[tuple[int, int]]:
        return [
            (
                edge.source,
                edge.target,
            )
            for edge in self.edges
        ]

    def stats(self) -> dict[str, int]:
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
        }
