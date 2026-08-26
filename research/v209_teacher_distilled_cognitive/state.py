from __future__ import annotations

from dataclasses import dataclass


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
    action: i
    for i, action in enumerate(ACTIONS)
}


@dataclass
class Node:
    concept: str
    activation: float
    role: int
    persistent: bool = False


@dataclass
class Edge:
    source: str
    relation: str
    target: str
    activation: float
    persistent: bool = False


@dataclass
class State:
    nodes: list[Node]
    edges: list[Edge]

    def clone(self) -> "State":
        return State(
            nodes=[
                Node(
                    concept=n.concept,
                    activation=n.activation,
                    role=n.role,
                    persistent=n.persistent,
                )
                for n in self.nodes
            ],
            edges=[
                Edge(
                    source=e.source,
                    relation=e.relation,
                    target=e.target,
                    activation=e.activation,
                    persistent=e.persistent,
                )
                for e in self.edges
            ],
        )

    def node(self, concept: str) -> Node | None:
        for node in self.nodes:
            if node.concept == concept:
                return node
        return None

    def add_node(
        self,
        concept: str,
        activation: float,
        role: int,
        persistent: bool = False,
    ) -> None:
        existing = self.node(concept)
        if existing is not None:
            existing.activation = max(
                existing.activation,
                activation,
            )
            existing.persistent |= persistent
            return

        self.nodes.append(
            Node(
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
            Edge(
                source=source,
                relation=relation,
                target=target,
                activation=activation,
                persistent=persistent,
            )
        )

    def has_edge(
        self,
        source: str,
        relation: str,
        target: str,
        active_only: bool = True,
    ) -> bool:
        return any(
            edge.source == source
            and edge.relation == relation
            and edge.target == target
            and (
                not active_only
                or edge.activation > 0.5
            )
            for edge in self.edges
        )

    def signature(self) -> dict:
        return {
            "nodes": [
                {
                    "concept": node.concept,
                    "activation": node.activation,
                    "role": node.role,
                    "persistent": node.persistent,
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "relation": edge.relation,
                    "target": edge.target,
                    "activation": edge.activation,
                    "persistent": edge.persistent,
                }
                for edge in self.edges
            ],
        }

    def apply(
        self,
        action_id: int,
        *,
        source: str | None = None,
        target: str | None = None,
        relation: str | None = None,
    ) -> "State":
        state = self.clone()
        action = ACTIONS[action_id]

        if action == "NOOP":
            return state

        if action == "REUSE":
            if target:
                node = state.node(target)
                if node is not None:
                    node.activation = 1.0
            return state

        if action == "CREATE":
            state.add_node(
                f"created_{len(state.nodes)}",
                activation=0.85,
                role=6,
            )
            return state

        if action == "BRANCH":
            if source and relation:
                branch = (
                    f"{source}#branch{len(state.nodes)}"
                )
                state.add_node(
                    branch,
                    activation=0.8,
                    role=7,
                )
                state.add_edge(
                    source,
                    relation,
                    branch,
                    activation=0.8,
                )
            return state

        if action == "INHIBIT":
            if target:
                node = state.node(target)
                if node is not None:
                    node.activation *= 0.05
            return state

        if action == "BIND":
            if source and target and relation:
                source_node = state.node(source)
                target_node = state.node(target)

                if source_node is not None:
                    source_node.activation = max(
                        source_node.activation,
                        1.0,
                    )

                if target_node is not None:
                    target_node.activation = max(
                        target_node.activation,
                        1.0,
                    )

                state.add_edge(
                    source,
                    relation,
                    target,
                    activation=1.0,
                )
            return state

        if action == "COMMIT":
            for node in state.nodes:
                if node.activation > 0.5:
                    node.persistent = True
            for edge in state.edges:
                if edge.activation > 0.5:
                    edge.persistent = True
            return state

        raise ValueError(
            f"Unknown action ID: {action_id}"
        )
