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

ACTION_TO_ID = {action: i for i, action in enumerate(ACTIONS)}


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
            nodes=[Node(n.concept, n.activation, n.role, n.persistent) for n in self.nodes],
            edges=[Edge(e.source, e.relation, e.target, e.activation, e.persistent) for e in self.edges],
        )

    def node(self, concept: str) -> Node | None:
        return next((n for n in self.nodes if n.concept == concept), None)

    def add_node(self, concept: str, activation: float, role: int, persistent: bool = False) -> None:
        existing = self.node(concept)
        if existing is not None:
            existing.activation = max(existing.activation, activation)
            existing.persistent |= persistent
            return
        self.nodes.append(Node(concept, activation, role, persistent))

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        activation: float = 1.0,
        persistent: bool = False,
    ) -> None:
        for edge in self.edges:
            if edge.source == source and edge.relation == relation and edge.target == target:
                edge.activation = max(edge.activation, activation)
                edge.persistent |= persistent
                return
        self.edges.append(Edge(source, relation, target, activation, persistent))

    def has_edge(self, source: str, relation: str, target: str, active_only: bool = True) -> bool:
        return any(
            e.source == source
            and e.relation == relation
            and e.target == target
            and (not active_only or e.activation > 0.5)
            for e in self.edges
        )

    def signature(self) -> dict:
        return {
            "nodes": [vars(n) for n in self.nodes],
            "edges": [vars(e) for e in self.edges],
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
            state.add_node(f"created_{len(state.nodes)}", 0.85, 6)
            return state

        if action == "BRANCH":
            if source and relation:
                branch = f"{source}#branch{len(state.nodes)}"
                state.add_node(branch, 0.8, 7)
                state.add_edge(source, relation, branch, 0.8)
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
                    source_node.activation = max(source_node.activation, 1.0)
                if target_node is not None:
                    target_node.activation = max(target_node.activation, 1.0)
                state.add_edge(source, relation, target, 1.0)
            return state

        if action == "COMMIT":
            for node in state.nodes:
                if node.activation > 0.5:
                    node.persistent = True
            for edge in state.edges:
                if edge.activation > 0.5:
                    edge.persistent = True
            return state

        raise ValueError(f"Unknown action ID: {action_id}")
