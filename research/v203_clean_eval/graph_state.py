from __future__ import annotations

from dataclasses import dataclass, field


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
    name: index
    for index, name in enumerate(ACTIONS)
}


@dataclass
class Node:
    concept: str
    role: int = 0
    activation: float = 0.0
    persistent: bool = False


@dataclass
class Edge:
    source: int
    relation_id: int
    target: int
    activation: float = 0.0
    persistent: bool = False


@dataclass
class GraphState:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def clone(self) -> "GraphState":
        return GraphState(
            nodes=[
                Node(
                    concept=n.concept,
                    role=n.role,
                    activation=n.activation,
                    persistent=n.persistent,
                )
                for n in self.nodes
            ],
            edges=[
                Edge(
                    source=e.source,
                    relation_id=e.relation_id,
                    target=e.target,
                    activation=e.activation,
                    persistent=e.persistent,
                )
                for e in self.edges
            ],
        )

    def node_id(self, concept: str) -> int:
        for i, node in enumerate(self.nodes):
            if node.concept == concept:
                return i
        return -1

    def add_node(
        self,
        concept: str,
        *,
        role: int = 0,
        activation: float = 0.0,
        persistent: bool = False,
    ) -> int:
        existing = self.node_id(concept)
        if existing >= 0:
            self.nodes[existing].activation = max(
                self.nodes[existing].activation,
                activation,
            )
            if role:
                self.nodes[existing].role = role
            if persistent:
                self.nodes[existing].persistent = True
            return existing

        self.nodes.append(
            Node(
                concept=concept,
                role=role,
                activation=activation,
                persistent=persistent,
            )
        )
        return len(self.nodes) - 1

    def add_edge(
        self,
        source: int,
        relation_id: int,
        target: int,
        *,
        activation: float = 1.0,
        persistent: bool = False,
    ) -> None:
        for edge in self.edges:
            if (
                edge.source == source
                and edge.relation_id == relation_id
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
                relation_id=relation_id,
                target=target,
                activation=activation,
                persistent=persistent,
            )
        )

    def has_edge(
        self,
        source: int,
        relation_id: int,
        target: int,
        *,
        active_only: bool = False,
    ) -> bool:
        return any(
            e.source == source
            and e.relation_id == relation_id
            and e.target == target
            and (
                not active_only
                or e.activation > 0.5
            )
            for e in self.edges
        )

    def apply(
        self,
        action_id: int,
        *,
        source: int = -1,
        target: int = -1,
        relation_id: int = -1,
    ) -> "GraphState":
        """
        Generic environment transition.

        The controller predicts an action and arguments.
        The environment applies them deterministically.
        """
        state = self.clone()
        action = ACTIONS[action_id]

        if action == "NOOP":
            return state

        if action == "REUSE":
            if 0 <= target < len(state.nodes):
                state.nodes[target].activation = max(
                    state.nodes[target].activation,
                    1.0,
                )
            return state

        if action == "CREATE":
            state.add_node(
                "CREATED_NODE",
                role=6,
                activation=0.9,
            )
            return state

        if action == "BRANCH":
            if 0 <= source < len(state.nodes):
                branch_id = state.add_node(
                    f"{state.nodes[source].concept}#branch",
                    role=7,
                    activation=0.8,
                )
                if relation_id >= 0:
                    state.add_edge(
                        source,
                        relation_id,
                        branch_id,
                        activation=0.8,
                    )
            return state

        if action == "INHIBIT":
            if 0 <= target < len(state.nodes):
                state.nodes[target].activation *= 0.05
            return state

        if action == "BIND":
            if (
                0 <= source < len(state.nodes)
                and 0 <= target < len(state.nodes)
                and relation_id >= 0
            ):
                state.nodes[source].activation = max(
                    state.nodes[source].activation,
                    1.0,
                )
                state.nodes[target].activation = max(
                    state.nodes[target].activation,
                    1.0,
                )
                state.add_edge(
                    source,
                    relation_id,
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
            f"Unknown action: {action_id}"
        )

    def signature(self) -> tuple:
        return (
            tuple(
                (
                    n.concept,
                    n.role,
                    round(n.activation, 3),
                    n.persistent,
                )
                for n in self.nodes
            ),
            tuple(
                (
                    e.source,
                    e.relation_id,
                    e.target,
                    round(e.activation, 3),
                    e.persistent,
                )
                for e in self.edges
            ),
        )

    def active_node_count(self) -> int:
        return sum(
            n.activation > 0.5
            for n in self.nodes
        )

    def active_edge_count(self) -> int:
        return sum(
            e.activation > 0.5
            for e in self.edges
        )
