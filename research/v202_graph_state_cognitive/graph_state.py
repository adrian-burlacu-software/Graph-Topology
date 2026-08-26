from __future__ import annotations

from dataclasses import dataclass, field

# Action IDs are generic. They are not semantic relation labels.
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


@dataclass
class GraphNode:
    concept: str
    role: int = 0
    activation: float = 0.0
    persistent: bool = False


@dataclass
class GraphEdge:
    source: int
    relation_id: int
    target: int
    activation: float = 0.0
    persistent: bool = False


@dataclass
class GraphState:
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    def clone(self) -> "GraphState":
        return GraphState(
            nodes=[
                GraphNode(
                    concept=node.concept,
                    role=node.role,
                    activation=node.activation,
                    persistent=node.persistent,
                )
                for node in self.nodes
            ],
            edges=[
                GraphEdge(
                    source=edge.source,
                    relation_id=edge.relation_id,
                    target=edge.target,
                    activation=edge.activation,
                    persistent=edge.persistent,
                )
                for edge in self.edges
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
            GraphNode(
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
                if persistent:
                    edge.persistent = True
                return

        self.edges.append(
            GraphEdge(
                source=source,
                relation_id=relation_id,
                target=target,
                activation=activation,
                persistent=persistent,
            )
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
        Apply a generic designer action.

        This is deliberately deterministic so the learning target is
        graph-derived. The model predicts the action; the environment applies it.
        """
        next_state = self.clone()

        action = ACTIONS[action_id]

        if action == "NOOP":
            return next_state

        if action == "REUSE":
            if 0 <= source < len(next_state.nodes):
                next_state.nodes[source].activation = max(
                    next_state.nodes[source].activation,
                    1.0,
                )
            return next_state

        if action == "CREATE":
            next_state.add_node(
                "CREATED_NODE",
                role=0,
                activation=0.75,
            )
            return next_state

        if action == "BRANCH":
            if 0 <= source < len(next_state.nodes):
                concept = (
                    next_state.nodes[source].concept
                    + "#branch"
                )
                branch_id = next_state.add_node(
                    concept,
                    role=0,
                    activation=0.70,
                )
                if relation_id >= 0:
                    next_state.add_edge(
                        source,
                        relation_id,
                        branch_id,
                        activation=0.70,
                    )
            return next_state

        if action == "INHIBIT":
            if 0 <= target < len(next_state.nodes):
                next_state.nodes[target].activation *= 0.05
            return next_state

        if action == "BIND":
            if (
                0 <= source < len(next_state.nodes)
                and 0 <= target < len(next_state.nodes)
                and relation_id >= 0
            ):
                next_state.nodes[source].activation = max(
                    next_state.nodes[source].activation,
                    1.0,
                )
                next_state.nodes[target].activation = max(
                    next_state.nodes[target].activation,
                    1.0,
                )
                next_state.add_edge(
                    source,
                    relation_id,
                    target,
                    activation=1.0,
                )
            return next_state

        if action == "COMMIT":
            for node in next_state.nodes:
                if node.activation > 0.5:
                    node.persistent = True
            for edge in next_state.edges:
                if edge.activation > 0.5:
                    edge.persistent = True
            return next_state

        raise ValueError(
            f"Unknown action id: {action_id}"
        )

    def tensor_signature(
        self,
    ) -> tuple:
        node_sig = tuple(
            (
                node.concept,
                node.role,
                round(node.activation, 4),
                node.persistent,
            )
            for node in self.nodes
        )
        edge_sig = tuple(
            (
                edge.source,
                edge.relation_id,
                edge.target,
                round(edge.activation, 4),
                edge.persistent,
            )
            for edge in self.edges
        )
        return node_sig, edge_sig

    def stats(self) -> dict:
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "active_nodes": sum(
                node.activation > 0.05
                for node in self.nodes
            ),
            "active_edges": sum(
                edge.activation > 0.05
                for edge in self.edges
            ),
        }
