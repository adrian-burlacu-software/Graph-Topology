
from dataclasses import dataclass

ACTIONS = ("NOOP", "REUSE", "CREATE", "BRANCH", "INHIBIT", "BIND", "COMMIT")
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}

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

    def clone(self):
        return State(
            [Node(n.concept, n.activation, n.role, n.persistent) for n in self.nodes],
            [Edge(e.source, e.relation, e.target, e.activation, e.persistent) for e in self.edges],
        )

    def node(self, concept):
        return next((n for n in self.nodes if n.concept == concept), None)

    def add_node(self, concept, activation, role, persistent=False):
        n = self.node(concept)
        if n is None:
            self.nodes.append(Node(concept, activation, role, persistent))
        else:
            n.activation = max(n.activation, activation)
            n.persistent = n.persistent or persistent

    def add_edge(self, source, relation, target, activation=1.0, persistent=False):
        for e in self.edges:
            if (e.source, e.relation, e.target) == (source, relation, target):
                e.activation = max(e.activation, activation)
                e.persistent = e.persistent or persistent
                return
        self.edges.append(Edge(source, relation, target, activation, persistent))

    def has_edge(self, source, relation, target, active_only=True):
        return any(
            e.source == source and e.relation == relation and e.target == target
            and (not active_only or e.activation > 0.5)
            for e in self.edges
        )

    def signature(self):
        return {
            "nodes": [vars(n) for n in self.nodes],
            "edges": [vars(e) for e in self.edges],
        }

    def apply(self, action_id, source=None, target=None, relation=None):
        s = self.clone()
        action = ACTIONS[action_id]

        if action == "NOOP":
            pass
        elif action == "REUSE":
            if target and s.node(target):
                s.node(target).activation = 1.0
        elif action == "CREATE":
            s.add_node(f"created_{len(s.nodes)}", 0.85, 6)
        elif action == "BRANCH":
            if source and relation:
                branch = f"{source}#branch{len(s.nodes)}"
                s.add_node(branch, 0.8, 7)
                s.add_edge(source, relation, branch, 0.8)
        elif action == "INHIBIT":
            if target and s.node(target):
                s.node(target).activation *= 0.05
        elif action == "BIND":
            if source and target and relation:
                s.add_edge(source, relation, target, 1.0)
        elif action == "COMMIT":
            for n in s.nodes:
                if n.activation > 0.5:
                    n.persistent = True
            for e in s.edges:
                if e.activation > 0.5:
                    e.persistent = True

        return s
