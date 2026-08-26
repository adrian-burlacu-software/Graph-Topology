from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class WMNode:
    value: str
    activation: float = 1.0
    source: str = "working"

@dataclass
class WMEdge:
    source: str
    relation: str
    target: str
    confidence: float = 1.0

@dataclass
class WorkingMemory:
    nodes: dict[str, WMNode] = field(default_factory=dict)
    edges: list[WMEdge] = field(default_factory=list)
    episode_id: int = 0

    def begin(self, episode_id: int) -> None:
        self.clear()
        self.episode_id = episode_id

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()

    def activate(self, value: str, activation: float = 1.0, source: str = "working") -> None:
        key = value.lower()
        node = self.nodes.get(key)
        if node is None:
            self.nodes[key] = WMNode(value, activation, source)
        else:
            node.activation = max(node.activation, activation)

    def bind(self, source: str, relation: str, target: str, confidence: float = 1.0) -> None:
        self.activate(source)
        self.activate(target)
        edge = WMEdge(source.lower(), relation, target.lower(), confidence)
        if not any(
            e.source == edge.source and e.relation == edge.relation and e.target == edge.target
            for e in self.edges
        ):
            self.edges.append(edge)

    def inhibit(self, value: str) -> bool:
        node = self.nodes.get(value.lower())
        if node is None:
            return False
        node.activation = 0.0
        return True

    def has_edge(self, source: str, relation: str, target: str) -> bool:
        return any(
            e.source == source.lower() and e.relation == relation and e.target == target.lower()
            for e in self.edges
        )

    def snapshot(self) -> dict:
        return {
            "nodes": [
                {"value": n.value, "activation": n.activation, "source": n.source}
                for n in self.nodes.values()
            ],
            "edges": [
                {"source": e.source, "relation": e.relation, "target": e.target,
                 "confidence": e.confidence}
                for e in self.edges
            ],
        }
