from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from .long_term_memory import LongTermMemory

@dataclass
class AttentionModule:
    name: str
    decay: float = 0.60
    propagation_scale: float = 0.65
    max_active: int = 24
    activations: dict[int, float] = field(default_factory=dict)
    inhibited_until: dict[int, int] = field(default_factory=dict)
    tick: int = 0

    def clear(self) -> None:
        self.activations.clear()
        self.inhibited_until.clear()
        self.tick = 0

    def seed(self, memory: LongTermMemory, concept: str, value: float = 1.0) -> bool:
        node = memory.nodes_by_name.get(concept.lower())
        if node is None:
            return False
        self.activations[node.node_id] = max(self.activations.get(node.node_id, 0.0), value)
        return True

    def inhibit(self, node_id: int, duration: int = 1) -> None:
        self.activations[node_id] = 0.0
        self.inhibited_until[node_id] = self.tick + max(1, duration)

    def step(self, memory: LongTermMemory) -> list[tuple[int, float]]:
        self.tick += 1
        for node_id in list(self.activations):
            self.activations[node_id] *= self.decay
        for node_id, until in list(self.inhibited_until.items()):
            if until < self.tick:
                del self.inhibited_until[node_id]

        incoming = defaultdict(float)
        for source_id, activation in self.activations.items():
            if activation <= 0:
                continue
            for edge in memory.outgoing(source_id):
                if self.inhibited_until.get(edge.target, -1) >= self.tick:
                    continue
                incoming[edge.target] += (
                    activation * self.propagation_scale * edge.effective_weight
                )

        for node_id, signal in incoming.items():
            self.activations[node_id] = self.activations.get(node_id, 0.0) + signal

        ranked = sorted(self.activations.items(), key=lambda x: x[1], reverse=True)
        keep = ranked[:self.max_active]
        self.activations = {k: v for k, v in keep if v > 1e-6}
        return keep

    def active_concepts(self, memory: LongTermMemory, limit: int = 20) -> list[tuple[str, float]]:
        ranked = sorted(self.activations.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [(memory.nodes_by_id[node_id].concept, value) for node_id, value in ranked]
