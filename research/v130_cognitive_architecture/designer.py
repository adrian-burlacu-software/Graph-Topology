from __future__ import annotations
from dataclasses import dataclass, field
from .long_term_memory import LongTermMemory
from .working_memory import WorkingMemory

@dataclass
class DesignerEvent:
    operation: str
    details: dict

@dataclass
class GraphDesigner:
    memory: LongTermMemory
    working: WorkingMemory
    events: list[DesignerEvent] = field(default_factory=list)

    def _emit(self, operation: str, **details) -> None:
        self.events.append(DesignerEvent(operation, details))

    def reuse(self, concept: str, activation: float = 1.0) -> bool:
        node = self.memory.nodes_by_name.get(concept.lower())
        if node is None:
            return False
        self.working.activate(concept, activation, "long_term")
        node.visit_count += 1
        self._emit("REUSE", concept=concept, activation=activation)
        return True

    def create(self, concept: str, activation: float = 0.8) -> None:
        self.working.activate(concept, activation, "created")
        self._emit("CREATE", concept=concept, activation=activation)

    def branch(self, source: str, branch: str, relation: str = "BRANCH") -> None:
        self.working.bind(source, relation, branch, 0.8)
        self._emit("BRANCH", source=source, branch=branch, relation=relation)

    def inhibit(self, concept: str) -> bool:
        changed = self.working.inhibit(concept)
        if changed:
            self._emit("INHIBIT", concept=concept)
        return changed

    def bind(self, source: str, relation: str, target: str, confidence: float = 1.0) -> None:
        self.working.bind(source, relation, target, confidence)
        self._emit("BIND", source=source, relation=relation, target=target, confidence=confidence)

    def accumulate(self, source: str, relation: str, target: str, amount: float = 0.05) -> None:
        self.memory.reinforce(source, relation, target, amount)
        self._emit("ACCUMULATE", source=source, relation=relation, target=target, amount=amount)

    def reinforce_semantic_edges(self, amount: float = 0.02) -> int:
        count = 0
        for edge in self.working.edges:
            if edge.relation in {"IsA", "CapableOf", "HasProperty", "UsedFor", "HasA", "PartOf"}:
                self.accumulate(edge.source, edge.relation, edge.target, amount * edge.confidence)
                count += 1
        return count
