from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path
import json, math, sqlite3

RELATION_BIAS = {
    "IsA": 1.25, "CapableOf": 1.20, "HasProperty": 1.18,
    "UsedFor": 1.18, "HasA": 1.10, "PartOf": 1.10,
    "Causes": 1.05, "AtLocation": 0.90, "MadeOf": 0.90,
    "ReceivesAction": 0.95, "HasPrerequisite": 0.95,
    "Synonym": 1.00, "Antonym": 0.95, "RelatedTo": 0.70,
    "SimilarTo": 0.80, "DefinedAs": 1.00, "HasContext": 0.75,
}

@dataclass
class LTNode:
    concept: str
    node_id: int
    visit_count: float = 0.0

@dataclass
class LTEdge:
    source: int
    relation: str
    target: int
    weight: float
    base_weight: float
    reinforcement: float = 0.0
    use_count: int = 0

    @property
    def effective_weight(self) -> float:
        return max(0.0, self.weight * (1.0 + self.reinforcement))

@dataclass
class LongTermMemory:
    nodes_by_name: dict[str, LTNode] = field(default_factory=dict)
    nodes_by_id: dict[int, LTNode] = field(default_factory=dict)
    edges_by_source: dict[int, list[LTEdge]] = field(
        default_factory=lambda: defaultdict(list)
    )
    edge_index: dict[tuple[int, str, int], LTEdge] = field(default_factory=dict)

    def get_or_create_node(self, concept: str) -> LTNode:
        concept = concept.strip().lower()
        if concept in self.nodes_by_name:
            return self.nodes_by_name[concept]
        node = LTNode(concept, len(self.nodes_by_name))
        self.nodes_by_name[concept] = node
        self.nodes_by_id[node.node_id] = node
        return node

    def add_edge(self, source: str, relation: str, target: str, weight: float) -> LTEdge:
        s = self.get_or_create_node(source)
        t = self.get_or_create_node(target)
        key = (s.node_id, relation, t.node_id)
        if key in self.edge_index:
            edge = self.edge_index[key]
            edge.weight = max(edge.weight, weight)
            edge.base_weight = max(edge.base_weight, weight)
            return edge
        edge = LTEdge(s.node_id, relation, t.node_id, float(weight), float(weight))
        self.edge_index[key] = edge
        self.edges_by_source[s.node_id].append(edge)
        return edge

    def outgoing(self, source_id: int) -> list[LTEdge]:
        return self.edges_by_source.get(source_id, [])

    def edge(self, source: str, relation: str, target: str) -> LTEdge | None:
        s = self.nodes_by_name.get(source.lower())
        t = self.nodes_by_name.get(target.lower())
        if s is None or t is None:
            return None
        return self.edge_index.get((s.node_id, relation, t.node_id))

    def reinforce(self, source: str, relation: str, target: str, amount: float = 0.05) -> None:
        edge = self.edge(source, relation, target)
        if edge is None:
            edge = self.add_edge(source, relation, target, max(0.01, amount))
        edge.reinforcement += amount
        edge.use_count += 1

    def stats(self) -> dict:
        rc = Counter()
        for edges in self.edges_by_source.values():
            for edge in edges:
                rc[edge.relation] += 1
        return {
            "nodes": len(self.nodes_by_name),
            "edges": len(self.edge_index),
            "relation_counts": dict(rc),
            "reinforced_edges": sum(e.use_count > 0 for e in self.edge_index.values()),
        }

    def save(self, path: Path) -> None:
        payload = {
            "nodes": [
                {"id": n.node_id, "concept": n.concept, "visit_count": n.visit_count}
                for n in self.nodes_by_id.values()
            ],
            "edges": [
                {"source": e.source, "relation": e.relation, "target": e.target,
                 "weight": e.weight, "base_weight": e.base_weight,
                 "reinforcement": e.reinforcement, "use_count": e.use_count}
                for edges in self.edges_by_source.values() for e in edges
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_conceptnet(cls, db_path: Path, dictionary_words: set[str],
                        min_edge_weight: float = 1.0,
                        max_edges_per_word: int = 60) -> "LongTermMemory":
        memory = cls()
        db = sqlite3.connect(str(db_path))
        db.row_factory = sqlite3.Row
        try:
            placeholders = ",".join("?" for _ in dictionary_words)
            if not placeholders:
                return memory
            rows = db.execute(
                f"""SELECT start, relation, end, weight FROM edge
                    WHERE start IN ({placeholders}) AND weight >= ?
                    ORDER BY start, weight DESC""",
                (*dictionary_words, min_edge_weight),
            )
            counts = Counter()
            for row in rows:
                start = row["start"]
                if counts[start] >= max_edges_per_word:
                    continue
                relation = row["relation"]
                weight = math.log1p(max(0.01, float(row["weight"]))) * RELATION_BIAS.get(relation, 0.75)
                memory.add_edge(start, relation, row["end"], weight)
                counts[start] += 1
        finally:
            db.close()
        return memory
