from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sqlite3


RELATIONS = (
    "IsA",
    "CapableOf",
    "HasProperty",
    "UsedFor",
    "HasA",
    "PartOf",
    "RelatedTo",
    "SimilarTo",
    "Antonym",
    "Causes",
    "AtLocation",
    "MadeOf",
    "ReceivesAction",
    "HasPrerequisite",
    "DefinedAs",
    "HasContext",
)

RELATION_TO_ID = {
    relation: index
    for index, relation in enumerate(RELATIONS)
}


@dataclass(frozen=True)
class SemanticEdge:
    source: str
    relation: str
    target: str
    weight: float


class ConceptNetMemory:
    """Read-only semantic memory backed by the compact ConceptNet SQLite DB."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def relation_id(self, relation: str) -> int:
        return RELATION_TO_ID.get(relation, -1)

    def edges_for_source(
        self,
        source: str,
        max_edges: int = 64,
    ) -> list[SemanticEdge]:
        rows = self.conn.execute(
            """
            SELECT start, relation, end, weight
            FROM edge
            WHERE start = ?
            ORDER BY weight DESC
            LIMIT ?
            """,
            (source, max_edges),
        )

        return [
            SemanticEdge(
                source=row["start"],
                relation=row["relation"],
                target=row["end"],
                weight=float(row["weight"]),
            )
            for row in rows
        ]

    def edge(
        self,
        source: str,
        relation: str,
        target: str,
    ) -> SemanticEdge | None:
        row = self.conn.execute(
            """
            SELECT start, relation, end, weight
            FROM edge
            WHERE start = ?
              AND relation = ?
              AND end = ?
            LIMIT 1
            """,
            (source, relation, target),
        ).fetchone()

        if row is None:
            return None

        return SemanticEdge(
            source=row["start"],
            relation=row["relation"],
            target=row["end"],
            weight=float(row["weight"]),
        )

    def local_subgraph(
        self,
        seed: str,
        max_nodes: int = 24,
        edges_per_node: int = 12,
    ) -> tuple[list[str], list[SemanticEdge]]:
        """
        Small graph neighborhood used as a transformer working-set source.
        This is intentionally bounded so training stays cheap.
        """
        seen = {seed}
        frontier = [seed]
        edges: list[SemanticEdge] = []

        while frontier and len(seen) < max_nodes:
            source = frontier.pop(0)
            outgoing = self.edges_for_source(
                source,
                max_edges=edges_per_node,
            )

            for edge in outgoing:
                edges.append(edge)

                if edge.target not in seen:
                    seen.add(edge.target)
                    frontier.append(edge.target)

                if len(seen) >= max_nodes:
                    break

        nodes = sorted(seen)
        return nodes, edges

    def candidate_targets(
        self,
        source: str,
        relation: str,
        limit: int = 32,
    ) -> list[tuple[str, float]]:
        rows = self.conn.execute(
            """
            SELECT end, weight
            FROM edge
            WHERE start = ?
              AND relation = ?
            ORDER BY weight DESC
            LIMIT ?
            """,
            (source, relation, limit),
        )

        return [
            (row["end"], float(row["weight"]))
            for row in rows
        ]
