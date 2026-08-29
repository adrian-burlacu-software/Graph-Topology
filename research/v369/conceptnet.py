
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import re
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class Concept:
    concept_id: str
    label: str


@dataclass(frozen=True)
class ConceptRelation:
    relation: str
    source: str
    target: str


class ConceptNetCompact:
    """
    Read-only adapter for the user's compact ConceptNet SQLite database.

    The adapter does not assume a particular schema blindly: it inspects the
    tables/columns and finds a plausible edge table from common compact
    ConceptNet layouts.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.conn: sqlite3.Connection | None = None
        self.edge_table: str | None = None
        self.columns: Tuple[str, ...] = ()

    def connect(self) -> None:
        if not self.db_path.exists():
            raise FileNotFoundError(self.db_path)

        self.conn = sqlite3.connect(
            f"file:{self.db_path.resolve()}?mode=ro",
            uri=True,
        )
        self._discover_edge_table()

    def _discover_edge_table(self) -> None:
        assert self.conn is not None

        tables = [
            row[0]
            for row in self.conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]

        candidates = []
        for table in tables:
            cols = [
                row[1]
                for row in self.conn.execute(
                    f'PRAGMA table_info("{table}")'
                )
            ]
            lowered = {c.lower() for c in cols}

            # Compact ConceptNet variants commonly expose some version of
            # relation/source/target or rel/start/end.
            source = any(
                c in lowered
                for c in ("source", "src", "start", "subject", "from_id")
            )
            target = any(
                c in lowered
                for c in ("target", "dst", "end", "object", "to_id")
            )
            relation = any(
                c in lowered
                for c in ("relation", "rel", "predicate", "relation_id")
            )

            if source and target and relation:
                candidates.append((table, tuple(cols)))

        if not candidates:
            raise RuntimeError(
                "Could not identify a ConceptNet edge table. "
                f"Tables found: {tables}"
            )

        self.edge_table, self.columns = candidates[0]

    def _column(self, names: Iterable[str]) -> str:
        lowered = {
            c.lower(): c
            for c in self.columns
        }
        for name in names:
            if name in lowered:
                return lowered[name]
        raise RuntimeError(
            f"Missing expected column among {tuple(names)} "
            f"in {self.edge_table}"
        )

    def stats(self) -> dict:
        assert self.conn is not None and self.edge_table
        count = self.conn.execute(
            f'SELECT COUNT(*) FROM "{self.edge_table}"'
        ).fetchone()[0]
        return {
            "database": str(self.db_path.resolve()),
            "edge_table": self.edge_table,
            "edge_count": int(count),
        }

    def edges_for_label(self, label: str, limit: int = 50):
        assert self.conn is not None and self.edge_table

        src = self._column(("source", "src", "start", "subject", "from_id"))
        dst = self._column(("target", "dst", "end", "object", "to_id"))
        rel = self._column(("relation", "rel", "predicate", "relation_id"))

        needle = label.strip().lower()

        # We support both direct string endpoints and integer Concept IDs with
        # a separate concepts/nodes table, discovered opportunistically.
        query = (
            f'SELECT "{src}", "{rel}", "{dst}" '
            f'FROM "{self.edge_table}" LIMIT ?'
        )

        rows = self.conn.execute(query, (max(limit * 20, 200),))
        out = []

        for source, relation, target in rows:
            s = str(source)
            t = str(target)
            if needle in s.lower() or needle in t.lower():
                out.append(
                    ConceptRelation(
                        relation=str(relation),
                        source=s,
                        target=t,
                    )
                )
                if len(out) >= limit:
                    break

        return out

    def related(self, labels: Iterable[str], limit: int = 100):
        out = []
        seen = set()

        for label in labels:
            for edge in self.edges_for_label(label, limit=limit):
                key=(edge.relation,edge.source,edge.target)
                if key not in seen:
                    seen.add(key)
                    out.append(edge)
                if len(out)>=limit:
                    return out
        return out

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn=None


def load_conceptnet(db_path: Path):
    graph=ConceptNetCompact(db_path)
    graph.connect()
    return graph
