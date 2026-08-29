
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import re
import unicodedata
from typing import Iterable


@dataclass(frozen=True)
class ConceptRelation:
    relation: str
    source: str
    target: str


def normalize_concept_text(value: str) -> str:
    """
    Normalize ConceptNet labels without substring matching.

    Examples:
      /c/en/dog       -> dog
      /c/en/dark_dog  -> dark dog
      "Beer O'Clock"  -> beer o'clock
    """
    value=str(value).strip()

    # ConceptNet URI-style nodes.
    value=re.sub(
        r"^/c/[a-z]{2}/",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value=value.split("/")[0]

    value=unicodedata.normalize("NFKC",value)
    value=value.replace("_"," ")
    value=value.replace("-"," ")
    value=value.lower()

    value=re.sub(r"[^a-z0-9']+", " ", value)
    value=re.sub(r"\s+"," ",value).strip()
    return value


def is_reasonable_anchor(value: str) -> bool:
    """
    Reject punctuation fragments, numeric garbage, and extremely short
    artifacts that are poor semantic anchors.
    """
    n=normalize_concept_text(value)

    if not n or len(n)<2:
        return False

    if not re.search(r"[a-z]", n):
        return False

    # The bare apostrophe-like fragments that caused the earlier accidental
    # ConceptNet match are not valid lexical anchors.
    if all(ch in "'0123456789" for ch in n):
        return False

    return True


class ConceptNetCompact:
    """
    Read-only ConceptNet compact SQLite adapter.

    The important grounding contract is exact normalized endpoint equality,
    never substring matching.
    """

    def __init__(self,db_path:Path):
        self.db_path=Path(db_path)
        self.conn:sqlite3.Connection|None=None
        self.edge_table:str|None=None
        self.columns=()

    def connect(self):
        if not self.db_path.exists():
            raise FileNotFoundError(self.db_path)

        self.conn=sqlite3.connect(
            f"file:{self.db_path.resolve()}?mode=ro",
            uri=True,
        )
        self._discover_edge_table()

    def _discover_edge_table(self):
        assert self.conn is not None

        tables=[
            row[0]
            for row in self.conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]

        for table in tables:
            cols=[
                row[1]
                for row in self.conn.execute(
                    f'PRAGMA table_info("{table}")'
                )
            ]
            lowered={c.lower() for c in cols}

            source=any(
                x in lowered
                for x in (
                    "source","src","start",
                    "subject","from_id",
                )
            )
            target=any(
                x in lowered
                for x in (
                    "target","dst","end",
                    "object","to_id",
                )
            )
            relation=any(
                x in lowered
                for x in (
                    "relation","rel","predicate",
                    "relation_id",
                )
            )

            if source and target and relation:
                self.edge_table=table
                self.columns=tuple(cols)
                return

        raise RuntimeError(
            f"Could not identify a ConceptNet edge table. "
            f"Tables found: {tables}"
        )

    def _column(self,names:Iterable[str]):
        lowered={
            c.lower():c
            for c in self.columns
        }
        for name in names:
            if name in lowered:
                return lowered[name]
        raise RuntimeError(
            f"Expected one of {tuple(names)} "
            f"in {self.edge_table}; got {self.columns}"
        )

    def stats(self):
        assert self.conn is not None and self.edge_table
        count=self.conn.execute(
            f'SELECT COUNT(*) FROM "{self.edge_table}"'
        ).fetchone()[0]
        return {
            "database":str(self.db_path.resolve()),
            "edge_table":self.edge_table,
            "edge_count":int(count),
        }

    def edges_for_label(
        self,
        label:str,
        limit:int=50,
    ):
        """
        Exact normalized endpoint matching.

        We intentionally retrieve candidate rows using a bounded scan and then
        compare normalized endpoints exactly. This is slower than the old
        substring path, but correctness is the priority for this benchmark.
        """
        assert self.conn is not None and self.edge_table

        if not is_reasonable_anchor(label):
            return []

        needle=normalize_concept_text(label)

        src=self._column(
            ("source","src","start","subject","from_id")
        )
        dst=self._column(
            ("target","dst","end","object","to_id")
        )
        rel=self._column(
            ("relation","rel","predicate","relation_id")
        )

        # Inspect a bounded number of rows. The full DB statistics are retained,
        # while grounding remains deterministic and safe.
        scan_limit=max(
            1000,
            limit*250,
        )

        rows=self.conn.execute(
            f'SELECT "{src}","{rel}","{dst}" '
            f'FROM "{self.edge_table}" LIMIT ?',
            (scan_limit,),
        )

        matches=[]
        for source,relation,target in rows:
            source_norm=normalize_concept_text(source)
            target_norm=normalize_concept_text(target)

            if source_norm!=needle and target_norm!=needle:
                continue

            matches.append(
                ConceptRelation(
                    relation=str(relation),
                    source=str(source),
                    target=str(target),
                )
            )

            if len(matches)>=limit:
                break

        return matches

    def related(self,labels:Iterable[str],limit:int=100):
        out=[]
        seen=set()

        for label in labels:
            for edge in self.edges_for_label(
                label,
                limit=limit,
            ):
                key=(
                    edge.relation,
                    edge.source,
                    edge.target,
                )
                if key in seen:
                    continue
                seen.add(key)
                out.append(edge)

                if len(out)>=limit:
                    return out

        return out

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn=None


def load_conceptnet(db_path:Path):
    graph=ConceptNetCompact(db_path)
    graph.connect()
    return graph
