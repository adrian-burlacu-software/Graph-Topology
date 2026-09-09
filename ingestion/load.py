"""Put an ingested source into a copy of the store, so it can be measured.

    python -m ingestion.load --source genericskb --into data/v684_genericskb.sqlite

This is deliberately *not* `research/v687/build.py`. That assembles the store
from nothing and takes as long as it takes; this copies an existing store and
adds one source to it, which is what an experiment needs: the only difference
between the two files is the source under test, so the audit's before and
after are comparable by construction.

A full rebuild is still the durable path -- `build.py` is where a source
belongs once it has earned its place. Nothing here writes to the store the
server reads.

## Attaching a word to a sense

GenericsKB states facts about word strings, exactly as Ascent++ and ConceptNet
do, so the same problem applies and gets the same answer: the term is joined
to the sense the build's own evidence made primary, and every row is flagged
`sense_assumed` so the join stays visible in the provenance rather than being
silently believed. A term with no lemma in the store is dropped.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STORE = REPOSITORY_ROOT / "data" / "v684_reasoning.sqlite"

SOURCES = ("genericskb",)


def primary_senses(connection) -> dict:
    """lemma -> the concept the build's evidence chose for it.

    `primary_sense` first, then WordNet's own order, which is the same
    precedence `Reasoner.senses_of` reads and so the same sense a question
    about that word would resolve to.
    """
    chosen: dict = {}
    for lemma, concept, primary, rank in connection.execute(
            "SELECT lemma, concept, primary_sense, sense_rank FROM lemmas "
            "ORDER BY primary_sense DESC, sense_rank"):
        if lemma not in chosen and ".n." in concept:
            chosen[lemma] = concept
    return chosen


def load(source: str, into: Path, floor: float = 0.0,
         report_every: int = 200_000) -> dict:
    """Copy the store, add one source's facts, and say what happened."""
    if source not in SOURCES:
        raise SystemExit(f"unknown source {source!r}; known: {SOURCES}")
    from . import genericskb

    into = Path(into)
    if into.exists():
        into.unlink()
    started = time.time()
    shutil.copy2(STORE, into)

    connection = sqlite3.connect(into)
    before = connection.execute("SELECT count(*) FROM facts").fetchone()[0]
    senses = primary_senses(connection)

    added = unresolved = seen = 0
    batch = []
    for term, relation, obj, name, confidence in genericskb.facts(floor=floor):
        seen += 1
        concept = senses.get(term)
        if concept is None:
            unresolved += 1
            continue
        batch.append((concept, relation, obj, name, confidence, 1))
        if len(batch) >= 20_000:
            connection.executemany(
                "INSERT OR IGNORE INTO facts (concept, relation, object, "
                "source, confidence, sense_assumed) VALUES (?,?,?,?,?,?)",
                batch)
            added += len(batch)
            batch.clear()
    if batch:
        connection.executemany(
            "INSERT OR IGNORE INTO facts (concept, relation, object, source, "
            "confidence, sense_assumed) VALUES (?,?,?,?,?,?)", batch)
        added += len(batch)
    connection.commit()
    after = connection.execute("SELECT count(*) FROM facts").fetchone()[0]
    kinds = dict(connection.execute(
        "SELECT relation, count(*) FROM facts WHERE source = ? "
        "GROUP BY relation ORDER BY 2 DESC", (source,)).fetchall())
    concepts = connection.execute(
        "SELECT count(DISTINCT concept) FROM facts WHERE source = ?",
        (source,)).fetchone()[0]
    connection.close()
    return {"store": str(into), "source": source,
            "facts_before": before, "facts_after": after,
            "rows_offered": seen, "rows_inserted": added,
            "rows_added_net": after - before,
            "term_had_no_sense": unresolved,
            "concepts_touched": concepts,
            "relations": kinds,
            "seconds": round(time.time() - started, 1)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="genericskb",
                        help="one of " + ", ".join(SOURCES))
    parser.add_argument("--into", default=str(
        REPOSITORY_ROOT / "data" / "v684_genericskb.sqlite"))
    parser.add_argument("--floor", type=float, default=0.0,
                        help="drop rows scoring below this")
    options = parser.parse_args(argv)
    import json

    print(json.dumps(load(options.source, Path(options.into), options.floor),
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
