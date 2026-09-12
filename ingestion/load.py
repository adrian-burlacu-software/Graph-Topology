"""Put an ingested source into a copy of the store, so it can be measured.

    python -m ingestion.load --source genericskb --into data/v684_genericskb.sqlite
    python -m ingestion.load --source definitions --into data/v684_definitions.sqlite

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

SOURCES = ("genericskb", "definitions")

#: Where `research/v689/learn_definitions.py` keeps what it read.
DEFINITIONS = REPOSITORY_ROOT / "state" / "v689-definitions.sqlite"


def definition_rows(path: Path = DEFINITIONS, agreed_only: bool = False):
    """(concept, relation, object, source, confidence, sense_assumed) for
    every fact read out of a gloss that the teacher did not dispute.

    The concept is the synset whose gloss it is, so nothing is assumed about
    its sense; the object is free text, like every crawled source's.
    """
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    sql = "SELECT concept, relation, object FROM defined WHERE "
    sql += ("checked = 'agreed'" if agreed_only else
            "(checked IS NULL OR checked != 'disputed')")
    for concept, relation, obj in connection.execute(sql):
        yield concept, relation, obj, "definition", 0.9, 0
    connection.close()


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
         report_every: int = 200_000, agreed_only: bool = False) -> dict:
    """Copy the store, add one source's facts, and say what happened."""
    if source not in SOURCES:
        raise SystemExit(f"unknown source {source!r}; known: {SOURCES}")

    into = Path(into)
    if into.exists():
        into.unlink()
    started = time.time()
    shutil.copy2(STORE, into)

    connection = sqlite3.connect(into)
    before = connection.execute("SELECT count(*) FROM facts").fetchone()[0]
    senses = primary_senses(connection)

    if source == "definitions":
        offered = definition_rows(agreed_only=agreed_only)
    else:
        from . import genericskb

        offered = ((senses.get(term), relation, obj, name, confidence, 1)
                   for term, relation, obj, name, confidence
                   in genericskb.facts(floor=floor))
        source = "genericskb"

    added = unresolved = seen = 0
    batch = []
    for concept, relation, obj, name, confidence, assumed in offered:
        seen += 1
        if concept is None:
            unresolved += 1
            continue
        batch.append((concept, relation, obj, name, confidence, assumed))
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
    written = "definition" if source == "definitions" else source
    kinds = dict(connection.execute(
        "SELECT relation, count(*) FROM facts WHERE source = ? "
        "GROUP BY relation ORDER BY 2 DESC", (written,)).fetchall())
    concepts = connection.execute(
        "SELECT count(DISTINCT concept) FROM facts WHERE source = ?",
        (written,)).fetchone()[0]
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
