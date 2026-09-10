"""Ask the model about properties the store does not record, and keep the
ones it is sure of.

    python -m ingestion.teach --dry-run          # what it would ask, and of what
    python -m ingestion.teach --into data/v684_taught.sqlite

This is the first thing in this repository that *writes* a fact nobody
crawled. `research/v688/AUDIT.md` §14 is why it is allowed to:

    floor   false asserted   true asserted   true claims it is sure of
    0.00         10.2%           82.5%              100%
    0.99          1.0%           95.2%             41.5%

At 0.99, with the `careful` system prompt, the model's over-affirmation rate
matches the store's own -- 1.0% against 1.0% -- while reaching 41.5% of true
claims where the crawl reaches 17.2%. Below that floor it is not safe and
this refuses to write.

## What it asks about

Not the concept space, which is 45,219 wide and mostly nothing anybody would
ask. The store already knows which properties belong near a concept: the
things its *category siblings* hold. If a robin, a crow and a wren all have
feathers, `does a sparrow have feathers` is a question worth putting, and the
answer is one the store lacks.

That is the loop's own curiosity logic run offline, and it is deliberately
the same shape: v688 generates questions from what it does not know, and the
audit says 81% of them hit absence. This turns that stream into writes.

## What it will not touch

`research/v688/holdout.py` reserves `bird`, `tool` and `fruit` -- 95 concepts
of 521. Nothing here writes to them, so that whatever teaching does to the
rest can be compared against a part it never saw. That comparison is the only
evidence that teaching generalises rather than memorises, and it is
unavailable once everything is taught.

Every fact written carries `source = "taught"` and the model's own confidence,
so it can be told apart from a crawled row, priced separately by
`confidence.py`, and deleted wholesale if it turns out to be a bad idea.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import shutil
import sqlite3
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STORE = REPOSITORY_ROOT / "data" / "v684_reasoning.sqlite"
SENSES = REPOSITORY_ROOT / "data" / "xcslb" / "concept_senses.csv"

#: Below this, nothing is written. AUDIT.md §14.
FLOOR = 0.99

#: How many sibling properties one concept may be asked about. The category
#: with the most is `animal` at 59 members holding some 900 properties
#: between them, and asking all of them of all of them is 300,000 questions
#: for a store that would refuse most of the answers anyway.
PER_CONCEPT = 60


def categories() -> dict:
    with SENSES.open(encoding="utf-8") as handle:
        return {row["concept"]: row["category"]
                for row in csv.DictReader(handle)}


def plan(per_concept: int = PER_CONCEPT) -> list:
    """(concept, subject, property, question) for everything worth asking.

    A property is worth asking about when the concept's category siblings
    hold it and the concept is not already recorded with it. Sorted by how
    many siblings hold it, so the most characteristic properties are asked
    first and `--limit` cuts the tail rather than a random slice.
    """
    from research.v687 import corpora
    from research.v688 import audit, holdout

    where = categories()
    article = audit.articles()
    listed: dict = {concept: set(features)
                    for concept, features in corpora.load_xcslb().items}
    by_category: dict = collections.defaultdict(collections.Counter)
    for concept, features in listed.items():
        for feature in features:
            by_category[where.get(concept, "?")][feature] += 1

    out = []
    for concept in sorted(listed):
        if holdout.held(concept):
            continue                       # reserved, and reserved is reserved
        subject = article.get(concept, "")
        if not subject:
            continue
        siblings = by_category.get(where.get(concept, "?"), {})
        wanted = [(count, feature) for feature, count in siblings.items()
                  if feature not in listed[concept] and count >= 2]
        wanted.sort(reverse=True)
        for _count, feature in wanted[:per_concept]:
            question = audit.phrase(subject, feature)
            if question:
                out.append((concept, subject, feature, question))
    return out


def relation_of(feature: str) -> tuple:
    """A property as a relation and an object, the store's own shape.

    The same leading-word test `ingestion/genericskb.py` uses, for the same
    reason: the determiner separates a kind from a property and no tagger is
    needed to see it.
    """
    words = (feature or "").strip().split()
    if not words:
        return "", ""
    head, rest = words[0].lower(), " ".join(words[1:]).strip()
    if head in ("has", "have"):
        return "has_a", rest
    if head in ("can", "could"):
        if rest.lower().startswith(("be made of ", "be made from ")):
            return "made_of", rest.split("of ", 1)[-1].split("from ", 1)[-1]
        return "capable_of", rest
    if head in ("is", "are", "was"):
        if rest.lower().startswith(("used for ", "used to ", "used as ")):
            return "used_for", rest.split(" ", 1)[-1]
        if rest.lower().startswith(("made of ", "made from ")):
            return "made_of", rest.split("of ", 1)[-1].split("from ", 1)[-1]
        if rest.lower().startswith(("a ", "an ", "the ")):
            return "is_a", rest.split(" ", 1)[-1]
        return "has_property", rest
    return "capable_of", feature


def primary_senses(connection) -> dict:
    chosen: dict = {}
    for lemma, concept, _primary, _rank in connection.execute(
            "SELECT lemma, concept, primary_sense, sense_rank FROM lemmas "
            "ORDER BY primary_sense DESC, sense_rank"):
        if lemma not in chosen and ".n." in concept:
            chosen[lemma] = concept
    return chosen


def teach(into: Path, limit: int = 0, floor: float = FLOOR,
          per_concept: int = PER_CONCEPT) -> dict:
    """Ask, keep what clears the floor, write it into a copy of the store."""
    from research.v688.teacher import Teacher

    asking = plan(per_concept)
    if limit:
        asking = asking[:limit]
    teacher = Teacher()
    if not teacher.available:
        raise SystemExit(f"no teacher: {teacher.error}")

    into = Path(into)
    if into.exists():
        into.unlink()
    started = time.time()
    shutil.copy2(STORE, into)
    connection = sqlite3.connect(into)
    before = connection.execute("SELECT count(*) FROM facts").fetchone()[0]
    senses = primary_senses(connection)

    asked = kept = unresolved = 0
    relations: collections.Counter = collections.Counter()
    batch = []
    for concept, _subject, feature, question in asking:
        asked += 1
        supports, confidence, _cached = teacher.judge("", "", question)
        if not supports or confidence < floor:
            continue
        target = senses.get(concept.replace("_", " ")) or senses.get(concept)
        if target is None:
            unresolved += 1
            continue
        relation, obj = relation_of(feature)
        if not relation or not obj:
            continue
        relations[relation] += 1
        batch.append((target, relation, obj, "taught", round(confidence, 4), 1))
        kept += 1
        if len(batch) >= 5000:
            connection.executemany(
                "INSERT OR IGNORE INTO facts (concept, relation, object, "
                "source, confidence, sense_assumed) VALUES (?,?,?,?,?,?)",
                batch)
            connection.commit()
            batch.clear()
    if batch:
        connection.executemany(
            "INSERT OR IGNORE INTO facts (concept, relation, object, source, "
            "confidence, sense_assumed) VALUES (?,?,?,?,?,?)", batch)
    connection.commit()
    after = connection.execute("SELECT count(*) FROM facts").fetchone()[0]
    touched = connection.execute(
        "SELECT count(DISTINCT concept) FROM facts WHERE source = 'taught'"
    ).fetchone()[0]
    connection.close()
    return {"store": str(into), "asked": asked, "accepted": kept,
            "accepted_share": round(kept / asked, 4) if asked else 0.0,
            "written": after - before, "concepts_touched": touched,
            "term_had_no_sense": unresolved, "floor": floor,
            "relations": dict(relations.most_common()),
            "seconds": round(time.time() - started, 1)}


def main(argv=None) -> int:
    from research.v688 import holdout

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--into", default=str(
        REPOSITORY_ROOT / "data" / "v684_taught.sqlite"))
    parser.add_argument("--limit", type=int, default=0,
                        help="ask only this many questions")
    parser.add_argument("--floor", type=float, default=FLOOR)
    parser.add_argument("--per-concept", type=int, default=PER_CONCEPT)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be asked, and stop")
    options = parser.parse_args(argv)

    if options.dry_run:
        asking = plan(options.per_concept)
        held = holdout.concepts()
        touched = {concept for concept, _s, _f, _q in asking}
        print(json.dumps({
            "questions": len(asking),
            "concepts": len(touched),
            "held_out": sorted(holdout.CATEGORIES),
            "held_out_concepts": len(held),
            "holdout_reached": sorted(touched & held),
            "examples": [q for _c, _s, _f, q in asking[:8]],
        }, indent=2))
        return 0

    print(json.dumps(teach(Path(options.into), options.limit, options.floor,
                           options.per_concept), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
