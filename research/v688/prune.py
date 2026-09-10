"""Ask which class-level facts are claims about the class, and demote the rest.

`AUDIT.md` §19 measured where v687's confident falsehoods come from, and it is
not where "fix the crawl" suggests. A fact **stated** of the concept asked
about affirms 163 true claims against 23 false -- **87.6% precision**. The
same machinery **inheriting** a fact from an ancestor affirms 343 true against
132 false -- **72.2%**. Inheritance is simultaneously where two thirds of the
correct coverage comes from and where 85% of the errors do.

And the errors are not the crawl being wrong. `animal capable_of "be used in
research"` is true; `animal capable_of "be riddled with bullet"` is true of
some animal, somewhere, in whatever sentence Ascent++ read it out of. Neither
is a claim *about animals*, and R1 hands both to all 4,016 descendants.

So the target is not fact accuracy. It is **whether a sentence attached to a
class node generalises over that class**, which is a different question and
one the `careful` prompt already asks: *say yes only if the property is
typical of that kind.*

## What is judged, and why so little of it

1,375,339 facts are inheritable and sit on a node with descendants -- 70% of
the store, and 95% of them from Ascent++. Judging all of it is ten GPU-hours,
which is not the place to start. Blast radius is:

    descendants under the node    inheritable facts
    1-9                                     686,408
    10-99                                   525,010
    100-999                                 137,563
    1000+                                    26,358

A wrong fact on `animal.n.01` reaches 4,016 concepts; a wrong fact on a node
with three children reaches three. **The 26,358 facts on the widest nodes are
22 GPU-minutes and carry more damage per fact than the other 1.3 million put
together**, so they are what `--build` asks about by default. `--band` widens
it when that pays.

## Demotion, not deletion

A demoted fact stays in the store and still answers about the concept it was
recorded of. It is skipped only where `distance` is non-zero -- `reason.py`'s
R30 -- so the change is exactly "this sentence does not descend" and nothing
else. The store is never written to and `data/demoted_facts.json` can be
deleted to undo the whole thing.

That matters more here than it did for the distilled norms, because this
subtracts. A wrong demotion silently removes a true inherited answer, and §19
says inherited answers are two thirds of the coverage. The floor is set
accordingly: a fact is demoted only when the judge is confident it is *not* a
class claim, so silence and uncertainty both leave the fact alone.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

from research.v687 import rules
from . import audit, densify, teacher

ROOT = Path(__file__).resolve().parents[2]
DEMOTED = ROOT / "derived" / "demoted_facts.json"

#: Descendants a node must have before its facts are worth judging. See the
#: blast-radius table in the docstring.
WIDEST = 1000

#: How sure the judge must be that a fact is *not* a class claim before the
#: fact stops descending. This is a subtraction, so the floor guards the
#: opposite direction from `densify.FLOOR`: there, silence cost nothing and
#: the risk was adding noise; here, a confident-but-wrong `no` deletes a true
#: inherited answer, and inherited answers are two thirds of the coverage.
FLOOR = 0.9


def candidates(engine, widest: int = WIDEST) -> list:
    """(concept, relation, object) on the nodes with the widest reach."""
    conn = engine.reasoner.connection
    size = {row[0]: (row[1] or 0) for row in
            conn.execute("SELECT id, descendants FROM concepts")}
    out = []
    for concept, relation, obj in conn.execute(
            "SELECT concept, relation, object FROM facts"):
        if size.get(concept, 0) < widest:
            continue
        if not rules.inheritable(relation) or relation not in teacher.READS:
            continue
        out.append((concept, relation, str(obj)))
    out.sort()
    return out


def question(concept: str, relation: str, obj: str) -> str:
    """Is this typical of the class? Asked of the class, not of a member.

    The same bare-claim phrasing `densify.py` uses, so a judgement caches
    once and the two files cannot disagree about what was asked.
    """
    name = densify.plain(concept.rsplit(".", 2)[0])
    return densify.question(name, relation, obj)


def build(engine=None, floor: float = FLOOR, widest: int = WIDEST,
          limit: int = 0, judge=None) -> dict:
    from research.v687.reasoning import ReasoningEngine

    engine = engine or ReasoningEngine(audit.STORE)
    rows = candidates(engine, widest)
    if limit:
        rows = rows[:limit]
    judge = judge or teacher.Teacher()
    if not getattr(judge, "available", False):
        return {"error": getattr(judge, "error", "") or "no teacher"}

    demoted, counts = [], collections.Counter()
    by_concept: dict = collections.Counter()
    started, fresh = time.time(), 0
    with judge.batch():
        for index, (concept, relation, obj) in enumerate(rows):
            text = question(concept, relation, obj)
            if not text:
                counts["unphrasable"] += 1
                continue
            holds, weight, cached = judge.judge("", "", text)
            fresh += not cached
            if fresh and not fresh % 2000:
                judge._save()                       # noqa: SLF001
            counts["asked"] += 1
            if not holds and weight >= floor:
                demoted.append([concept, relation, obj])
                by_concept[concept] += 1
                counts["demoted"] += 1
            elif not holds:
                counts["denied_under_floor"] += 1
            else:
                counts["kept"] += 1
            if index and not index % 1000:
                rate = fresh / max(time.time() - started, 1e-9)
                print(f"[prune] {index}/{len(rows)}, {fresh} fresh at "
                      f"{rate:.1f}/s, {counts['demoted']} demoted", flush=True)

    DEMOTED.write_text(json.dumps(sorted(demoted), indent=1),
                       encoding="utf-8")
    return {"candidates": len(rows), "floor": floor, "widest": widest,
            "counts": dict(counts),
            "share_demoted": (round(counts["demoted"] / counts["asked"], 4)
                              if counts["asked"] else 0.0),
            "worst_nodes": by_concept.most_common(10),
            "seconds": round(time.time() - started, 1),
            "written_to": str(DEMOTED)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Demote class-node facts that are not claims about the "
                    "class.")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--plan", action="store_true",
                        help="size the candidate set and stop; no model")
    parser.add_argument("--floor", type=float, default=FLOOR)
    parser.add_argument("--band", type=int, default=WIDEST,
                        help="fewest descendants a node needs before its "
                             "facts are judged")
    parser.add_argument("--limit", type=int, default=0)
    options = parser.parse_args(argv)

    if options.plan:
        from research.v687.reasoning import ReasoningEngine

        engine = ReasoningEngine(audit.STORE)
        rows = candidates(engine, options.band)
        where = collections.Counter(row[0] for row in rows)
        print(f"candidates on nodes with >= {options.band} descendants: "
              f"{len(rows)}")
        print(f"nodes: {len(where)}   at 20/s: "
              f"{len(rows) / 20 / 60:.0f} GPU-minutes")
        for node, count in where.most_common(8):
            print(f"  {node:<28}{count:>7} facts")
        print("\nsample questions:")
        for row in rows[:8]:
            print(f"   {question(*row)}")
        return 0
    if options.build:
        print(json.dumps(build(floor=options.floor, widest=options.band,
                               limit=options.limit), indent=2))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
