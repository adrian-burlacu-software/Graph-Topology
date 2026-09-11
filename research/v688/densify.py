"""Fill the cells R19 actually reads, so its silences stop meaning "no".

§17 established the case on AwA2: free listing gives R19 the wrong verdict
more than half the time, because `corroboration` counts a kind as not bearing
a term out when nobody was ever asked, and XCSLB is 0.66% dense. This builds
the fix for the real store.

## Which cells, and why not the ones §17 proposed

§17 proposed a greedy set cover over taxonomically narrow classes. Recording
what R19 is *actually* asked -- 1,503 consultations over 1,468 audit
questions -- says that was the wrong target. The ancestors R19 reaches are
basic level and chosen by where the crawl put the fact, not by how narrow the
class is:

    animal.n.01      317 calls   143 norm-covered kinds
    plant.n.02       142          52
    furniture.n.01   131          17
    device.n.01       87          59
    bird.n.01         85          29

118 ancestors in all, and **not one of them has more than 150 norm-covered
kinds**. So there is no set cover to do and no new concepts to norm. The 477
concepts XCSLB and AwA2 already cover are the kinds R19 already consults;
they are simply nearly empty.

Restricting to the 35 ancestors with 8 to 150 kinds -- 8 is R19's own
refusal floor, and above 150 nothing is ever consulted -- gives **18,066
cells covering 87% of observed consultations**, about fifteen GPU-minutes.
Of those cells the existing norms bear out **10.1%**. The other 89.9% are
the silences.

The 83 ancestors with fewer than 8 kinds account for the remaining 13%, and
they need new *concepts* rather than new properties. That is a different and
harder job, deliberately not done here.

## How a cell is asked and what is stored

A cell is a member concept and one inheritable fact on an ancestor above it
-- exactly the fact R19 would be corroborating. Both halves of the phrasing
are settled by measurements already made:

- **The question is the bare claim**, `audit.phrase` over the fact read
  plainly: `does a leopard hunt at night`. That is the path §16 calibrated
  and §17 validated at 90% on cells humans agree about. The `PROMPT` path
  that supplies a fact is for adjudication, not for asking whether something
  is true.
- **What is stored is `teacher.stated(relation, object)`** -- `has_a` +
  `wing` becomes `has a wing`. It has to be a phrase and not the bare term,
  because `_hit` matches a query word against *any word of a predicate*, so
  storing the readable phrase makes every word in it reachable. That was
  condition 3 of §17 and it is answered by construction here rather than
  hoped for.

## What it does not touch

`identify.stated` is left exactly as it is, and so are the predicate trie,
`_from_below`, and every display path. Distilled norms land in their own
dict that only `Profiles.corroboration` consults, which is the one place §17
gives evidence for changing. That keeps the provenance separable -- condition
1 -- without a migration, and it means the ablation is a single flag.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import time
from pathlib import Path

from research.v687 import rules
from . import audit, teacher

ROOT = Path(__file__).resolve().parents[2]
NORMS = ROOT / "derived" / "distilled_norms.json"
CALLS = ROOT / "derived" / "r19-calls.json"

#: Nothing above 150 norm-covered kinds was ever consulted in 1,503 recorded
#: calls, so above that a cell is never read.
#:
#: The floor was 8 -- R19's own refusal threshold -- on the reasoning that
#: below it R19 cannot speak, so filling those cells buys nothing. That was
#: right for R19 as a veto and wrong for R19 as a precondition
#: (`profile.CORROBORATION_REQUIRED`), where an ancestor with three covered
#: kinds and nothing borne out is a *refusal*. `dog.n.01` has three, none of
#: them was ever asked whether it barks, and `does a beagle bark` became
#: UNKNOWN. The 83 ancestors below 8 kinds are 411 cells -- twenty seconds --
#: so there is no reason to leave them empty.
FEWEST_KINDS = 1
MOST_KINDS = 150

#: What a distilled norm must be worth before it is written, and the whole
#: difference between this being worth shipping and not. Measured on the
#: recorded sample, `corroborated` against screened gold:
#:
#:     floor   coverage   accuracy   over-affirmed   R19 verdicts flipped
#:     off        20.3%      88.9%            3.8%                     --
#:     0.99       20.5%      89.0%            3.8%                  11.8%
#:     0.90       20.5%      88.0%            4.0%                  34.3%
#:
#: 0.90 buys the same coverage and costs a point of accuracy, because the
#: extra flips are the crawl's noise rather than anybody's typicality:
#: `animal capable_of used` goes from 14 of 143 to 129. 0.99 keeps the flips
#: that are real -- `mammal has fur` 13 -> 50 of 65, `mammal is warm
#: blooded` 4 -> 48, `bird eats food` 2 -> 27 of 29 -- and leaves
#: over-affirmation flat on both the screened foils and §14's model-free
#: corrupted claims.
#:
#: It matches `teacher.TEACHING_FLOOR` by coincidence rather than by
#: argument: teaching invents a fact the store will assert, while this
#: supplies one vote of eight or more in a ratio that has to clear a third.
#: The reasoning is different and the number came out the same.
FLOOR = 0.99


def observed(path: Path = CALLS) -> dict:
    """ancestor -> the terms R19 was recorded being asked about it.

    `research/v688/record_r19.py` writes this. Without it every inheritable term on every
    ancestor is fair game, which is 824,431 cells and eleven GPU-hours.
    """
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8")).get("calls", [])
    out: dict = collections.defaultdict(set)
    for row in rows:
        out[row["ancestor"]].add(row["term"])
    return dict(out)


def cells(engine, terms: dict) -> list:
    """(member, ancestor, relation, object) for every cell worth asking.

    One fact per (ancestor, term): the highest-confidence inheritable fact
    whose object names the term, which is the one R19's own `better` sort
    would have reached for.
    """
    profiles = engine.profiles
    identifier = profiles.identifier
    kinds_of: dict = collections.defaultdict(list)
    for name, above in profiles._lineage().items():        # noqa: SLF001
        if profiles.stated.get(name):
            for node in above:
                kinds_of[node].append(name)

    out, seen = [], set()
    for ancestor, wanted in sorted(terms.items()):
        kinds = kinds_of.get(ancestor, [])
        if not (FEWEST_KINDS <= len(kinds) <= MOST_KINDS):
            continue
        # A relation `teacher.READS` has no frame for phrases as the bare
        # object, and the question comes out malformed -- `has_prerequisite`
        # + `cold or warm water` asks `does a gown cold or warm water`. The
        # model answers such a question rather than refusing it, so the norm
        # would be noise wearing a confidence score. 100 cells of 17,627.
        facts = [fact for fact in engine.reasoner.facts_of(ancestor)
                 if rules.inheritable(fact.relation)
                 and fact.relation in teacher.READS]
        for term in sorted(wanted):
            best = None
            for fact in facts:
                if identifier._hit(term, frozenset({str(fact.object)})):
                    if best is None or fact.confidence > best.confidence:
                        best = fact
            if best is None:
                continue
            for member in kinds:
                key = (member, best.relation, str(best.object))
                if key in seen:
                    continue
                seen.add(key)
                out.append((member, ancestor, best.relation,
                            str(best.object)))
    return out


#: A sense-tagged object as a reader would say it: `animal tissue.n.01` ->
#: `animal tissue`. Relations that point at a concept rather than at free
#: text carry the synset id, and asking the model `does a bat have animal
#: tissue.n.01` measures the model's patience rather than its knowledge.
SENSED = re.compile(r"^(.*)\.[nvasr]\.\d{2}$")


def plain(obj: str) -> str:
    found = SENSED.match((obj or "").strip())
    return found.group(1) if found else (obj or "").strip()


def question(member: str, relation: str, obj: str) -> str:
    """The cell as the bare claim `audit.phrase` would build."""
    return audit.phrase(teacher.article(member),
                        teacher.stated(relation, plain(obj)))


#: Relations that would be put to `teacher.CAPABLE` instead of `CAREFUL`.
#: **Empty, and that is the finding** -- `AUDIT.md` §24.
#:
#: The capability prompt is the better *capability judge* by a wide margin.
#: Against 400 true and 400 false capability claims from the screened
#: benchmark, at the 0.99 floor these cells are written at, `careful` affirms
#: 33.8% of the true ones and `capable` 73.0%, both with **zero** false
#: positives. Both answer all eight of R19's guard claims correctly.
#:
#: Distilled with it, every artifact grew about threefold -- 1,454 predicates
#: to 4,789 -- and the system got worse: 92.1% accuracy to 90.9%, and
#: over-affirmation 2.2% to 2.5%. Sweeping the corroboration floor to 0.9 did
#: not recover it.
#:
#: **R19 does not want a capability judge.** Its question is whether an
#: inherited fact is a claim *about the class*, and `can fall into a hole` is
#: not a property of canines in any useful sense however true it is of every
#: one of them. Typicality is the right test even for `capable_of`, and
#: `careful` was answering the right question all along.
#:
#: Kept as a name rather than deleted because §21 predicted the opposite and
#: someone will want to try it again.
CAPABILITY: frozenset = frozenset()


def style_for(relation: str) -> str:
    return "capable" if relation in CAPABILITY else ""


def build(engine=None, floor: float = FLOOR, limit: int = 0,
          judge=None) -> dict:
    """Ask every cell and write the norms that clear the floor."""
    from research.v687.reasoning import ReasoningEngine

    engine = engine or ReasoningEngine(audit.STORE)
    terms = observed()
    if not terms:
        return {"error": f"no {CALLS}; run record_r19.py first"}
    grid = cells(engine, terms)
    if limit:
        grid = grid[:limit]
    judge = judge or teacher.Teacher()
    if not getattr(judge, "available", False):
        return {"error": getattr(judge, "error", "") or "no teacher"}

    written: dict = collections.defaultdict(set)
    counts = collections.Counter()
    started, fresh = time.time(), 0
    with judge.batch():
        for index, (member, _ancestor, relation, obj) in enumerate(grid):
            text = question(member, relation, obj)
            if not text:
                counts["unphrasable"] += 1
                continue
            holds, weight, cached = judge.judge("", "", text,
                                                style_for(relation))
            fresh += not cached
            if fresh and not fresh % 2000:
                judge._save()                       # noqa: SLF001
            counts["asked"] += 1
            counts[f"asked_{style_for(relation) or 'careful'}"] += 1
            if holds and weight >= floor:
                written[member].add(teacher.stated(relation, plain(obj)))
                counts["written"] += 1
            elif holds:
                counts["under_floor"] += 1
            else:
                counts["denied"] += 1
            if index and not index % 1000:
                rate = fresh / max(time.time() - started, 1e-9)
                print(f"[densify] {index}/{len(grid)}, {fresh} fresh at "
                      f"{rate:.1f}/s, {counts['written']} written", flush=True)

    NORMS.write_text(json.dumps(
        {name: sorted(predicates) for name, predicates in sorted(
            written.items())}, indent=1), encoding="utf-8")
    return {"cells": len(grid), "floor": floor, "concepts": len(written),
            "counts": dict(counts),
            "predicates": sum(len(v) for v in written.values()),
            "seconds": round(time.time() - started, 1),
            "written_to": str(NORMS)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Distil the norms R19 reads, for the concepts it "
                    "already consults.")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--plan", action="store_true",
                        help="size the grid and stop; no model")
    parser.add_argument("--floor", type=float, default=FLOOR)
    parser.add_argument("--limit", type=int, default=0)
    options = parser.parse_args(argv)

    if options.plan:
        from research.v687.reasoning import ReasoningEngine

        engine = ReasoningEngine(audit.STORE)
        terms = observed()
        grid = cells(engine, terms)
        print(f"ancestors recorded: {len(terms)}")
        print(f"cells in band {FEWEST_KINDS}..{MOST_KINDS}: {len(grid)}")
        print(f"concepts touched: {len({row[0] for row in grid})}")
        print(f"at 20/s: {len(grid) / 20 / 60:.0f} GPU-minutes")
        for row in grid[:8]:
            print(f"   {question(row[0], row[2], row[3])}")
        return 0
    if options.build:
        print(json.dumps(build(floor=options.floor, limit=options.limit),
                         indent=2))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
