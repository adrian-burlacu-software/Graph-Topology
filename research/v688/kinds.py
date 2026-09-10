"""Give R19 kinds to count, where the norms cover too few to speak.

`densify.py` added *properties* to the concepts the norms already cover.
This adds *concepts*, which is the other half and the one deferred twice.

Every norm this project has used covers **571 concepts** -- 521 from XCSLB and
50 from AwA2 -- and `Profiles.corroboration` draws its denominator from
exactly those. So when an ancestor has fewer than `CORROBORATION_MIN_KINDS`
of them beneath it, R19 cannot speak at all: **83 of the 118 consulted
ancestors, 195 of 1,503 recorded calls**. `dog.n.01` has three covered kinds.
`turtle.n.02` has one.

That silence is why `AUDIT.md` §19's best unshipped result is unshipped.
Making R19 a precondition rather than a veto measured +2.1 accuracy for −1.7
coverage, and it turned `does a beagle bark` into UNKNOWN -- not because
beagles are quiet, but because three dogs is not a sample and nobody had
asked any of them.

## Dense mode: witnesses that can testify together

`--build` asks each witness only about the terms the recording saw, which is
cheap and leaves most witnesses unable to speak about most terms. That is
safe (see below) and nearly useless for a **precondition** rule, because
`CORROBORATION_MIN_KINDS` needs *eight witnesses able to speak about the same
term at once*. Witnesses each covering a random half of an ancestor's facts
give an expected 4 speakers out of 8, and R19 stays silent.

So `--dense` organises the work **per ancestor rather than per witness**: up
to `WITNESSES` concepts under each consulted ancestor, each asked about
**every** inheritable fact on that ancestor. They then share an identical
`asked` set by construction and can always testify together.

That also collapses the artifact. R19 is only ever consulted at `(A, term)`
when `term` matched a fact on `A` -- that is why it was consulted -- so a
witness asked about all of `A`'s facts can speak to *any* term R19 raises
there. The artifact therefore records **which ancestors a witness was asked
at**, not the thousands of claims it was asked; `asked_at` is a list of
synsets and stays a few kilobytes.

## A kind counts only where it has testimony

Adding kinds is more dangerous than adding properties. A new kind would enter
the **denominator for every term asked at that ancestor**, including the ones
it was never asked about, where it contributes nothing: five thin kinds under
`dog.n.01` take `bearing / 3` to `bearing / 8` and make refusal *more* likely.
That is the sparsity trap one level down -- the exact thing §17 diagnosed --
and it would be self-inflicted.

Asking each new kind about everything would avoid it and costs 11 GPU-hours:
806,597 cells, a median of 2,621 inheritable facts per concept's ancestry.
Capping that is arbitrary and only bounds the damage rather than removing it.

So a distilled kind is counted **only for terms it was actually asked
about**. It is a witness about specific claims, not a warm body in a
denominator. `dog.n.01` gains five kinds for `transport`, which it was asked,
and gains none for `bark`, which it was not -- so R19 speaks where there is
evidence and stays silent where there is none, which is what it should have
done all along. The artifact therefore records what was **asked** as well as
what was affirmed, and both are read.

The honest limitation: this helps on the terms the recording saw and nowhere
else. `does a beagle bark` stays UNKNOWN under a precondition rule, because
`bark` is not a term COMPS ever asks and so is not a term anybody was asked.
Widening that means widening the recording, not this file.

## Which concepts

A greedy cover over the 66 starved ancestors that have enough candidates
beneath them to reach eight: concepts already in the store, carrying facts,
not already covered by the norms. 319 of them close every one of the 66.

Sense ambiguity -- this project's most persistent defect -- does not arise:
candidates are synset ids taken from the taxonomy, so `mouse` and `sheep`
cannot resolve to a pointing device and a docile person on the way in.

## Provenance, again

Written to `derived/distilled_kinds.json` and read only by
`Profiles.corroboration`, beside `distilled_norms.json` and for the same
reason: `identify.stated` stays exactly what people were asked, so the
predicate trie, `_from_below`, the profile display and the audit's `shipped`
control are untouched. `V687_NO_DISTILLED_KINDS=1` ablates it.

The one thing it does change beyond the arithmetic is the wording of R19's
note, which used to say "the kinds of X **the norms cover**". It now says
"on record", because with this loaded that sentence would otherwise be false.
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
KINDS = ROOT / "derived" / "distilled_kinds.json"

#: How deep below a starved ancestor to look for candidates. Four levels
#: reaches breeds and varieties without wandering into a different kind of
#: thing.
DEPTH = 4

#: Same floor as `densify.FLOOR`, and for the same measured reason: at 0.90
#: the extra predicates are the crawl's noise rather than typicality.
FLOOR = 0.99

#: Witnesses per ancestor in `--dense`. Eight is `CORROBORATION_MIN_KINDS`
#: exactly and leaves no margin: one unphrasable question and the ancestor
#: drops below the floor and goes silent again. Ten costs 11% more and 94 of
#: the 118 consulted ancestors can still be fed at least eight.
WITNESSES = 10

#: Cells between artifact writes. `--dense` is a nine-hour run and the
#: judgement cache alone is not enough to recover it: the cache makes a
#: re-run fast, but only a written artifact is usable.
SAVE_EVERY = 40000


def context(engine) -> tuple:
    """(kinds_of, terms, weight) from the recorded R19 calls."""
    profiles = engine.profiles
    kinds_of: dict = collections.defaultdict(list)
    for name, above in profiles._lineage().items():         # noqa: SLF001
        if profiles.stated.get(name):
            for node in above:
                kinds_of[node].append(name)
    terms: dict = collections.defaultdict(set)
    weight: dict = collections.Counter()
    for row in json.loads(densify.CALLS.read_text(encoding="utf-8"))["calls"]:
        terms[row["ancestor"]].add(row["term"])
        weight[row["ancestor"]] += row["hits"]
    return kinds_of, terms, weight


def descendants(engine, node: str, depth: int = DEPTH) -> set:
    conn = engine.reasoner.connection
    seen, edge = set(), {node}
    for _ in range(depth):
        marks = ",".join("?" * len(edge))
        edge = {row[0] for row in conn.execute(
            f"SELECT child FROM taxonomy WHERE parent IN ({marks})",
            tuple(edge))} - seen
        if not edge:
            break
        seen |= edge
    return seen


def cover(engine) -> tuple:
    """The fewest new concepts that let R19 speak at every starved ancestor.

    Greedy, and ties broken toward the concept that serves *fewer* ancestors,
    so a specific concept is preferred to one sitting under half the taxonomy.
    """
    from research.v687.profile import CORROBORATION_MIN_KINDS as MIN

    profiles = engine.profiles
    conn = engine.reasoner.connection
    kinds_of, _terms, weight = context(engine)
    have_facts = {row[0] for row in
                  conn.execute("SELECT DISTINCT concept FROM facts")}
    normed = {profiles.synset[name] for name in profiles.stated
              if name in profiles.synset}

    pool_of, short = {}, {}
    for ancestor in weight:
        have = len(kinds_of.get(ancestor, []))
        if have >= MIN:
            continue
        pool = [c for c in descendants(engine, ancestor)
                if c in have_facts and c not in normed]
        if len(pool) >= MIN - have:
            pool_of[ancestor] = pool
            short[ancestor] = MIN - have

    serves: dict = collections.defaultdict(set)
    for ancestor, pool in pool_of.items():
        for concept in pool:
            serves[concept].add(ancestor)

    need, chosen = dict(short), []
    while need:
        best = max(serves,
                   key=lambda c: (len(serves[c] & need.keys()), -len(serves[c])),
                   default=None)
        if best is None or not (serves[best] & need.keys()):
            break
        chosen.append(best)
        for ancestor in list(serves[best] & need.keys()):
            need[ancestor] -= 1
            if need[ancestor] <= 0:
                del need[ancestor]
        del serves[best]
    return chosen, short, need


def cells(engine, chosen: list) -> list:
    """(concept, ancestor, relation, object) for every new kind.

    The terms come from **every** recorded ancestor above the concept, not
    only the starved one it was chosen for, so a kind chosen for `dog.n.01`
    can also witness at `carnivore.n.01` when it was asked something there.
    """
    _kinds_of, terms, _weight = context(engine)
    identifier = engine.profiles.identifier
    out, seen = [], set()
    for concept in chosen:
        above = {node for node, distance, _ in engine.reasoner.ascend(concept)
                 if distance}
        for ancestor in sorted(above & terms.keys()):
            facts = [fact for fact in engine.reasoner.facts_of(ancestor)
                     if rules.inheritable(fact.relation)
                     and fact.relation in teacher.READS]
            for term in sorted(terms[ancestor]):
                best = None
                for fact in facts:
                    if identifier._hit(term, frozenset({str(fact.object)})):
                        if best is None or fact.confidence > best.confidence:
                            best = fact
                if best is None:
                    continue
                key = (concept, best.relation, str(best.object))
                if key in seen:
                    continue
                seen.add(key)
                out.append((concept, ancestor, best.relation,
                            str(best.object)))
    return out


def dense_plan(engine, witnesses: int = WITNESSES) -> tuple:
    """(ancestor -> its witnesses, ancestor -> its inheritable facts).

    Per ancestor, so every witness under one shares an identical `asked` set
    and they can testify together. Concepts already in `distilled_kinds` are
    preferred, so a dense run extends the cheap one rather than replacing it.
    """
    from research.v687.profile import CORROBORATION_MIN_KINDS as MIN

    profiles = engine.profiles
    conn = engine.reasoner.connection
    _kinds_of, _terms, weight = context(engine)
    have_facts = {row[0] for row in
                  conn.execute("SELECT DISTINCT concept FROM facts")}
    normed = {profiles.synset[name] for name in profiles.stated
              if name in profiles.synset}
    already = set(profiles.distilled_kinds)

    chosen, facts = {}, {}
    for ancestor in sorted(weight, key=lambda node: -weight[node]):
        pool = [c for c in descendants(engine, ancestor)
                if c in have_facts and c not in normed]
        if len(pool) < MIN:
            continue
        pool.sort(key=lambda c: (c not in already, c))
        on = [fact for fact in engine.reasoner.facts_of(ancestor)
              if rules.inheritable(fact.relation)
              and fact.relation in teacher.READS]
        if not on:
            continue
        chosen[ancestor] = pool[:witnesses]
        facts[ancestor] = on
    return chosen, facts


def dense_cells(chosen: dict, facts: dict) -> list:
    """(witness, ancestor, relation, object), busiest ancestor first.

    Order matters for a run that may be interrupted: finishing the ancestors
    R19 is consulted at most often first means a partial run is still worth
    something.
    """
    out = []
    for ancestor, members in chosen.items():
        for fact in facts[ancestor]:
            for member in members:
                out.append((member, ancestor, fact.relation, str(fact.object)))
    return out


def build_dense(engine=None, floor: float = FLOOR, witnesses: int = WITNESSES,
                limit: int = 0, judge=None) -> dict:
    """Ask every witness about every fact on the ancestor it witnesses for."""
    from research.v687.reasoning import ReasoningEngine

    engine = engine or ReasoningEngine(audit.STORE)
    chosen, facts = dense_plan(engine, witnesses)
    grid = dense_cells(chosen, facts)
    if limit:
        grid = grid[:limit]
    judge = judge or teacher.Teacher()
    if not getattr(judge, "available", False):
        return {"error": getattr(judge, "error", "") or "no teacher"}

    # Seeded from whatever is already there, so a dense run *extends* the
    # cheap one. Without this an interrupted nine-hour run would leave the
    # artifact holding only the ancestors it happened to finish, which is
    # strictly worse than the 312 witnesses it started with.
    from research.v687 import corpora

    affirmed: dict = collections.defaultdict(set)
    asked_at: dict = collections.defaultdict(set)
    asked: dict = collections.defaultdict(set)
    for concept, one in corpora.load_distilled_kinds().items():
        affirmed[concept] |= set(one["predicates"])
        asked_at[concept] |= set(one["asked_at"])
        asked[concept] |= set(one["asked"])
    counts: dict = collections.Counter()
    started, fresh = time.time(), 0

    def save() -> None:
        rows = {concept: {"name": name_of(engine, concept),
                          "asked_at": sorted(asked_at[concept]),
                          "asked": sorted(asked[concept]),
                          "predicates": sorted(affirmed.get(concept, ()))}
                for concept in sorted(set(asked_at) | set(asked))}
        KINDS.write_text(json.dumps(rows, indent=1, sort_keys=True),
                         encoding="utf-8")

    # An ancestor is usable only once *all* its witnesses have been asked, so
    # `asked_at` is credited when its block finishes rather than per cell. A
    # half-asked ancestor would claim witnesses that cannot actually testify
    # together, which is the failure this mode exists to fix.
    done: dict = collections.Counter()
    want = {node: len(facts[node]) * len(chosen[node]) for node in chosen}
    with judge.batch():
        for index, (member, ancestor, relation, obj) in enumerate(grid):
            text = densify.question(name_of(engine, member), relation, obj)
            if text:
                holds, weight, cached = judge.judge("", "", text)
                fresh += not cached
                counts["asked"] += 1
                if holds and weight >= floor:
                    affirmed[member].add(
                        teacher.stated(relation, densify.plain(obj)))
                    counts["written"] += 1
                elif holds:
                    counts["under_floor"] += 1
                else:
                    counts["denied"] += 1
            else:
                counts["unphrasable"] += 1
            done[ancestor] += 1
            if done[ancestor] == want[ancestor]:
                for one in chosen[ancestor]:
                    asked_at[one].add(ancestor)
                counts["ancestors_done"] += 1
            if fresh and not fresh % 2000:
                judge._save()                       # noqa: SLF001
            if index and not index % SAVE_EVERY:
                save()
            if index and not index % 5000:
                rate = fresh / max(time.time() - started, 1e-9)
                left = (len(grid) - index) / max(rate, 1e-9) / 3600
                print(f"[kinds] {index}/{len(grid)} ({index / len(grid):.1%}),"
                      f" {fresh} fresh at {rate:.1f}/s, "
                      f"{counts['ancestors_done']} ancestors done, "
                      f"~{left:.1f}h left", flush=True)
    save()
    return {"mode": "dense", "ancestors": len(chosen),
            "ancestors_complete": counts["ancestors_done"],
            "witnesses": len(asked_at), "cells": len(grid), "floor": floor,
            "counts": dict(counts),
            "predicates": sum(len(one) for one in affirmed.values()),
            "hours": round((time.time() - started) / 3600, 2),
            "written_to": str(KINDS)}


def name_of(engine, concept: str) -> str:
    return densify.plain(concept.rsplit(".", 2)[0])


def build(engine=None, floor: float = FLOOR, limit: int = 0,
          judge=None) -> dict:
    from research.v687.reasoning import ReasoningEngine

    engine = engine or ReasoningEngine(audit.STORE)
    chosen, short, still = cover(engine)
    grid = cells(engine, chosen)
    if limit:
        grid = grid[:limit]
    judge = judge or teacher.Teacher()
    if not getattr(judge, "available", False):
        return {"error": getattr(judge, "error", "") or "no teacher"}

    written: dict = collections.defaultdict(set)
    asked: dict = collections.defaultdict(set)
    counts = collections.Counter()
    started, fresh = time.time(), 0
    with judge.batch():
        for index, (concept, _ancestor, relation, obj) in enumerate(grid):
            text = densify.question(name_of(engine, concept), relation, obj)
            if not text:
                counts["unphrasable"] += 1
                continue
            holds, weight, cached = judge.judge("", "", text)
            fresh += not cached
            if fresh and not fresh % 2000:
                judge._save()                       # noqa: SLF001
            counts["asked"] += 1
            asked[concept].add(teacher.stated(relation, densify.plain(obj)))
            if holds and weight >= floor:
                written[concept].add(
                    teacher.stated(relation, densify.plain(obj)))
                counts["written"] += 1
            elif holds:
                counts["under_floor"] += 1
            else:
                counts["denied"] += 1
            if index and not index % 500:
                rate = fresh / max(time.time() - started, 1e-9)
                print(f"[kinds] {index}/{len(grid)}, {fresh} fresh at "
                      f"{rate:.1f}/s, {counts['written']} written", flush=True)

    # Both halves are written. `asked` is the denominator -- the claims this
    # kind can testify about at all -- and `predicates` is the numerator. A
    # kind that answered "no" to everything is still a witness and still
    # counts against, which is the whole point; one that was never asked
    # about a term does not appear in that term's arithmetic at all.
    rows = {concept: {"name": name_of(engine, concept),
                      "asked": sorted(asked.get(concept, ())),
                      "predicates": sorted(written.get(concept, ()))}
            for concept in chosen}
    KINDS.write_text(json.dumps(rows, indent=1, sort_keys=True),
                     encoding="utf-8")
    return {"ancestors_fed": len(short) - len(still),
            "ancestors_still_short": len(still),
            "concepts": len(chosen), "cells": len(grid), "floor": floor,
            "counts": dict(counts),
            "predicates": sum(len(one["predicates"]) for one in rows.values()),
            "seconds": round(time.time() - started, 1),
            "written_to": str(KINDS)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Distil new kinds for the ancestors R19 cannot speak at.")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--dense", action="store_true",
                        help="ask every witness about every fact on the "
                             "ancestor it witnesses for, so eight of them can "
                             "testify about the same term at once")
    parser.add_argument("--witnesses", type=int, default=WITNESSES)
    parser.add_argument("--plan", action="store_true",
                        help="size the cover and stop; no model")
    parser.add_argument("--floor", type=float, default=FLOOR)
    parser.add_argument("--limit", type=int, default=0)
    options = parser.parse_args(argv)

    if options.plan and options.dense:
        from research.v687.reasoning import ReasoningEngine

        engine = ReasoningEngine(audit.STORE)
        chosen, facts = dense_plan(engine, options.witnesses)
        grid = dense_cells(chosen, facts)
        witnesses = {one for members in chosen.values() for one in members}
        print(f"ancestors with >= 8 witnesses: {len(chosen)}")
        print(f"distinct witnesses: {len(witnesses)}")
        print(f"cells: {len(grid):,}   {len(grid) / 20 / 3600:.1f} GPU-hours")
        busiest = list(chosen)[:6]
        for node in busiest:
            print(f"   {node:<26}{len(chosen[node]):>3} witnesses x "
                  f"{len(facts[node]):>5} facts")
        return 0
    if options.plan:
        from research.v687.reasoning import ReasoningEngine

        engine = ReasoningEngine(audit.STORE)
        chosen, short, still = cover(engine)
        grid = cells(engine, chosen)
        print(f"starved ancestors that can be fed: {len(short)}")
        print(f"  still short after the cover: {len(still)}")
        print(f"concepts to add: {len(chosen)}")
        print(f"cells to ask: {len(grid)}   "
              f"{len(grid) / 20 / 60:.0f} GPU-minutes")
        per = collections.Counter(row[0] for row in grid)
        if per:
            counts = sorted(per.values())
            print(f"  per concept: median {counts[len(counts) // 2]}, "
                  f"max {counts[-1]}")
        for concept in chosen[:8]:
            print(f"   {concept:<28}{per.get(concept, 0):>4} questions")
        return 0
    if options.dense:
        print(json.dumps(build_dense(floor=options.floor,
                                     witnesses=options.witnesses,
                                     limit=options.limit), indent=2))
        return 0
    if options.build:
        print(json.dumps(build(floor=options.floor, limit=options.limit),
                         indent=2))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
