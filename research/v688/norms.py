"""Distil property norms with the model, and check them against a closed matrix.

Yesterday's teaching wrote 3,363 leaf facts over 409 concepts for +0.7
coverage and no transfer, because a fact about a beagle helps beagles. The
proposal this file tests is that the same GPU hours spent on **norms** would
transfer, because a norm is not an answer -- it is the evidence R19 weighs
before believing an inherited fact, and one norm-covered sibling serves every
question about every member of its class.

Two things have to be true for that to be worth doing, and neither was known:

1. **A distilled norm has to be about as good as an elicited one.** The model
   is a worse epistemic agent than the store (`AUDIT.md` §13). If its norms
   are noise, R19 weighs noise.
2. **Density has to be the thing that was wrong.** If R19 already gets the
   right answer from XCSLB, none of this matters.

AwA2 settles both, because it is the only **closed** gold here: 50 classes
each scored on every one of 85 attributes, so the matrix states what is false
as well as what is true. That is precisely what XCSLB lacks and what
`screen.py` used one level up.

## What is measured

    distil      the model's yes/no on all 50 x 83 cells -- a dense matrix
    validate    that matrix against AwA2's, per confidence floor
    corroborate R19's actual verdict, computed three ways over the same
                ancestors and attributes: from AwA2 (right by construction),
                from the distilled matrix, and from XCSLB's free listing

The third is the one that matters. R19 asks `bearing / kinds >= 1/3` over an
ancestor's norm-covered kinds, and the question is not whether a norm source
is accurate in the abstract but **whether it makes R19 say the same thing the
closed matrix would.** A source can be individually noisy and still give the
right verdict, because a third of eight siblings is a blunt threshold; a
source can be individually accurate and still give the wrong one, if what it
omits is correlated. Free listing omits exactly what is typical -- people
describing a leopard say `can pounce`, not `has four legs` -- which is the
correlation that hurts most, since typicality is the whole of what R19 tests.

## What this deliberately does not do

It does not touch `identify.stated`, `profile.corroboration`, or anything
v687 answers from. It is a measurement, and it simulates R19's arithmetic
rather than calling it, for one reason worth stating: real R19 reaches its
evidence through `_hit`, a whole-word stem match between a fact's object and
a norm's predicate. Feeding it AwA2 attributes would measure that matcher as
much as the norms. Here both sides are attribute names, so the **evidence**
question is isolated from the **matching** question. If the evidence question
comes out well, the matching one is the next thing to answer and not this
file's business.
"""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import time
from pathlib import Path

from . import screen

ROOT = Path(__file__).resolve().parents[2]
AWA2 = ROOT / "data" / "awa2" / "Animals_with_Attributes2"
STORE = ROOT / "data" / "v684_reasoning.sqlite"
OUT = ROOT / "research" / "v688" / "audit-out"

#: R19's constants, imported so the simulation cannot drift from the rule it
#: simulates. They were copied at first, `test_distil.py` asserted the copies
#: matched, and §19 moved the floor from 1/3 to 0.5 and the assertion caught
#: it -- which is the argument for importing. `profile` pulls in no store, so
#: this file still runs without building a reasoner.
#:
#: **§17's tables were measured at a floor of 1/3.** Re-running `--corroborate`
#: now uses 0.5 and will not reproduce them exactly; the comparison between
#: the three evidence bases is what that section rests on, and it is scored
#: under whatever floor is current, the same one for all three.
from research.v687.profile import (                      # noqa: E402
    CORROBORATION_FLOOR as FLOOR,
    CORROBORATION_MIN_KINDS as MIN_KINDS)

#: How far up from a class to look for ancestors. `profile._lineage` uses
#: `reasoner.ascend`, which is the same walk.
DEPTH = 12

#: An ancestor is worth corroborating at when it is specific enough to say
#: something. `AUDIT.md` §16's follow-up measured that 61% of concepts get
#: corroborated only against 1,000+ siblings, which is the `animal.n.01
#: "wings" -> 20 of 143` case the rule was written to avoid.
NARROW = 300


# -- the three sources ----------------------------------------------------

def read(path: Path) -> list[str]:
    return [line.split("\t")[-1].strip() if "\t" in line else line.split()[-1]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def classes() -> list[str]:
    return [name.replace("+", " ") for name in read(AWA2 / "classes.txt")]


def attributes() -> list[str]:
    """The 83 AwA2 attributes `screen.py` can phrase as a question.

    `newworld` and `oldworld` are excluded there and so here: biogeographic
    range is not a property anybody would answer about a beaver, and a judge
    that gets them wrong is not making the kind of mistake being measured.
    """
    return [name for name in read(AWA2 / "predicates.txt")
            if name in screen.ATTRIBUTES]


def gold() -> dict:
    """(class, attribute) -> True/False, from the published binary matrix.

    Closed: every cell was scored, so a False here is a judgement and not a
    silence. That is the entire reason this experiment is possible.
    """
    names, predicates = classes(), read(AWA2 / "predicates.txt")
    rows = [line.split() for line in
            (AWA2 / "predicate-matrix-binary.txt")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    keep = set(attributes())
    return {(name, attribute): float(value) > 0
            for name, flags in zip(names, rows)
            for attribute, value in zip(predicates, flags)
            if attribute in keep}


def confidence_of_gold() -> dict:
    """(class, attribute) -> the continuous rating behind the binary cell.

    Used to split the validation into cells humans were sure about and cells
    they were not. A model disagreeing about `is a leopard big` is a
    different event from one disagreeing about `can a leopard fly`.
    """
    names, predicates = classes(), read(AWA2 / "predicates.txt")
    rows = [[float(value) for value in line.split()] for line in
            (AWA2 / "predicate-matrix-continuous.txt")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    keep = set(attributes())
    return {(name, attribute): score
            for name, scores in zip(names, rows)
            for attribute, score in zip(predicates, scores)
            if attribute in keep}


def question(name: str, attribute: str) -> str:
    subject = f"{screen.article(name)} {name}"
    return screen.ATTRIBUTES[attribute].format(s=subject)


def distil(teacher=None, note: str = "distil") -> dict:
    """(class, attribute) -> (holds, confidence). All 50 x 83 cells.

    Dense by construction, which is the point: the model is asked about every
    cell, so its zeros are answers. Judgements cache by claim in
    `llm/adjudications.json` and most are already there -- `screen.py` asked
    the extremes of this same grid to calibrate itself.
    """
    from .teacher import Teacher

    teacher = teacher if teacher is not None else Teacher()
    if not teacher.available:
        return {}
    cells, started, fresh = {}, time.time(), 0
    every = [(name, attribute) for name in classes()
             for attribute in attributes()]
    with teacher.batch():
        for index, (name, attribute) in enumerate(every):
            supports, weight, cached = teacher.judge(
                "", "", question(name, attribute))
            cells[(name, attribute)] = (bool(supports), float(weight))
            fresh += not cached
            if fresh and not fresh % screen.FLUSH:
                teacher._save()                     # noqa: SLF001
            if index and not index % 500:
                rate = fresh / max(time.time() - started, 1e-9)
                print(f"[norms] {note} {index}/{len(every)}, "
                      f"{fresh} fresh at {rate:.1f}/s", flush=True)
    return cells


def free_listing() -> dict:
    """(class, attribute) -> True/False from XCSLB, for the classes it covers.

    The comparison arm, and the reason it is small: XCSLB phrases properties
    as sentences and AwA2 as single words, so the two only meet where a
    lexicon feature says the same thing. `EQUIVALENT` is that join, written by
    hand against `feature_lexicon.csv` rather than fuzzy-matched, because a
    fuzzy match here would be measuring the matcher.

    A False is what nobody happened to list, which is the whole finding.
    """
    from research.v687 import corpora

    listed = {name: set(properties)
              for name, properties in corpora.load_xcslb().items}
    out = {}
    for name in classes():
        if name not in listed:
            continue
        for attribute, features in EQUIVALENT.items():
            out[(name, attribute)] = any(f in listed[name] for f in features)
    return out


#: AwA2 attribute -> the XCSLB features that state the same thing, verified
#: present in `feature_lexicon.csv`. Only attributes with a real equivalent
#: appear; the rest have no XCSLB expression and are left out rather than
#: approximated. 20 of 83, which is why the free-listing arm is reported as
#: a subset and not as a rate over the whole grid.
EQUIVALENT = {
    "quadrapedal": ("has four legs",),
    "bipedal": ("has two legs",),
    "walks": ("can walk", "can walk long distances"),
    "swims": ("can swim",),
    "flys": ("can fly", "can fly fast"),
    "hops": ("can hop",),
    "tail": ("has a tail",),
    "horns": ("has horns",),
    "claws": ("has claws",),
    "hooves": ("has hooves",),
    "paws": ("has paws",),
    "furry": ("has fur",),
    "spots": ("has spots on its body", "has black spots", "has brown spots"),
    "stripes": ("has striped patterns", "can have striped patterns",
                "has black stripes", "has white stripes",
                "has black and white stripes"),
    "agility": ("is agile",),
    "fast": ("can be fast", "can run fast"),
    # `strong` has no XCSLB expression. The lexicon's nearest entries are
    # `has strong muscles` -- which is AwA2's separate `muscle` attribute --
    # and `has a strong smell`, which is something else entirely. Left out
    # rather than approximated; that is what the 20 in the docstring means.
    "domestic": ("is domesticated",),
    "nocturnal": ("is nocturnal",),
    "hibernate": ("can hibernate",),
}


# -- validating the distilled matrix --------------------------------------

def validate(cells: dict, floors: tuple = (0.5, 0.75, 0.9, 0.95, 0.99)) -> dict:
    """The distilled matrix against AwA2's, per confidence floor.

    Below a floor the model's answer is discarded and the cell goes back to
    being unknown, which is what makes the trade legible: a higher floor buys
    accuracy on what remains and loses density, and density is the thing this
    is for.
    """
    truth, sure = gold(), confidence_of_gold()
    rows = []
    for floor in floors:
        kept = {key: holds for key, (holds, weight) in cells.items()
                if weight >= floor and key in truth}
        agree = sum(1 for key, holds in kept.items() if holds == truth[key])
        yes = [key for key, holds in kept.items() if holds]
        hit = sum(1 for key in yes if truth[key])
        actual = [key for key in truth if truth[key]]
        found = sum(1 for key in actual if kept.get(key))
        # the cells humans were not close to unanimous on, excluded
        clear = {key for key, score in sure.items()
                 if score <= 5.0 or score >= 60.0}
        kept_clear = {key: holds for key, holds in kept.items()
                      if key in clear}
        agree_clear = sum(1 for key, holds in kept_clear.items()
                          if holds == truth[key])
        rows.append({
            "floor": floor,
            "density": round(len(kept) / max(len(truth), 1), 4),
            "accuracy": round(agree / len(kept), 4) if kept else 0.0,
            "precision": round(hit / len(yes), 4) if yes else 0.0,
            "recall": round(found / len(actual), 4) if actual else 0.0,
            "n": len(kept),
            "accuracy_on_clear": (round(agree_clear / len(kept_clear), 4)
                                  if kept_clear else 0.0),
            "n_clear": len(kept_clear)})
    return {"cells": len(cells), "gold_cells": len(truth), "rows": rows}


def free_listing_density() -> dict:
    """How dense the two sources are on the same cells. The headline contrast."""
    truth, free = gold(), free_listing()
    shared = [key for key in free if key in truth]
    listed = sum(1 for key in shared if free[key])
    holds = sum(1 for key in shared if truth[key])
    return {"cells_compared": len(shared),
            "awa2_says_yes": holds,
            "xcslb_listed": listed,
            "recall_of_free_listing": (round(listed / holds, 4)
                                       if holds else 0.0)}


# -- the measurement that matters: R19's verdict --------------------------

def synset_of(names: list[str], store: Path = STORE) -> dict:
    """class -> synset, restricted to things under `animal.n.01`.

    Sense selection is this project's most persistent defect -- `sheep`
    resolves to "a docile and vulnerable person" and `mouse` to a pointing
    device -- so the restriction is not tidiness. Every one of these 50 is an
    animal, so saying so removes the whole class of error rather than
    detecting it.
    """
    conn = sqlite3.connect(store)
    parents = collections.defaultdict(set)
    for child, parent in conn.execute("SELECT child, parent FROM taxonomy"):
        parents[child].add(parent)

    def above(node: str) -> set:
        seen, edge = set(), {node}
        for _ in range(DEPTH + 6):
            edge = {p for n in edge for p in parents.get(n, ())} - seen
            if not edge:
                break
            seen |= edge
        return seen

    out = {}
    for name in names:
        rows = conn.execute(
            "SELECT concept FROM lemmas WHERE lemma = ? ORDER BY sense_rank",
            (name,)).fetchall()
        for (concept,) in rows:
            if "animal.n.01" in above(concept) or concept == "animal.n.01":
                out[name] = concept
                break
    conn.close()
    return out


def ancestors_of(synsets: dict, store: Path = STORE) -> dict:
    """ancestor -> the classes beneath it, for ancestors R19 could use.

    Kept only where at least `MIN_KINDS` of the 50 sit beneath, because that
    is R19's own refusal condition, and where the ancestor is narrow enough
    to be worth asking at.
    """
    conn = sqlite3.connect(store)
    parents = collections.defaultdict(set)
    for child, parent in conn.execute("SELECT child, parent FROM taxonomy"):
        parents[child].add(parent)
    size = {row[0]: (row[1] or 0) for row in
            conn.execute("SELECT id, descendants FROM concepts")}
    conn.close()

    under: dict = collections.defaultdict(list)
    for name, concept in synsets.items():
        seen, edge = set(), {concept}
        for _ in range(DEPTH):
            edge = {p for n in edge for p in parents.get(n, ())} - seen
            if not edge:
                break
            seen |= edge
        for node in seen:
            under[node].append(name)
    return {node: sorted(members) for node, members in under.items()
            if len(members) >= MIN_KINDS}, size


def verdict(members: list, attribute: str, source: dict) -> str | None:
    """R19, simulated: believed, refused, or None when it cannot speak.

    `source` maps (class, attribute) to a bool. A class the source does not
    cover is not counted in the denominator -- which is what
    `profile.corroboration` does, since it draws its kinds from the norms.
    """
    kinds = [name for name in members if (name, attribute) in source]
    if len(kinds) < MIN_KINDS:
        return None
    bearing = sum(1 for name in kinds if source[(name, attribute)])
    return "believed" if bearing / len(kinds) >= FLOOR else "refused"


def tally(where: dict, attributes_wanted: list, truth: dict,
          source: dict) -> dict:
    """One evidence base scored against gold, over some ancestors.

    `lost` and `over` are the two ways R19 can be wrong and they are not
    interchangeable: refusing a true inheritance costs coverage, believing a
    false one is the confident falsehood the rule exists to prevent.
    """
    same = collections.Counter()
    pairs = 0
    for members in where.values():
        for attribute in attributes_wanted:
            right = verdict(members, attribute, truth)
            if right is None:
                continue
            pairs += 1
            mine = verdict(members, attribute, source)
            if mine is None:
                same["silent"] += 1
            elif mine == right:
                same["agreed"] += 1
            elif right == "believed":
                same["lost"] += 1              # a true inheritance refused
            else:
                same["over"] += 1              # a false one believed
    total = pairs or 1
    return {"pairs": pairs,
            "agreed": round(same["agreed"] / total, 4),
            "lost_true": round(same["lost"] / total, 4),
            "over_affirmed": round(same["over"] / total, 4),
            "silent": round(same["silent"] / total, 4),
            "counts": dict(same)}


def corroborate(cells: dict, floor: float = 0.5,
                narrow: int = NARROW) -> dict:
    """R19's verdict from three evidence bases, over the same questions.

    The comparison this file exists for. `gold` is right by construction, so
    the other two are scored against it: how often would R19 have reached the
    same conclusion, and when it did not, which way did it go wrong.

    Reported per ancestor as well as in total, because the aggregate hides
    how thin the base is. AwA2's 50 classes are taxonomically spread --
    whales, bats, primates, rodents, ungulates -- so only two ancestors under
    300 descendants hold eight of them, and a two-ancestor headline is a
    claim about `ruminant` and `even-toed ungulate`. The ladder up to
    `entity.n.01` says whether the result survives changing the ancestor.

    **The ladder does not measure the altitude effect and must not be read as
    doing so.** Every row here draws its kinds from the same 50 classes, so
    `animal.n.01` is scored over 50 mammals rather than over the 143
    norm-covered kinds real R19 would find there, spanning birds, insects and
    fish. The dilution that makes `animal.n.01 "wings" -> 20 of 143` useless
    cannot appear in this table by construction. That gold believes about as
    much at the top as at the bottom is an artefact of the restriction, not
    evidence against altitude mattering; the altitude measurement is separate
    and lives in `AUDIT.md` §17.
    """
    truth = gold()
    distilled = {key: holds for key, (holds, weight) in cells.items()
                 if weight >= floor}
    free = free_listing()
    synsets = synset_of(classes())
    groups, size = ancestors_of(synsets)
    every = attributes()
    mapped = [name for name in every if name in EQUIVALENT]

    ladder = []
    for node in sorted(groups, key=lambda n: size.get(n, 0)):
        one = {node: groups[node]}
        believed = sum(1 for attribute in every
                       if verdict(groups[node], attribute, truth) == "believed")
        ladder.append({
            "ancestor": node, "members": len(groups[node]),
            "descendants": size.get(node, 0),
            "gold_believes": believed,
            "distilled": tally(one, every, truth, distilled),
            "free_listing": tally(one, mapped, truth, free)})

    close = {node: members for node, members in groups.items()
             if size.get(node, 10 ** 9) < narrow}
    return {"narrow_cut": narrow,
            "ancestors_narrow": len(close),
            "ancestors_all": len(groups),
            "classes_resolved": len(synsets),
            "floor": floor,
            "narrow_distilled": tally(close, every, truth, distilled),
            "narrow_free_listing": tally(close, mapped, truth, free),
            "narrow_distilled_same_subset": tally(close, mapped, truth,
                                                  distilled),
            "all_distilled": tally(groups, every, truth, distilled),
            "all_free_listing": tally(groups, mapped, truth, free),
            "all_distilled_same_subset": tally(groups, mapped, truth,
                                               distilled),
            "ladder": ladder}


# -- reports ---------------------------------------------------------------

def as_text(report: dict, kind: str) -> str:
    lines = ["", "=" * 76]
    if kind == "validate":
        lines += ["DISTILLED NORMS vs AwA2's CLOSED MATRIX", "=" * 76, "",
                  "  density   share of the 50 x 83 grid the model still",
                  "            answers once the floor is applied",
                  "  clear     accuracy on cells humans were not split on", "",
                  f"{'floor':>7}{'density':>10}{'accuracy':>10}"
                  f"{'precision':>11}{'recall':>9}{'clear':>9}{'n':>8}",
                  "-" * 76]
        for row in report["rows"]:
            lines.append(f"{row['floor']:>7.2f}{row['density']:>10.1%}"
                         f"{row['accuracy']:>10.1%}{row['precision']:>11.1%}"
                         f"{row['recall']:>9.1%}"
                         f"{row['accuracy_on_clear']:>9.1%}{row['n']:>8}")
        free = report.get("free_listing") or {}
        if free:
            lines += ["", "  For contrast, XCSLB free listing on the 20 "
                          "attributes it can express:",
                      f"    of {free['awa2_says_yes']} cells AwA2 says are "
                      f"true, free listing mentions {free['xcslb_listed']}"
                      f" -- {free['recall_of_free_listing']:.1%}"]
    else:
        lines += ["R19's VERDICT, FROM THREE EVIDENCE BASES", "=" * 76, "",
                  f"  {report['ancestors_all']} ancestors hold >= {MIN_KINDS}"
                  f" of the 50 classes; {report['ancestors_narrow']} are"
                  f" narrower than {report['narrow_cut']}",
                  "  scored against AwA2, which is right by construction", "",
                  "  lost     a true inheritance R19 would have refused",
                  "  over     a false one it would have believed", ""]
        for scope, title in (("narrow", f"narrower than "
                                        f"{report['narrow_cut']} -- where "
                                        f"corroboration is worth doing"),
                             ("all", "every ancestor, including the useless "
                                     "ones near the top")):
            lines += [f"-- {title}",
                      f"{'evidence':<30}{'pairs':>7}{'agreed':>9}{'lost':>9}"
                      f"{'over':>9}{'silent':>9}", "-" * 76]
            for suffix, label in (("distilled", "distilled"),
                                  ("free_listing", "free listing"),
                                  ("distilled_same_subset",
                                   "distilled, same subset")):
                one = report[f"{scope}_{suffix}"]
                lines.append(f"{label:<30}{one['pairs']:>7}"
                             f"{one['agreed']:>9.1%}{one['lost_true']:>9.1%}"
                             f"{one['over_affirmed']:>9.1%}"
                             f"{one['silent']:>9.1%}")
            lines.append("")
        lines += ["-- by ancestor, narrowest first. `gold believes` is how "
                  "many of the",
                  "   83 attributes the closed matrix says are typical of the "
                  "class:",
                  f"{'ancestor':<26}{'kin':>5}{'desc':>7}{'gold':>6}"
                  f"{'distilled':>11}{'free':>9}", "-" * 76]
        for one in report["ladder"]:
            free = one["free_listing"]
            free_cell = (f"{free['agreed']:>9.1%}" if free["pairs"]
                         else f"{'-':>9}")
            lines.append(f"{one['ancestor']:<26}{one['members']:>5}"
                         f"{one['descendants']:>7}{one['gold_believes']:>6}"
                         f"{one['distilled']['agreed']:>11.1%}{free_cell}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Distil AwA2's norms with the model and check them "
                    "against the closed matrix.")
    parser.add_argument("--distil", action="store_true",
                        help="fill the 50 x 83 grid (cached; ~4,150 asks)")
    parser.add_argument("--validate", action="store_true",
                        help="the distilled matrix against AwA2's")
    parser.add_argument("--corroborate", action="store_true",
                        help="R19's verdict from each evidence base -- the "
                             "measurement this file is for")
    parser.add_argument("--floor", type=float, default=0.5,
                        help="confidence a distilled cell needs to count")
    parser.add_argument("--narrow", type=int, default=NARROW,
                        help="an ancestor this wide or wider is too broad "
                             "for corroboration to say anything")
    parser.add_argument("--questions", action="store_true",
                        help="print the grid's questions and stop; no model")
    options = parser.parse_args(argv)

    if options.questions:
        every = classes()
        print(f"{len(every)} classes x {len(attributes())} attributes = "
              f"{len(every) * len(attributes())} cells")
        for name in every[:3]:
            for attribute in attributes()[:6]:
                print(f"   {question(name, attribute)}")
        print("\nfree-listing arm covers "
              f"{len(EQUIVALENT)} attributes: {sorted(EQUIVALENT)}")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    cells = distil() if (options.distil or options.validate
                         or options.corroborate) else {}
    if not cells:
        print("[norms] no judgements; is the model available?")
        return 1
    if options.validate:
        report = validate(cells)
        report["free_listing"] = free_listing_density()
        print(as_text(report, "validate"))
        (OUT / "norms-validate.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
    if options.corroborate:
        report = corroborate(cells, options.floor, options.narrow)
        print(as_text(report, "corroborate"))
        (OUT / "norms-corroborate.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
