"""Should the teacher be asked about a yes that nothing bore out?

The teacher is asked only what the store left open (`teacher.unsettled`), so
the store's own over-affirmation is never put to it. `can a fish walk on land`
comes back VERIFIED on one Ascent++ row -- `fish capable_of "walk on land"`,
which is mudskippers -- the run finds nothing to corroborate or contradict
it, and the page says `unchallenged`. Asked the bare question, the model says
no at 0.996.

`Teacher.challenge` asks it. Before letting that change a single badge, this
measures what the model would say about the yes answers it would be shown:
every polar yes resting on a crawled row, in the most recent audit run.

That is a superset of what the loop challenges -- the loop also skips a yes
its run bore out -- so the cost measured here is an upper bound.

    listed      the claim is a feature XCSLB lists. True. A confident no here
                is the teacher disputing a correct record: the cost.
    corrupted   built to be false (`audit.corrupted`). No model had a hand in
                it. A confident no here is over-affirmation caught.
    foil        COMPS' other side, whose truth is open. The screened gold
                these rows were run on was itself filtered by this model, so
                a dispute here is described, not scored.

What the gold cannot show is the case that motivated this. XCSLB is almost
all leaves, and the over-affirmation that matters is a crawled row filed on a
class -- `animal`, `fish`, `mammal` -- so the class-level questions are
listed apart and read one at a time.

No store and no GPU: the rows are `audit.py`'s, and the judgements are the
teacher's cache. A question the cache does not have is counted as unasked
unless `--load` is given, which needs the GPU the server holds.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from . import audit
from .teacher import SETTLING_FLOOR, Teacher

ROOT = Path(__file__).resolve().parents[2]
ROWS = ROOT / "research" / "v688" / "audit-out" / "dense-now"
OUT = ROOT / "research" / "v688" / "audit-out" / "challenge.json"

#: The sources `teacher.challengeable` accepts as crawled.
CRAWLED = frozenset({"ascentpp", "conceptnet"})

#: Class-level questions, with the truth a person would give. The false ones
#: are the shape the gold cannot see; the true ones are there so a teacher
#: that says no to everything is caught here too.
GUARDS = (("can a fish walk on land", False), ("can an animal fly", False),
          ("does an animal have wings", False), ("can a mammal fly", False),
          ("can a person fly", False), ("can a rock swim", False),
          ("does a cat lay eggs", False), ("can a fish walk", False),
          ("does a dog bark", True), ("can a bird fly", True),
          ("does an animal breathe", True), ("can a fish swim", True),
          ("does a mammal have fur", True), ("can a person walk", True),
          ("does a dog have a tail", True), ("can a horse run", True))


def truths() -> dict:
    """key -> True (listed), False (corrupted) or None (foil), for the gold
    the rows were run on."""
    out: dict = {}
    with audit.gold_file("screened").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            out.setdefault(f"{row['unacceptable_concept']}|{row['property']}",
                           None)
            out[f"{row['acceptable_concept']}|{row['property']}"] = True
    return out


def rows(where: Path) -> list[dict]:
    found: dict = {}
    for path in sorted(where.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                found[row["key"]] = row
    return list(found.values())


def read(teacher: Teacher, question: str, floor: float) -> str:
    """disputed | agreed | below | unasked"""
    supports, confidence, cached = teacher.judge("", "", question)
    if not cached and not teacher.available:
        return "unasked"
    if confidence < floor:
        return "below"
    return "agreed" if supports else "disputed"


def run(where: Path = ROWS, floor: float = SETTLING_FLOOR,
        load: bool = False) -> dict:
    teacher = Teacher(load=load)
    truth = truths()
    sets: dict = collections.defaultdict(list)
    for row in rows(where):
        if (row.get("outcome") != "verified"
                or (row.get("source") or "").lower() not in CRAWLED):
            continue
        key = row["key"]
        name = ("corrupted" if key.startswith("!") else
                "listed" if truth.get(key) is True else "foil")
        sets[name].append(row)

    report: dict = {"rows": str(where), "floor": floor, "sets": {},
                    "guards": []}
    with teacher.batch():
        for name, chosen in sets.items():
            counts: collections.Counter = collections.Counter()
            by_distance: dict = collections.defaultdict(collections.Counter)
            disputed = []
            for row in chosen:
                said = read(teacher, row["question"], floor)
                counts[said] += 1
                by_distance["stated" if not row.get("distance")
                            else "inherited"][said] += 1
                if said == "disputed":
                    disputed.append({"question": row["question"],
                                     "rule": row.get("rule"),
                                     "source": row.get("source"),
                                     "distance": row.get("distance")})
            asked = sum(counts[one] for one in ("disputed", "agreed", "below"))
            report["sets"][name] = {
                "n": len(chosen), "asked": asked, **dict(counts),
                "disputed_rate": (round(counts["disputed"] / asked, 4)
                                  if asked else 0.0),
                "agreed_rate": (round(counts["agreed"] / asked, 4)
                                if asked else 0.0),
                "by_distance": {one: dict(value)
                                for one, value in by_distance.items()},
                "disputed_examples": disputed[:25]}
        for question, holds in GUARDS:
            supports, confidence, cached = teacher.judge("", "", question)
            report["guards"].append({
                "question": question, "truth": holds,
                "asked": cached or teacher.available,
                "supports": supports, "confidence": round(confidence, 4),
                "reading": read(teacher, question, floor)})
    return report


def as_text(report: dict) -> str:
    lines = ["", "=" * 78,
             "CHALLENGING A YES: WHAT THE TEACHER SAYS OF CRAWLED YES ANSWERS",
             "=" * 78, "", f"rows  {report['rows']}",
             f"floor {report['floor']}", "",
             f"  {'set':<10} {'n':>5} {'asked':>6} {'disputed':>9} "
             f"{'agreed':>8} {'below':>6}"]
    for name in ("listed", "corrupted", "foil"):
        one = report["sets"].get(name)
        if not one:
            continue
        lines.append(f"  {name:<10} {one['n']:>5} {one['asked']:>6} "
                     f"{one['disputed_rate']:>8.1%} {one['agreed_rate']:>8.1%}"
                     f" {one.get('below', 0):>6}")
    for name in ("listed", "corrupted", "foil"):
        one = report["sets"].get(name)
        if not one:
            continue
        lines += ["", f"  {name}: by where the row was", ""]
        for where, counts in sorted(one["by_distance"].items()):
            lines.append(f"    {where:<10} {counts}")
        lines += ["", f"  {name}: disputed", ""]
        for example in one["disputed_examples"]:
            lines.append(f"    {example['question']:<48} {example['rule']} "
                         f"{example['source']} d={example['distance']}")
    lines += ["", "  class-level guards, one at a time", ""]
    for one in report["guards"]:
        lines.append(f"    {'T' if one['truth'] else 'F'} "
                     f"{one['question']:<30} "
                     f"{'yes' if one['supports'] else 'no ':<3} "
                     f"{one['confidence']:.3f}  {one['reading']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path, default=ROWS)
    parser.add_argument("--floor", type=float, default=SETTLING_FLOOR)
    parser.add_argument("--load", action="store_true",
                        help="load the model for questions the cache lacks")
    options = parser.parse_args(argv)
    report = run(options.rows, options.floor, options.load)
    print(as_text(report))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
