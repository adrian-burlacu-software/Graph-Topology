"""Calibrate a second prompt for claims about what a thing can do.

`AUDIT.md` §21 left one thing open. `teacher.CAREFUL` says *say yes only if
the property is typical of that kind*, and that is the right instruction for
`has_a` and `has_property`. It is the wrong one for `capable_of`: all
seventeen canine witnesses denied `can a beagle fall into a hole`, because
falling into holes is not characteristic of beagles even though every one of
them can, and R19 then refused a true claim.

The fix is a second prompt. The risk is that a looser prompt is just looser:
`a fish walks`, `a rock swims` and `a pig flies` have to stay no, and those
are exactly the refusals R19 exists to produce.

## The gold nobody had to build

§16's screened benchmark is labelled on both sides, and **4,766 of its 20,925
pairs are capability-shaped** -- the property begins with `can`. The held side
is what people listed, so those claims are true; the foil side survived the
screen, so a calibrated judge called them false. That gives **2,775 true and
4,008 false capability claims** without annotating anything.

AwA2 was the obvious alternative and is not usable here: its capability
attributes are scored for *typicality* too, which is the whole confusion --
`swims` is 0 for collie and dalmatian, and dogs swim. Using it would measure
the prompt against the belief the prompt is meant to correct.

## What is compared

Both prompts over the same claims, reported the way §13 reported the first
four: how often each says yes to what is true, how often to what is false,
and the separation between them. A prompt that gains on true claims and
gains as much on false ones has bought nothing.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import audit

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "v688" / "audit-out" / "capability.json"

#: The prompts to compare. `""` is `CAREFUL`, under which every existing
#: judgement was cached.
STYLES = ("", "capable")

#: Claims R19's refusals rest on, kept out of the sample and reported
#: separately. A prompt that scores well and lets `a fish walks` through is
#: not usable, and an aggregate over thousands of claims will not show it.
GUARDS = (("can a fish walk on land", False),
          ("can a rock swim", False),
          ("can a pig fly", False),
          ("can a dog fall into a hole", True),
          ("can a beagle fall into a hole", True),
          ("can a dog swim", True),
          ("can a penguin fly", False),
          ("can a leopard hunt at night", True))


def claims(limit: int = 400) -> tuple:
    """(true, false) capability claims from the screened benchmark."""
    article = audit.articles()
    held: dict = {}
    foil: dict = {}
    for pair in audit.pairs(0, "screened"):
        if pair.prop.split()[:1] != ["can"]:
            continue
        text = audit.phrase(article.get(pair.held, ""), pair.prop)
        if text:
            held.setdefault(text, pair)
        text = audit.phrase(article.get(pair.foil, ""), pair.prop)
        if text:
            foil.setdefault(text, pair)
    # Strided, like every other sample here, so a small run is a subset of a
    # large one and two runs are comparable.
    return (audit.pairs and sample(sorted(held), limit),
            sample(sorted(foil), limit))


def sample(rows: list, limit: int) -> list:
    if not limit or limit >= len(rows):
        return rows
    stride = len(rows) / limit
    return [rows[int(index * stride)] for index in range(limit)]


def run(limit: int = 400, floors: tuple = (0.5, 0.9, 0.99)) -> dict:
    from .teacher import Teacher

    teacher = Teacher()
    if not teacher.available:
        return {"error": teacher.error or "no teacher"}
    true_claims, false_claims = claims(limit)
    answers: dict = {}
    started = time.time()
    with teacher.batch():
        for style in STYLES:
            for name, rows in (("true", true_claims), ("false", false_claims),
                               ("guard", [text for text, _ in GUARDS])):
                for index, text in enumerate(rows):
                    supports, weight, _cached = teacher.judge(
                        "", "", text, style)
                    answers[(style, text)] = (supports, weight)
                    if index and not index % 400:
                        print(f"[capability] {style or 'careful'} {name} "
                              f"{index}/{len(rows)}", flush=True)

    def says_yes(style: str, rows: list, floor: float) -> float:
        hit = [answers[(style, text)] for text in rows]
        return sum(1 for supports, weight in hit
                   if supports and weight >= floor) / max(len(hit), 1)

    report: dict = {"n_true": len(true_claims), "n_false": len(false_claims),
                    "rows": [], "guards": {},
                    "seconds": round(time.time() - started, 1)}
    for style in STYLES:
        for floor in floors:
            yes_true = says_yes(style, true_claims, floor)
            yes_false = says_yes(style, false_claims, floor)
            report["rows"].append({
                "style": style or "careful", "floor": floor,
                "yes_to_true": round(yes_true, 4),
                "yes_to_false": round(yes_false, 4),
                "separation": round(yes_true - yes_false, 4)})
        report["guards"][style or "careful"] = [
            {"claim": text, "truth": truth,
             "says": answers[(style, text)][0],
             "confidence": round(answers[(style, text)][1], 3),
             "right": answers[(style, text)][0] == truth}
            for text, truth in GUARDS]
    return report


def as_text(report: dict) -> str:
    if "error" in report:
        return f"[capability] {report['error']}\n"
    lines = ["", "=" * 72,
             "A SECOND PROMPT FOR WHAT A THING CAN DO", "=" * 72, "",
             f"  {report['n_true']} true and {report['n_false']} false "
             f"capability claims, from the screened benchmark", "",
             f"{'prompt':<10}{'floor':>7}{'yes to true':>14}"
             f"{'yes to false':>14}{'separation':>13}", "-" * 72]
    for row in report["rows"]:
        lines.append(f"{row['style']:<10}{row['floor']:>7.2f}"
                     f"{row['yes_to_true']:>14.1%}"
                     f"{row['yes_to_false']:>14.1%}"
                     f"{row['separation']:>13.1%}")
    lines += ["", "  the claims R19's refusals rest on:", ""]
    for style, rows in report["guards"].items():
        wrong = [one for one in rows if not one["right"]]
        lines.append(f"  {style}: {len(rows) - len(wrong)}/{len(rows)} right")
        for one in rows:
            mark = " " if one["right"] else "  <- WRONG"
            lines.append(f"     {one['claim']:<34}"
                         f"{'yes' if one['says'] else 'no ':<5}"
                         f"{one['confidence']:.2f}{mark}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare the careful and capability prompts on labelled "
                    "capability claims.")
    parser.add_argument("--limit", type=int, default=400)
    options = parser.parse_args(argv)
    report = run(options.limit)
    print(as_text(report))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
