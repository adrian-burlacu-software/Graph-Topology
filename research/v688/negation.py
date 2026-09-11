"""Ask a question both ways, so that a model's no can be counted.

`teacher.CAREFUL` makes the model safe to believe when it says yes and unsafe
to believe when it says no. It is told most claims are false, and it denies
true things nobody would list: 5 of 13 `careful` witnesses said a dog does
not breathe (`AUDIT.md` §25). So the loop counts only its yes.

Two ways to get a no worth counting, measured side by side:

* **negation** -- ask `does a dog breathe` and `does a dog not breathe`. A
  model that says no to both is showing the prompt's lean rather than an
  opinion, and abstains; one that says no and then yes means it. The known
  risk is that small models read "not" poorly and answer both the same way.
* **mirror** -- ask the same question under `teacher.CREDULOUS`, a prompt
  leaning the other way. A no from a model told most claims are true is a no
  that survived the lean. No grammar is involved.

Scored on the gold sets `screen.truth_sets` already has -- AwA2's two closed
ends, XCSLB's listed features and the model-free corrupted claims -- plus a
handful of claims reported one at a time, because an aggregate over
thousands will not show that `does a dog breathe` came back no.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import screen

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "v688" / "audit-out" / "negation.json"

#: Openers that take a plain `not`. `can` is handled apart: `can a dog not
#: swim` asks whether it is able to refrain, which is not the negation.
NEGATES = {"is", "are", "was", "does", "do", "did", "has", "will", "must",
           "may"}

#: (question, concept, truth).
GUARDS = (("does a dog breathe", "dog", True),
          ("can a dog swim", "dog", True),
          ("does a dog have legs", "dog", True),
          ("can a dog fall into a hole", "dog", True),
          ("can a leopard hunt at night", "leopard", True),
          ("does a cow eat plants", "cow", True),
          ("does a pig have wings", "pig", False),
          ("can a pig fly", "pig", False),
          ("can a fish walk on land", "fish", False),
          ("can a rock swim", "rock", False),
          ("can a penguin fly", "penguin", False),
          ("is a bat a bird", "bat", False))

RULES = ("shipped", "careful", "negation", "mirror")


def negate(question: str, concept: str) -> str:
    """`does a dog breathe` -> `does a dog not breathe`, or "" if unsure.

    The subject is found by its concept rather than parsed, because every
    claim measured here was built from a template around a known concept.
    """
    head, _, tail = question.strip().rstrip("?").partition(" ")
    at = f" {tail} ".find(f" {concept} ")
    if not tail or not concept or at < 0:
        return ""
    end = at + len(concept)
    subject, rest = tail[:end].strip(), tail[end:].strip()
    if not rest:
        return ""
    if head == "can":
        return f"is {subject} unable to {rest}"
    if head in NEGATES:
        return f"{head} {subject} not {rest}"
    return ""


def yes_mass(teacher, question: str, style: str = "") -> float:
    """The probability the model puts on yes, whichever way it answered."""
    supports, confidence, _cached = teacher.judge("", "", question, style)
    return confidence if supports else 1.0 - confidence


def decide(rule: str, asked: float, negated: float, mirrored: float,
           floor: float):
    """"yes", "no" or None (abstain) under one rule at one floor."""
    yes, no = asked >= floor, asked <= 1.0 - floor
    if rule == "shipped":
        return "yes" if yes else None
    if rule == "careful":
        return "yes" if yes else "no" if no else None
    if rule == "negation":
        if yes and negated < floor:
            return "yes"
        if no and negated >= floor:
            return "no"
        return None
    if rule == "mirror":
        denied = mirrored <= 1.0 - floor
        if yes and not denied:
            return "yes"
        if denied and not yes:
            return "no"
        return None
    raise ValueError(rule)


def run(limit: int = 400, floors: tuple = (0.9, 0.99)) -> dict:
    from .teacher import Teacher

    teacher = Teacher()
    if not teacher.available:
        return {"error": teacher.error or "no teacher"}
    sets = {name: screen.sample(claims, limit)
            for name, claims in screen.truth_sets(limit).items()}
    sets["guards"] = [screen.Claim(question, truth, "guard", concept)
                      for question, concept, truth in GUARDS]

    probed: dict = {}
    skipped: dict = {}
    started = time.time()
    with teacher.batch():
        for name, claims in sets.items():
            for index, claim in enumerate(claims):
                if claim.question in probed:
                    continue
                negated = negate(claim.question, claim.concept)
                if not negated:
                    skipped[name] = skipped.get(name, 0) + 1
                    continue
                probed[claim.question] = (
                    yes_mass(teacher, claim.question),
                    yes_mass(teacher, negated),
                    yes_mass(teacher, claim.question, "credulous"),
                    negated)
                if index and not index % 100:
                    print(f"[negation] {name} {index}/{len(claims)} "
                          f"({time.time() - started:.0f}s)", flush=True)

    report: dict = {"limit": limit, "skipped": skipped, "rows": [],
                    "echo": {}, "guards": [],
                    "seconds": round(time.time() - started, 1)}
    for name, claims in sets.items():
        rows = [probed[claim.question] for claim in claims
                if claim.question in probed]
        truth = claims[0].truth if claims else None
        if name != "guards" and rows:
            # How often the negation is simply the same answer again: no to
            # both, or yes to both. High means the model is not reading "not".
            report["echo"][name] = {
                "n": len(rows),
                "no_to_both": round(sum(a < 0.5 and n < 0.5
                                        for a, n, _, _ in rows) / len(rows), 4),
                "yes_to_both": round(sum(a >= 0.5 and n >= 0.5
                                         for a, n, _, _ in rows) / len(rows),
                                     4)}
            for floor in floors:
                for rule in RULES:
                    said = [decide(rule, a, n, m, floor)
                            for a, n, m, _ in rows]
                    report["rows"].append({
                        "set": name, "truth": truth, "floor": floor,
                        "rule": rule, "n": len(rows),
                        "yes": round(said.count("yes") / len(rows), 4),
                        "no": round(said.count("no") / len(rows), 4)})
    for question, concept, truth in GUARDS:
        if question not in probed:
            continue
        asked, negated, mirrored, text = probed[question]
        report["guards"].append({
            "question": question, "truth": truth, "negated": text,
            "asked": round(asked, 4), "negated_yes": round(negated, 4),
            "mirrored": round(mirrored, 4),
            "decisions": {rule: decide(rule, asked, negated, mirrored, 0.99)
                          for rule in RULES}})
    return report


def as_text(report: dict) -> str:
    if "error" in report:
        return f"[negation] {report['error']}\n"
    lines = ["", "=" * 78, "COUNTING A NO: NEGATION AGAINST A MIRRORED PROMPT",
             "=" * 78, "",
             "  right = yes on a true set, no on a false set", ""]
    floors = sorted({row["floor"] for row in report["rows"]})
    names = [name for name in report["echo"]]
    for floor in floors:
        lines.append(f"floor {floor}")
        lines.append(f"  {'rule':<10}" + "".join(
            f"{name:>22}" for name in names))
        lines.append(f"  {'':<10}" + "".join(
            f"{'right / wrong':>22}" for _ in names))
        for rule in RULES:
            cells = []
            for name in names:
                row = next(one for one in report["rows"]
                           if one["set"] == name and one["floor"] == floor
                           and one["rule"] == rule)
                right, wrong = ((row["yes"], row["no"]) if row["truth"]
                                else (row["no"], row["yes"]))
                cells.append(f"{right:>13.1%} / {wrong:<6.1%}")
            lines.append(f"  {rule:<10}" + "".join(cells))
        lines.append("")
    lines.append("  the negation answered the same as the question:")
    for name, echo in report["echo"].items():
        lines.append(f"    {name:<18} n={echo['n']:<5} no to both "
                     f"{echo['no_to_both']:.1%}, yes to both "
                     f"{echo['yes_to_both']:.1%}")
    lines += ["", "  one at a time, floor 0.99 (P(yes): asked / negated / "
              "mirrored):", ""]
    for one in report["guards"]:
        said = " ".join(f"{rule}={one['decisions'][rule] or '-'}"
                        for rule in RULES)
        lines.append(f"    {'T' if one['truth'] else 'F'} "
                     f"{one['question']:<30} {one['asked']:.3f} / "
                     f"{one['negated_yes']:.3f} / {one['mirrored']:.3f}  "
                     f"{said}")
        lines.append(f"      negated as: {one['negated']}")
    if report["skipped"]:
        lines += ["", f"  not negatable, skipped: {report['skipped']}"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure whether a negated question or a mirrored prompt "
                    "makes the teacher's no worth counting.")
    parser.add_argument("--limit", type=int, default=400)
    options = parser.parse_args(argv)
    report = run(options.limit)
    print(as_text(report))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
