"""Turn COMPS' foils into denials, or drop them.

`AUDIT.md` §1 found that XCSLB's zeros are silences: `concept_matrix.txt` is
521 x 3,644 and 1.58% dense, so a zero means no participant listed the
feature, not that anyone judged it false. COMPS draws every foil from those
zeros. `stocking NOT absorbs sweat` is a foil, and stockings absorb sweat.

That is not a small flaw. It has now blocked three separate conclusions --
GenericsKB's cost, R28 at distance zero, and whether teaching helped -- each
time in the same way: a system that correctly affirms something true about a
foil is scored wrong for it, so the accuracy column punishes knowing things.

This file is the fix. Every claim on both sides of every pair is put to a
judge; a pair is kept only where the foil side is **denied**, and the rest of
COMPS is set aside as unlabelled rather than counted as negative.

## Why a screen can be trusted when a judge cannot

The judge is SmolLM3 with the `careful` prompt, which `AUDIT.md` §13 measured
as a *worse* epistemic agent than the store -- 13.1% confident falsehoods,
never abstains. Using it to answer the benchmark would be circular, and using
it to label the benchmark looks like the same mistake.

It is not the same, for one reason: **a screen's error rate is measurable and
an answer's is not.** The screen is one bit -- does the model deny this claim
at confidence >= theta -- and there are two sets here whose truth is known
independently of both the model and XCSLB, so both of the screen's error
rates can be counted rather than assumed:

    known true      XCSLB's ones. A listed feature is an assertion.
                    AwA2's high continuous scores, which are ratings.
    known false     AwA2's low continuous scores. The matrix is *closed* --
                    every one of 50 classes was scored on every one of 85
                    attributes -- so a low score is a judgement that the
                    attribute is absent, which is exactly what XCSLB lacks.

`corpora.denied_awa2` has said so all along; what is new here is phrasing
those 85 attributes as questions, so the closed matrix can calibrate anything
that answers questions.

## What the calibration buys beyond a threshold

Sensitivity and specificity also give the thing nobody here has been able to
state: **what share of COMPS foils are false at all.** If the screen fires on
a fraction F of foils, and fires on a fraction R of claims known false and K
of claims known true, then the share p of genuinely false foils satisfies

    F = p*R + (1 - p)*K        so      p = (F - K) / (R - K)

which is prevalence from a screening test, and the precision of the kept set
is p*R / F. Both are printed per rung of the foil ladder, because the ladder
is the one place the flaw was always going to be worst: a taxonomic foil is
near by construction, so it is the most likely to be quietly true.

## What this does not fix

The calibration sets are easier than the taxonomic rung. AwA2's denials are
natural claims about real animals, which is far better than the synthetic
corruptions `audit.corrupted` builds, but `is a beaver blue` is not
`a stocking absorbs sweat`. **Sensitivity on the hardest rung is not directly
measurable here**, so the prevalence estimate for `taxonomic` is the weakest
number in the report and is marked as such.

And the screen shares a model with `teacher.py`. Measuring a *taught* store
against a model-screened benchmark is circular in a way measuring an untaught
one is not: the same judge decided what to write and what counts as false.
`audit.corrupted` stays in the report for that reason -- it is model-free, and
it is the arbiter whenever teaching is what is being weighed.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AWA2 = ROOT / "data" / "awa2" / "Animals_with_Attributes2"
COMPS = ROOT / "data" / "xcslb" / "comps_base.jsonl"
SCREENED = ROOT / "data" / "xcslb" / "comps_screened.jsonl"

#: A foil is kept only if the judge denies it at least this confidently.
#:
#: Chosen from the survey, which is in `AUDIT.md` §16. The trade is residual
#: contamination against keeping the near rungs populated:
#:
#:     theta   foils kept   precision, random -> taxonomic
#:     0.90        19,308   98.7  90.3  85.5  85.7
#:     0.95        16,916   98.9  91.7  87.4  87.6
#:     0.99        10,745   99.1  94.1  90.5  90.1
#:
#: 0.95 leaves about 6% of kept foils still true, against roughly 40% before
#: screening, and still keeps 2,194 taxonomic pairs -- enough that the audit's
#: per-rung sample is not the binding constraint. 0.99 is better material and
#: half the size; it is one flag away and the table is here so the choice can
#: be remade without another GPU hour.
THRESHOLD = 0.95

#: AwA2's 85 attributes as questions. The matrix is the only closed-world
#: gold in the repository and it has been unusable for calibration because
#: `flys` and `strainteeth` are not English; this is the whole of what was
#: missing. `{s}` is the subject with its article -- "a beaver".
#:
#: `newworld` and `oldworld` are left out on purpose. They are biogeographic
#: range, not a property anybody would answer about a beaver, and a judge
#: that gets them wrong is not making the kind of mistake being measured.
ATTRIBUTES = {
    "black": "is {s} black", "white": "is {s} white", "blue": "is {s} blue",
    "brown": "is {s} brown", "gray": "is {s} gray",
    "orange": "is {s} orange", "red": "is {s} red",
    "yellow": "is {s} yellow",
    "patches": "does {s} have patches", "spots": "does {s} have spots",
    "stripes": "does {s} have stripes",
    "furry": "is {s} furry", "hairless": "is {s} hairless",
    "toughskin": "does {s} have tough skin",
    "big": "is {s} big", "small": "is {s} small",
    "bulbous": "is {s} bulbous", "lean": "is {s} lean",
    "flippers": "does {s} have flippers", "hands": "does {s} have hands",
    "hooves": "does {s} have hooves", "pads": "does {s} have paw pads",
    "paws": "does {s} have paws", "longleg": "does {s} have long legs",
    "longneck": "does {s} have a long neck", "tail": "does {s} have a tail",
    "chewteeth": "does {s} have teeth for chewing",
    "meatteeth": "does {s} have teeth for tearing meat",
    "buckteeth": "does {s} have buck teeth",
    "strainteeth": "does {s} strain its food from the water",
    "horns": "does {s} have horns", "claws": "does {s} have claws",
    "tusks": "does {s} have tusks", "smelly": "is {s} smelly",
    "flys": "can {s} fly", "hops": "does {s} hop", "swims": "can {s} swim",
    "tunnels": "does {s} dig tunnels", "walks": "does {s} walk",
    "fast": "is {s} fast", "slow": "is {s} slow", "strong": "is {s} strong",
    "weak": "is {s} weak", "muscle": "is {s} muscular",
    "bipedal": "does {s} walk on two legs",
    "quadrapedal": "does {s} walk on four legs",
    "active": "is {s} active", "inactive": "is {s} inactive",
    "nocturnal": "is {s} nocturnal", "hibernate": "does {s} hibernate",
    "agility": "is {s} agile",
    "fish": "does {s} eat fish", "meat": "does {s} eat meat",
    "plankton": "does {s} eat plankton",
    "vegetation": "does {s} eat plants", "insects": "does {s} eat insects",
    "forager": "does {s} forage for food", "grazer": "does {s} graze",
    "hunter": "does {s} hunt", "scavenger": "does {s} scavenge",
    "skimmer": "does {s} skim its food from the water",
    "stalker": "does {s} stalk its prey",
    "arctic": "does {s} live in the arctic",
    "coastal": "does {s} live on the coast",
    "desert": "does {s} live in the desert",
    "bush": "does {s} live in the bush",
    "plains": "does {s} live on the plains",
    "forest": "does {s} live in forests",
    "fields": "does {s} live in fields",
    "jungle": "does {s} live in the jungle",
    "mountains": "does {s} live in the mountains",
    "ocean": "does {s} live in the ocean",
    "ground": "does {s} live on the ground",
    "water": "does {s} live in water", "tree": "does {s} live in trees",
    "cave": "does {s} live in caves",
    "fierce": "is {s} fierce", "timid": "is {s} timid",
    "smart": "is {s} smart", "group": "does {s} live in groups",
    "solitary": "is {s} solitary", "nestspot": "does {s} build a nest",
    "domestic": "is {s} domesticated",
}

#: The continuous matrix is an average human rating out of 100, and the
#: published binary matrix thresholds it at about 20.8. Calibration wants
#: claims nobody would argue about, so it takes only the ends: a rating at or
#: under 5 is a denial, one at or over 60 is an assertion, and the wide band
#: between them -- where `is a leopard big` lives -- is discarded rather than
#: guessed at.
DENIED_AT = 5.0
HELD_AT = 60.0

#: Fresh judgements between saves of the cache. See `judge_all`.
FLUSH = 2000

#: The four rungs, nearest last. Repeated from `audit.py` rather than
#: imported, because importing `audit` pulls in the engine pool and this file
#: is meant to be runnable with nothing but the model.
LADDER = ("random", "co-occurrence", "overlap", "taxonomic")


@dataclass(frozen=True)
class Claim:
    """One question whose answer is known, or is the thing being asked."""

    question: str
    truth: bool | None          # None where that is the open question
    origin: str                 # awa2 | xcslb | corrupted | foil
    concept: str = ""
    prop: str = ""


# -- the gold that is closed ----------------------------------------------

def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def awa2_claims(denied_at: float = DENIED_AT,
                held_at: float = HELD_AT) -> list[Claim]:
    """AwA2's extremes as questions, both truths, from a closed matrix.

    This is the calibration set the audit never had: 50 classes scored on
    every one of 85 attributes, so a low score is somebody saying no rather
    than nobody saying anything.
    """
    def read(path: Path) -> list[str]:
        return [line.split("\t")[-1].strip() if "\t" in line
                else line.split()[-1]
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    classes = read(AWA2 / "classes.txt")
    predicates = read(AWA2 / "predicates.txt")
    rows = [[float(value) for value in line.split()]
            for line in (AWA2 / "predicate-matrix-continuous.txt")
            .read_text(encoding="utf-8").splitlines() if line.strip()]

    out = []
    for animal, scores in zip(classes, rows):
        name = animal.replace("+", " ")
        subject = f"{article(name)} {name}"
        for attribute, score in zip(predicates, scores):
            template = ATTRIBUTES.get(attribute)
            if not template:
                continue
            if score <= denied_at:
                truth = False
            elif score >= held_at:
                truth = True
            else:
                continue
            out.append(Claim(template.format(s=subject), truth, "awa2",
                             name, attribute))
    out.sort(key=lambda claim: claim.question)
    return out


# -- the gold that is open, and the questions themselves -------------------

def comps_claims(side: str) -> list[Claim]:
    """One side of every COMPS pair, deduplicated and phrased.

    `acceptable` is 12,335 claims that are all true -- a listed feature is an
    assertion, which is the half of XCSLB that was never in doubt.
    `unacceptable` is 36,701 whose truth is the open question.
    """
    from .audit import articles, phrase

    field = ("acceptable_concept" if side == "acceptable"
             else "unacceptable_concept")
    truth = True if side == "acceptable" else None
    origin = "xcslb" if side == "acceptable" else "foil"
    article_of = articles()
    seen: set = set()
    out = []
    with COMPS.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            concept, prop = row[field], row["property"]
            if (concept, prop) in seen:
                continue
            seen.add((concept, prop))
            text = phrase(article_of.get(concept, ""), prop)
            if text:
                out.append(Claim(text, truth, origin, concept, prop))
    out.sort(key=lambda claim: claim.question)
    return out


def xcslb_claims(limit: int = 0) -> list[Claim]:
    """The acceptable side. Used to count how often the screen denies
    something true, which is the error that silently deletes good foils."""
    return sample(comps_claims("acceptable"), limit)


def foil_claims(limit: int = 0) -> list[Claim]:
    """The unacceptable side. Truth unknown: that is the point."""
    return sample(comps_claims("unacceptable"), limit)


def corrupted_claims(limit: int = 0) -> list[Claim]:
    """`audit.corrupted` as claims. Known false, model-free, and easy.

    Kept in the calibration because it is the one negative set no model had a
    hand in, and reported apart from AwA2 because the difference between the
    two sensitivities is itself the warning: a screen that is perfect on
    `does an arm have a bubble tube` and merely good on `is a beaver blue`
    will be worse again on `does a stocking absorb sweat`.
    """
    from .audit import articles, corrupted, phrase

    article_of = articles()
    out = [Claim(phrase(article_of.get(pair.held, ""), pair.prop), False,
                 "corrupted", pair.held, pair.prop)
           for pair in corrupted()]
    out = [claim for claim in out if claim.question]
    out.sort(key=lambda claim: claim.question)
    return sample(out, limit)


def sample(rows: list, limit: int) -> list:
    """A strided walk, so a small run is a sub-sample of a large one.

    The same rule `audit.pairs` uses and for the same reason: two runs at
    different sizes stay comparable, and a run repeats.
    """
    if not limit or limit >= len(rows):
        return rows
    stride = len(rows) / limit
    return [rows[int(index * stride)] for index in range(limit)]


def rungs() -> dict:
    """(foil concept, property) -> which rung of the ladder COMPS put it on."""
    out: dict = {}
    with COMPS.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            out.setdefault((row["unacceptable_concept"], row["property"]),
                           row["negative_sample_type"])
    return out


# -- asking ----------------------------------------------------------------

def judge_all(claims: list, teacher=None, note: str = "") -> dict:
    """question -> (denied, confidence). One GPU, so one pass, in order.

    Cached judgements are free and are counted separately in the progress
    line, because a run that is 90% cache is a different kind of evidence
    from one that is not.

    The batch is flushed every `FLUSH` fresh judgements rather than only at
    the end. A full foil pass is 36,701 questions and half an hour, and
    holding the file open for all of it means a crash at minute twenty-nine
    costs the whole run; saving fifteen times instead of thirty thousand
    keeps the saving cheap and bounds what an interruption can lose.
    """
    from .teacher import Teacher

    teacher = teacher if teacher is not None else Teacher()
    out: dict = {}
    started, fresh, saved = time.time(), 0, 0
    with teacher.batch():
        for index, claim in enumerate(claims):
            if claim.question in out:
                continue
            supports, weight, cached = teacher.judge("", "", claim.question)
            out[claim.question] = (not supports, float(weight))
            fresh += not cached
            if fresh - saved >= FLUSH:
                saved = fresh
                teacher._save()                     # noqa: SLF001
            if note and index and not index % 1000:
                rate = fresh / max(time.time() - started, 1e-9)
                print(f"[screen] {note} {index}/{len(claims)}, "
                      f"{fresh} fresh at {rate:.1f}/s", flush=True)
    return out


def denial_rate(claims: list, judged: dict, threshold: float) -> tuple:
    """(n, share denied at or above `threshold`)."""
    rows = [judged[claim.question] for claim in claims
            if claim.question in judged]
    if not rows:
        return 0, 0.0
    denied = sum(1 for was_denied, weight in rows
                 if was_denied and weight >= threshold)
    return len(rows), denied / len(rows)


def prevalence(observed: float, sensitivity: float,
               denies_truth: float) -> float | None:
    """The share of an unlabelled pool that is genuinely false.

    `observed` is how often the screen fires on it, `sensitivity` how often it
    fires on claims known false, `denies_truth` how often it fires on claims
    known true. Rogan-Gladen, clamped to [0, 1]; None where the two rates are
    within five points of each other, because a screen that does not separate
    its own gold cannot estimate anything and the honest answer is to say so.
    """
    span = sensitivity - denies_truth
    if span <= 0.05:
        return None
    return max(0.0, min(1.0, (observed - denies_truth) / span))


# -- the reports -----------------------------------------------------------

#: The thresholds worth trying. 0.5 is "the model leaned no", which is where
#: adjudication runs; the rest are floors.
THRESHOLDS = (0.5, 0.75, 0.9, 0.95, 0.99)


def truth_sets(limit: int = 1500) -> dict:
    """The four sets whose answers are known, largest useful size."""
    every = awa2_claims()
    return {"awa2_false": [c for c in every if c.truth is False],
            "awa2_true": [c for c in every if c.truth is True],
            "xcslb_true": xcslb_claims(limit),
            "corrupted_false": corrupted_claims(limit)}


def calibrate(limit: int = 1500, thresholds: tuple = THRESHOLDS) -> dict:
    """What the screen costs and catches, at each threshold worth trying."""
    from .teacher import Teacher

    teacher = Teacher()
    if not teacher.available:
        return {"error": teacher.error or "no teacher"}
    sets = truth_sets(limit)
    judged: dict = {}
    for name, claims in sets.items():
        judged.update(judge_all(claims, teacher, note=name))

    rows = []
    for threshold in thresholds:
        one: dict = {"threshold": threshold}
        for name, claims in sets.items():
            count, share = denial_rate(claims, judged, threshold)
            one[name] = round(share, 4)
            one[f"n_{name}"] = count
        rows.append(one)
    return {"sets": {name: len(claims) for name, claims in sets.items()},
            "rows": rows}


def survey(limit: int = 0, thresholds: tuple = THRESHOLDS,
           truth_limit: int = 1500) -> dict:
    """Run the screen over the foils and say what COMPS is actually made of.

    The prevalence estimate per rung is the number this whole file exists to
    produce: it says, with an error rate behind it, how much of the benchmark
    was never a negative.
    """
    from .teacher import Teacher

    teacher = Teacher()
    if not teacher.available:
        return {"error": teacher.error or "no teacher"}
    truths = truth_sets(truth_limit)
    judged: dict = {}
    for name, claims in truths.items():
        judged.update(judge_all(claims, teacher, note=name))
    foils = foil_claims(limit)
    judged.update(judge_all(foils, teacher, note="foils"))

    where = rungs()
    by_rung: dict = collections.defaultdict(list)
    for claim in foils:
        by_rung[where.get((claim.concept, claim.prop), "?")].append(claim)

    out: dict = {"n_foils": len(foils), "thresholds": []}
    for threshold in thresholds:
        _, sensitivity = denial_rate(truths["awa2_false"], judged, threshold)
        _, easy = denial_rate(truths["corrupted_false"], judged, threshold)
        _, loss = denial_rate(truths["xcslb_true"], judged, threshold)
        rows = {}
        for name in LADDER:
            count, share = denial_rate(by_rung.get(name, []), judged,
                                       threshold)
            estimate = prevalence(share, sensitivity, loss)
            rows[name] = {
                "n": count, "denied": round(share, 4),
                "false_share": (None if estimate is None
                                else round(estimate, 4)),
                "precision": (None if not share or estimate is None else
                              round(min(1.0, estimate * sensitivity / share),
                                    4))}
        out["thresholds"].append({
            "threshold": threshold,
            "sensitivity_awa2": round(sensitivity, 4),
            "sensitivity_corrupted": round(easy, 4),
            "denies_truth": round(loss, 4),
            "by_rung": rows})
    return out


# -- the artifact ----------------------------------------------------------

def build(threshold: float = THRESHOLD, limit: int = 0) -> dict:
    """Write the screened pair set, so the audit needs no GPU to use it.

    One line per surviving COMPS pair in COMPS' own shape, plus what the
    judge was asked and how sure it was, so a later reader can see what was
    kept and why without rerunning anything.
    """
    foils = foil_claims(limit)
    judged = judge_all(foils, note="foils")
    kept = {}
    for claim in foils:
        denied, weight = judged.get(claim.question, (False, 0.0))
        if denied and weight >= threshold:
            kept[(claim.concept, claim.prop)] = (claim.question, weight)

    seen = written = 0
    by_rung: dict = collections.Counter()
    with SCREENED.open("w", encoding="utf-8") as out:
        with COMPS.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                seen += 1
                found = kept.get((row["unacceptable_concept"],
                                  row["property"]))
                if not found:
                    continue
                question, weight = found
                row["screen_question"] = question
                row["screen_confidence"] = round(weight, 4)
                out.write(json.dumps(row) + "\n")
                written += 1
                by_rung[row["negative_sample_type"]] += 1
    return {"threshold": threshold, "pairs_in": seen, "pairs_kept": written,
            "foils_in": len(foils), "foils_kept": len(kept),
            "by_rung": dict(by_rung), "written_to": str(SCREENED)}


def as_text(report: dict, kind: str) -> str:
    if "error" in report:
        return f"[screen] no judge: {report['error']}\n"
    lines = ["", "=" * 76]
    if kind == "calibrate":
        lines += ["SCREEN CALIBRATION -- how often the judge says no,",
                  "  on claims whose answer is already known.", "=" * 76, "",
                  "  awa2 F / corrupt F   it should deny these",
                  "  awa2 T / xcslb T     it should not", "",
                  f"{'theta':>7}{'awa2 F':>10}{'corrupt F':>11}"
                  f"{'awa2 T':>10}{'xcslb T':>10}", "-" * 76]
        for row in report["rows"]:
            lines.append(f"{row['threshold']:>7.2f}"
                         f"{row['awa2_false']:>10.1%}"
                         f"{row['corrupted_false']:>11.1%}"
                         f"{row['awa2_true']:>10.1%}"
                         f"{row['xcslb_true']:>10.1%}")
        lines += ["", "  sizes: " + ", ".join(
            f"{name}={count}" for name, count in report["sets"].items())]
    else:
        lines += ["WHAT COMPS' FOILS ARE MADE OF", "=" * 76, "",
                  "  denied     how often the screen calls the foil false",
                  "  false      the estimate of how many really are",
                  "  precision  of the kept ones, how many are really false",
                  f"  n foils    {report['n_foils']}", ""]
        for one in report["thresholds"]:
            lines += [f"-- theta {one['threshold']:.2f}   sensitivity "
                      f"{one['sensitivity_awa2']:.1%} awa2 / "
                      f"{one['sensitivity_corrupted']:.1%} corrupted, "
                      f"denies truth {one['denies_truth']:.1%}",
                      f"{'rung':<16}{'n':>8}{'denied':>10}{'false':>10}"
                      f"{'precision':>12}"]
            for name, row in one["by_rung"].items():
                share = f"{row['false_share']:>10.1%}" \
                    if row["false_share"] is not None else f"{'-':>10}"
                exact = f"{row['precision']:>12.1%}" \
                    if row["precision"] is not None else f"{'-':>12}"
                lines.append(f"{name:<16}{row['n']:>8}"
                             f"{row['denied']:>10.1%}{share}{exact}")
            lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Screen COMPS' foils for whether they are denials.")
    parser.add_argument("--calibrate", action="store_true",
                        help="the judge's error rates on claims already "
                             "known true and known false")
    parser.add_argument("--survey", action="store_true",
                        help="run the screen over the foils and estimate "
                             "what share of them were ever negatives")
    parser.add_argument("--build", action="store_true",
                        help="write data/xcslb/comps_screened.jsonl")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--limit", type=int, default=0,
                        help="claims per set (0 = all)")
    parser.add_argument("--questions", action="store_true",
                        help="print a sample of each claim set and stop; "
                             "no model, so it costs nothing")
    options = parser.parse_args(argv)

    if options.questions:
        for name, claims in (("awa2", awa2_claims()),
                             ("xcslb", xcslb_claims()),
                             ("corrupted", corrupted_claims()),
                             ("foil", foil_claims())):
            print(f"-- {name}: {len(claims)}")
            for claim in sample(claims, 6):
                print(f"   {str(claim.truth):<5} {claim.question}")
        return 0

    where = ROOT / "research" / "v688" / "audit-out"
    where.mkdir(parents=True, exist_ok=True)
    if options.calibrate:
        report = calibrate(options.limit or 1500)
        print(as_text(report, "calibrate"))
        (where / "screen-calibration.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
    if options.survey:
        report = survey(options.limit)
        print(as_text(report, "survey"))
        (where / "screen-survey.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
    if options.build:
        print(json.dumps(build(options.threshold, options.limit), indent=2))
    if not (options.calibrate or options.survey or options.build):
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
