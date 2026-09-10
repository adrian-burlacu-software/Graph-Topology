"""An audit of v688 against the one thing here that has labels.

Everything in v687 and v688 has been judged by hand, one question at a time.
Seventeen page examples carry the whole case for the loop, and one of them --
`does a beagle swim`, the example v688 was built around -- turned out to rest
on a misreading of AwA2's zeros. That is the argument for this file: a claim
checked on seventeen chosen cases is a claim about seventeen chosen cases.

## What the labels can and cannot say

XCSLB, through COMPS. 521 concepts, 3,592 properties, 12,335 pairs it
asserts, each with four foils.

The first thing this audit found was about its own gold set, so it is written
down here before anything else:

**XCSLB's zeros are not denials.** `concept_matrix.txt` is 521 x 3,644 binary
and 1.58% dense -- 30,009 ones in 1.9M cells. That is a free-listing norm.
A zero means no participant listed the feature, not that anyone judged it
false. COMPS draws its foils from those zeros, which is why they do not read
as denials:

    stocking  NOT absorbs sweat     (taxonomic)
    potato    NOT absorbs water     (co-occurrence)

Stockings absorb sweat. So `corpora.denied_xcslb`'s claim to be "the only
place in any of this data where absence is stated rather than merely
observed" is false, and it is the same trap as the AwA2 zeros one level up.
`profile.py` merges it into `Profiles.denied`, which is what R17 answers
DENIED from.

This is why the audit scores two ways and, under `--gold base`, never scores
a foil as a denial:

    absolute    positives only. A feature somebody listed for a concept *is*
                an assertion, so "did the store confirm it" is a fair
                question. This is the sparsity number.
    relative    the minimal pair, which is what COMPS is for: does the system
                rank `sock / absorbs sweat` above `stocking / absorbs sweat`?
                That comparison is sound however absent the foil's zero is,
                because it needs only the *relative* claim.

`screen.py` buys a third way. It puts every claim on both sides of every pair
to a judge whose error rates are measured against two sets whose truth is
known independently -- AwA2's closed matrix and XCSLB's own ones -- and keeps
only the pairs whose foil that judge denies. Run with `--gold screened` the
foil side is a claim somebody says is false, so a fourth column appears:

    denials     the foil side, scored as denial. `verified` is an error,
                `unknown` is honest. This is over-affirmation on thousands of
                natural near misses at four measured distances, where
                `corrupted` is 521 synthetic ones at no distance at all.

`--gold base` remains the default, and the two are not comparable: they are
different rows. Report which one a number came from.

`negative_sample_type` orders the foils by how near they are -- random,
co-occurrence, overlap, taxonomic. Accuracy should fall along that ladder,
and if it does not, the method is measuring noise rather than knowledge.
That gradient is the audit's own control.

## The ablation

The norms are gold, so for the sweep they stop being an *answer*. The seam is
`IdentifyingEngine.about`, which already returns `None` to hand a question
back to the fact store -- the documented path taken today by `pig`, `rock`,
`tree` and `car`, not a new one opened for the audit.

    shipped     everything on. A control, not a result: the norms answer
                from the rows the question was built from.
    crawl       `about` -> None, `corroborate` -> identity. 1.9M crawled
                facts, unaided. **This is the number that generalises**,
                because it is what the other 44,678 concepts get.
    corroborated`about` -> None, R19 on. What last session's fix bought.
    loop        the same, driven by v688.

`profiles.route` is deliberately *not* disabled. Two other call sites read it
-- one to label which words a pin applies to, one as `_is_backwards`' guard
that a norm-covered concept is not an inverse question -- and silencing those
would change routing for reasons unrelated to the ablation.

**`corroborated` and `loop` are advantaged and their numbers are not a
generalisation.** R19 checks an inherited fact against the ancestor's other
kinds, and here those siblings are other rows of the same instrument. That is
not a flaw in R19, which was built to use the norms as a check; it means the
delta from `crawl` describes the shipped system on the 521 concepts the norms
cover and says nothing about the remaining 98.8%.

## The phrasing

A gold pair is a concept and a predicate, and the question is built by a
transform over the predicate's leading word. 3,499 of the lexicon's 3,927
features open with `is`, `can` or `has`; the remaining 109 leading words are
third-person singular verbs taking one shape. The de-inflection rule was
checked against all 109 and is right on 108 -- `used` is the exception and is
handled by name. The article comes from XCSLB's own `concept_senses.csv`, so
`a`/`an` is read rather than guessed. All 3,927 features phrase;
`--phrasing` prints that without building an engine.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from research.v687 import corpora
from . import confidence
from .holdout import held as holdout_held

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "data" / "v684_reasoning.sqlite"

#: Overridden by `--store`, so a source added by `ingestion.load` can be
#: measured against the same questions. Module-level because the shard
#: children re-import rather than inherit.
USING = STORE
COMPS = ROOT / "data" / "xcslb" / "comps_base.jsonl"
SCREENED = ROOT / "data" / "xcslb" / "comps_screened.jsonl"

#: `base` is COMPS as it ships, whose foils are absence; `screened` is the
#: subset `screen.py` kept, whose foils a calibrated judge denied. Same
#: module-level treatment as `USING`, and for the same reason -- the shard
#: children re-import and re-derive their questions rather than inherit them.
GOLD = "base"

#: The four configurations, in the order the report reads them.
CONFIGS = ("shipped", "crawl", "lenient", "stated", "pinned",
           "corroborated", "loop", "llm")

#: Predicate openers that are already a question's auxiliary.
AUXILIARY = {"is", "can", "was", "are", "does", "has", "have", "will",
             "must", "may"}

#: COMPS' foils, from furthest to nearest. Accuracy is expected to fall along
#: this order; that it does is what says the measurement is real.
LADDER = ("random", "co-occurrence", "overlap", "taxonomic")


# -- the phrasing ---------------------------------------------------------

def stem(verb: str) -> str:
    """Third person singular back to the bare form.

    Checked against every one of the 109 verb-initial features in the
    lexicon: right on 108. `used` is the one that is not a third-person verb
    at all, and `phrase` handles it before this is reached.
    """
    if verb.endswith("ies") and len(verb) > 4:
        return verb[:-3] + "y"
    if verb.endswith(("ches", "shes", "sses", "xes", "zes", "oes")):
        return verb[:-2]
    if verb.endswith("s") and not verb.endswith("ss"):
        return verb[:-1]
    return verb


def phrase(subject: str, feature: str) -> str:
    """A concept and a property as a question, or "" if it does not apply.

    `subject` is the article and concept together -- "a bat", "an owl" --
    exactly as XCSLB ships it, so the determiner is read and not guessed.
    """
    words = (feature or "").strip().lower().split()
    if not subject or not words:
        return ""
    head, rest = words[0], " ".join(words[1:])
    if not rest:
        # A bare verb is still a question: `slithers` -> `does a snake
        # slither`. Only an auxiliary on its own is not.
        return "" if head in AUXILIARY else f"does {subject} {stem(head)}"
    if head in ("has", "have"):
        return f"does {subject} have {rest}"
    if head == "used":                      # "used for X" -- one row
        return f"is {subject} used {rest}"
    if head in AUXILIARY:
        return f"{head} {subject} {rest}"
    return f"does {subject} {stem(head)} {rest}"


# -- the gold set ---------------------------------------------------------

@dataclass(frozen=True)
class Pair:
    """One COMPS minimal pair: a property, a concept, and a foil."""

    prop: str
    held: str               # the concept the norms list the property for
    foil: str               # the concept COMPS picked as the near miss
    foil_kind: str          # random | co-occurrence | overlap | taxonomic
    kind: str               # the property's feature type

    def keys(self) -> tuple:
        return (f"{self.held}|{self.prop}", f"{self.foil}|{self.prop}")


def store_senses() -> dict:
    """concept -> the synset XCSLB says it means, in the store's spelling.

    The audit was asking `is a donkey a mammal` in bare English and letting
    v687 choose, and v687 chose `donkey.n.01` -- "the symbol of the Democratic
    Party", which sits under `emblem -> symbol -> abstraction`. R27 then
    denied the claim at 0.95, correctly, about the wrong animal. Six of the
    ten confident false denials in the first sweep were that.

    XCSLB ships the sense key, so none of that guessing is necessary:
    `donkey%1:05:00::` resolves to `domestic_ass.n.01`. `identify.py` makes
    this join already but keeps only the lemma half of the key and drops the
    sense index, so it is redone here from the key itself.

    The `pinned` configuration is what this is for. It is `crawl` with the
    reader's sense supplied, so `pinned - crawl` separates a store that does
    not know from a reasoner that looked in the wrong place.
    """
    from nltk.corpus import wordnet

    out = {}
    for concept, key in corpora.senses().items():
        try:
            name = wordnet.lemma_from_key(key).synset().name()
        except Exception:                           # noqa: BLE001
            continue
        out[concept.replace("_", " ")] = name.replace("_", " ")
    return out


def pins_for(concept: str, senses: dict) -> dict:
    """The pin for one concept, keyed the way the parser will ask for it.

    `pins.of` is looked up by the parsed subject, so a two-word concept is
    pinned under both its full name and its head word. A pin that matches
    neither goes unused, and v687 already reports unused pins rather than
    swallowing them.
    """
    name = concept.replace("_", " ")
    sense = senses.get(name)
    if not sense:
        return {}
    pinned = {name: sense}
    if " " in name:
        pinned[name.rsplit(" ", 1)[-1]] = sense
    return pinned


def articles() -> dict:
    path = ROOT / "data" / "xcslb" / "concept_senses.csv"
    with path.open(encoding="utf-8") as handle:
        return {row["concept"]: (row["article"] or "").strip()
                for row in csv.DictReader(handle)}


def gold_file(gold: str = "") -> Path:
    """The pair file one gold setting reads.

    `screened` falls back to `base` with a warning rather than an error,
    because the screened file is a build product: a clone of the repository
    has `comps_base.jsonl` and does not have `comps_screened.jsonl` until
    somebody with a GPU runs `screen.py --build`.
    """
    if (gold or GOLD) != "screened":
        return COMPS
    if SCREENED.exists():
        return SCREENED
    print("[audit] no comps_screened.jsonl; run `python -m "
          "research.v688.screen --build`. Falling back to base.", flush=True)
    return COMPS


def pairs(limit: int = 0, gold: str = "") -> list[Pair]:
    """COMPS' minimal pairs, both sides phrasable.

    `limit` caps each rung of the foil ladder separately, so a small sample
    still estimates all four. Sampling is a strided walk over a sorted list,
    not a random draw: the same `limit` gives the same items on every machine
    and every run, which is what makes two configurations comparable.

    Under `--gold screened` the file read is the subset `screen.py` kept,
    where the foil side is a claim a calibrated judge denied rather than one
    nobody happened to list. Nothing else about the sampling changes, so a
    screened run and a base run are the same procedure over different rows.
    """
    article = articles()
    kinds = corpora.feature_types()
    rungs: dict = {name: [] for name in LADDER}
    with gold_file(gold).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            held, foil = row["acceptable_concept"], row["unacceptable_concept"]
            prop = row["property"]
            if not (phrase(article.get(held, ""), prop)
                    and phrase(article.get(foil, ""), prop)):
                continue
            rung = rungs.get(row["negative_sample_type"])
            if rung is not None:
                rung.append(Pair(prop, held, foil,
                                 row["negative_sample_type"],
                                 kinds.get(prop, "NA")))

    out = []
    for name in LADDER:
        rows = sorted(rungs[name], key=lambda p: (p.prop, p.held, p.foil))
        if limit and limit < len(rows):
            stride = len(rows) / limit
            rows = [rows[int(index * stride)] for index in range(limit)]
        out += rows
    return out


def questions_for(chosen: list) -> dict:
    """key -> question, one per distinct (concept, property) either side.

    The held side is shared by up to four pairs, so asking by key rather than
    by pair is most of the saving: 4N pairs need well under 2N questions.
    """
    article = articles()
    out = {}
    for pair in chosen:
        for concept in (pair.held, pair.foil):
            key = f"{concept}|{pair.prop}"
            if key not in out:
                out[key] = phrase(article.get(concept, ""), pair.prop)
    return out


#: Properties held inside exactly one XCSLB category are the safe ones to
#: move: `has feathers` is only ever a bird's, so a spanner does not have
#: them. Single-word properties are excluded because they are the generic
#: ones -- `is small`, `is used` -- which travel.
def corrupted(limit: int = 0, seed: int = 0) -> list:
    """Claims that are false, which nothing else here provides.

    Every negative this file had was a COMPS foil, and §1 established those
    are absence rather than denial: `carp can be a trophy` is a foil and is
    true. That makes a **false-assertion rate unmeasurable**, which is the one
    number that matters for over-affirmation -- and over-affirmation is what
    the audit that started all this found by hand.

    So negatives are built by corruption. Take a property only one category
    ever holds and put it on a concept from another: `does an arm have a
    bubble tube`. Verifiably false, cheap, and not drawn from anybody's
    absence.

    It also lifts §12's limitation. XCSLB is leaf-heavy and cannot see
    class-level over-generalisation; a corrupted claim can be built for any
    concept, so a config that over-affirms shows up here whatever it
    over-affirms about.
    """
    import csv as _csv

    path = ROOT / "data" / "xcslb" / "concept_senses.csv"
    with path.open(encoding="utf-8") as handle:
        where = {row["concept"]: row["category"]
                 for row in _csv.DictReader(handle)}
    article = articles()
    kinds = corpora.feature_types()
    holders: dict = collections.defaultdict(set)
    listed: dict = collections.defaultdict(set)
    for concept, features in corpora.load_xcslb().items:
        for feature in features:
            holders[feature].add(where.get(concept, "?"))
            listed[concept].add(feature)
    # Held by one category *and* by several of its members. One concept's
    # idiosyncratic feature is not a category marker and does not travel
    # safely: `has various styles` is listed only for an armchair, and an
    # armchair is not the only thing that has them. Requiring three holders
    # keeps `has feathers` and drops that.
    counted: dict = collections.Counter()
    for _concept, features in corpora.load_xcslb().items:
        for feature in features:
            counted[feature] += 1
    exclusive = {feature: next(iter(cats))
                 for feature, cats in holders.items()
                 if len(cats) == 1 and len(feature.split()) >= 2
                 and counted[feature] >= 3}

    rng = random.Random(seed)
    out = []
    for concept in sorted(listed):
        subject = article.get(concept, "")
        if not subject:
            continue
        mine = where.get(concept, "?")
        far = sorted(feature for feature, cat in exclusive.items()
                     if cat != mine and feature not in listed[concept])
        if not far:
            continue
        feature = far[rng.randrange(len(far))]
        text = phrase(subject, feature)
        if text:
            out.append(Pair(feature, concept, concept, "corrupted",
                            kinds.get(feature, "NA")))
    rng.shuffle(out)
    return out[:limit] if limit else out


def corrupted_questions(chosen: list) -> dict:
    """key -> question for corrupted claims. Keyed apart from the pair set so
    a concept can appear in both without one overwriting the other."""
    article = articles()
    out = {}
    for pair in chosen:
        key = f"!{pair.held}|{pair.prop}"
        if key not in out:
            out[key] = phrase(article.get(pair.held, ""), pair.prop)
    return out


def score_corrupted(answers: dict, chosen: list) -> dict:
    """How often a config asserts something built to be false.

    The over-affirmation number, at last measurable. `verified` here is
    always wrong; `unknown` is right, and so is `denied`.
    """
    rows = [answers[f"!{p.held}|{p.prop}"] for p in chosen
            if f"!{p.held}|{p.prop}" in answers]
    counts = collections.Counter(r["outcome"] for r in rows)
    total = len(rows) or 1
    spoke = counts["verified"] + counts["denied"]
    return {"n": len(rows), "outcomes": dict(counts),
            "asserted": round(counts["verified"] / total, 4),
            "refused": round(counts["denied"] / total, 4),
            "silent": round(counts["unknown"] / total, 4),
            "wrong_when_it_spoke": (round(counts["verified"] / spoke, 4)
                                    if spoke else 0.0)}


def score_denials(answers: dict, chosen: list) -> dict:
    """The foil side, once the foils are denials -- the point of the screen.

    Under `--gold base` this is meaningless and is not reported: a COMPS foil
    is a zero in a free listing, so asserting it is not an error. Under
    `--gold screened` every foil here is a claim a calibrated judge denied,
    which makes `verified` wrong, `denied` right, and `unknown` honest.

    It is the same question `score_corrupted` asks, on better material.
    `corrupted` has 521 synthetic claims built by moving a category-exclusive
    property (`does an arm have a bubble tube`) and they are easy; these are
    thousands of natural near misses, sorted into four rungs by how near.
    Over-affirmation that only shows up on the taxonomic rung is the kind
    this project keeps finding by hand, and this is the first column that can
    see it.
    """
    def bucket() -> dict:
        return {"n": 0, "asserted": 0, "refused": 0}

    overall, by_foil = bucket(), {}
    seen: dict = {}
    for pair in chosen:
        key = f"{pair.foil}|{pair.prop}"
        if key in seen or key not in answers:
            continue
        seen[key] = pair
        outcome = answers[key]["outcome"]
        for target in (overall, by_foil.setdefault(pair.foil_kind, bucket())):
            target["n"] += 1
            target["asserted"] += outcome == "verified"
            target["refused"] += outcome == "denied"

    def finish(one: dict) -> dict:
        spoke = one["asserted"] + one["refused"]
        total = one["n"] or 1
        return {**one,
                "asserted_share": round(one["asserted"] / total, 4),
                "refused_share": round(one["refused"] / total, 4),
                "silent_share": round((one["n"] - spoke) / total, 4),
                "wrong_when_it_spoke": (round(one["asserted"] / spoke, 4)
                                        if spoke else 0.0)}

    return {"overall": finish(overall),
            "by_foil": {name: finish(by_foil[name])
                        for name in LADDER if name in by_foil}}


def all_questions(limit: int = 0, corrupt: int = 0, gold: str = "") -> dict:
    """Every question one configuration is asked: the pairs and the
    corrupted claims, keyed apart so neither can shadow the other."""
    asked = questions_for(pairs(limit, gold))
    asked.update(corrupted_questions(corrupted(corrupt)))
    return asked


def phrasing_report() -> dict:
    """What the transform reaches, without building an engine."""
    lexicon = {}
    path = ROOT / "data" / "xcslb" / "feature_lexicon.csv"
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lexicon[row["feature"]] = row["feature_type"]
    article = articles()
    chosen = pairs()
    return {"features": len(lexicon),
            "unphrasable_features": sorted(
                f for f in lexicon if not phrase("a thing", f)),
            "concepts_without_article": sorted(
                c for c, _ in corpora.load_xcslb().items if not article.get(c)),
            "pairs": len(chosen),
            "distinct_questions": len(questions_for(chosen)),
            "examples": [{"feature": f, "question": phrase("a beagle", f)}
                         for f in ("has legs", "can fly", "is a vehicle",
                                   "eats meat", "carries water", "goes fast",
                                   "slithers", "used for cooking")]}


# -- the ablation ---------------------------------------------------------

def engine_class(config: str):
    """The v687 engine one configuration asks with.

    Audit apparatus, so it lives here: subclasses that turn parts off, which
    nothing in the shipped system reads.
    """
    from research.v687.reasoning import ReasoningEngine

    if config == "shipped":
        return ReasoningEngine

    class NoNorms(ReasoningEngine):
        """The norms answer nothing. R19 still checks with them."""

        def about(self, question: str):
            return None

    if config in ("corroborated", "loop"):
        return NoNorms

    if config == "lenient":
        class NoR28(NoNorms):
            """Crawl alone, with R28's plain-match requirement removed.

            R28 holds that a qualified fact does not affirm the bare claim:
            `leopard capable_of "hunt at night"` is not `a leopard hunts`. It
            is the rule that keeps `fish capable_of "walk on land"` from
            saying fish walk, and it is why `can a person run` is UNKNOWN.

            The audit found that 54% of the gold positives the crawl misses
            have a row in the store sharing a content word, and many are this
            shape -- `hose used_for "washing car"` for `is used to wash`. So
            the question is what R28 costs and what it buys, and `verify`
            reads its standard off `matcher.plain`: a matcher without one
            leaves every match plain, which is R28 off.
            """

            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                lenient = self.parser.matcher()
                if hasattr(lenient, "plain"):
                    del lenient.plain
                self.match = lenient

            def corroborate(self, answer, target):
                return answer

        return NoR28

    if config == "stated":
        class R28OnInheritedOnly(NoNorms):
            """R28 kept for inherited facts, dropped for stated ones.

            `lenient` showed R28 costs a third of the reachable coverage for
            three points of accuracy. This asks whether the trade is really
            about qualification or about *inheritance*: the case R28 was
            written for -- `fish capable_of "walk on land"` -- is a class fact
            carried down to a member, and at distance zero there is nothing to
            carry.
            """

            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.reasoner.R28_ON_STATED = False

            def corroborate(self, answer, target):
                return answer

        return R28OnInheritedOnly

    class CrawlAlone(NoNorms):
        """...and nothing checks the crawl against them either.

        `crawl` and `pinned` share this: they differ only in whether the
        reader's sense is supplied with the question.
        """

        def corroborate(self, answer, target):
            return answer

    return CrawlAlone


# -- asking ---------------------------------------------------------------

def read_answer(payload: dict, verdict: str) -> dict:
    """One v687 payload as the few fields the audit scores."""
    weight = confidence.of_answer(payload, verdict)
    evidence = (payload or {}).get("evidence") or []
    lead = evidence[0] if evidence else {}
    return {"verdict": verdict, "outcome": weight.outcome,
            "confidence": round(weight.value, 3), "band": weight.band,
            "rule": confidence.decided_by(payload),
            "source": lead.get("source") or "",
            "distance": int(lead.get("distance") or 0)}


def ask_llm(asked: list) -> list:
    """The control this whole file needs and did not have.

    A store that reaches 19% of what people list is only worth building if it
    beats asking a model the same question. So: same gold, same scoring, no
    graph at all -- SmolLM3 is handed the question `audit.py` would have put
    to v687 and answers yes or no.

    It always answers, so its coverage is 100% by construction and the number
    that matters is accuracy. A model that is right more often than the store
    *and* never silent is a model that makes the store redundant; one that is
    confidently wrong where the store is honestly absent is the opposite.

    One process, because it is one GPU. `--shards` is ignored here.
    """
    from .teacher import Teacher

    teacher = Teacher()
    rows = []
    for key, text in asked:
        started = time.time()
        if not teacher.available:
            rows.append({"key": key, "question": text, "outcome": "error",
                         "confidence": 0.0, "error": teacher.error,
                         "elapsed": 0.0})
            continue
        # The gold question as it stands. `judge` wants a fact to weigh, so
        # the claim is put to it directly instead.
        supports, confidence, cached = teacher.judge("", "", text)
        rows.append({"key": key, "question": text,
                     "verdict": "VERIFIED" if supports else "CONTRADICTED",
                     "outcome": "verified" if supports else "denied",
                     "confidence": round(confidence, 3),
                     "band": "high" if confidence >= 2 / 3 else
                             ("medium" if confidence >= 1 / 3 else "low"),
                     "rule": "LLM", "source": "smollm3", "distance": 0,
                     "cached": cached,
                     "elapsed": round(time.time() - started, 3)})
    return rows


def ask_shard(config: str, asked: list, workers: int, cycles: int) -> list:
    """Answer a slice of the question set. `asked` is a list of (key, text)."""
    if config == "llm":
        return ask_llm(asked)
    from .pool import EnginePool

    pool = EnginePool(USING, workers=workers,
                      engine_class=engine_class(config))
    senses = store_senses() if config == "pinned" else {}

    def pinned_for(key: str) -> dict:
        return pins_for(key.split("|")[0], senses) if senses else {}

    rows = []
    try:
        if config == "loop":
            from .attention import Curiosity
            from .loop import Loop

            curiosity = Curiosity(pool.engines[0].profiles.plan)
            loop = Loop(pool, curiosity, max_cycles=cycles)
            for key, text in asked:
                started = time.time()
                try:
                    run = loop.run(text, pinned_for(key) or None)
                    summary = run.summary or {}
                    read = {"verdict": summary.get("verdict") or "",
                            "outcome": summary.get("outcome") or "unknown",
                            "confidence": float(summary.get("confidence") or 0),
                            "band": summary.get("band") or "low",
                            "rule": "", "source": "", "distance": 0,
                            "asked": sum(len(c.answers) for c in run.cycles),
                            "cycles": len(run.cycles)}
                except Exception as bad:            # noqa: BLE001
                    read = {"outcome": "error", "confidence": 0.0,
                            "error": f"{type(bad).__name__}: {bad}"}
                rows.append({"key": key, "question": text, **read,
                             "elapsed": round(time.time() - started, 2)})
        else:
            from concurrent.futures import ThreadPoolExecutor

            def one(item):
                key, text = item
                return key, text, pool.ask_one(
                    text, pinned_for(key) or None)

            with ThreadPoolExecutor(max_workers=pool.workers) as runner:
                for key, text, answer in runner.map(one, asked):
                    if answer.error:
                        read = {"outcome": "error", "confidence": 0.0,
                                "error": answer.error}
                    else:
                        read = read_answer(answer.payload, answer.verdict)
                    rows.append({"key": key, "question": text, **read,
                                 "elapsed": round(answer.elapsed, 3)})
    finally:
        pool.close()
    return rows


# -- scoring --------------------------------------------------------------

def stance(row: dict) -> float:
    """One answer as a number on a line, so two can be compared.

    Verified is positive, denied is negative, and anything unsettled is zero
    -- *not* a small positive. An `unknown` is the store declining to speak,
    and a system that declines on both sides of a pair has expressed no
    preference, which is what the tie count reports.
    """
    if not row:
        return 0.0
    value = float(row.get("confidence") or 0.0)
    if row.get("outcome") == "verified":
        return value
    if row.get("outcome") == "denied":
        return -value
    return 0.0


def only_held(chosen: list) -> list:
    """The pairs whose concept was reserved from teaching.

    After anything is taught, the number that matters is this one: the store
    answering about concepts nothing wrote to. `holdout.py` says why, and why
    the reservation is over writing rather than over answering.
    """
    from . import holdout

    return [pair for pair in chosen if holdout.held(pair.held)]


def score_absolute(answers: dict, chosen: list) -> dict:
    """Positives only: did the store confirm what people listed?

    Sound in a way the foils are not -- a listed feature is an assertion.
    """
    seen = {}
    for pair in chosen:
        seen.setdefault(f"{pair.held}|{pair.prop}", pair)
    rows = [answers[key] for key in seen if key in answers]
    counts = collections.Counter(r["outcome"] for r in rows)
    total = len(rows) or 1
    reached = total - counts["unknown"] - counts["error"]
    return {"n": len(rows), "outcomes": dict(counts),
            "coverage": round(reached / total, 4),
            "confirmed": round(counts["verified"] / total, 4),
            "contradicted": round(counts["denied"] / total, 4),
            "right_when_reached": (round(counts["verified"] / reached, 4)
                                   if reached else 0.0)}


def score_pairs(answers: dict, chosen: list) -> dict:
    """The minimal pair, which is what COMPS is actually for.

    A pair is *decided* when the two sides do not score identically. Ties are
    reported rather than split, because a coin flip on a store that said
    nothing would flatter every configuration equally and hide the only thing
    that separates them here, which is how often it can speak at all.
    """
    def bucket() -> dict:
        return {"n": 0, "decided": 0, "right": 0}

    overall, by_foil, by_kind = bucket(), {}, {}
    edges = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 2.1))
    margins = {f"{low:.1f}-{min(high, 1.0):.1f}": bucket()
               for low, high in edges}
    for pair in chosen:
        held_key, foil_key = pair.keys()
        if held_key not in answers or foil_key not in answers:
            continue
        margin = stance(answers[held_key]) - stance(answers[foil_key])
        rung = next(margins[f"{low:.1f}-{min(high, 1.0):.1f}"]
                    for low, high in edges if low <= abs(margin) < high)
        for target in (overall,
                       by_foil.setdefault(pair.foil_kind, bucket()),
                       by_kind.setdefault(pair.kind, bucket()),
                       *([rung] if margin else [])):
            target["n"] += 1
            if margin != 0:
                target["decided"] += 1
                target["right"] += margin > 0

    def finish(one: dict) -> dict:
        return {**one,
                "decided_share": (round(one["decided"] / one["n"], 4)
                                  if one["n"] else 0.0),
                "accuracy": (round(one["right"] / one["decided"], 4)
                             if one["decided"] else None)}

    return {"overall": finish(overall),
            "by_foil": {name: finish(by_foil[name])
                        for name in LADDER if name in by_foil},
            "by_margin": [{"margin": label, **finish(one)}
                          for label, one in margins.items() if one["n"]],
            "by_feature_type": {k: finish(v) for k, v in sorted(
                by_kind.items(), key=lambda kv: -kv[1]["n"])}}


def reliability(answers: dict, chosen: list) -> list:
    """Does a higher number mean a better answer? The point of the exercise.

    Scored on the positives, where a label is sound: each settled answer is
    bucketed by its stated confidence and checked against the norms listing
    the property. `confidence.py` is five judgement constants and this is the
    first evidence for or against them.
    """
    held = {f"{p.held}|{p.prop}" for p in chosen}
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    out = []
    for low, high in zip(edges, edges[1:]):
        mine = [answers[k] for k in held
                if k in answers
                and answers[k].get("outcome") in ("verified", "denied")
                and low <= float(answers[k].get("confidence") or 0) < high]
        right = sum(1 for r in mine if r["outcome"] == "verified")
        out.append({"band": f"{low:.1f}-{min(high, 1.0):.1f}", "n": len(mine),
                    "accuracy": round(right / len(mine), 4) if mine else None})
    return out


def by_field(answers: dict, chosen: list, field: str) -> dict:
    """Accuracy on the positives, split by one field of the answer."""
    held = {f"{p.held}|{p.prop}" for p in chosen}
    out: dict = {}
    for key in held:
        row = answers.get(key)
        if not row or row.get("outcome") not in ("verified", "denied"):
            continue
        one = out.setdefault(str(row.get(field) or "-"), {"n": 0, "right": 0})
        one["n"] += 1
        one["right"] += row["outcome"] == "verified"
    for one in out.values():
        one["accuracy"] = round(one["right"] / one["n"], 4) if one["n"] else None
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["n"]))


# -- running --------------------------------------------------------------

def out_dir(where: str | None) -> Path:
    path = Path(where) if where else ROOT / "research" / "v688" / "audit-out"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_config(config: str, limit: int, shards: int, workers: int,
               cycles: int, where: Path, corrupt: int = 0,
               held_only: bool = False) -> dict:
    """Fan one configuration across processes, then score what comes back.

    One process per shard, each with its own pool, because the engines are
    the cost and they do not share. Five shards of four is twenty engines.
    """
    chosen = pairs(limit, GOLD)
    bad = corrupted(corrupt)
    asked = all_questions(limit, corrupt, GOLD)
    started = time.time()
    if config == "llm":
        shards = 1                      # one GPU, so one process

    procs = [subprocess.Popen(
        [sys.executable, "-m", "research.v688.audit",
         "--config", config, "--limit", str(limit),
         "--shard", str(index), "--shards", str(shards),
         "--workers", str(workers), "--cycles", str(cycles),
         "--corrupt", str(corrupt), "--gold", GOLD, "--out", str(where)]
        + (["--store", str(USING)] if USING != STORE else []), cwd=str(ROOT))
        for index in range(shards)]
    failed = [index for index, proc in enumerate(procs) if proc.wait() != 0]

    answers: dict = {}
    for index in range(shards):
        target = where / f"{config}.{index}.jsonl"
        if target.exists():
            for line in target.read_text(encoding="utf-8").splitlines():
                if line:
                    row = json.loads(line)
                    answers[row["key"]] = row

    if held_only:
        chosen = only_held(chosen)
        bad = [one for one in bad if holdout_held(one.held)]
    report = {"config": config, "gold": GOLD, "pairs": len(chosen),
              "questions": len(asked), "answered": len(answers),
              "shards_failed": failed,
              "wall_seconds": round(time.time() - started, 1),
              "absolute": score_absolute(answers, chosen),
              "corrupted": score_corrupted(answers, bad),
              "denials": (score_denials(answers, chosen)
                          if GOLD == "screened" else {}),
              "relative": score_pairs(answers, chosen),
              "reliability": reliability(answers, chosen),
              "by_source": by_field(answers, chosen, "source"),
              "by_rule": by_field(answers, chosen, "rule")}
    (where / f"{config}.json").write_text(json.dumps(report, indent=2),
                                          encoding="utf-8")
    return report


def as_text(reports: list) -> str:
    """The report as something readable in a terminal."""
    gold = reports[0].get("gold", "base") if reports else "base"
    lines = ["", "=" * 76,
             "v688 AUDIT -- COMPS/XCSLB as gold, the norms path off",
             "=" * 76, "",
             f"gold: {gold}" + (
                 "  -- COMPS as it ships. Its foils are absence, not denial,"
                 "\n        so the relative accuracy column punishes a system"
                 " for\n        knowing true things about a foil. Do not read"
                 " it as error."
                 if gold == "base" else
                 "  -- the subset `screen.py` kept, where a calibrated"
                 "\n            judge denied the foil. Smaller and easier than"
                 " `base`, so\n            the levels are not comparable"
                 " between the two; the\n            differences between"
                 " configurations are."),
             "",
             "ABSOLUTE -- positives only, where a label is sound.",
             "  coverage  how much of it the store reached at all",
             "  confirm   of all of it, how much came back verified", "",
             f"{'config':<14}{'n':>7}{'coverage':>10}{'confirm':>9}"
             f"{'contra':>9}{'wall':>9}"]
    lines.append("-" * 76)
    for r in reports:
        a = r["absolute"]
        lines.append(f"{r['config']:<14}{a['n']:>7}{a['coverage']:>10.1%}"
                     f"{a['confirmed']:>9.1%}{a['contradicted']:>9.1%}"
                     f"{r['wall_seconds']:>8.0f}s")

    lines += ["", "CORRUPTED -- claims built to be false by moving a",
              "  category-exclusive property: `does an arm have a bubble",
              "  tube`. Synthetic, easy, and the only negatives here no model",
              "  had a hand in -- so read it as an ordering between configs,",
              "  not as a level. The DENIED column is the level.",
              "  asserted  how much of it the config claimed is true", "",
              f"{'config':<14}{'n':>7}{'asserted':>10}{'refused':>9}"
              f"{'silent':>9}{'wrong when it spoke':>22}"]
    lines.append("-" * 76)
    for r in reports:
        c = r.get("corrupted") or {}
        if not c.get("n"):
            continue
        lines.append(f"{r['config']:<14}{c['n']:>7}{c['asserted']:>10.1%}"
                     f"{c['refused']:>9.1%}{c['silent']:>9.1%}"
                     f"{c['wrong_when_it_spoke']:>22.1%}")

    if any(r.get("denials", {}).get("overall", {}).get("n") for r in reports):
        lines += ["", "DENIED -- the foil side, once a calibrated judge has",
                  "  said the foil is actually false. Natural near misses,",
                  "  sorted by how near, which `corrupted` cannot be.",
                  "  asserted  claimed a screened-false thing is true", "",
                  f"{'config':<14}{'n':>7}{'asserted':>10}{'refused':>9}"
                  f"{'silent':>9}{'wrong when it spoke':>22}", "-" * 76]
        for r in reports:
            d = (r.get("denials") or {}).get("overall") or {}
            if not d.get("n"):
                continue
            lines.append(f"{r['config']:<14}{d['n']:>7}"
                         f"{d['asserted_share']:>10.1%}"
                         f"{d['refused_share']:>9.1%}"
                         f"{d['silent_share']:>9.1%}"
                         f"{d['wrong_when_it_spoke']:>22.1%}")

    lines += ["", "RELATIVE -- the minimal pair, which is what COMPS is for.",
              "  decided   pairs where the two sides did not score the same",
              "  accuracy  of those, how often the listed concept won", "",
              f"{'config':<14}{'pairs':>7}{'decided':>10}{'accuracy':>10}"]
    lines.append("-" * 76)
    for r in reports:
        o = r["relative"]["overall"]
        acc = o["accuracy"]
        lines.append(f"{r['config']:<14}{o['n']:>7}{o['decided_share']:>10.1%}"
                     f"{(acc if acc is not None else 0):>10.1%}")

    for r in reports:
        lines += ["", f"-- {r['config']} " + "-" * (72 - len(r['config'])),
                  "   foil ladder (near foils should be harder):"]
        for name in LADDER:
            one = r["relative"]["by_foil"].get(name)
            if one and one["decided"]:
                lines.append(f"     {name:<15} n={one['n']:<6} "
                             f"decided={one['decided_share']:<7.1%} "
                             f"{one['accuracy']:.1%}")
        denied = (r.get("denials") or {}).get("by_foil") or {}
        if denied:
            lines.append("   over-affirmation by how near the foil is:")
            for name in LADDER:
                one = denied.get(name)
                if one and one["n"]:
                    lines.append(f"     {name:<15} n={one['n']:<6} "
                                 f"asserted={one['asserted_share']:<7.1%} "
                                 f"silent={one['silent_share']:.1%}")
        margins = r["relative"].get("by_margin") or []
        if margins:
            lines.append("   calibration: does a wider margin mean more often "
                         "right?")
            for one in margins:
                lines.append(f"     margin {one['margin']}  "
                             f"n={one['decided']:<6} {one['accuracy']:.1%}")
        lines.append("   reliability on positives (stated confidence vs. how "
                     "often right):")
        for one in r["reliability"]:
            if one["n"]:
                lines.append(f"     {one['band']}  n={one['n']:<6} "
                             f"{one['accuracy']:.1%}")
        for field in ("by_source", "by_rule"):
            shown = [(k, v) for k, v in r[field].items() if v["n"] >= 20][:6]
            if shown:
                lines.append(f"   {field.replace('by_', 'by ')}:")
                for name, one in shown:
                    lines.append(f"     {name:<18} n={one['n']:<6} "
                                 f"{one['accuracy']:.1%}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    global GOLD, USING          # both are re-read by the shard children

    parser = argparse.ArgumentParser(
        description="Audit v688 against COMPS/XCSLB, with the norms path off.")
    parser.add_argument("--config", default="",
                        help="one of " + ", ".join(CONFIGS))
    parser.add_argument("--limit", type=int, default=750,
                        help="pairs per rung of the foil ladder (0 = all)")
    parser.add_argument("--loop-limit", type=int, default=60,
                        help="pairs per rung for the loop, which is slower")
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4,
                        help="engines per shard process")
    parser.add_argument("--cycles", type=int, default=12,
                        help="most internal cycles one loop run may take")
    parser.add_argument("--shard", type=int, default=-1,
                        help=argparse.SUPPRESS)   # set by the parent
    parser.add_argument("--only-holdout", action="store_true",
                        help="score only the concepts reserved from "
                             "teaching; the reading that matters after any "
                             "of it has happened")
    parser.add_argument("--corrupt", type=int, default=0,
                        help="corrupted false claims to ask as well "
                             "(0 = all 521); the over-affirmation measure")
    parser.add_argument("--gold", default=GOLD, choices=("base", "screened"),
                        help="`base` is COMPS as it ships, whose foils are "
                             "absence rather than denial; `screened` is the "
                             "subset `screen.py` kept, and is the only one "
                             "whose accuracy column means what it says")
    parser.add_argument("--store", default="",
                        help="a store other than the built one, e.g. one "
                             "`ingestion.load` wrote")
    parser.add_argument("--out", default="")
    parser.add_argument("--phrasing", action="store_true",
                        help="report the transform's coverage and stop")
    options = parser.parse_args(argv)
    where = out_dir(options.out)
    GOLD = options.gold
    if options.store:
        USING = Path(options.store)

    if options.phrasing:
        print(json.dumps(phrasing_report(), indent=2))
        return 0

    def limit_for(config: str) -> int:
        return options.loop_limit if config == "loop" else options.limit

    # -- a child: answer one slice and write it out ------------------------
    if options.shard >= 0:
        # `--limit` as given, never re-resolved: the parent has already chosen
        # the effective limit for this config and a child that recomputes it
        # answers a different sample than the parent scores.
        asked = sorted(all_questions(options.limit, options.corrupt,
                                     options.gold).items())
        mine = [item for index, item in enumerate(asked)
                if index % options.shards == options.shard]
        rows = ask_shard(options.config, mine, options.workers, options.cycles)
        (where / f"{options.config}.{options.shard}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return 0

    # -- the parent: every configuration, then the report ------------------
    wanted = [options.config] if options.config else list(CONFIGS)
    # `llm` is the same model that screened the gold, so on screened pairs it
    # is marking its own paper: it would deny every foil it was kept for, by
    # construction, and score near 100%. Dropped from the sweep rather than
    # reported with an asterisk, because an asterisk in a table gets read as
    # a number. Its honest measurement is the `base` run in AUDIT.md §13.
    if GOLD == "screened" and "llm" in wanted:
        wanted = [name for name in wanted if name != "llm"]
        print("[audit] skipping `llm`: it screened this gold set, so its "
              "score on it would be circular. See AUDIT.md §13 for the "
              "measurement that is not.", flush=True)
        if not wanted:
            return 0
    reports = []
    for config in wanted:
        limit = limit_for(config)
        print(f"[audit] {config}: {options.shards} shards x "
              f"{options.workers} engines, {limit} pairs per rung", flush=True)
        reports.append(run_config(config, limit, options.shards,
                                  options.workers, options.cycles, where,
                                  options.corrupt, options.only_holdout))
        last = reports[-1]
        print(f"[audit] {config}: {last['answered']} questions in "
              f"{last['wall_seconds']}s", flush=True)

    text = as_text(reports)
    print(text)
    (where / "audit.txt").write_text(text, encoding="utf-8")
    (where / "audit.json").write_text(json.dumps(reports, indent=2),
                                      encoding="utf-8")
    print(f"[audit] written to {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
