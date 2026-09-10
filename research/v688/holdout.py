"""Concepts nothing is allowed to teach, so that teaching can be measured.

The moment a model writes facts into the store, `AUDIT.md` stops meaning what
it means today. Its numbers say something about the store's knowledge only
because XCSLB is independent of it; a store taught by a model and measured
against a set that model has also read is a store measuring its own teacher.

So three categories are reserved before the first fact is written.

**This holds at teaching, not at answering.** A held-out concept is asked,
walked, inherited through and adjudicated exactly as any other; the only
thing it never gets is a *written* fact. A store that refused to answer about
sparrows would be a broken store, not a controlled one -- the control is over
what was learned, not over what may be said. `does a sparrow have feathers`
answers from the crawl and the taxonomy after teaching precisely as it did
before, which is the point: any change in that answer came from somewhere
else.

Whatever teaching does to `bird`, `tool` and `fruit` is therefore what it
would do to a concept it has not seen, and that is the only honest estimate
available.

    bird    36 concepts    a natural kind
    tool    31             an artifact
    fruit   28             neither, and the category most entangled with food

95 of 521, about 18%. Three categories rather than a random 18% because the
question is whether teaching *generalises*, and a random sample of concepts
leaves near neighbours on both sides of the line: teach `sparrow` and `robin`
and holding out `wren` measures very little.

## This is not a defence against contamination

It cannot be. SmolLM3 has very likely read CSLB, COMPS or something built
from them, so a high score on the holdout may be memory rather than
generalisation, and no partition of *our* data changes what is in *its*
weights. Nor does swapping which categories are taught: the model's beliefs
come from training, not from the store, so partitioning the data does not
partition the beliefs.

What the holdout does measure is narrower and still worth having: whether
teaching one part of the store improves answers about a part it did not
touch, and whether it damages them. That is the question that decides whether
teaching is worth doing, and it is unanswerable once everything is taught.

## Using it

    from research.v688 import holdout

    holdout.held(concept)          # never write to this one
    holdout.concepts()             # the 95, for measuring on

`audit.py` takes `--only-holdout` to score them alone, which is the reading
that matters after teaching.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SENSES = ROOT / "data" / "xcslb" / "concept_senses.csv"

#: Reserved 2026-09-09, before anything was taught. Changing this list
#: invalidates every before-and-after comparison that used it, so it is a
#: constant and not an option.
CATEGORIES = ("bird", "tool", "fruit")


@lru_cache(maxsize=1)
def concepts() -> frozenset:
    """The held-out concepts, by XCSLB's own category column."""
    if not SENSES.exists():
        return frozenset()
    with SENSES.open(encoding="utf-8") as handle:
        return frozenset(
            row["concept"] for row in csv.DictReader(handle)
            if row["category"] in CATEGORIES)


@lru_cache(maxsize=1)
def words() -> frozenset:
    """The same set spelled as the words a question would use."""
    return frozenset(name.replace("_", " ") for name in concepts())


def held(concept: str) -> bool:
    """Is this concept reserved? Anything that writes must ask first."""
    name = (concept or "").strip().lower()
    if not name:
        return False
    return name in concepts() or name.replace("_", " ") in words()
