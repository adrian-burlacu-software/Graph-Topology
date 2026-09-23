"""Goals made from the store, for the proposer to learn from.

v693 made maths goals from objects it built. The open world's goals are
made from what the store says things are for: a row `used_for cut paper`
is a goal, *cut the paper*, and `used_for keep food cold` is another,
*make the food cold*. Nothing is written by hand; which rows become goals
is decided by the words -- a verb VerbNet has, a noun WordNet has, an
adjective after `keep` or `make`.

    labelled(count, seed)   (goal, the way of its simplest design) pairs

The label is the designer's own answer, checked in the imagined world --
the same as v693's.
"""
from __future__ import annotations

import random

from research.v694 import knowing as K
from research.v694.goals import Goal


def _noun(word: str) -> bool:
    """A particular physical thing: what a designer can do something to.
    *Sail* and *church* are, *obligation* and *nigeria* are not."""
    first = K.first_sense(word)
    return first is not None and not K.lives(word) and \
        not K.category(word) and K.is_a(word, "physical_entity.n.01") \
        and not K.is_a(word, "location.n.01")


def _physical(verb: str) -> bool:
    """A verb that does something to a thing: one VerbNet gives an
    instrument (`cut`, `open`, `warm`)."""
    from research.v694.goals import takes_instrument
    return takes_instrument(verb)


def _adjective(word: str) -> bool:
    try:
        from nltk.corpus import wordnet
        return bool(wordnet.synsets(word, wordnet.ADJ))
    except Exception:                              # noqa: BLE001
        return False


def candidates() -> list:
    """Every literal the store's rows make a goal of, once each."""
    from research.v689 import change
    frames = change.frames()
    found = set()
    for relation in K.SERVING:
        for said, in K.connection().execute(
                "SELECT DISTINCT object FROM facts WHERE relation = ?",
                (relation,)):
            words = said.lower().split()
            if len(words) == 2 and words[0] in frames and \
                    _physical(words[0]) and _noun(words[1]):
                found.add(f"{words[0]} {K.name_of(words[1])}")
            elif (len(words) == 3 and words[0] in ("keep", "make")
                  and _noun(words[1]) and _adjective(words[2])
                  and words[2] not in frames):
                found.add(f"{words[2]} {K.name_of(words[1])}")
    return sorted(found)


def labelled(count: int, seed: int, exclude=()) -> list:
    """(goal, the way its simplest design takes, or None) for `count`
    goals drawn from the store's rows."""
    from research.v694 import designing
    rng = random.Random(seed)
    pool = [one for one in candidates() if one not in set(exclude)]
    rng.shuffle(pool)
    out = []
    for literal in pool[:count]:
        goal = Goal((literal,))
        found = designing.design(goal)
        best = found.parts[0].best if found.parts else None
        out.append((goal, best.form if best is not None else None))
    return out
