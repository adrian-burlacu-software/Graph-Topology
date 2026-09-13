"""Retrieval: which one, when several fit.

Every place that picks one thing out of several used to pick in its own way.
`discourse.py` filtered by gender, then preferred people, then the word it was
introduced by, then ranked by salience and asked for a lead; `story.py` took
the latest occurrence, the last place, the last one to have given it; I1 took
the last told. They are one operation (`v690/DESIGN.md` §4.4), after ACT-R's
retrieval (Anderson et al. 2004): among the candidates that fit, the most
active is recalled if it is far enough ahead, and otherwise nothing is, or the
question is which one.

    preferences   what the candidate should be, strongest first: each keeps
                  the candidates that satisfy it, unless none does. A
                  preference narrows; it never empties the field (a filter
                  that may empty it is the caller's, before retrieval).
    activation    how live each candidate is: salience for an individual,
                  recency for what was told or happened.
    lead          how far ahead the most active must be. None: ahead at all,
                  as a pronoun needs. A number: that many times the next, as
                  a description needs, because two beagles a turn apart are
                  genuinely ambiguous.

**Recency is base-level activation.** ACT-R's base level for something met
`n` times is `ln Σ t_j^-d`; met once, it falls as its age grows, so the most
recent is the most active and "the latest" is retrieval with no margin. That
is why I1's last-told and story's latest-first are this function rather than
rules of their own: where the members of a kind disagree, what was told last
is the more active evidence (qa16: bAbI's answer 59 times in 64).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

#: ACT-R's default decay for base-level learning.
DECAY = 0.5


@dataclass
class Retrieved:
    """What retrieval came to, and on what."""

    chosen: object | None
    #: the candidates left once the preferences had narrowed them, most
    #: active first
    ranked: list = field(default_factory=list)
    top: float = 0.0
    second: float = 0.0
    #: the preferences that narrowed the field, in order
    kept_by: list = field(default_factory=list)

    @property
    def ambiguous(self) -> bool:
        return self.chosen is None and len(self.ranked) > 1


def preferred(pool: Iterable, preferences=()) -> tuple[list, list]:
    """(the candidates left, the names of the preferences that narrowed)."""
    pool = list(pool)
    kept_by = []
    for name, test in preferences:
        fitting = [one for one in pool if test(one)]
        if fitting and len(fitting) < len(pool):
            pool = fitting
            kept_by.append(name)
    return pool, kept_by


def retrieve(pool: Iterable, activation: Callable[[object], float],
             order: Callable[[object], float] | None = None,
             lead: float | None = None, preferences=()) -> Retrieved:
    """The most active candidate, if it is far enough ahead.

    Ties in activation go to the higher `order` (the later introduced, the
    later told). One candidate left is recalled whatever its activation:
    nothing else could be meant.
    """
    pool, kept_by = preferred(pool, preferences)
    if not pool:
        return Retrieved(None, [], kept_by=kept_by)
    if len(pool) == 1:
        return Retrieved(pool[0], pool, activation(pool[0]), 0.0, kept_by)
    ranked = sorted(pool, key=lambda one: (-activation(one),
                                           -(order(one) if order else 0)))
    top, second = activation(ranked[0]), activation(ranked[1])
    ahead = top > second if lead is None else top >= lead * second
    chosen = ranked[0] if top > 0 and ahead else None
    return Retrieved(chosen, ranked, top, second, kept_by)


def latest(pool: Iterable, when: Callable[[object], object]):
    """The most recent: retrieval by recency, where the most active is always
    ahead because no two things were met at the same moment. `when` may be a
    tuple -- (place in the story, place in the telling). None for an empty
    pool."""
    return max(pool, key=when, default=None)


def base_level(presentations: Iterable[float], now: float,
               decay: float = DECAY) -> float:
    """ACT-R's base-level activation: `ln Σ (now - t)^-d` over the times
    something was met, each at least a moment ago. -inf if never met."""
    ages = [max(now - one, 1e-3) for one in presentations]
    if not ages:
        return -math.inf
    return math.log(sum(age ** -decay for age in ages))
