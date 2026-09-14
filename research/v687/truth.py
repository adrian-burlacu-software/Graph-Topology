"""Truth as evidence: what a link is worth, and what three answers are read off.

A verdict used to be decided wherever it was reached. R19 counted kinds and
compared a ratio to a floor, below a minimum sample it declined to judge; R5
multiplied a confidence per level; R8 and R20 kept three values; v688 priced a
claim by its ground and moved it by what the loop found. Each is one account
of evidence (`v690/DESIGN.md` §4.2), after NARS (Wang 2013):

    positive, negative    the evidence for and against, as counts
    frequency  f = positive / (positive + negative)
    confidence c = total / (total + K)

Three answers are read off it, never stored: silent while `c` is below what it
takes to speak, stated when `f` clears the floor, denied when it falls under
the floor's mirror, and mixed in between. Silence is `c` too small, not `f`
at 0.5 -- absent, not false.

**K is a sample size, not a probability.** At K = 1, eight kinds give
c = 8/9, and "a sample worth refusing on" (R19's eight kinds) is c ≥ 8/9. The
constants that were sample sizes keep their numbers; what changes is that
they are now one quantity, and one fit can move them together.

Nothing here reads the store.
"""
from __future__ import annotations

from dataclasses import dataclass

#: The evidential horizon: how much evidence counts as much as none. One, as
#: NARS has it; every threshold below is stated in samples at this K.
K = 1.0

TRUE, FALSE, UNKNOWN = "TRUE", "FALSE", "UNKNOWN"


@dataclass(frozen=True)
class Evidence:
    positive: float = 0.0
    negative: float = 0.0

    @property
    def total(self) -> float:
        return self.positive + self.negative

    @property
    def frequency(self) -> float | None:
        """The share of the evidence that is for it; None with no evidence."""
        return self.positive / self.total if self.total else None

    def confidence(self, k: float = K) -> float:
        return self.total / (self.total + k) if self.total else 0.0

    def revise(self, other: "Evidence") -> "Evidence":
        """Two independent bodies of evidence pooled (NARS revision)."""
        return Evidence(self.positive + other.positive,
                        self.negative + other.negative)

    def as_dict(self) -> dict:
        frequency = self.frequency
        return {"positive": self.positive, "negative": self.negative,
                "frequency": None if frequency is None else round(frequency, 3),
                "confidence": round(self.confidence(), 3)}


def counted(bearing: int, kinds: int) -> Evidence:
    """Induction from a kind's members: `bearing` of `kinds` bear it out."""
    return Evidence(float(bearing), float(max(kinds - bearing, 0)))


def speaks_at(samples: float, k: float = K) -> float:
    """The confidence `samples` pieces of evidence reach: a sample size said
    as a confidence, so a threshold written as a count reads the same."""
    return samples / (samples + k) if samples > 0 else 0.0


def judge(evidence: Evidence, floor: float, speaks: float,
          k: float = K) -> str:
    """TRUE, FALSE or UNKNOWN: silent below `speaks`, stated at `floor` or
    above, denied at `1 - floor` or below, and unsettled between."""
    if evidence.confidence(k) < speaks or evidence.frequency is None:
        return UNKNOWN
    if evidence.frequency >= floor:
        return TRUE
    if evidence.frequency <= 1.0 - floor:
        return FALSE
    return UNKNOWN


def borne_out(evidence: Evidence, floor: float) -> bool:
    """Does the share for it reach the floor? Whatever the sample size --
    whether the sample may speak at all is `judge`'s other half."""
    return evidence.frequency is not None and evidence.frequency >= floor


def deduced(value: float, distance: int, decay: float) -> float:
    """R5: a value borrowed from `distance` levels up keeps `decay` of itself
    per level. Deduction down the taxonomy spends confidence, not frequency."""
    return value * (decay ** distance)
