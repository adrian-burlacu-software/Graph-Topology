"""One outcome and one number, for a system that had seventeen of the first
and none of the second.

v687 answers with a vocabulary that grew a word at a time as the audits found
distinctions worth keeping -- VERIFIED, HELD, INHERITED, CONTRADICTED,
DENIED, UNKNOWN, UNRECORDED, NO_MATCH, UNSUPPORTED, UNKNOWN_WORD, UNPARSED,
AMBIGUOUS, MIXED, LISTING, DEFINED, PROFILE, IDENTIFIED. Every one of them
earns its place inside the reasoner, where the difference between "absent"
and "refused by name" chooses the repair. None of them belongs on a badge a
person reads in a second.

So: four readings, and a number beside them.

    verified   the claim holds
    denied     the claim fails
    unknown    the store does not settle it, for any of nine reasons
    retrieved  nothing was claimed -- a definition, a listing, a profile

The seventeen are not thrown away. `gap.kind_of` still routes the repair,
`Loop.trust` still says in a phrase what went wrong, and both travel
alongside. This is the reading, not the reasoning.

## The number

It is a product of factors, each of which is either a measured distribution,
a constant that already existed in this codebase, or a count of answers the
loop actually got back. Every factor is reported with its reason, so a reader
can see which one cost the claim its confidence rather than being handed a
float to trust.

The *ground* is what the derivation stood on, and the four things an answer
can rest on here are not equally good:

    R1, R27   0.95   WordNet's taxonomy. Curated by lexicographers, and the
                     answer is a walk over it rather than a match against it.
    R29       0.95   Synset to synset, again WordNet, again no string matched.
    R17       0.85   The feature norms. Elicited from people, one fixed
                     question per concept -- but only 541 concepts, so a
                     little below the taxonomy.
    R4, R3    per    A crawled fact, and this is where calibration is needed,
                     because the three sources write confidence in three
                     incomparable ways.

`percentile` is that calibration. Ascent++ has a real distribution and a
fact's standing is its position in it; ConceptNet writes 0.35 on all 88,683
of its rows and WordNet 0.95 on all 66,376, so for those the number carries
no information at all and the source is the whole of the signal. This is the
same finding that made `does a beagle swim` stop being called weakly held:
0.42 sounds low and is Ascent++'s 78th percentile.

Then the claim is moved from that ground by what the loop went and found
out: distance costs it, doubts cost it, kin bearing it out earns it back.
Those multipliers are the one place here that is judgement rather than
measurement, so they are few, they are named together, and none of them is
further from 1 than 0.5.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import gap

#: The four readings. Seventeen verdicts collapse onto these, and nothing
#: else in the loop reads this map -- `gap.kind_of` still chooses repairs.
OUTCOME = {
    "VERIFIED": "verified", "HELD": "verified", "INHERITED": "verified",
    "CONTRADICTED": "denied", "DENIED": "denied",
    "LISTING": "retrieved", "DEFINED": "retrieved",
    "PROFILE": "retrieved", "IDENTIFIED": "retrieved",
}

#: Ascent++'s confidence distribution, measured over all 1,808,006 of its
#: rows rather than guessed. A fact's standing is where it falls in this.
ASCENT = ((0.081, 0.05), (0.125, 0.10), (0.157, 0.25), (0.264, 0.50),
          (0.394, 0.75), (0.513, 0.90), (0.588, 0.95), (0.763, 0.99))

#: Sources whose confidence column is one number for every row, so the
#: number is not a signal and the source is.
FLAT = {"conceptnet": 0.50, "wordnet": 0.90}

#: What the derivation stood on, before anything the loop found moved it.
GROUND = {"R1": 0.95, "R27": 0.95, "R29": 0.95, "R26": 0.90,
          "R17": 0.85, "R16": 0.85, "R21": 0.85, "R20": 0.80}

#: Rules whose walk *is* the proof, so distance from the concept costs them
#: nothing. Subsumption is transitive and exact: `is a beagle a dog` is no
#: less true for the dog being three levels up, and decaying it there read
#: 0.55 -- medium confidence that a beagle is a dog.
#:
#: R2 and R4 are the opposite and are absent on purpose. An inherited
#: *property* is defeasible, which is the whole of what R3 exists to
#: override, so every level of borrowing costs it.
EXACT = frozenset({"R1", "R27", "R29"})

#: Where the bands fall. Thirds, deliberately: any finer split would claim a
#: precision the inputs do not have.
LOW, HIGH = 1 / 3, 2 / 3

#: The judgement in this file, kept small and kept together.
PER_LEVEL = 0.85          # R5's own decay, one level of borrowing
DOUBT_COSTS = 0.80        # each undermining doubt, compounding
BORNE_OUT = 1.25          # the family was asked and most of it agreed
OVERTURNED = 0.55         # the loop had to correct what v687 answered
CONFLICTED = 0.50         # its own family denied it


@dataclass
class Weight:
    """A confidence, and every factor that made it."""

    value: float
    outcome: str
    factors: list = field(default_factory=list)

    @property
    def band(self) -> str:
        if self.value >= HIGH:
            return "high"
        return "medium" if self.value >= LOW else "low"

    def as_dict(self) -> dict:
        return {"outcome": self.outcome, "confidence": round(self.value, 2),
                "band": self.band,
                "factors": [{"name": name, "factor": round(factor, 2),
                             "why": why}
                            for name, factor, why in self.factors]}


def outcome_of(verdict: str) -> str:
    """One of the four. Anything unlisted reads as unknown, which is the safe
    default: a verdict this does not know about has settled nothing."""
    return OUTCOME.get((verdict or "").upper(), "unknown")


def percentile(source: str, value: float) -> tuple:
    """Where a fact stands within its own source, and why."""
    source = (source or "").strip().lower()
    if source in FLAT:
        return FLAT[source], (
            f"{source} writes the same confidence on every row, so the "
            f"number says nothing and the source is the whole of the signal")
    if source != "ascentpp":
        return 0.5, f"{source or 'an unnamed source'} is not calibrated here"
    low_value, low_rank = 0.0, 0.0
    for edge, rank in ASCENT:
        if value <= edge:
            span = edge - low_value
            share = (value - low_value) / span if span else 0.0
            place = low_rank + share * (rank - low_rank)
            return place, (f"{value:.2f} is about the {place * 100:.0f}th "
                           f"percentile of Ascent++")
        low_value, low_rank = edge, rank
    return 0.99, f"{value:.2f} is in the top percentile of Ascent++"


def decided_by(payload: dict) -> str:
    """The rule of the step that actually settled it.

    The last step that matched something, because that is the one that ended
    the walk. Falling back to the last step with any rule at all covers the
    paths that answer without a match -- a definition, a refusal.
    """
    steps = (payload or {}).get("steps") or []
    for step in reversed(steps):
        if step.get("matched"):
            return step.get("rule") or ""
    for step in reversed(steps):
        if step.get("rule"):
            return step.get("rule") or ""
    return ""


def ground(payload: dict) -> tuple:
    """The strength of whatever the answer actually rested on."""
    rule = decided_by(payload)
    if rule in EXACT:
        return GROUND.get(rule, 0.95), (
            f"{rule}: the walk over WordNet's taxonomy is the proof, not a "
            f"fact matched against it")
    evidence = (payload or {}).get("evidence") or []
    if evidence:
        lead = evidence[0]
        return percentile(lead.get("source") or "",
                          float(lead.get("confidence") or 0.0))
    # No fact means the norms or a definition answered. The rule of the last
    # step that did any work is what to price.
    for step in reversed((payload or {}).get("steps") or []):
        if (step.get("rule") or "") in GROUND:
            return GROUND[step["rule"]], (
                f"{step['rule']} answered it, not a crawled fact")
    return 0.5, "nothing in the derivation says what this rested on"


def of_answer(payload: dict, verdict: str = "") -> Weight:
    """What one answer is worth, from its own payload and nothing else.

    Every row in the table gets this. The headline gets `of_run`, which also
    knows what the rest of the loop found out about it.
    """
    payload = payload or {}
    outcome = outcome_of(verdict or payload.get("verdict") or "")
    if outcome == "unknown":
        # Silence is not a claim held weakly, it is no claim at all, and a
        # number beside it would be read as one.
        return Weight(0.0, outcome,
                      [("no claim", 0.0, "nothing was settled either way")])
    value, why = ground(payload)
    factors = [("ground", value, why)]
    evidence = payload.get("evidence") or []
    distance = int(evidence[0].get("distance") or 0) if evidence else 0
    if distance and decided_by(payload) in EXACT:
        factors.append(("exact", 1.0,
                        f"{distance} level(s) up, and subsumption is "
                        f"transitive: the distance is the proof"))
    elif distance:
        step = PER_LEVEL ** distance
        value *= step
        factors.append(("inherited", step,
                        f"borrowed from {distance} level(s) up, at R5's own "
                        f"{PER_LEVEL} per level"))
    return Weight(max(min(value, 1.0), 0.0), outcome, factors)


def of_run(headline, buffer, conflicts, overturned: bool,
           corrected=None) -> Weight:
    """What the run is worth once the loop has had its say.

    The answer's own payload is the ground; everything the loop went and
    found out moves it. This is the number the badge shows, and it is the
    reason the loop exists: a single `ask` cannot produce it, because a
    single ask has nothing to compare its answer against.
    """
    if headline is None:
        return Weight(0.0, "unknown", [("nothing asked", 0.0, "no answer")])
    speaking = corrected or headline
    weight = of_answer(speaking.payload, speaking.verdict)
    if weight.outcome == "unknown":
        return weight
    value, factors = weight.value, list(weight.factors)

    if overturned:
        value *= OVERTURNED
        factors.append(("overturned", OVERTURNED,
                        "the loop had to correct what v687 answered"))
    if any(bad.question == headline.question for bad in conflicts):
        value *= CONFLICTED
        factors.append(("its family denied it", CONFLICTED,
                        "the kin this claim was put to disagreed"))
    doubts = sorted({one.reason for one in buffer.seen_doubts
                     if one.question == headline.question
                     and one.reason in gap.UNDERMINING})
    if doubts:
        step = DOUBT_COSTS ** len(doubts)
        value *= step
        factors.append((f"{len(doubts)} doubt(s)", step, ", ".join(doubts)))
    if buffer.borne_out(headline.question):
        value *= BORNE_OUT
        factors.append(("corroborated", BORNE_OUT,
                        "the family was asked and most of it agreed"))
    return Weight(max(min(value, 1.0), 0.0), weight.outcome, factors)
