"""The inference policy: what descends a taxonomy, and what stops it.

Thirteen rules. Each is stated once, here, so that an answer can be traced back
to the rule that produced it rather than to a heuristic buried in a query.

R1  Subsumption closure     is_a is transitive; WordNet is acyclic so the
                            closure terminates without a visited-set hack.
R2  Property lift           a subtype inherits a supertype's facts, but only
                            for relations in INHERITABLE. Membership is a
                            claim about the relation, not a convenience.
R3  Exception blocking      a fact stated closer to the concept overrides an
                            inherited one, and an explicit negation blocks the
                            positive. Inheritance is defeasible, not monotonic.
R4  Specificity preference  when several ancestors answer, the nearest wins.
R5  Confidence decay        a fact borrowed from d levels up is worth less
                            than one stated directly.
R6  Sense scoping           inference runs per sense, never per word string.
R7  Relation gating         `related_to` and other contentless relations never
                            participate.
R8  Answer synthesis        a claim is VERIFIED, CONTRADICTED or UNKNOWN --
                            never "probably".
R9  Relation families       has_a and has_part answer for each other.
R10 Redundancy elimination  a fact an ancestor states is not stored twice.
R11 Hoisting                a fact every child states moves to the parent.
R12 Breadth gating          a word-level fact does not inherit from a concept
                            too general for the word to have meant it. This
                            sets facts aside; it does not stop the walk.
R13 Range typing            a relation's object must be the kind of thing the
                            relation takes.

The rule that matters most is R2. `research/v683/diagnose.py` measured what
happens without it: trusting every relation over every edge yields a mean of
1,548 derived facts per concept, most of them wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .links import LINKS, named
from .rulebook import rule, texts

# Which relation is which -- inheritable, gated, a negation, a family -- is
# each relation's row in `links.py`, with the reasons. The names below are
# read off that table, so the rules and the table cannot disagree.

# -- R2: which relations descend ------------------------------------------
#: A subtype has whatever the supertype has. If mammals can breathe, dogs can.
INHERITABLE: frozenset[str] = named(lambda one: one.inherited)

#: Relations that do NOT descend, with the reason each is excluded.
NOT_INHERITABLE: dict[str, str] = {
    name: one.why_not for name, one in LINKS.items() if one.why_not}

# -- R7: relations that never participate ---------------------------------
GATED: frozenset[str] = named(lambda one: one.gated)

#: R3: an assertion of the key blocks inheritance of the value, and vice versa.
NEGATIONS: dict[str, str] = {
    name: one.denies for name, one in LINKS.items() if one.denies}
POSITIVES: dict[str, str] = {value: key for key, value in NEGATIONS.items()}

# -- R9: relations that answer for each other ------------------------------
FAMILIES: tuple[frozenset[str], ...] = tuple(dict.fromkeys(
    one.family for one in LINKS.values() if one.family))


def family(relation: str) -> list[str]:
    """Every relation that can answer a question asked about `relation`."""
    for group in FAMILIES:
        if relation in group:
            return sorted(group)
    return [relation]


#: R5: what one level of borrowing costs. A fact five levels up retains
#: 0.85**5 = 0.44 of its confidence, so `thing capable_of fall down` ranks
#: below anything stated about dogs directly.
DECAY = rule("R5").parameters["decay"]

#: R5: below this, a derived fact is not worth reporting.
FLOOR = rule("R5").parameters["floor"]

#: R12: how wide a subtree makes a word-level fact untrustworthy to inherit.
#:
#: Ascent++ and ConceptNet talk about words, and a word in an open-domain
#: corpus never means a top-of-taxonomy abstraction. Their `artifact` is a
#: thing dug out of a tomb, not WordNet's "man-made object taken as a whole";
#: their `person` is somebody, not the class of all persons. Attached to those
#: synsets and inherited, 8,815 such facts reach 10,000+ descendants each --
#: which is why asking about a hammer returned `at_location tomb`.
#:
#: Six concepts sit above this line: entity, living thing, organism, causal
#: agent, artifact, person. `animal` (4,016 descendants, 1,408 facts) and
#: `plant` (4,487) sit below it and keep inheriting, because there the word
#: and the class really do mean the same thing. The gap in the data between
#: those two groups is where the threshold goes.
BREADTH_LIMIT = rule("R12").parameters["breadth_limit"]


def inheritable(relation: str) -> bool:
    """R2 + R7 in one predicate."""
    return relation not in GATED and relation in INHERITABLE


#: R13: what a relation's object is allowed to be.
#:
#: A relation has a range, and a corpus that stated facts about words did not
#: check it. 26.9% of the `at_location` rows name something that is not a
#: place at all -- `concept at_location play`, `obligation at_location
#: writing`, `people at_location way` -- and asking where a hammer is returned
#: `communication` and `high quality` because of it.
#:
#: The check uses WordNet's own top-level split, so it needs no new data: the
#: object is resolved to a sense and that sense must fall under the named root.
#: A word the ontology does not know passes -- the rule only fires on what it
#: can actually judge, which is why it costs 11.4% of returned facts and
#: empties 12 answers in 770 while keeping every deep generalisation tested
#: (`crested screamer` -> `bird`, `timber rattlesnake` -> `reptile`).
#:
#: Only word-level facts are checked. WordNet's own are already sense-tagged.
RANGES: dict[str, str] = {
    name: one.range for name, one in LINKS.items() if one.range}


def inheritable_from(relation: str, breadth: int, sense_assumed: bool) -> bool:
    """R2 + R7 + R12: may this fact descend from a concept this general?

    `breadth` is the size of the ancestor's subtree. A fact WordNet states
    about a sense inherits however general the sense is; a fact a corpus
    stated about a word does not, once the sense covers more of the world
    than the word ever meant.
    """
    if not inheritable(relation):
        return False
    return not (sense_assumed and breadth >= BREADTH_LIMIT)


def why_not_inheritable(relation: str) -> str:
    """The stated reason a relation does not descend, for the UI."""
    if relation in GATED:
        return "gated: carries no usable semantics"
    return NOT_INHERITABLE.get(relation, "not marked inheritable")


def confidence_at(base: float, distance: int) -> float:
    """R5: deduction down the taxonomy spends confidence (`truth.py`)."""
    from .truth import deduced
    return deduced(base, distance, DECAY)


#: R3: words that turn an object phrase into a denial of itself.
#:
#: The sources write negation in two places. `not_has_part` puts it in the
#: relation, which R3 has always read; `has_part "no legs"` puts it in the
#: object, which nothing read at all. Lemma overlap then scored "no legs" as
#: a full answer to "legs", and `does a fish have legs` came back VERIFIED on
#: the fact that a fish has none.
DENIERS = frozenset({"no", "not", "never", "cannot", "without", "lack",
                     "lacks", "lacking"})


def denial_in(fact_object: str, target: str | None, matcher) -> bool:
    """Does this object state the target only to deny it?

    The denier has to lead, and what follows it has to be the thing asked --
    otherwise "no longer hungry" would deny every question about an apple
    rather than the one about hunger.
    """
    if not target:
        return False
    words = fact_object.strip().split()
    if len(words) < 2 or words[0].lower().strip(",.;:") not in DENIERS:
        return False
    return bool(matcher(" ".join(words[1:]), target))


def blocks(stated: str, candidate: str) -> bool:
    """R3: does a directly stated relation block an inherited one?"""
    return NEGATIONS.get(stated) == candidate or POSITIVES.get(stated) == candidate


@dataclass
class Step:
    """One move the reasoner made, replayable in the UI."""

    index: int
    #: `block` and `stop` end the walk; `skip` sets facts aside and carries on.
    kind: str                      # resolve | ascend | check | match |
                                   # block | stop | skip
    concept: str
    distance: int
    rule: str
    detail: str
    facts_checked: int = 0
    matched: dict[str, Any] | None = None
    parents: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "kind": self.kind, "concept": self.concept,
            "distance": self.distance, "rule": self.rule, "detail": self.detail,
            "facts_checked": self.facts_checked, "matched": self.matched,
            "parents": self.parents,
        }


#: What the page shows for each rule, from its row (`rulebook.py`).
RULE_TEXT: dict[str, str] = texts("R1", "R2", "R3", "R4", "R5", "R6", "R7",
                                  "R8", "R9", "R10", "R11", "R12", "R13")
