"""The inference policy: what descends a taxonomy, and what stops it.

Eight rules. Each is stated once, here, so that an answer can be traced back to
the rule that produced it rather than to a heuristic buried in a query.

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

The rule that matters most is R2. `research/v683/diagnose.py` measured what
happens without it: trusting every relation over every edge yields a mean of
1,548 derived facts per concept, most of them wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# -- R2: which relations descend ------------------------------------------
#: A subtype has whatever the supertype has. If mammals can breathe, dogs can.
INHERITABLE: frozenset[str] = frozenset({
    "capable_of", "has_property", "has_a", "has_part", "receives_action",
    "used_for", "desires", "not_desires", "not_capable_of",
    "not_has_property", "has_prerequisite", "has_subevent",
    "motivated_by_goal", "causes", "at_location", "part_of", "has_attribute",
    "entails", "located_near",
})

#: Relations that do NOT descend, with the reason each is excluded.
NOT_INHERITABLE: dict[str, str] = {
    "made_of": "a subtype may be made of something else entirely -- a chair is "
               "furniture, but furniture is not made of wood",
    "similar_to": "similarity is not transitive through subtyping",
    "instance_of": "an instance's membership says nothing about a subclass",
    "created_by": "the maker of a kind is not the maker of every subkind",
    "symbol_of": "symbolism attaches to the specific thing, not the category",
    "defined_as": "a definition is about that concept alone",
    "manner_of": "manner relates two actions, it does not descend a hierarchy",
}

# -- R7: relations that never participate ---------------------------------
#: `related_to` is 1,678,150 of v633's 3.9M edges and carries no semantics --
#: no direction, no relation type, just co-occurrence. Inheriting it floods
#: every answer. It is excluded from storage and from inference.
GATED: frozenset[str] = frozenset({"related_to", "has_context", "form_of",
                                   "derived_from", "etymologically_related_to",
                                   "synonym", "antonym", "has_sense",
                                   "definition", "usage_count", "distinct_from",
                                   "verb_group"})

#: R3: an assertion of the key blocks inheritance of the value, and vice versa.
NEGATIONS: dict[str, str] = {
    "not_capable_of": "capable_of",
    "not_has_property": "has_property",
    "not_desires": "desires",
}
POSITIVES: dict[str, str] = {value: key for key, value in NEGATIONS.items()}

# -- R9: relations that answer for each other ------------------------------
#: Sources disagree about which relation a fact belongs under. WordNet files
#: "a dog has a tail" as `has_part`; Ascent++ files it as `has_a`. A question
#: asking about one must see the other, or the answer depends on which source
#: happened to record it. Only genuinely interchangeable relations appear here
#: -- `part_of` is NOT a family member of `has_part`, it is its inverse.
FAMILIES: tuple[frozenset[str], ...] = (
    frozenset({"has_a", "has_part"}),
    frozenset({"at_location", "located_near"}),
    frozenset({"has_property", "has_attribute"}),
)


def family(relation: str) -> list[str]:
    """Every relation that can answer a question asked about `relation`."""
    for group in FAMILIES:
        if relation in group:
            return sorted(group)
    return [relation]


#: R5: what one level of borrowing costs. A fact five levels up retains
#: 0.85**5 = 0.44 of its confidence, so `thing capable_of fall down` ranks
#: below anything stated about dogs directly.
DECAY = 0.85

#: R5: below this, a derived fact is not worth reporting.
FLOOR = 0.05


def inheritable(relation: str) -> bool:
    """R2 + R7 in one predicate."""
    return relation not in GATED and relation in INHERITABLE


def why_not_inheritable(relation: str) -> str:
    """The stated reason a relation does not descend, for the UI."""
    if relation in GATED:
        return "gated: carries no usable semantics"
    return NOT_INHERITABLE.get(relation, "not marked inheritable")


def confidence_at(base: float, distance: int) -> float:
    """R5."""
    return base * (DECAY ** distance)


def blocks(stated: str, candidate: str) -> bool:
    """R3: does a directly stated relation block an inherited one?"""
    return NEGATIONS.get(stated) == candidate or POSITIVES.get(stated) == candidate


@dataclass
class Step:
    """One move the reasoner made, replayable in the UI."""

    index: int
    kind: str                      # resolve | ascend | check | match | block | stop
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


RULE_TEXT: dict[str, str] = {
    "R1": "Subsumption closure: is_a is transitive over an acyclic taxonomy.",
    "R2": "Property lift: a subtype inherits a supertype's facts, for "
          "inheritable relations only.",
    "R3": "Exception blocking: a closer statement, or an explicit negation, "
          "overrides an inherited fact.",
    "R4": "Specificity preference: the nearest ancestor that answers wins.",
    "R5": f"Confidence decay: each level of borrowing multiplies confidence "
          f"by {DECAY}.",
    "R6": "Sense scoping: inference runs per WordNet sense, never per word.",
    "R7": "Relation gating: contentless relations such as related_to never "
          "participate.",
    "R8": "Answer synthesis: VERIFIED, CONTRADICTED or UNKNOWN.",
    "R9": "Relation families: has_a and has_part answer for each other, "
          "because the sources disagree about which one a fact belongs under.",
}
