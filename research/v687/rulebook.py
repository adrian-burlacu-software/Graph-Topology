"""Rules as data: every named rule, in one table.

A rule used to live where it was applied: R5's decay beside the walk, R12's
limit in a constant, R19's floor in the profile, the page's text in three
dictionaries across two versions, v688's price for each rule in a fourth. Each
is a row here (`v690/DESIGN.md` §4.5, §6):

    becomes     what it is in the architecture: a link type's field, a walk
                along a link type, a rewrite, a truth function, an index, an
                attention term, an operator, the executive, reading, memory
    links       the link types it reads or derives along (`links.py`)
    field       the link-type field whose rows decide it, where one does
    truth       the function in `truth.py` it applies, where one does
    parameters  its numbers -- what fitting would move
    source      where its content comes from: WordNet, VerbNet, a norm set,
                or the rule as written
    where       the module that applies it
    flagged     what it is kept named for, where it is a convention rather
                than English

The code that applies a rule reads its parameters and its text from its row,
so a number and its explanation cannot disagree, and a test pins that every
rule the code cites has one. What a rule does is still the code named in
`where`; a rule whose content already is data -- T4 from VerbNet's frames,
R2 from the link table -- says so in `source` and `field`.

Nothing here reads the store.
"""
from __future__ import annotations

import dataclasses
import os


@dataclasses.dataclass
class Rule:
    name: str
    title: str
    becomes: str
    statement: str = ""
    links: tuple = ()
    field: str = ""
    truth: str = ""
    parameters: dict = dataclasses.field(default_factory=dict)
    source: str = "written"
    where: str = ""
    flagged: str = ""

    def text(self) -> str:
        """What the page shows: the statement, with its parameters."""
        return self.statement.format(**self.parameters)


def _rows(*rules: Rule) -> dict[str, Rule]:
    return {one.name: one for one in rules}


#: The dimensions S1-S4 order along (`links.py`).
DIMENSIONS = ("north", "east", "above", "right", "bigger")

RULES: dict[str, Rule] = _rows(
    # -- v684's thirteen: inheritance down the taxonomy ---------------------
    Rule("R1", "Subsumption closure", "walk",
         "Subsumption closure: is_a is transitive over an acyclic taxonomy.",
         links=("is_a",), field="transitive", source="WordNet",
         parameters={"ground": 0.95, "exact": True}, where="reason.py"),
    Rule("R2", "Property lift", "rewrite",
         "Property lift: a subtype inherits a supertype's facts, for "
         "inheritable relations only.",
         links=("is_a",), field="inherited", truth="deduced",
         where="reason.py"),
    Rule("R3", "Exception blocking", "truth",
         "Exception blocking: a closer statement, or an explicit negation, "
         "overrides an inherited fact.",
         field="denies", where="reason.py"),
    Rule("R4", "Specificity preference", "truth",
         "Specificity preference: the nearest ancestor that answers wins.",
         links=("is_a",), where="reason.py"),
    Rule("R5", "Confidence decay", "truth",
         "Confidence decay: each level of borrowing multiplies confidence "
         "by {decay}.",
         truth="deduced", parameters={"decay": 0.85, "floor": 0.05},
         where="rules.py"),
    Rule("R6", "Sense scoping", "memory",
         "Sense scoping: inference runs per WordNet sense, never per word.",
         source="WordNet", where="reason.py"),
    Rule("R7", "Relation gating", "link type",
         "Relation gating: contentless relations such as related_to never "
         "participate.",
         field="gated", where="rules.py"),
    Rule("R8", "Answer synthesis", "truth",
         "Answer synthesis: VERIFIED, CONTRADICTED or UNKNOWN.",
         truth="judge", where="reason.py"),
    Rule("R9", "Relation families", "link type",
         "Relation families: has_a and has_part answer for each other, "
         "because the sources disagree about which one a fact belongs under.",
         field="family", where="rules.py"),
    Rule("R10", "Redundancy elimination", "index",
         "Redundancy elimination: a fact an ancestor already states is not "
         "stored twice -- R2 rebuilds it. Lossless.",
         links=("is_a",), where="compress.py"),
    Rule("R11", "Hoisting", "rewrite",
         "Hoisting: a fact every child states moves to the parent. A "
         "generalisation, not a deduction, so hoisted rows are marked.",
         links=("is_a",), truth="counted", where="compress.py"),
    Rule("R12", "Breadth gating", "attention",
         "Breadth gating: a word-level fact does not inherit from a concept "
         "with {breadth_limit:,}+ descendants, where the word and the class "
         "have stopped meaning the same thing.",
         links=("is_a",), parameters={"breadth_limit": 8000},
         where="reason.py"),
    Rule("R13", "Range typing", "link type",
         "Range typing: a relation's object must be the kind of thing the "
         "relation takes. A location has to be a place.",
         field="range", source="WordNet", where="rules.py"),

    # -- v685 ------------------------------------------------------------------
    Rule("R14", "Sibling exclusion", "rewrite",
         "Sibling exclusion: a fact about a co-hyponym of the anchor does "
         "not transfer to it. An ancestor's property descends (R2); a "
         "sibling's does not. Two concepts are siblings when their nearest "
         "shared class has fewer than {sibling_limit:,} descendants -- "
         "`violin` and `drum` meet at `musical instrument` (163), while "
         "`violin` and `bow` only meet at `device` (2,764).",
         links=("is_a",), parameters={"sibling_limit": 1000},
         where="relevance.py"),
    Rule("R15", "Bridging", "operator",
         "Bridging: a question with two subjects is answered about the "
         "second, but only after a route to it is found in the fact graph. "
         "Each hop is a stored fact, and the route is shown so it can be "
         "judged.",
         where="relevance.py"),

    # -- v686: the trie --------------------------------------------------------
    Rule("R16", "Identification", "index",
         "Identification: a description is answered by walking the trie "
         "down instead of storing into it. Properties are taken general "
         "first, so each step narrows visibly -- round things, then the "
         "round thing with hexagons. Rarest-first would identify in fewer "
         "questions but answer the whole thing at step one. What a norm "
         "states about a thing outranks what it inherits.",
         source="McRae, CSLB and XCSLB norms", parameters={"ground": 0.85},
         where="identify.py"),
    Rule("R17", "Retrieval", "index",
         "Retrieval is the same walk backwards. An individual sits at a "
         "leaf, so walking from that leaf to the origin recovers exactly "
         "the predicates it was stored with: the ones met first are shared "
         "with nothing, the ones met last with half the corpus. Nodes where "
         "nothing branched are collapsed. A property the norms scored false "
         "is a denial and not a silence; one they never mention is looked "
         "for above the leaf, in what the concept inherits.",
         source="McRae, CSLB and XCSLB norms", parameters={"ground": 0.85},
         where="profile.py"),

    # -- v687 ------------------------------------------------------------------
    Rule("R18", "Question-shape gating", "executive",
         "Question-shape gating: a construction no rule covers is refused "
         "by name, not answered from the part of it that happens to be "
         "understandable. Comparatives, superlatives, counts of parts, "
         "counterfactuals, dates, questions about words rather than senses, "
         "facts about named individuals and antonyms are named and "
         "declined -- every silent wrong answer found in the v686 and v687 "
         "audits came from answering an easier question than the one asked. "
         "A pinned sense whose part of speech cannot complete the sentence "
         "is refused the same way: `can a dog bark` answered VERIFIED with "
         "`bark` pinned to the covering of a tree, and to a three-masted "
         "sailing ship, because the norms match the word and never resolve "
         "it. Where a pin merely fails to bite rather than contradicting "
         "the sentence, the answer says which pins it did not use.",
         where="reasoning.py"),
    Rule("R19", "Corroboration", "truth",
         "Corroboration: an inherited fact is put to the ancestor's other "
         "kinds before it is believed. `bird capable_of fly` is borne out "
         "by 21 of 29 birds in the norms and is inherited; `animal has a "
         "wing` by 20 of 143 and is refused. One crawled sentence is not a "
         "property of a category. This governs both answering paths. It was "
         "written for the norms and wired only into them, so the fact store "
         "believed `mammal capable_of fly` on one sentence and answered `do "
         "pigs fly` yes; a crawled fact about a class says some of its "
         "members do this, and the norms are what tell an existential from "
         "a universal.",
         links=("is_a",), truth="counted",
         parameters={"floor": float(os.environ.get(
             "V687_CORROBORATION_FLOOR") or 0.6), "min_kinds": 8},
         source="McRae, CSLB and XCSLB norms, distilled", where="profile.py"),
    Rule("R20", "Three-valued composition", "truth",
         "Three-valued composition: a question with structure is evaluated "
         "in Kleene's logic, because silence is not falsehood. One false "
         "conjunct settles a conjunction, one true disjunct settles a "
         "disjunction, and an unknown part suspends the whole. Quantifiers "
         "ask every kind beneath a concept and count.",
         truth="judge", parameters={"ground": 0.80}, where="logic.py"),
    Rule("R21", "Contrast", "index",
         "Contrast: what two concepts share, where they part, how alike "
         "they are and how typical one is are one operation -- the lowest "
         "common ancestor of two branches. Semantic overlap and trie "
         "prefix are both reported, because they disagree: a trie built "
         "for storage does not group by similarity.",
         links=("is_a",), parameters={"ground": 0.85}, where="contrast.py"),
    Rule("R22", "Inverse traversal", "link type",
         "Inverse traversal: the graph is read from the object as well as "
         "the subject, so `what is made of wood` is answerable and not only "
         "`what is a hammer made of`. Relations that pair (`has_part` and "
         "`part_of`) are read from both columns.",
         field="converse", where="inverse.py"),
    Rule("R23", "Scripts and abduction", "rewrite",
         "Scripts and abduction: prerequisites, subevents and effects are "
         "walked in script order -- before, during, after. An observation "
         "is explained by ranking causes as competing hypotheses, scored by "
         "specificity, directness and confidence. A cause that causes forty "
         "things explains none of them.",
         source="ConceptNet", where="causal.py"),
    Rule("R24", "Analogy", "operator",
         "Analogy over the norms only: a role is approximated by feature "
         "type plus standing within the concept. `bark : dog :: ? : cat` "
         "gives meow and purr. The scraped graph cannot support this and is "
         "not asked -- its `part_of` is largely taxonomy misfiled.",
         source="McRae, CSLB and XCSLB norms", where="shapes.py"),
    Rule("R25", "Counting kinds", "operator",
         "Counting kinds: the ontology holds no numbers, so it cannot count "
         "a dog's legs -- but it can count what stands beneath a concept in "
         "the taxonomy, which is what `how many kinds of dog` asks. Two "
         "counts are reported: every descendant WordNet records, and the "
         "far smaller number the feature norms actually describe, because "
         "only the second can be reasoned about.",
         links=("is_a",), source="WordNet", where="shapes.py"),
    Rule("R26", "Definition", "operator",
         "Definition: `what is a robin` is answered from the taxonomy "
         "itself -- which sense is meant, the gloss WordNet gives it, what "
         "it is a kind of, and what kinds it has. Before this it fell "
         "through to a property listing and returned `helpful, passionate, "
         "professional` for a robin, because ConceptNet holds those of the "
         "name Robin. A definition is the one question a taxonomy answers "
         "by being a taxonomy.",
         links=("is_a",), source="WordNet", parameters={"ground": 0.90},
         where="reasoning.py"),
    Rule("R27", "Taxonomic exclusion", "link type",
         "Taxonomic exclusion: absence is not denial, except between the "
         "top branches of the taxonomy, where it is. Nothing is both a "
         "plant and an animal, or both an artifact and an abstraction, so "
         "`is a dog a plant` is a no with a reason and not a silence. Held "
         "to those branches on purpose: WordNet's middle does not record "
         "that a dog is a pet, so `is a dog a pet` stays unknown. Every "
         "sense of the target has to be excluded, because `plant` also "
         "means a factory.",
         links=("is_a",), field="disjoint", source="WordNet",
         parameters={"ground": 0.95, "exact": True}, where="reason.py"),
    Rule("R28", "Qualified claims", "reading",
         "Qualified claims: a fact that carries the question inside a wider "
         "claim does not answer it. `rock capable_of “go for swim”` "
         "is about a place people swim and `fish capable_of “walk on "
         "land”` is about the fish that do, and both answered yes. "
         "v687 already declined the mirror of this -- denying a qualified "
         "property does not deny the property -- and this is the same "
         "reading applied to yes. The match itself is whole-word now: "
         "`fly` was named by “attract butterfly”, `walk` by "
         "“block the sidewalk” and `run` by “get drunk”, "
         "so `can a tree fly` was VERIFIED on a butterfly.",
         where="reason.py"),
    Rule("R29", "Sense-to-sense lookup", "memory",
         "Sense-to-sense lookup: where both ends of a question are synsets, "
         "it is answered between them and no string is matched. Only "
         "WordNet writes an object as a synset id -- has_part, part_of, "
         "similar_to, entails and causes, 36,283 concepts -- and that is "
         "the only place a pin on the object has anything to bind to. "
         "`does a car have an accelerator` is UNKNOWN through the words, "
         "because the accelerator is recorded only as a synset, and "
         "VERIFIED through the graph. It is tried first and it is not "
         "authoritative: the synset rows are patchy (they have a car's "
         "wheel and a dog's tail, and not a fish's gills), so when the "
         "graph is silent the words still answer and the note says which "
         "of the two spoke.",
         source="WordNet", parameters={"ground": 0.95, "exact": True},
         where="reasoning.py"),
    Rule("R30", "Recorded of a class", "truth",
         "Recorded of a class: a sentence stated of a class node is about "
         "the class, not about each kind under it, and does not descend.",
         links=("is_a",), where="reason.py"),
    Rule("R31", "Magnitudes", "link type",
         "Magnitudes: bigger, smaller, heavier and lighter are compared on "
         "the scales THINGSplus had people rate for 1,854 objects -- "
         "real-world size, anchored from a grain of sand to an aircraft "
         "carrier, and heaviness from 1 to 7. Two objects rated too close "
         "together are called too close, not ordered by the noise. Faster, "
         "older and better have no scale, and R18 still refuses them.",
         field="axis", source="THINGSplus", parameters={"ground": 0.85},
         where="magnitudes.py"),
    Rule("R32", "Only the living", "link type",
         "Only the living: a thing rated not alive is not the doer of a "
         "verb VerbNet gives an animate subject in every class it is in -- "
         "breathe, eat, drink, think. Neither source says a rock does not "
         "breathe; together they do. It stands down when the store states "
         "that the thing does exactly that, which is the disagreement test "
         "AwA2's zeros face, and when any reading of the word is alive.",
         field="range", source="VerbNet 3.3 and THINGSplus",
         parameters={"ground": 0.80}, where="rated.py"),

    # -- v689: individuals, events and relations ------------------------------
    Rule("E1", "A quality does not descend", "link type",
         "A quality does not descend from a kind to an individual: it is "
         "answered from what was said of that individual, with the kind's "
         "tendency beside it. A quality toward something descends as a "
         "doing does.",
         field="quality", where="episodic.py"),
    Rule("E2", "What carried it did it", "rewrite",
         "An action done while carried belongs to what carries it, within "
         "one episode, and is given back when nothing carrying it does it.",
         links=("at_location",), field="in_time", where="episodic.py"),
    Rule("T1", "Order from tense", "reading",
         "Told in order, happened in order: a simple past moves the story "
         "on, a progressive is in progress at the time it is at, a past "
         "perfect is before it; links and anchors say otherwise.",
         where="timeline.py"),
    Rule("T2", "Before", "walk",
         "Before is a partial order, walked like the taxonomy: yes, no, or "
         "not told.",
         links=("before",), field="transitive", where="timeline.py"),
    Rule("T3", "A state within its episode", "link type",
         "A state holds within its episode until something ends it; nothing "
         "told of one episode answers another; one place at a time.",
         links=("at_location",), field="exclusive", where="timeline.py"),
    Rule("T4", "What an occurrence changes", "rewrite",
         "What an occurrence changes, from VerbNet's event structure: what "
         "holds at a frame's end or result holds after, what holds at its "
         "start held before.",
         source="VerbNet 3.3", where="change.py"),
    Rule("T5", "An occurrence is an individual", "index",
         "An occurrence is an individual of its verb: who, when, what "
         "happened and how many times are identification on the occurrence "
         "trie with one slot read out.",
         where="timeline.py"),
    Rule("T6", "Nothing happens by inheritance", "link type",
         "Nothing happens by inheritance: what a kind does is a tendency; "
         "that one of them did it is an occurrence.",
         field="episodic", where="session.py"),
    Rule("S1", "Converse", "link type",
         "A relation and its converse are one fact.",
         links=DIMENSIONS, field="converse", where="relations.py"),
    Rule("S2", "Order along a dimension", "walk",
         "Along a dimension, a partial order walked like before.",
         links=DIMENSIONS, field="axis", where="relations.py"),
    Rule("S3", "In line along a direction", "rewrite",
         "A direction puts the two things in line: north of means neither "
         "east nor west of.",
         links=DIMENSIONS, field="across", where="relations.py",
         flagged="bAbI's convention, not English's"),
    Rule("S4", "The compass as a map", "operator",
         "The compass is a map: a way from one place to another is a path "
         "over direction links, a direction a step.",
         links=DIMENSIONS, field="compass", where="relations.py"),
    Rule("I1", "Induction", "truth",
         "Nothing told of one individual's value: the others of its kind "
         "told of here, all of them where they agree and the last told "
         "where they do not, said as probably.",
         truth="counted", where="session.py"),
    Rule("motives", "Motives", "attention",
         "A state moves one to what the store says it motivates, needs or "
         "leads to; a place is for what its kind is used for; the two meet "
         "on a lemma.",
         source="ConceptNet", where="motives.py",
         flagged="meets on a shared lemma; spreading activation (DESIGN "
                 "§4.4) changes answers, so it waits for fitting"),
)


def rule(name: str) -> Rule:
    return RULES[name]


def texts(*names: str) -> dict[str, str]:
    """The page's text for these rules, in this order."""
    return {name: RULES[name].text() for name in names}


def grounds() -> dict[str, float]:
    """What v688 prices a derivation by each rule at, before the loop moves
    it."""
    return {name: one.parameters["ground"] for name, one in RULES.items()
            if "ground" in one.parameters}


def exact() -> frozenset[str]:
    """Rules whose walk is the proof, so distance costs them nothing."""
    return frozenset(name for name, one in RULES.items()
                     if one.parameters.get("exact"))
