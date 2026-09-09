"""The trie read *upwards*: what does this thing have, and does it have that?

`identify.py` walks the trie down -- a description in, a thing out. This is the
other direction, and it is the one that checks the answer:

    what attributes does a blue whale have   -> its whole path, origin to leaf
    is a blue whale furry                    -> no, and the norms say so

An individual sits at a leaf. Walking from that leaf up to the origin collects
exactly the predicates it was stored with, in coverage order, so the walk is
both the retrieval and the explanation: the predicates met first (nearest the
leaf) are the ones almost nothing else carries, and the ones met last (nearest
the origin) are shared with half the corpus. That is Appendix 3's compression
seen from inside one branch -- the prefix is shared, the tail is what makes it
this thing.

Two things are needed that identification did not need:

    branch points   Most nodes on a path are pass-through: nothing else leaves
                    the branch there. The nodes worth drawing are the ones
                    where the field actually split, so consecutive
                    non-branching nodes are collapsed the way a radix tree
                    collapses them. `blue whale` carries 28 predicates and has
                    far fewer branch points than that.

    denial          "Is a blue whale furry" is not answered by silence. AwA2
                    scored every class on every one of its 85 attributes, so a
                    zero is a denial; COMPS ships an unacceptable concept
                    beside every acceptable one. Both are read here and only
                    here -- the trie stores what a thing has, so what it lacks
                    has to come from the corpus rather than from the structure.

Above the leaf the taxonomy continues where the norms stop. `blue whale` is a
whale is a cetacean, and v684 knows what those do, so the profile ends with
what the concept inherits, nearest ancestor first, each fact attributed to the
ancestor that supplies it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .ordering import adaptive_coverage
from .substrate import Corpus
from .trie import PredicateTrie
from . import corpora, logic, pins, rules

#: Facts shown per ancestor. One ancestor can carry hundreds; six is enough to
#: see what a level contributes without burying the level below it.
FACTS_PER_ANCESTOR = 6

#: How far up the taxonomy to read, matching `identify.INHERIT_DEPTH`.
ANCESTOR_DEPTH = 6

#: Trie neighbours shown per branch point: the things that shared the prefix
#: this far and then went elsewhere.
NEIGHBOURS_SHOWN = 5

#: Denials listed on a profile. A concept can be denied hundreds of
#: properties, and "here are 300 things a stocking is not" is not a profile.
DENIALS_SHOWN = 12

#: A property can be phrased as a denial: XCSLB states `cannot fly` of a
#: penguin and `has no legs` of a snake, and v684 carries `not_capable_of`
#: beside `capable_of`. Matching one of those on "does a penguin fly" and
#: reporting it as a yes is the one way this can be confidently wrong, so the
#: phrasing is read as well as the match.
NEGATORS = frozenset("""
not cannot cant never no none nor without lacks lack lacking non
""".split())

#: Which inherited relation makes the better answer when several match.
#: "Does a robin fly" is answered by `capable_of fly`, not by `desires fly`,
#: which is a ConceptNet crawl of somebody's sentence.
RELATION_RANK = {"capable_of": 0, "has_a": 1, "has_part": 1, "has_property": 2,
                 "has_attribute": 2, "used_for": 3, "receives_action": 4,
                 "part_of": 5, "entails": 6, "has_subevent": 7,
                 "has_prerequisite": 7, "causes": 8, "motivated_by_goal": 8,
                 "desires": 9, "at_location": 10, "located_near": 10}

#: R19. An inherited fact is corpus free text, and one sentence about
#: `animal.n.01` made every dog winged. Before such a fact is believed it is
#: put to the ancestor's other kinds, which the norms *did* elicit: at least
#: this share of them must bear it out.
#:
#:     bird.n.01     "fly"      21 of 29 kinds   72%   believed
#:     animal.n.01   "wings"    20 of 143        14%   refused
#:     carnivore.n.01 "wings"    0 of 24          0%   refused
CORROBORATION_FLOOR = 1 / 3

#: ...and refusal needs a sample worth refusing on. `whale.n.02` has four
#: kinds in the norms; one of them singing is not evidence that whales do not
#: sing. Below this, an inherited fact is taken as it was before.
CORROBORATION_MIN_KINDS = 8

#: Words that carry no property in a question about a named thing.
ASIDE = frozenset("""
what which is are was were be been does do did has have had can could would
a an the of its it their there any some this that these those
tell me about you i us we know show give
attribute attributes property properties feature features characteristic
characteristics trait traits quality qualities like describe list description
thing things kind kinds sort sorts type types really actually also too and or
with into onto out off up down over under through across around at in on to
from for by
""".split())


def _clone(node: logic.Node) -> logic.Node:
    """A fresh copy of the tree, because `evaluate` writes its result onto it
    and a quantified question evaluates the same tree once per kind."""
    return logic.Node(op=node.op, term=node.term,
                      children=[_clone(child) for child in node.children])


@dataclass
class Segment:
    """One branch point on the path: what was taken, and who left there."""
    predicates: list[str]
    depth: int
    shared: int
    dropped: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"predicates": self.predicates, "depth": self.depth,
                "shared": self.shared, "dropped": self.dropped}


@dataclass
class Held:
    """One attribute the concept has, and where on the path it sits."""
    predicate: str
    depth: int
    shared: int

    def as_dict(self) -> dict[str, Any]:
        return {"predicate": self.predicate, "depth": self.depth,
                "shared": self.shared}


@dataclass
class Ancestry:
    """One level above the leaf, and what it lends."""
    concept: str
    distance: int
    gloss: str | None
    facts: list[dict]
    total: int
    #: Every inheritable fact the level supplies, not just the few shown.
    #: `verify` reads this: an answer that happens to sit seventh is still
    #: the answer, and truncating for the display must not hide it.
    all_facts: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"concept": self.concept, "distance": self.distance,
                "gloss": self.gloss, "facts": self.facts, "total": self.total}


@dataclass
class Verdict:
    """The answer to "does this thing have that"."""
    term: str
    verdict: str                # HELD | DENIED | INHERITED | MIXED | UNRECORDED
    detail: str
    predicate: str | None = None
    source: str | None = None
    distance: int = 0
    depth: int = 0
    shared: int = 0
    #: When the answer came from the kinds below rather than the concept
    #: itself: what each of them said, and with which property.
    members: list[dict] = field(default_factory=list)
    #: When the question had structure -- `a tail and wings`, `furry or
    #: purple`, `all birds` -- the evaluated expression and the quantifier,
    #: so the page can show which half of an `and` is the one that failed.
    tree: dict | None = None
    quantifier: str | None = None
    #: Each term's own verdict, keyed by term.
    parts: dict[str, dict] = field(default_factory=dict)
    #: Whether the feature norms were consulted and had enough kinds to bear
    #: on this. False means the answer came from one crawled sentence at an
    #: ancestor with too few described kinds to check it against -- an answer
    #: the norms did not contribute to, and one the fact graph gives better.
    corroborated: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"term": self.term, "verdict": self.verdict,
                "detail": self.detail, "predicate": self.predicate,
                "source": self.source, "distance": self.distance,
                "depth": self.depth, "shared": self.shared,
                "members": self.members, "tree": self.tree,
                "quantifier": self.quantifier, "parts": self.parts,
                "corroborated": self.corroborated}


@dataclass
class Description:
    """Everything the architecture holds about one named individual."""
    name: str
    concept: str | None = None
    gloss: str | None = None
    source: str = "?"
    path: list[Held] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    inherited: list[Ancestry] = field(default_factory=list)
    denied: list[str] = field(default_factory=list)
    denied_total: int = 0
    asked: Verdict | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "concept": self.concept,
                "gloss": self.gloss, "source": self.source,
                "path": [held.as_dict() for held in self.path],
                "segments": [s.as_dict() for s in self.segments],
                "inherited": [a.as_dict() for a in self.inherited],
                "denied": self.denied, "denied_total": self.denied_total,
                "asked": self.asked.as_dict() if self.asked else None}


class Profiles:
    """Attribute retrieval over the same trie identification searches.

    Built from the identifier so both read one copy of the norms and one open
    reasoner: the trie here is the storage plan `run_v686.py` measures, and
    the ancestry is the taxonomy `identify.py` inherits through.
    """

    def __init__(self, identifier) -> None:
        self.identifier = identifier
        self.reasoner = identifier.reasoner
        self.stated = identifier.stated
        self.synset = identifier.synset
        self.origin = identifier.origin

        self._ancestors: dict[str, set[str]] | None = None
        # AwA2 only. `denied_xcslb` used to be merged in here and it is not a
        # set of denials: XCSLB's matrix is 521 x 3,644 and 1.58% dense, a free
        # listing whose zeros are what nobody happened to say, and COMPS builds
        # its foils by sampling concepts out of them. Nothing judged them
        # false. `violin` is the foil for `can be made of ivory`, which is how
        # `is a violin made of wood` came back CONTRADICTED.
        #
        # AwA2 is different in the way that matters: its matrix *is* closed --
        # every class was scored on every one of the 85 attributes -- so its
        # zeros are real judgements and the only question is what they mean.
        # That is a semantics problem on data that exists, which is what
        # `_zero_that_is_not_a_no` is for. XCSLB's zeros were never judgements
        # at all, so there is nothing to guard and nothing to keep.
        self.denied: dict[str, frozenset[str]] = {}
        for name, properties in corpora.denied_awa2().items():
            self.denied[name] = self.denied.get(name, frozenset()) | properties

        corpus = Corpus("norms", tuple(sorted(
            (name, predicates) for name, predicates in self.stated.items()
            if predicates)))
        self.plan = adaptive_coverage(corpus)
        self.trie = PredicateTrie()
        self.node_of: dict[str, int] = {}
        # Who sits under each point of the trie, keyed by the prefix that
        # reaches it. The counts are what make the walk say something: a
        # predicate shared with forty concepts is a corridor, one shared with
        # none is the thing itself.
        self.members: dict[tuple, list[str]] = {}
        for individual, path in self.plan:
            self.node_of[individual] = self.trie.insert(individual, path).node
            for cut in range(len(path) + 1):
                self.members.setdefault(tuple(path[:cut]), []).append(individual)

    # -- the walk ----------------------------------------------------------
    def knows(self, name: str) -> bool:
        return name in self.node_of

    def walk_up(self, name: str) -> list[Held]:
        """Leaf to origin: the predicates this individual was stored with.

        Reading upwards is the retrieval. The first predicate met is the one
        that singled the thing out, and each one after it is shared with more
        concepts than the last, until the origin, which everything shares.
        """
        route = self.trie.path(self.node_of[name])
        climb = [Held(predicate=str(route[cut - 1]), depth=cut,
                      shared=len(self.members.get(tuple(route[:cut]), ())))
                 for cut in range(len(route), 0, -1)]
        return climb

    def describe(self, name: str) -> Description | None:
        """The whole profile: the trie path, then what sits above the leaf."""
        if not self.knows(name):
            return None
        concept = self.synset.get(name)
        found = Description(
            name=name, concept=concept,
            gloss=self.reasoner.gloss(concept) if concept else None,
            source=self.origin.get(name, "?"))
        found.path = list(reversed(self.walk_up(name)))     # origin first

        route = self.trie.path(self.node_of[name])
        segment: list[str] = []
        for depth in range(1, len(route) + 1):
            here = set(self.members.get(tuple(route[:depth]), ()))
            above = set(self.members.get(tuple(route[:depth - 1]), ()))
            segment.append(str(route[depth - 1]))
            left = sorted(above - here)
            if left or depth == len(route):
                found.segments.append(Segment(
                    predicates=segment, depth=len(found.segments),
                    shared=len(here), dropped=self._nearest(name, left)))
                segment = []

        found.inherited = self.ancestry(name)
        denied = sorted(self.denied.get(name, ()))
        found.denied_total = len(denied)
        found.denied = denied[:DENIALS_SHOWN]
        return found

    def _nearest(self, name: str, left: list[str]) -> list[str]:
        """The concepts that shared this prefix and went elsewhere, closest first.

        Sorting by name gives the alphabet. What is worth seeing beside a
        branch point is the thing most like this one that still left, which is
        overlap of stored predicates -- the same ranking the ruled-out rivals
        of an identification get.
        """
        target = self.stated.get(name, frozenset())
        return sorted(left, key=lambda other: (
            -len(self.stated.get(other, frozenset()) & target), other)
        )[:NEIGHBOURS_SHOWN]

    # -- below the leaf ----------------------------------------------------
    def subtypes(self, name: str) -> list[str]:
        """The concepts in the norms that are a kind of this one.

        `whale` is a concept in its own right here and carries 27 properties,
        none of them about fur -- so "is a whale furry" was silence, while
        four kinds of whale sat under it with the attribute scored and denied.
        A class question is answerable from what is stored beneath it.
        """
        concept = (pins.of(name) or self.synset.get(name)
                   or self.class_concept(name))
        if not concept:
            return []
        return sorted(other for other, above in self._lineage().items()
                      if other != name and concept in above
                      and self.stated.get(other))

    def class_concept(self, word: str) -> str | None:
        """A class the norms do not cover but WordNet does.

        `bird` is not one of XCSLB's 521 concepts, yet 30 of them are birds.
        Without this, "do all birds fly" -- the question quantifiers exist for
        -- had nothing to quantify over, and fell through to a single
        ConceptNet sentence about bird.n.01 that answered it wrongly.
        """
        chosen = pins.of(word)
        if chosen:
            return chosen              # the reader overruled the guess
        row = self.reasoner.connection.execute(
            "SELECT concept FROM lemmas WHERE lemma = ? "
            "ORDER BY primary_sense DESC LIMIT 1", (word,)).fetchone()
        return row[0] if row else None

    def _lineage(self) -> dict[str, set[str]]:
        """Every individual's ancestors, built once and kept.

        `identify._within` walks the taxonomy per query, which is a second per
        question. Inverting it costs the same walk once and turns "what is a
        kind of this" into a lookup.
        """
        if self._ancestors is None:
            self._ancestors = {
                name: {node for node, distance, _
                       in self.reasoner.ascend(concept) if distance > 0}
                for name, concept in self.synset.items()}
        return self._ancestors

    def _from_below(self, name: str, terms: list[str]) -> Verdict | None:
        """Ask the kinds below when the concept itself does not say.

        This is induction, not inheritance, and it is deliberately placed
        above the taxonomy walk rather than below it: the kinds are scored
        norm data and the ancestors are corpus free text, so four whales
        scored `hairless` is better evidence than anything `mammal.n.01`
        happens to say about fur.

        Only *stated* votes count. A kind that merely inherits its answer is
        not a witness -- it is the same ancestor's sentence read again, and
        three dog breeds re-inheriting `animal.n.01 has a wing` counted as
        three independent votes for winged dogs. Votes have to be independent
        or the count means nothing.
        """
        votes = [(other, self.verify_one(other, terms[0], descend=False))
                 for other in self.subtypes(name)]
        held = [(other, answer) for other, answer in votes
                if answer.verdict == "HELD"]
        denied = [(other, answer) for other, answer in votes
                  if answer.verdict == "DENIED"]
        if not held and not denied:
            return None
        members = [{"name": other, "verdict": answer.verdict,
                    "predicate": answer.predicate}
                   for other, answer in held + denied]
        kinds = len(votes)
        term = (held + denied)[0][1].term
        if held and not denied:
            return Verdict(
                term=term, verdict="HELD", source="kinds", members=members,
                predicate=held[0][1].predicate,
                detail=f"Not recorded of {name} itself, but all "
                       f"{len(held)} of the {kinds} kinds of {name} the norms "
                       f"cover state it.")
        if denied and not held:
            return Verdict(
                term=term, verdict="DENIED", source="kinds", members=members,
                predicate=denied[0][1].predicate,
                detail=f"Not recorded of {name} itself, but all "
                       f"{len(denied)} of the {kinds} kinds of {name} the "
                       f"norms cover deny it.")
        return Verdict(
            term=term, verdict="MIXED", source="kinds", members=members,
            predicate=held[0][1].predicate,
            detail=f"{len(held)} of the {kinds} kinds of {name} the norms "
                   f"cover state it and {len(denied)} deny it, so the class "
                   f"does not settle it. Defeasible, which is R3's point.")

    #: Confidence at or above which a fact is worth setting against an AwA2
    #: zero: each source's own median, because the three are not comparable.
    #: Ascent++'s is 0.264 over 1,808,006 rows; ConceptNet and WordNet write
    #: one number on every row, so any row of theirs counts.
    TYPICAL = {"ascentpp": 0.264}

    #: Relations that assert something of the concept, for that test, split
    #: by whether a subtype inherits them.
    #:
    #: A capability and a part are the two that can be set against a zero.
    #: Dogs swim, so a collie swims, and `dog capable_of "swim"` is worth
    #: putting against AwA2's zero on the collie.
    #:
    #: `has_property` is left out, and that is the whole of the judgement
    #: here. AwA2's adjectives -- colours, sizes, temperaments -- are where
    #: its zeros are least reliable, and they are also where the crawl is:
    #: some cats are white and a bobcat is not, `bear has_property "small"`
    #: is about sun bears and not grizzlies. Setting one noisy source against
    #: another buys nothing, and letting it made `is a bobcat white` a yes.
    ASSERTING = ("capable_of", "has_a", "has_part")

    def _zero_that_is_not_a_no(self, name: str, terms: list[str]) -> bool:
        """Is this denial an AwA2 zero that another source contradicts?

        AwA2 annotates 85 attributes per class from a strength matrix, and a
        zero means the attribute is not *characteristic* of the class. Read as
        a denial it says a collie has no claws, no muscle, is never black and
        is not found in fields -- and that a collie does not swim, which is
        how `does a dog swim` came back CONTRADICTED with all three kinds of
        dog the norms cover lined up behind it.

        XCSLB is not reached here because it is no longer a denial source at
        all. This docstring used to claim its negatives "were elicited from
        people as negatives"; they were not. They are COMPS foils sampled out
        of a 1.58%-dense free listing, and `Profiles.__init__` says why they
        are gone rather than guarded.

        The test is disagreement, not overruling: an AwA2 zero stands unless
        another source positively asserts the same thing of this concept, at
        or above that source's own median. `dog capable_of swim` is Ascent++
        at 0.68, its 97th percentile, so the zero on `swims` is a
        disagreement between sources and not a no. Nothing in the store says
        a dog flies, so the zero on `flys` is left alone and `can a dog fly`
        stays CONTRADICTED.
        """
        if self.origin.get(name) != "awa2":
            return False
        concept = self.synset.get(name)
        if not concept:
            return False
        wanted = {self.identifier.stem(word) for word in terms}
        if not wanted:
            return False
        marks = ",".join("?" * len(self.ASSERTING))
        for node, _distance, _parents in self.reasoner.ascend(concept):
            # A class too wide to speak for its members cannot rescue a zero
            # either. `animal capable_of swim` would clear every AwA2 denial
            # of swimming there is; `dog capable_of swim` is about dogs.
            if self.reasoner.too_broad(node) or self._wide(node):
                continue
            for row in self.reasoner.connection.execute(
                    f"SELECT object, source, confidence FROM facts WHERE "
                    f"concept = ? AND relation IN ({marks}) "
                    f"ORDER BY confidence DESC LIMIT 400",
                    (node, *self.ASSERTING)):
                if row["confidence"] < self.TYPICAL.get(row["source"], 0.0):
                    continue
                said = {self.identifier.stem(word)
                        for word in re.findall(r"[a-z]+",
                                               row["object"].lower())
                        if word not in self.identifier.FRAME
                        and word not in self.identifier.LOCATORS}
                # R28's standard, and for R28's reason. `whale capable_of
                # "walk on land"` carries `walk` and is about the whales that
                # do; letting it clear AwA2's zero on `walks` said a killer
                # whale might walk. The fact has to state the thing and add
                # nothing to it, which `dog capable_of "swim"` does.
                if said == wanted:
                    return True
        return False

    #: Where a class stops speaking for its members. `animal` has 4,016
    #: descendants and R19 already refuses to inherit its rows.
    WIDE = 1000

    def _wide(self, concept: str) -> bool:
        row = self.reasoner.connection.execute(
            "SELECT descendants FROM concepts WHERE id = ?",
            (concept,)).fetchone()
        return bool(row and (row[0] or 0) >= self.WIDE)

    def corroboration(self, ancestor: str, term: str) -> tuple[int, int]:
        """R19: how many of an ancestor's norm-covered kinds bear a fact out.

        The norms elicited a fixed question about every concept they cover, so
        a property they record for 21 of 29 birds is a property of birds,
        while one recorded for 20 of 143 animals is a property of some animals
        and not of the category. That is the difference between `bird.n.01
        capable_of fly`, which every robin should inherit, and `animal.n.01
        has a wing`, which no dog should.
        """
        kinds = [name for name, above in self._lineage().items()
                 if ancestor in above and self.stated.get(name)]
        bearing = sum(1 for name in kinds
                      if self.identifier._hit(term, self.stated[name]))
        return bearing, len(kinds)

    # -- above the leaf ----------------------------------------------------
    def ancestry(self, name: str) -> list[Ancestry]:
        """What the concept inherits, nearest ancestor first.

        A fact met at two levels is kept at the nearer one only: R10's
        redundancy elimination applied to the display. Without it every mammal
        fact is repeated under `vertebrate` and again under `animal`, and the
        specific level -- the one that says something about whales -- is
        buried under the generic one.
        """
        concept = self.synset.get(name)
        if not concept:
            return []
        seen: set[tuple[str, str]] = set()
        levels: list[Ancestry] = []
        for node, distance, _ in self.reasoner.ascend(concept):
            if distance == 0 or distance > ANCESTOR_DEPTH:
                continue
            if self.reasoner.too_broad(node):
                break                                       # R12
            fresh = []
            for fact in sorted(self.reasoner.facts_of(node),
                               key=lambda f: -f.confidence):
                if not rules.inheritable(fact.relation):
                    continue
                key = (fact.relation, fact.object)
                if key in seen:
                    continue
                seen.add(key)
                fresh.append(fact)
            if not fresh:
                continue
            levels.append(Ancestry(
                concept=node, distance=distance,
                gloss=self.reasoner.gloss(node),
                facts=[fact.as_dict() for fact in fresh[:FACTS_PER_ANCESTOR]],
                total=len(fresh),
                all_facts=[fact.as_dict() for fact in fresh]))
        return levels

    # -- the polar question ------------------------------------------------
    def verify_one(self, name: str, term: str,
                   descend: bool = True,
                   asked: list[str] | None = None) -> Verdict:
        """Does this thing have that *one* property?

        Stated, denied, inherited or unrecorded.

        The order is evidence quality, not convenience. What the norms state
        about the thing itself is the best answer; what they state it lacks is
        the next best, and it is a real "no" rather than a silence. Only when
        the norms are silent both ways is the taxonomy asked, because
        inherited facts are corpus free text and matching one word inside one
        is loose enough to make a marble fly.
        """
        terms = [term]
        if not self.knows(name):
            return Verdict(term=term,
                           verdict="UNRECORDED",
                           detail=f"“{name}” is not one of the concepts the "
                                  f"norms cover.")
        stated = self.stated.get(name, frozenset())
        climb = self.walk_up(name)
        sharing = {held.predicate: held.shared for held in climb}
        depths = {held.predicate: held.depth for held in climb}
        for term in terms:
            # Every predicate the word names, not the first: one of them may
            # be a narrower denial that answers a different question, and
            # returning on it hid the plain statement behind it.
            found = self.identifier.hits(term, stated)
            denials = [hit for hit in found
                       if self._denies(hit)
                       and self.identifier.denies_term(asked or terms, hit)]
            plain = [hit for hit in found if not self._denies(hit)]
            hit = denials[0] if denials else (plain[0] if plain else None)
            if hit is not None and self._denies(hit):
                return Verdict(
                    term=term, verdict="DENIED", predicate=hit,
                    source="stated", depth=depths.get(hit, 0),
                    shared=sharing.get(hit, 0),
                    detail=f"The norms state “{hit}” of {name}. The property "
                           f"they record is the denial itself, so this is a "
                           f"no with evidence rather than a silence.")
            if hit is not None:
                # `shared` counts who is still on the branch at that node, not
                # who carries the predicate anywhere: flippers is common, but
                # by the time this branch reaches it only these are left.
                shared = sharing.get(hit, 0)
                company = ("by then the branch holds nothing but this one"
                           if shared <= 1 else
                           f"{shared - 1} other concept"
                           f"{'' if shared == 2 else 's'} "
                           f"{'is' if shared == 2 else 'are'} still on the "
                           f"branch there")
                return Verdict(
                    term=term, verdict="HELD", predicate=hit, source="stated",
                    depth=depths.get(hit, 0), shared=shared,
                    detail=f"The norms state “{hit}” of {name}. It sits at "
                           f"depth {depths.get(hit, 0)} of its trie path, and "
                           f"{company}.")
        # A denial answers this question only if the question covers what the
        # denial claims. `has small ears` being false of a beaver is not the
        # beaver having no ears, and reading it that way was the one place
        # this system gave a confident wrong answer.
        hit, narrower = self.identifier.denial_hit(
            asked or terms, self.denied.get(name, frozenset()))
        if hit is not None and not self._zero_that_is_not_a_no(name, asked
                                                              or terms):
            return Verdict(
                term=term, verdict="DENIED", predicate=hit,
                source=self.origin.get(name, "?"),
                detail=f"The norms record “{hit}” as false of "
                       f"{name}: scored and denied, not merely absent.")
        if descend:
            below = self._from_below(name, terms)
            if below is not None:
                return below
        # Nearest ancestor first, then the plainest fact it offers. The
        # ancestor is specificity; the length is quality control -- "fly"
        # matches `capable of fly` and `capable of fly in the water` equally,
        # and the second is corpus noise that reads as an answer.
        matches = []
        for level in self.ancestry(name):
            for fact in level.all_facts:
                text = f"{fact['relation'].replace('_', ' ')} {fact['object']}"
                negated = (fact["relation"].startswith("not_")
                           or self._denies(text))
                # A crawled sentence that negates something narrower than the
                # question is evidence for neither side. `capable of not eat
                # bone of contention` says nothing about whether dogs eat.
                if negated and not self.identifier.denies_term(
                        asked or terms,
                        fact["object"] if fact["relation"].startswith("not_")
                        else text):
                    continue
                for term in terms:
                    if self.identifier._hit(term, frozenset({text})):
                        matches.append((level, fact, text, term))
                        break
        if matches:
            asked_stems = {self.identifier.stem(word)
                           for word in (asked or terms)}

            def better(match) -> tuple:
                level, fact, text, _ = match
                relation = fact["relation"]
                # R3: at the level that answers, a denial blocks the positive.
                negative = relation.startswith("not_") or self._denies(text)
                # How much of the question this one fact accounts for. Length
                # alone was the tiebreak, on the grounds that `capable of fly`
                # beats `capable of fly in the water` for a question about
                # flying -- true, and it made `can a dog fall into a hole`
                # cite `capable of fall victim` over `capable of fall into
                # hole`, which is the same shortness rule reading the shorter
                # of two facts when the longer answered the question. Coverage
                # first, then shortness among equals: the `fly` case is
                # untouched because both cover the one word asked.
                covered = len(asked_stems & {self.identifier.stem(word)
                                             for word in text.split()})
                return (level.distance, 0 if negative else 1,
                        -covered,
                        RELATION_RANK.get(relation.removeprefix("not_"), 11),
                        len(text), -fact["confidence"])

            refused: list[tuple] = []
            for level, fact, text, term in sorted(matches, key=better):
                if (fact["relation"].startswith("not_")
                        or self._denies(text, term)):
                    return Verdict(
                        term=term, verdict="DENIED", predicate=text,
                        source=level.concept, distance=level.distance,
                        detail=f"Not in the norms for {name}, and "
                               f"{level.concept} — {level.distance} level(s) "
                               f"up — {text}.")
                # R19: put the borrowed fact to the ancestor's other kinds.
                bearing, kinds = self.corroboration(level.concept, term)
                if (kinds >= CORROBORATION_MIN_KINDS
                        and bearing / kinds < CORROBORATION_FLOOR):
                    refused.append((level, text, bearing, kinds))
                    continue
                # Only report corroboration where it was actually consulted.
                # Below the minimum R19 declines to judge, and printing "0 of
                # 7 bear it out" beside a yes reads as the answer arguing with
                # itself -- which is what `can a dog fall into a hole` did.
                if kinds >= CORROBORATION_MIN_KINDS:
                    support = (f" {bearing} of {kinds} kinds of "
                               f"{level.concept.split('.')[0]} in the norms "
                               f"bear it out.")
                elif kinds:
                    support = (f" The norms describe only {kinds} kind"
                               f"{'' if kinds == 1 else 's'} of "
                               f"{level.concept.split('.')[0]}, too few to "
                               f"corroborate either way.")
                else:
                    support = ""
                return Verdict(
                    term=term, verdict="INHERITED", predicate=text,
                    source=level.concept, distance=level.distance,
                    corroborated=kinds >= CORROBORATION_MIN_KINDS,
                    detail=f"Not in the norms for {name}, but {level.concept} "
                           f"— {level.distance} level(s) up — {text}."
                           + support)
            if refused:
                level, text, bearing, kinds = refused[0]
                return Verdict(
                    term=terms[0], verdict="UNRECORDED",
                    source=level.concept, distance=level.distance,
                    detail=f"{level.concept} is recorded as “{text}”, but "
                           f"only {bearing} of its {kinds} kinds in the norms "
                           f"bear that out, so it is not inherited down to "
                           f"{name}. R19: one crawled sentence is not a "
                           f"property of a category.")
        if narrower:
            shown = ", ".join(f"“{text}”" for text in sorted(narrower)[:3])
            return Verdict(
                term=terms[0] if terms else "", verdict="UNRECORDED",
                source=self.origin.get(name, "?"),
                detail=f"The norms deny {shown} of {name}, but each of those "
                       f"is a narrower claim than the one asked: denying a "
                       f"qualified property does not deny the property. On "
                       f"the question as asked they are silent.")
        return Verdict(
            term=terms[0] if terms else "", verdict="UNRECORDED",
            detail=f"The norms neither state nor deny that of {name}, and "
                   f"nothing it is a kind of does either. Absent, not false.")

    # -- structured questions ----------------------------------------------
    #: How a single-property verdict reads as a truth value.
    AS_VALUE = {"HELD": logic.TRUE, "INHERITED": logic.TRUE,
                "DENIED": logic.FALSE, "MIXED": logic.UNKNOWN,
                "UNRECORDED": logic.UNKNOWN}
    #: And back again, for the badge.
    AS_VERDICT = {logic.TRUE: "HELD", logic.FALSE: "DENIED",
                  logic.UNKNOWN: "UNRECORDED"}

    def verify(self, name: str, terms: list[str],
               descend: bool = True) -> Verdict:
        """Several properties at once, read as a conjunction.

        v686 returned on the first term that matched, so `does a dog have a
        tail and wings` was VERIFIED on the strength of the tail. Asking for
        two things and being told about one is the failure R14 exists to
        prevent, one level up.
        """
        query = logic.Query(quantifier=None, tree=logic.Node(
            op="and", children=[logic.Node(op="term", term=term)
                                for term in terms]))
        return self.assess(name, query, descend=descend)

    def assess(self, name: str, query: logic.Query,
               descend: bool = True) -> Verdict:
        """Evaluate a whole question against one concept, or its kinds.

        The single-property verdicts are unchanged -- this only composes them,
        and composing is where the three values earn their keep: an unknown
        conjunct suspends the answer rather than sinking it.
        """
        if query.quantifier and descend and self.subtypes(name):
            return self._quantified(name, query)
        parts: dict[str, Verdict] = {}

        asked = query.tree.terms()

        def test(term: str) -> tuple[str, str]:
            answer = self.verify_one(name, term, descend=descend, asked=asked)
            parts[term] = answer
            return self.AS_VALUE.get(answer.verdict, logic.UNKNOWN), answer.detail

        value = logic.evaluate(query.tree, test)
        lead = next((answer for answer in parts.values()
                     if self.AS_VALUE.get(answer.verdict) == value), None)
        settled = self._why(query, parts, value)
        # Composition must not lose *how* a thing is true. `INHERITED` is a
        # yes, so it evaluates as one -- but a yes the concept itself never
        # states is a different answer from one it does, and flattening every
        # true part to HELD threw that away. When nothing but inheritance
        # carries it, the verdict says so.
        verdict = self.AS_VERDICT[value]
        if value == logic.TRUE and parts and all(
                answer.verdict == "INHERITED" for answer in parts.values()
                if self.AS_VALUE.get(answer.verdict) == logic.TRUE):
            verdict = "INHERITED"
        return Verdict(
            term=query.tree.terms()[0] if query.tree.terms() else "",
            verdict=verdict,
            predicate=lead.predicate if lead else None,
            source=lead.source if lead else None,
            distance=lead.distance if lead else 0,
            corroborated=lead.corroborated if lead else None,
            depth=lead.depth if lead else 0, shared=lead.shared if lead else 0,
            members=lead.members if lead else [],
            tree=query.tree.as_dict(), quantifier=query.quantifier,
            parts={term: answer.as_dict() for term, answer in parts.items()},
            detail=settled)

    def _why(self, query: logic.Query, parts: dict[str, Verdict],
             value: str) -> str:
        """Name the term that settled it, because that is the explanation.

        "No" to a conjunction is a claim about one conjunct, and which one is
        the whole content of the answer.
        """
        shape = logic.readable(query.tree)
        if len(parts) == 1:
            return next(iter(parts.values())).detail
        decisive = [term for term, answer in parts.items()
                    if self.AS_VALUE.get(answer.verdict) == value]
        if value == logic.FALSE:
            return (f"{shape}: no. " +
                    "; ".join(parts[term].detail for term in decisive[:2]))
        if value == logic.TRUE:
            lead = ("yes -- one true side settles a disjunction"
                    if query.tree.op == "or" else "yes, every part of it")
            return f"{shape}: {lead}. " + parts[decisive[0]].detail
        silent = [term for term, answer in parts.items()
                  if answer.verdict == "UNRECORDED"]
        return (f"{shape}: unsettled. Nothing is recorded either way about "
                + ", ".join(f"“{term}”" for term in silent[:3])
                + ", and an unknown part suspends the whole rather than "
                  "making it false.")

    def _quantified(self, name: str, query: logic.Query) -> Verdict:
        """`do all birds fly` -- put the question to every kind and count.

        A single lookup on `bird` answered this from one ConceptNet sentence.
        The kinds beneath it answer it properly, and the exceptions are the
        interesting half: most birds fly, and the two that do not are the
        reason `most` is a different word from `all`.
        """
        kinds = self.subtypes(name)
        holds, fails, silent = [], [], []
        members = []
        for other in kinds:
            answer = self.assess(other, logic.Query(None, _clone(query.tree)),
                                 descend=False)
            value = self.AS_VALUE.get(answer.verdict, logic.UNKNOWN)
            (holds if value == logic.TRUE else
             fails if value == logic.FALSE else silent).append(other)
            if value in (logic.TRUE, logic.FALSE):
                members.append({"name": other, "verdict": answer.verdict,
                                "predicate": answer.predicate})
        decided = len(holds) + len(fails)
        shape = logic.readable(query.tree)
        want = query.quantifier
        if not decided:
            value = logic.UNKNOWN
        elif want == "all":
            value = logic.FALSE if fails else logic.TRUE
        elif want == "some":
            value = logic.TRUE if holds else logic.FALSE
        elif want == "none":
            value = logic.FALSE if holds else logic.TRUE
        else:                                   # most: a count, not a claim
            value = logic.TRUE if len(holds) > len(fails) else logic.FALSE
        # Each quantifier is settled by a different set, and naming the wrong
        # one is how "some birds fly" came back citing the birds that do.
        counted = f"{len(holds)} of {decided} recorded kinds of {name} satisfy {shape}"

        def naming(names: list[str], shown: int = 4) -> str:
            """The first few, and honest about being the first few.

            `7 do not: chicken, cockerel, emu, magpie` names four and claims
            seven, which reads as an arithmetic error rather than a list cut
            short.
            """
            head = ", ".join(names[:shown])
            return head if len(names) <= shown else f"{head} and {len(names) - shown} more"

        if want == "all":
            decisive = (f", and {len(fails)} do not: {naming(fails)}"
                        if fails else ", with no exception")
        elif want == "some":
            decisive = (f", among them {naming(holds)}" if holds
                        else ", so none does")
        elif want == "none":
            decisive = (f", so the claim fails on {naming(holds)}"
                        if holds else ", so none does and the claim holds")
        else:
            decisive = (f" — a majority, and the {len(fails)} that do not are "
                        f"{naming(fails)}" if len(holds) > len(fails)
                        else f" — not a majority")
        detail = (f"Asked of every kind: {counted}{decisive}."
                  + (f" {len(silent)} more say nothing either way, and are "
                     f"counted for neither side." if silent else ""))
        return Verdict(
            term=query.tree.terms()[0] if query.tree.terms() else "",
            verdict=self.AS_VERDICT[value], source="kinds",
            members=members, tree=query.tree.as_dict(),
            quantifier=want, detail=detail)

    @staticmethod
    def _denies(text: str, target: str = "") -> bool:
        """Is this property phrased as a denial rather than a claim?

        Negation scopes forward. Without a target there is nothing to scope
        over and any negator counts, which is what every caller but one
        wants. Given one, a negator only denies if it comes *before* the
        thing asked: `capable of digest grass but humans cannot` asserts that
        cattle digest grass and denies it of humans, and reading the trailing
        `cannot` as a denial is how `does a cow eat grass` came back
        CONTRADICTED.
        """
        words = re.findall(r"[a-z]+", text.lower())
        negators = [i for i, word in enumerate(words) if word in NEGATORS]
        if not negators:
            return False
        if not target:
            return True
        wanted = set(re.findall(r"[a-z]+", target.lower()))
        hit = next((i for i, word in enumerate(words) if word in wanted), None)
        if hit is None:
            return True
        return negators[0] < hit

    # -- routing -----------------------------------------------------------
    #: "what attributes does X have", "what is X like", "describe X".
    PROFILE = (
        re.compile(r"^(?:what|which)\s+(?:attributes?|properties|property|"
                   r"features?|characteristics?|qualities|traits?)\b"),
        re.compile(r"^(?:what|which)\s+(?:is|are)\s+.+\s+like$"),
        re.compile(r"^(?:describe|list)\b"),
        re.compile(r"^(?:tell me|what do (?:we|you) know)\s+about\b"),
    )
    #: A polar question: "is a blue whale furry", "does a robin have wings".
    POLAR = re.compile(r"^(?:is|are|was|were|does|do|did|has|have|can|could)\b")

    def route(self, question: str) -> tuple[str, str, list[str]] | None:
        """(mode, individual, property words), or None to leave it alone.

        Either question is about a named thing, so the name is looked up
        before the shape is trusted: the longest concept the norms know that
        the question mentions. Nothing matches, nothing is taken, and v684
        answers as it always did.
        """
        text = (question or "").strip().lower().rstrip("?").strip()
        if not text:
            return None
        profile = any(pattern.search(text) for pattern in self.PROFILE)
        if not profile and not self.POLAR.match(text):
            return None
        name = self.subject(text)
        if name is None:
            return None
        spent = set(name.split())
        words = [word for word in re.findall(r"[a-z]+", text)
                 if word not in ASIDE and word not in spent]
        if profile:
            return ("profile", name, words)
        return ("verify", name, words) if words else None

    def subject(self, text: str) -> str | None:
        """The concept a question is about: a norm concept, or a class above.

        `do all birds fly` is the question quantifiers exist for and `bird` is
        not one of XCSLB's 521 concepts, so requiring one meant the showcase
        question fell through to a single ConceptNet sentence. A word the
        norms do not cover still counts if the taxonomy puts norm-covered
        kinds underneath it.
        """
        named = self.named(text)
        if named:
            return named
        for word in re.findall(r"[a-z]+", text):
            if word in ASIDE or word in logic.QUANTIFIERS or len(word) < 3:
                continue
            single = word[:-1] if word.endswith("s") and len(word) > 3 else word
            for candidate in dict.fromkeys((word, single)):
                if self.subtypes(candidate):
                    return candidate
        return None

    def named(self, text: str) -> str | None:
        """The longest concept the norms know that the question mentions.

        Longest wins for the reason it does in v684's parser: `blue whale` and
        `whale` are both concepts the data knows, and the question asked about
        the first.
        """
        best: str | None = None
        for name in self.stated:
            if best is not None and len(name) <= len(best):
                continue
            if re.search(r"\b" + re.escape(name) + r"\b", text):
                best = name
        return best
