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
from . import rules
from . import corpora

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

#: Words that carry no property in a question about a named thing.
ASIDE = frozenset("""
what which is are was were be been does do did has have had can could would
a an the of its it their there any some this that these those
tell me about you i us we know show give
attribute attributes property properties feature features characteristic
characteristics trait traits quality qualities like describe list description
thing things kind kinds sort sorts type types really actually also too and or
with
""".split())


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

    def as_dict(self) -> dict[str, Any]:
        return {"term": self.term, "verdict": self.verdict,
                "detail": self.detail, "predicate": self.predicate,
                "source": self.source, "distance": self.distance,
                "depth": self.depth, "shared": self.shared,
                "members": self.members}


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
        self.denied: dict[str, frozenset[str]] = {}
        for source in (corpora.denied_awa2(), corpora.denied_xcslb()):
            for name, properties in source.items():
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
        concept = self.synset.get(name)
        if not concept:
            return []
        return sorted(other for other, above in self._lineage().items()
                      if other != name and concept in above
                      and self.stated.get(other))

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
        """
        votes = [(other, self.verify(other, terms, descend=False))
                 for other in self.subtypes(name)]
        held = [(other, answer) for other, answer in votes
                if answer.verdict in ("HELD", "INHERITED")]
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
    def verify(self, name: str, terms: list[str],
               descend: bool = True) -> Verdict:
        """Does this thing have that? Stated, denied, inherited or unrecorded.

        The order is evidence quality, not convenience. What the norms state
        about the thing itself is the best answer; what they state it lacks is
        the next best, and it is a real "no" rather than a silence. Only when
        the norms are silent both ways is the taxonomy asked, because
        inherited facts are corpus free text and matching one word inside one
        is loose enough to make a marble fly.
        """
        if not self.knows(name):
            return Verdict(term=terms[0] if terms else "",
                           verdict="UNRECORDED",
                           detail=f"“{name}” is not one of the concepts the "
                                  f"norms cover.")
        stated = self.stated.get(name, frozenset())
        climb = self.walk_up(name)
        sharing = {held.predicate: held.shared for held in climb}
        depths = {held.predicate: held.depth for held in climb}
        for term in terms:
            hit = self.identifier._hit(term, stated)
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
        for term in terms:
            hit = self.identifier._hit(term, self.denied.get(name, frozenset()))
            if hit is not None:
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
                for term in terms:
                    if self.identifier._hit(term, frozenset({text})):
                        matches.append((level, fact, text, term))
                        break
        if matches:
            def better(match) -> tuple:
                level, fact, text, _ = match
                relation = fact["relation"]
                # R3: at the level that answers, a denial blocks the positive.
                negative = relation.startswith("not_") or self._denies(text)
                return (level.distance, 0 if negative else 1,
                        RELATION_RANK.get(relation.removeprefix("not_"), 11),
                        len(text), -fact["confidence"])

            level, fact, text, term = min(matches, key=better)
            if fact["relation"].startswith("not_") or self._denies(text):
                return Verdict(
                    term=term, verdict="DENIED", predicate=text,
                    source=level.concept, distance=level.distance,
                    detail=f"Not in the norms for {name}, and {level.concept} "
                           f"— {level.distance} level(s) up — {text}.")
            return Verdict(
                term=term, verdict="INHERITED", predicate=text,
                source=level.concept, distance=level.distance,
                detail=f"Not in the norms for {name}, but {level.concept} "
                       f"— {level.distance} level(s) up — {text}.")
        return Verdict(
            term=terms[0] if terms else "", verdict="UNRECORDED",
            detail=f"The norms neither state nor deny that of {name}, and "
                   f"nothing it is a kind of does either. Absent, not false.")

    @staticmethod
    def _denies(text: str) -> bool:
        """Is this property phrased as a denial rather than a claim?"""
        return any(word in NEGATORS
                   for word in re.findall(r"[a-z]+", text.lower()))

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
        name = self.named(text)
        if name is None:
            return None
        spent = set(name.split())
        words = [word for word in re.findall(r"[a-z]+", text)
                 if word not in ASIDE and word not in spent]
        if profile:
            return ("profile", name, words)
        return ("verify", name, words) if words else None

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
