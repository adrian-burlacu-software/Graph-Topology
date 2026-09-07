"""Two concepts at once: what they share, where they part, how alike they are.

Four question types that look unrelated and are one operation:

    what do a dog and a cat have in common     the shared prefix
    what is the difference between them        where the branch splits
    what is similar to a dog                   how long others walk with it
    is a penguin a typical bird                how far it walks with its class

All four are the lowest common ancestor of two branches. That is not a
metaphor about the trie -- it is what the trie is: an individual is a path,
two individuals share a prefix and then diverge, and every comparison anyone
wants to make about them is a question about where that happens.

Two measures are reported, not one, because they disagree and the
disagreement is a finding:

    shared      the predicates both concepts carry. This is semantic overlap
                and it is what a person means by "in common".

    together    how far the two walk down the *stored* trie before parting.
                This is the compression view, and it is usually much shorter,
                because `adaptive_coverage` orders predicates by how many
                individuals carry them and not by what any pair has in common.

The gap between those two numbers says something worth saying: a trie built
for storage does not group by similarity. `killer whale` and `blue whale`
share most of their properties and part at the first node. Storage order and
similarity order are as separate as storage order and question order were in
v686 -- the same finding, one axis over.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import pins
from typing import Any

#: A property is *core* to a class when at least this share of the class's
#: norm-covered kinds carry it. Typicality is measured against these, because
#: a property one bird in thirty has says nothing about what a bird is.
#:
#: A quarter, not a half, and the data chose the number: XCSLB elicitation is
#: sparse and free, so no two people list the same things. Across 29 birds the
#: most-produced property is `has talons`, at 10 -- 34%. At a half share the
#: core of every XCSLB class is empty and typicality cannot be computed at
#: all. AwA2's classes are dense enough that any floor works.
CORE_SHARE = 0.25

#: How many concepts a similarity question returns.
NEIGHBOURS = 8

#: Properties listed per side of a comparison.
SHOWN = 10


@dataclass
class Comparison:
    """Two concepts, and the shape of their agreement."""
    left: str
    right: str
    shared: list[str] = field(default_factory=list)
    only_left: list[str] = field(default_factory=list)
    only_right: list[str] = field(default_factory=list)
    shared_total: int = 0
    left_total: int = 0
    right_total: int = 0
    #: How far the two walk together down the stored trie before parting.
    together: list[str] = field(default_factory=list)
    parted_at: str | None = None
    jaccard: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"left": self.left, "right": self.right,
                "shared": self.shared, "only_left": self.only_left,
                "only_right": self.only_right,
                "shared_total": self.shared_total,
                "left_total": self.left_total, "right_total": self.right_total,
                "together": self.together, "parted_at": self.parted_at,
                "jaccard": round(self.jaccard, 3)}


@dataclass
class Typicality:
    """How ordinary a member is, measured against its own class."""
    member: str
    klass: str
    core: list[str] = field(default_factory=list)
    held: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    score: float = 0.0
    rank: int = 0
    of: int = 0
    ranking: list[dict] = field(default_factory=list)
    #: The best score any kind achieves. When even the most ordinary member of
    #: a class carries little of its core, the class has no core worth ranking
    #: against and the measure says so instead of pretending.
    support: float = 0.0
    #: Which corpus the comparison was drawn from, and how many of the class's
    #: kinds it left out. A member is only ever ranked against kinds described
    #: in the same vocabulary as itself.
    corpus: str | None = None
    excluded: int = 0

    @property
    def sound(self) -> bool:
        return self.support >= 0.5

    def as_dict(self) -> dict[str, Any]:
        return {"member": self.member, "klass": self.klass, "core": self.core,
                "held": self.held, "missing": self.missing,
                "score": round(self.score, 3), "rank": self.rank,
                "of": self.of, "ranking": self.ranking,
                "support": round(self.support, 3), "sound": self.sound,
                "corpus": self.corpus, "excluded": self.excluded}


class Contrast:
    """Comparisons over the same norms and the same trie `Profiles` reads."""

    def __init__(self, profiles) -> None:
        self.profiles = profiles
        self.stated = profiles.stated

    # -- two concepts ------------------------------------------------------
    def compare(self, left: str, right: str) -> Comparison | None:
        if not (self.profiles.knows(left) and self.profiles.knows(right)):
            return None
        here, there = self.stated[left], self.stated[right]
        shared = here & there
        both = here | there
        walk = self._together(left, right)
        return Comparison(
            left=left, right=right,
            shared=sorted(shared)[:SHOWN],
            only_left=self._telling(left, here - there),
            only_right=self._telling(right, there - here),
            shared_total=len(shared),
            left_total=len(here), right_total=len(there),
            together=[str(p) for p in walk],
            parted_at=self._parting(left, right, len(walk)),
            jaccard=len(shared) / len(both) if both else 0.0)

    def _telling(self, name: str, only: set[str]) -> list[str]:
        """A difference is more interesting the fewer other things share it.

        Sorting alphabetically buries `has stripes` under `has a body`. The
        properties worth reading are the ones that few other concepts carry,
        which is `anti_coverage` used as a presentation order.
        """
        return sorted(only, key=lambda p: (self._carriers(p), p))[:SHOWN]

    def _carriers(self, predicate: str) -> int:
        if not hasattr(self, "_counts"):
            counts: dict[str, int] = {}
            for predicates in self.stated.values():
                for predicate_ in predicates:
                    counts[predicate_] = counts.get(predicate_, 0) + 1
            self._counts = counts
        return self._counts.get(predicate, 0)

    def _together(self, left: str, right: str) -> tuple:
        """The longest common prefix of two branches: the trie's own LCA."""
        here = self.profiles.trie.path(self.profiles.node_of[left])
        there = self.profiles.trie.path(self.profiles.node_of[right])
        shared = []
        for mine, yours in zip(here, there):
            if mine != yours:
                break
            shared.append(mine)
        return tuple(shared)

    def _parting(self, left: str, right: str, depth: int) -> str | None:
        """The predicate one took and the other did not."""
        here = self.profiles.trie.path(self.profiles.node_of[left])
        return str(here[depth]) if depth < len(here) else None

    # -- one concept against everything -----------------------------------
    def nearest(self, name: str, limit: int = NEIGHBOURS) -> list[dict]:
        """What else is like this, by shared properties.

        The stored `similar_to` relation is not used. It has 21,877 facts and
        no inheritance, while this is computed from what the norms actually
        elicited about both concepts -- and it can say *why*, which a stored
        similarity edge cannot.
        """
        if not self.profiles.knows(name):
            return []
        here = self.stated[name]
        scored = []
        for other, predicates in self.stated.items():
            if other == name or not predicates:
                continue
            shared = here & predicates
            union = here | predicates
            scored.append((len(shared) / len(union), len(shared), other, shared))
        scored.sort(key=lambda row: (-row[0], -row[1], row[2]))
        return [{"name": other, "jaccard": round(score, 3), "shared": count,
                 "because": sorted(shared,
                                   key=lambda p: (self._carriers(p), p))[:4]}
                for score, count, other, shared in scored[:limit]]

    def covered(self, word: str) -> bool:
        return self.profiles.knows(word)

    def _up_to(self, name: str, node: str) -> int:
        """How many steps from a covered concept up to a shared ancestor."""
        concept = self.profiles.synset.get(name)
        if not concept:
            return 99
        for above, distance, _ in self.profiles.reasoner.ascend(concept):
            if above == node:
                return distance
        return 99

    def nearest_covered(self, word: str) -> tuple[str, int, str] | None:
        """The closest concept above this word that the norms do describe.

        A comparison needs two concepts with properties, and the norms cover
        541 things. When one side is not among them the useful thing to say is
        not "no" but "not this, though it is a kind of that".

        Every sense is tried, not just the primary one, and this is the case
        that shows why: the first sense of `bitch` is a difficulty -- life's a
        bitch -- whose ancestors are `condition`, `attribute`, `abstraction`.
        Nothing there is a dog. `bitch.n.04` is the female dog, and it is one
        step under `dog`. Taking the primary sense and stopping would report
        that nothing above `bitch` is covered, which is false of the sense the
        reader meant.
        """
        named: dict[str, str] = {}
        for name, synset in self.profiles.synset.items():
            named.setdefault(synset, name)
        pinned = pins.of(word)
        senses = ([pinned] if pinned else
                  [row["id"] for row in
                   self.profiles.reasoner.senses_of(word)]
                  or [self.profiles.class_concept(word)])
        lineage = self.profiles._lineage()
        best: tuple[str, int, str] | None = None
        for concept in senses:
            if not concept:
                continue
            for node, distance, _ in self.profiles.reasoner.ascend(concept):
                name = named.get(node)
                if name and self.covered(name):
                    found = (name, distance, concept)     # covered ancestor
                elif kin := [other for other, above in lineage.items()
                             if node in above and self.covered(other)]:
                    # Not above it, but beside it. WordNet files the female
                    # dog under `canine`, not under `dog`, so walking straight
                    # up from `bitch.n.04` passes canine, carnivore, placental
                    # and never meets a concept the norms describe. The answer
                    # worth giving is the nearest *relative* they do describe.
                    #
                    # Which one is nearest has to be measured. Counting each
                    # candidate's ancestors as a proxy for how general it is
                    # picked `fox` over `dog`, because `dog` is filed under
                    # `domestic animal` as well as `canine` and so has more
                    # ancestors than a fox. Distance up to the shared node is
                    # the thing actually being asked about.
                    found = (min(kin, key=lambda n: (self._up_to(n, node), n)),
                             distance, concept)
                else:
                    continue
                if best is None or found[1] < best[1]:
                    best = found
                break
        return best

    # -- counting over the taxonomy ---------------------------------------
    def kinds_of(self, word: str, limit: int = 40) -> dict | None:
        """How many kinds of a thing there are, and which.

        R18 refuses `how many legs does a dog have` on the grounds that the
        ontology holds no numbers, and says in the same breath that it can
        count *kinds*. It could not: nothing implemented this, so the question
        R18's own refusal offers as the answerable one fell through to a
        listing about the word `kind`.

        Two counts, because they are different questions and conflating them
        would be the same mistake:

            in the taxonomy   every descendant WordNet records. This is what
                              "how many kinds of dog are there" asks.
            in the norms      the ones the feature norms actually describe,
                              which is what every other rule here can reason
                              about.
        """
        # The reader's pin outranks the norm join as well as the guess: the
        # question "how many kinds of mouse" is about whichever mouse they
        # meant, and the norms cover only one of them.
        concept = (pins.of(word) or self.profiles.synset.get(word)
                   or self.profiles.class_concept(word))
        if not concept:
            return None
        reasoner = self.profiles.reasoner
        seen: set[str] = set()
        frontier = [concept]
        while frontier:
            node = frontier.pop()
            for row in reasoner.connection.execute(
                    "SELECT child FROM taxonomy WHERE parent = ?", (node,)):
                child = row["child"]
                if child not in seen:
                    seen.add(child)
                    frontier.append(child)
        direct = [row["child"] for row in reasoner.connection.execute(
            "SELECT child FROM taxonomy WHERE parent = ?", (concept,))]
        described = self.profiles.subtypes(word)
        return {"word": word, "concept": concept,
                "gloss": reasoner.gloss(concept),
                "total": len(seen), "direct": len(direct),
                "direct_kinds": sorted(direct)[:limit],
                "described": len(described),
                "described_kinds": described[:limit]}

    # -- a member against its class ---------------------------------------
    def core_of(self, klass: str,
                corpus: str | None = None) -> tuple[list[str], list[str]]:
        """The properties that most of a class's kinds carry, and the kinds.

        `corpus` restricts both to one source, and it matters more than it
        looks. The norms are two corpora with disjoint vocabularies: AwA2
        gives every one of its 50 animals the same 85 attributes, XCSLB is
        free elicitation where no two people write the same phrase. Pooled,
        `animal` has 143 kinds of which 43 are AwA2 -- and because those 43
        are dense and agree with each other word for word, only AwA2
        predicates ever clear the floor. The core came out `oldworld,
        quadrapedal, ground`, and all 100 XCSLB kinds scored zero against it:
        `is a dog a typical animal` answered no, at 0%, ranking 67 of 143.

        A class is only a class within one vocabulary, so the comparison is
        made within one.
        """
        kinds = self.profiles.subtypes(klass)
        if corpus:
            kinds = [kind for kind in kinds
                     if self.profiles.origin.get(kind) == corpus]
        if not kinds:
            return [], []
        counts: dict[str, int] = {}
        for kind in kinds:
            for predicate in self.stated.get(kind, ()):
                counts[predicate] = counts.get(predicate, 0) + 1
        floor = max(2, int(len(kinds) * CORE_SHARE))
        core = sorted((p for p, n in counts.items() if n >= floor),
                      key=lambda p: (-counts[p], p))
        return core, kinds

    def typicality(self, member: str, klass: str) -> Typicality | None:
        """How much of what makes a class a class does this member have?

        A penguin is a bird that fails most of what birds are for; a robin is
        a bird that passes. Nothing in the norms says "typical" -- it falls
        out of counting, which is the point: typicality is not stored, it is
        the shape of the class seen from one member.
        """
        corpus = self.profiles.origin.get(member)
        core, kinds = self.core_of(klass, corpus)
        if not core:
            return None
        excluded = len(self.profiles.subtypes(klass)) - len(kinds)

        def score(name: str) -> float:
            held = self.stated.get(name, frozenset()) & set(core)
            return len(held) / len(core)
        ranked = sorted(kinds, key=lambda name: (-score(name), name))
        mine = set(self.stated.get(member, frozenset())) & set(core)
        return Typicality(
            member=member, klass=klass, core=core[:SHOWN],
            held=sorted(mine)[:SHOWN],
            missing=sorted(set(core) - mine)[:SHOWN],
            score=score(member),
            rank=ranked.index(member) + 1 if member in ranked else 0,
            of=len(ranked),
            support=score(ranked[0]) if ranked else 0.0,
            corpus=corpus, excluded=excluded,
            ranking=[{"name": name, "score": round(score(name), 3)}
                     for name in ranked[:5]]
                    + [{"name": name, "score": round(score(name), 3)}
                       for name in ranked[-3:]])
