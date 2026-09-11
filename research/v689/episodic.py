"""Episodic memory: what a conversation has been told, held the way the store
holds what it knows and reasoned over by the same rules.

## Semantic memory and episodic memory

v687's store is semantic memory. Episodic memory is everything a conversation
adds to it, kept beside it and never written back:

    individuals     the pig you mentioned, under hog.n.03
    kinds           a wemble, under animal.n.01 -- a word the store never had
    taxonomy        a wemble is a kind of animal
    norms           beagles can't swim; wembles can fly
    episodes        he was flying; it was in an airplane

Every one of them is the store's own shape: a node, its parents, its facts,
with `told` as the source. `EpisodicReasoner` is v687's `Reasoner` with
`parents_of` and `facts_of` reading both memories, so the two are one
taxonomy to every rule in `rules.py`:

    it can't swim          not_capable_of swim on the pig     R3 at distance 0
    beagles can't swim     not_capable_of swim on beagle.n.01 R3 at distance 1,
                                                              before dog's row
    a wemble is an animal  wemble -> animal.n.01              R1 walks through it
    he was flying          capable_of fly on the pig          R4 at distance 0
    it has no tail         has_a "no tail"                    R3 in the object
    it chased the cat      capable_of "chase a cat", and which cat, beside it

Two rules are added.

**E1. A quality does not descend from a kind to an individual.** `is a beagle
black` is recorded, and beagles are tricoloured; `is the second beagle black`
is about one dog. R2 decides per relation what descends between kinds; E1
says `has_property` does not descend onto an individual at all.

**E2. An action done while carried belongs to what carries it.** `he was
flying` and `it was in an airplane`: an airplane flies, so the flying was the
airplane's, and the pig's `capable_of fly` is withdrawn rather than left
standing as an exception. Only what the individual was *seen doing* is
withdrawn -- `it can fly`, said outright, is a claim about the pig, and a
claim is not explained away by where the pig was. If what carried it cannot
do the thing either, the doing stands. The session applies it (`_carry`),
because whether an airplane flies is a question for the rules and for v688,
not for memory.

Two relations are added, and no rule reads either: `did_not` -- not doing a
thing is not being unable to -- and `carried`, what E2 withdrew.

## The trie, live

Appendix 3 stores individuals as goal nodes under ordered predicate paths.
The episodic trie is that structure over the conversation's individuals. Each
one's predicates are every kind above it -- taught kinds included, so `the
wemble` and `the animal` both reach it by R1's closure -- what it was told,
what it is called and whose it is. It is re-planned with `adaptive_coverage`
on every change, because "the topographical growth of tries is driven by
allocation", and every change reports what it allocated. A second beagle
allocates nothing: it shares every predicate with the first until one of them
is told something.

Resolving a description is identification, the trie read downwards, as
`identify.py` reads it: walk down until the description is exhausted, and
everyone stored beneath fits.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from research.v687 import rules
from research.v687.ordering import adaptive_coverage
from research.v687.reason import Answer, Fact, Reasoner
from research.v687.rules import Step
from research.v687.trie import ROOT, PredicateTrie

#: The source column of everything a conversation stores.
TOLD = "told"

#: E1: what a quality is.
QUALITIES = frozenset({"has_property", "has_attribute", "not_has_property"})

#: Not doing a thing. Read by the session for `does it`, never by a rule.
DID_NOT = "did_not"

#: What E2 withdrew. Read by nothing; kept so the page can say what happened.
CARRIED = "carried"

DENIERS = ("no ", "not ")


def name_of(node: str | None) -> str:
    """`hunting dog.n.01` -> `hunting dog`; an episodic node is its own name."""
    if not node:
        return ""
    return node.rsplit(".", 2)[0] if node.count(".") >= 2 else node


def _stem(obj: str) -> str:
    """An object with its denial taken off: `no tail` and `tail` are the
    same claim, asserted and denied."""
    text = (obj or "").lower().strip()
    for denier in DENIERS:
        if text.startswith(denier):
            return text[len(denier):].strip()
    return text


def _quoted(said: str) -> str:
    """Kept without closing punctuation: it is only ever shown in quotes."""
    return (said or "").strip().rstrip(".!?")


@dataclass
class Growth:
    """What one change to episodic memory did to the trie."""

    reason: str
    individuals: int
    cells: int            # individual-predicate pairs, stored flat
    nodes: int
    allocated: int        # nodes this change added; negative if it freed some

    def as_dict(self) -> dict:
        return {"reason": self.reason, "individuals": self.individuals,
                "cells": self.cells, "nodes": self.nodes,
                "allocated": self.allocated,
                "shared": (round(1 - self.nodes / self.cells, 3)
                           if self.cells else 0.0)}


@dataclass
class Identification:
    """A description walked down the trie, and who is stored beneath it."""

    wanted: list
    candidates: list
    visited: int
    #: every path at which the description was exhausted
    reached: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"wanted": list(self.wanted),
                "candidates": list(self.candidates),
                "visited": self.visited,
                "reached": [list(path) for path in self.reached]}


@dataclass
class Withdrawal:
    """What E2 took back from an individual, and what it was carried by."""

    node: str
    relation: str
    object: str
    carrier: str
    said: str
    #: the individual that carried it, when it was one
    carrier_id: str | None = None

    def as_dict(self) -> dict:
        return {"node": self.node, "relation": self.relation,
                "object": self.object, "carrier": self.carrier,
                "said": self.said, "carrier_id": self.carrier_id}


class EpisodicMemory:
    """Everything one conversation added to what the store knows."""

    def __init__(self, reasoner: Reasoner) -> None:
        self.base = reasoner
        #: node -> its parents in episodic memory: an individual's kind, a
        #: taught kind's parent, or an edge taught between two store kinds
        self.edges: dict[str, list[str]] = {}
        #: the nodes that are individuals, in the order they came up
        self.individuals: list[str] = []
        #: word -> node, for kinds the store has no sense for
        self.kinds: dict[str, str] = {}
        #: facts on any node: an individual, a taught kind, a store synset
        self.facts: dict[str, list[Fact]] = {}
        #: (node, relation, object) -> the utterance that told it
        self.said: dict[tuple, str] = {}
        #: (node, relation, object) -> `can` for a claim of ability, `does`
        #: for something seen done. E2 withdraws only the second.
        self.mode: dict[tuple, str] = {}
        #: (node, relation, object) -> v688's answer for the kind, when told
        self.against: dict[tuple, str] = {}
        #: (node, relation, object) -> the individuals the object named. `it
        #: chased the cat` stores `capable_of "chase a cat"`, which is what
        #: the rules can read, and keeps which cat here
        self.bound: dict[tuple, set] = {}
        #: individual -> `name rex`, `owner you`
        self.labels: dict[str, set[str]] = {}
        #: individual -> the word it was introduced by
        self.words: dict[str, str] = {}
        #: individual -> every kind it is, nearest first, as words
        self.lineage: dict[str, list[str]] = {}
        self.withdrawn: list[Withdrawal] = []
        self.trie = PredicateTrie()
        self.plan: list = []
        self.members: dict[tuple, list[str]] = {}
        self.growth: list[Growth] = []
        self.reasoner = EpisodicReasoner(reasoner, self)

    @property
    def parent(self) -> dict:
        """individual -> the kind it sits under."""
        return {node: (self.edges.get(node) or [None])[0]
                for node in self.individuals}

    def episodic_only(self, node: str | None) -> bool:
        """A node the store does not have: an individual or a taught kind."""
        return bool(node) and (node in self.individuals
                               or node in self.kinds.values())

    # -- writing -----------------------------------------------------------
    def kind_node(self, word: str, sense: str | None) -> str:
        """Where a kind word lives: its synset, or a kind taught here.

        A word the store has a sense for is that sense -- `beagles can't
        swim` is about beagle.n.01. A word it has none for becomes a node of
        its own the first time it is used: `wemble` is a kind the
        conversation taught, and stays one.
        """
        if sense:
            return sense
        word = (word or "").strip().lower()
        if word not in self.kinds:
            self.kinds[word] = word
            self.edges.setdefault(word, [])
            self.facts.setdefault(word, [])
        return self.kinds[word]

    def place(self, individual: str, word: str, node: str | None) -> Growth:
        """Put an individual under a kind, or move it under a narrower one."""
        if individual not in self.individuals:
            self.individuals.append(individual)
        self.edges[individual] = [node] if node else []
        self.words[individual] = word
        self.facts.setdefault(individual, [])
        self.labels.setdefault(individual, set())
        return self.store(f"{individual} is {word}")

    def relate(self, node: str, parent: str, said: str) -> Growth:
        """A taught edge: `a wemble is a kind of animal`."""
        parents = self.edges.setdefault(node, [])
        if parent not in parents:
            parents.append(parent)
        self.facts.setdefault(node, [])
        self.said[(node, "is_a", parent)] = _quoted(said)
        return self.store(f"{name_of(node)} is a kind of {name_of(parent)}")

    def tell(self, node: str, relation: str, obj: str, said: str,
             mode: str = "does", bound: str | None = None) -> Growth:
        """Record one fact about one node; what it contradicts is dropped.

        A fact replaces the same relation or its negation about the same
        object. `it can swim` after `it can't swim` is a correction, not two
        facts for R3 to adjudicate.

        `bound` is the individual the object named. The same fact told of
        another one adds it -- `it chased the first cat`, then `it chased the
        second cat` -- and the fact told of no one in particular is about any.
        """
        opposed = {relation}
        if relation in rules.NEGATIONS:
            opposed.add(rules.NEGATIONS[relation])
        if relation in rules.POSITIVES:
            opposed.add(rules.POSITIVES[relation])
        stem = _stem(obj)
        key = (node, relation, obj)
        before = self.facts.setdefault(node, [])
        again = any(fact.relation == relation and fact.object == obj
                    for fact in before)
        kept = []
        for fact in before:
            if fact.relation in opposed and _stem(fact.object) == stem:
                if (fact.relation, fact.object) != (relation, obj):
                    self.bound.pop((node, fact.relation, fact.object), None)
                continue
            kept.append(fact)
        kept.append(Fact(node, relation, obj, TOLD, 1.0, False))
        self.facts[node] = kept
        self.said[key] = _quoted(said)
        self.mode[key] = mode
        if bound:
            self.bound[key] = (set(self.bound.get(key, ())) if again
                               else set()) | {bound}
        else:
            self.bound.pop(key, None)
        return self.store(f"{name_of(node)} {relation} {obj}")

    def withdraw(self, node: str, relation: str, obj: str,
                 carrier: str, carrier_id: str | None = None) -> Withdrawal:
        """E2: take a fact back and keep it as `carried`."""
        said = self.said.get((node, relation, obj), "")
        self.facts[node] = [fact for fact in self.facts.get(node, [])
                            if not (fact.relation == relation
                                    and fact.object == obj)]
        self.facts[node].append(Fact(node, CARRIED, obj, TOLD, 1.0, False))
        self.said[(node, CARRIED, obj)] = said
        withdrawal = Withdrawal(node, relation, obj, carrier, said,
                                carrier_id)
        self.withdrawn.append(withdrawal)
        self.store(f"{node} {relation} {obj} withdrawn, carried by {carrier}")
        return withdrawal

    def restore(self, node: str, obj: str) -> Withdrawal | None:
        """E2 undone: nothing carrying it does the thing, so the doing was
        its own. The fact goes back as it was told."""
        withdrawal = next((one for one in reversed(self.withdrawn)
                           if one.node == node and one.object == obj), None)
        if withdrawal is None:
            return None
        self.withdrawn.remove(withdrawal)
        key = (node, withdrawal.relation, obj)
        self.facts[node] = [fact for fact in self.facts.get(node, [])
                            if not (fact.relation == CARRIED
                                    and fact.object == obj)]
        self.facts[node].append(Fact(node, withdrawal.relation, obj, TOLD,
                                     1.0, False))
        self.said[key] = withdrawal.said
        self.mode[key] = "does"
        self.store(f"{node} {withdrawal.relation} {obj} restored, "
                   f"{withdrawal.carrier} does not do it")
        return withdrawal

    def label(self, individual: str, predicate: str) -> Growth:
        """`name rex`, `owner you`: one of each kind, the latest kept."""
        head = predicate.split(" ", 1)[0] + " "
        self.labels[individual] = {one for one in self.labels[individual]
                                   if not one.startswith(head)} | {predicate}
        return self.store(f"{individual} {predicate}")

    def predicates_of(self, individual: str) -> frozenset:
        return frozenset(
            [f"is_a {kind}" for kind in self.lineage.get(individual, [])]
            + [f"{fact.relation} {fact.object.lower()}"
               for fact in self.facts.get(individual, [])]
            + list(self.labels.get(individual, ())))

    def store(self, reason: str) -> Growth:
        """Re-plan and rebuild: Appendix 3, run on every change.

        Every lineage is walked again first, because a taught edge anywhere
        above an individual changes what it is.
        """
        before = self.trie.node_count
        for individual in self.individuals:
            kinds = [self.words.get(individual, "")] + [
                name_of(node) for node, distance, _ in
                self.reasoner.ascend(individual) if distance]
            self.lineage[individual] = [kind for kind in
                                        dict.fromkeys(kinds) if kind]
        corpus = tuple(sorted((individual, self.predicates_of(individual))
                              for individual in self.individuals))
        self.plan = adaptive_coverage(corpus)
        self.trie, self.members = PredicateTrie(), {}
        for individual, path in self.plan:
            self.trie.insert(individual, path)
            for cut in range(len(path) + 1):
                self.members.setdefault(tuple(path[:cut]), []).append(
                    individual)
        growth = Growth(reason, len(corpus),
                        sum(len(predicates) for _, predicates in corpus),
                        self.trie.node_count, self.trie.node_count - before)
        self.growth.append(growth)
        return growth

    # -- reading -----------------------------------------------------------
    def identify(self, wanted) -> Identification:
        """Walk down the trie until the description is exhausted.

        A predicate on the path that the description does not name is walked
        through, not refused -- a black beagle is still a beagle -- and once
        every wanted predicate has been met, everyone stored beneath fits.
        """
        wanted = frozenset(wanted)
        found: set[str] = set()
        reached: list[tuple] = []
        visited = 0
        stack = [(ROOT, frozenset())]
        while stack:
            node, have = stack.pop()
            visited += 1
            if have >= wanted:
                path = self.trie.path(node)
                reached.append(path)
                found.update(self.members.get(path, ()))
                continue
            for symbol, child in self.trie.children(node).items():
                stack.append((child, have | (frozenset({symbol}) & wanted)))
        return Identification(sorted(wanted),
                              sorted(found, key=self.individuals.index),
                              visited, reached)

    def as_dict(self) -> dict:
        paths = dict(self.plan)
        last = self.growth[-1] if self.growth else None

        def facts(node: str) -> list:
            return [{"relation": fact.relation, "object": fact.object,
                     "said": self.said.get((node, fact.relation,
                                            fact.object), ""),
                     "bound": sorted(self.bound.get(
                         (node, fact.relation, fact.object), ()))}
                    for fact in self.facts.get(node, [])]

        taught = [node for node in dict.fromkeys(list(self.edges)
                                                 + list(self.facts))
                  if node not in self.individuals
                  and (self.edges.get(node) or self.facts.get(node))]
        return {"nodes": self.trie.node_count,
                "cells": last.cells if last else 0,
                "individuals": [
                    {"id": individual, "parent": self.parent[individual],
                     "path": list(paths.get(individual, ())),
                     "facts": facts(individual)}
                    for individual in self.individuals],
                "taught": [
                    {"node": node, "episodic": node in self.kinds.values(),
                     "parents": list(self.edges.get(node, [])),
                     "facts": facts(node)}
                    for node in taught],
                "withdrawn": [one.as_dict() for one in self.withdrawn]}


class EpisodicReasoner(Reasoner):
    """v687's `Reasoner`, reading episodic memory and the store as one.
    Shares the base reasoner's connection and caches."""

    def __init__(self, base: Reasoner, memory: EpisodicMemory) -> None:
        # Deliberately not `super().__init__`: that opens a second
        # connection to the store, and this is the same reasoner with two
        # lookups extended.
        self.__dict__.update(base.__dict__)
        self.memory = memory
        self._individual_only = False

    def parents_of(self, concept: str) -> list[str]:
        taught = list(self.memory.edges.get(concept, []))
        if self.memory.episodic_only(concept):
            return taught
        return taught + [parent for parent in super().parents_of(concept)
                         if parent not in taught]

    def facts_of(self, concept: str, relation: str | None = None) -> list:
        group = set(rules.family(relation)) if relation else None
        # Copies: `verify` writes distance and decayed confidence onto the
        # facts it is handed, and these are the memory's own rows. Told facts
        # come first, as the store's own order would put a confidence of 1.
        told = [Fact(fact.concept, fact.relation, fact.object, fact.source,
                     fact.confidence, False)
                for fact in self.memory.facts.get(concept, [])
                if group is None or fact.relation in group]
        if self.memory.episodic_only(concept):
            return told
        return told + super().facts_of(concept, relation)

    def gloss(self, concept: str) -> str | None:
        if concept in self.memory.individuals:
            return (f"an individual under "
                    f"{self.memory.parent[concept] or 'no known kind'}")
        if self.memory.episodic_only(concept):
            return "a kind taught in this conversation"
        return super().gloss(concept)

    def too_broad(self, concept: str) -> bool:
        return (False if self.memory.episodic_only(concept)
                else super().too_broad(concept))

    def ascend(self, concept: str):
        for node, distance, parents in super().ascend(concept):
            yield node, distance, parents
            if self._individual_only:
                return

    def classify(self, concept: str, target_lemma: str) -> Answer:
        """R1 as v687 has it, plus the kinds this conversation taught.

        v687 finds a target through the store's lemma table, where `wemble`
        is not; a taught kind is found by walking up to its node instead.
        """
        taught = self.memory.kinds.get((target_lemma or "").strip().lower())
        if taught is None:
            return super().classify(concept, target_lemma)
        answer = Answer(question="", verdict="UNKNOWN", concept=concept,
                        concept_gloss=self.gloss(concept))
        steps = answer.steps
        for node, distance, parents in self.ascend(concept):
            answer.chain.append(node)
            steps.append(Step(len(steps), "ascend" if distance else "check",
                              node, distance, "R1",
                              f"Generalise to {name_of(node)}." if distance
                              else f"Start from {name_of(node)}.",
                              parents=parents))
            if node == taught:
                fact = Fact(concept, "is_a", node, TOLD, 1.0, False, distance)
                answer.verdict = "VERIFIED"
                answer.evidence.append(fact)
                steps.append(Step(len(steps), "match", node, distance, "R1",
                                  f"{name_of(node)} is a kind taught here, "
                                  f"{distance} step(s) up. Subsumption is "
                                  f"transitive, so yes.",
                                  matched=fact.as_dict()))
                return answer
        answer.note = (f"“{target_lemma}” is a kind taught here, and it is "
                       f"not among the ancestors of {name_of(concept)}. "
                       f"Absent, not false.")
        return answer

    def verify(self, concept: str, relation: str, target: str, matcher):
        quality = (concept in self.memory.individuals
                   and relation in QUALITIES)
        self._individual_only = quality
        try:
            answer = super().verify(concept, relation, target, matcher)
        finally:
            self._individual_only = False
        if quality and answer.verdict == "UNKNOWN":
            kind = name_of(self.memory.parent.get(concept)) or "its kind"
            answer.steps.append(Step(
                len(answer.steps), "stop", concept, 0, "E1",
                f"`{relation}` is a quality, and a quality does not descend "
                f"from {kind} to one of them."))
            answer.note = (
                f"Nothing was said of this one being “{target}”. What holds "
                f"of {kind} in general is a tendency of the kind, and E1 does "
                f"not let it descend onto an individual.")
        return answer
