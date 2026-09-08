"""What is worth asking, and what it is worth asking about.

Curiosity over a trie with no attention is a breadth-first crawl of the whole
ontology: 11,707 nodes, every question defensible on its own, none of them
about anything. Attention is the bound. Three scores multiply into one rank,
and each answers a different objection:

    salience   is anything here about what was just said?
    gain       would the answer tell the trie apart, or merely fill a slot?
    urgency    is something blocked, or is this only interesting?

Salience decays. That is the only dynamic quantity in the system -- the
subsystem audit lists activation dynamics as the deepest missing piece, and it
arrives here as a side effect of having to bound curiosity rather than as a
feature anybody set out to build.

On `gain`: `identifiability.py` measures identification depth over the same
trie and calls it twenty questions. Its `anti_coverage` ordering -- rarest
predicate first -- is the greedy approximation to what is computed here.
The exact quantity is the expected reduction in log2 of the candidate set, and
it is maximised by a predicate that splits the field in half, not by one
almost nobody carries. A singleton predicate identifies one individual and
learns nothing about the other 540.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

#: How much of a concept's activation survives one cycle. At 0.6, something
#: mentioned once is still faintly live three cycles later and gone by six,
#: which is about the depth a single utterance runs to.
DECAY = 0.6

#: Below this a concept is not under attention at all and generates nothing.
FLOOR = 0.05

#: What each kind of incompleteness is worth pursuing. A blocked question
#: outranks an interesting one: `word` gaps make everything downstream
#: meaningless, so they are the only thing that should ever pre-empt a doubt.
URGENCY = {
    "word": 1.00,           # nothing can proceed until this is named
    "construction": 0.90,   # the question itself has to be repaired
    "sense": 0.85,          # a reading has to be chosen before facts mean anything
    "conflict": 0.80,       # the parts disagree; find out which part is wrong
    "coverage": 0.55,       # a real absence, informative but not blocking
    "doubt": 0.65,          # an answer that should not be trusted this far
    "curiosity": 0.30,      # nothing is wrong; this is only worth knowing
}

#: The rank is a weighted product, not a sum, so a zero in any one of the
#: three kills the question. That is deliberate: a maximally informative
#: question about something nobody mentioned is not a question worth asking.
WEIGHTS = {"salience": 1.0, "gain": 0.8, "urgency": 1.2}


@dataclass
class Activation:
    """Which concepts are live, and how strongly.

    One table per utterance. `bump` on mention, `decay` once per cycle, and
    the floor does the pruning: a concept that stops being mentioned stops
    generating questions within a few cycles without anything deciding to
    forget it.
    """

    table: dict[str, float] = field(default_factory=dict)
    #: cycle number -> what was bumped, for the page to replay.
    history: list[tuple[int, str, float]] = field(default_factory=list)

    def bump(self, concept: str, amount: float = 1.0, cycle: int = 0) -> None:
        concept = (concept or "").strip().lower()
        if not concept:
            return
        self.table[concept] = min(2.0, self.table.get(concept, 0.0) + amount)
        self.history.append((cycle, concept, self.table[concept]))

    def decay(self, rate: float = DECAY) -> None:
        for concept in list(self.table):
            self.table[concept] *= rate
            if self.table[concept] < FLOOR:
                del self.table[concept]

    def salience(self, concept: str) -> float:
        return self.table.get((concept or "").strip().lower(), 0.0)

    def live(self) -> list[tuple[str, float]]:
        """Concepts under attention, strongest first."""
        return sorted(self.table.items(), key=lambda kv: (-kv[1], kv[0]))

    def as_dict(self) -> dict:
        return {"table": dict(self.table), "live": self.live()}


class Curiosity:
    """Information gain over the predicate trie.

    Built once from `Profiles.plan` -- the same 541 individuals and 11,707
    nodes the reasoner already stores -- and read per question. Nothing here
    touches the store: it is arithmetic over the plan v687 computed.
    """

    def __init__(self, plan) -> None:
        #: predicate -> the individuals carrying it
        self.holders: dict[str, frozenset[str]] = {}
        holders: dict[str, set[str]] = defaultdict(set)
        #: individual -> its predicates, for sibling comparison
        self.carried: dict[str, frozenset[str]] = {}
        for individual, path in plan:
            name = str(individual)
            self.carried[name] = frozenset(str(p) for p in path)
            for predicate in path:
                holders[str(predicate)].add(name)
        self.holders = {p: frozenset(who) for p, who in holders.items()}
        self.universe = frozenset(self.carried)

    # -- the twenty-questions quantity ------------------------------------
    def gain(self, predicate: str, candidates: frozenset[str] | None = None
             ) -> float:
        """Expected bits of the candidate set this question removes.

        1.0 for a predicate that splits the field exactly in half, 0.0 for one
        every candidate carries or none does. Normalised so the page can show
        it as a bar without knowing how many candidates there were.
        """
        pool = candidates if candidates is not None else self.universe
        total = len(pool)
        if total < 2:
            return 0.0
        yes = len(pool & self.holders.get(predicate, frozenset()))
        no = total - yes
        if not yes or not no:
            return 0.0
        before = math.log2(total)
        after = (yes / total) * math.log2(yes) + (no / total) * math.log2(no)
        # The best any single question can do on this pool is one bit; divide
        # by that so `gain` is comparable across pools of different sizes.
        return (before - after)

    def rivals(self, concept: str, limit: int = 24) -> frozenset[str]:
        """The individuals a question about this concept is trying to tell it
        from: the ones sharing most of its predicates.

        Without a pool, gain is computed against all 541 and every question
        looks equally uninformative. The pool is what makes `can it swim`
        interesting among dogs and dull among all artifacts.
        """
        mine = self.carried.get(concept)
        if not mine:
            return self.universe
        scored = sorted(
            ((len(mine & other), name)
             for name, other in self.carried.items() if name != concept),
            reverse=True)
        return frozenset([concept] + [name for _, name in scored[:limit]])

    def candidates(self, concept: str, limit: int = 12,
                   pool: frozenset[str] | None = None
                   ) -> list[tuple[str, float]]:
        """The predicates worth asking about this concept, best first.

        Predicates it already carries are skipped -- the answer is recorded
        and asking it learns nothing. What is left is exactly the trie's
        growth frontier: a question here either confirms an absence or
        allocates a node, which is the paper's growth signal (Section 15).

        `pool` is the field the question is trying to split. It has to be
        passed in for a concept the trie has never heard of: `beagle` is not
        one of the 541, so its own rivals are all 541, and against that
        everything scores the same and the ranking says nothing.
        """
        pool = pool if pool is not None else self.rivals(concept)
        mine = self.carried.get(concept, frozenset())
        scored = []
        for predicate, who in self.holders.items():
            if predicate in mine:
                continue
            if not (pool & who):
                continue
            scored.append((self.gain(predicate, pool), predicate))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [(predicate, score) for score, predicate in scored[:limit]]

    def knows(self, concept: str) -> bool:
        return concept in self.carried

    def novelty(self, concept: str) -> float:
        """1.0 when nothing about this concept is in the trie at all.

        `beagle` is not one of the 541, so every answer about it allocates.
        That is the growth signal used as a reason to ask rather than as a
        statistic reported afterwards.
        """
        return 0.0 if self.knows(concept) else 1.0


def rank(salience: float, gain: float, urgency: float) -> float:
    """One number from the three, as a weighted product.

    A product rather than a sum so that a zero anywhere is fatal. A maximally
    informative question about something nobody mentioned should not be asked,
    and a sum would let a big enough `gain` carry it.
    """
    return (max(salience, 0.0) ** WEIGHTS["salience"]
            * max(gain, 0.0) ** WEIGHTS["gain"]
            * max(urgency, 0.0) ** WEIGHTS["urgency"])
