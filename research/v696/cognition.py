"""The mechanisms of cognitive search, with no domain in them.

`PLAN.md` Phase 2 says every mechanism goes into the architecture, not into
a code module, and is turned back on the planner and the designer (Phase 3).
These are they, over nothing but *features* (hashable predicates),
*signatures* (what a candidate does, as a tuple) and *operators* (anything
with a key). `search.py` uses them for code; v691's planner and v694's
designer use the same classes.

    Recognizer    a trie of what was solved, keyed by its features in the
                  order most common first (the paper's Appendix 3); a new
                  problem walks it down and gets what solved the problems
                  it reaches -- the unique-answer shortcut when one is left
    Equivalence   the forward trie: a candidate whose signature has been
                  seen is access, not allocation, and is not grown again
                  (`v687.trie.PredicateTrie`'s growth signal)
    Attention     one queue across every operator: the candidates whose
                  parts already resemble the answer first
    Control       the executive's learned control: how likely an operator
                  is, given the features, from what solved problems used
                  (the EFSM's classifier, naive Bayes here)
"""
from __future__ import annotations

import heapq
import itertools
import math
from collections import defaultdict

from research.v687.trie import ROOT, PredicateTrie


class Recognizer:
    """What was solved, found again by what it was like."""

    def __init__(self) -> None:
        self.trie = PredicateTrie()
        self.counts: dict = defaultdict(int)
        self.stored: dict = defaultdict(list)
        self.size = 0

    def order(self, features) -> list:
        """Most common first: a coarse predicate near the root rules out
        most of what is stored at once."""
        return sorted(features, key=lambda one: (-self.counts[one], str(one)))

    def remember(self, features, solution) -> bool:
        """Store a solution under its features; True if that allocated --
        a problem unlike any before (the paper's growth signal)."""
        for one in features:
            self.counts[one] += 1
        node, allocated = ROOT, False
        for one in self.order(features):
            node, fresh = self.trie.ensure(node, one)
            allocated = allocated or fresh
        self.stored[node].append(solution)
        self.size += 1
        return allocated

    def recognise(self, features) -> list:
        """Walk down with the features that are there, skipping the ones
        that are not: what is stored at the node reached and beneath it,
        nearest first."""
        node = ROOT
        for one in self.order(features):
            child = self.trie.children(node).get(one)
            if child is not None:
                node = child
        out, level = [], [node]
        while level:
            below = []
            for here in level:
                out += self.stored.get(here, [])
                below += list(self.trie.children(here).values())
            level = below
        return out


class Equivalence:
    """The forward trie: one entry per distinct signature."""

    def __init__(self) -> None:
        self.trie = PredicateTrie()
        self.accessed = 0

    def admit(self, signature) -> bool:
        """True if the signature is new -- allocated; False if it is access
        to what is already there."""
        node, allocated = ROOT, False
        for index, one in enumerate(signature):
            node, fresh = self.trie.ensure(node, (index, one))
            allocated = allocated or fresh
        if not allocated:
            self.accessed += 1
        return allocated


class Attention:
    """A queue across every operator, most promising first."""

    def __init__(self) -> None:
        self.heap = []
        self.count = itertools.count()

    def offer(self, promise: float, item) -> None:
        heapq.heappush(self.heap, (-promise, next(self.count), item))

    def take(self, many: int) -> list:
        out = []
        while self.heap and len(out) < many:
            out.append(heapq.heappop(self.heap)[2])
        return out

    def __len__(self) -> int:
        return len(self.heap)


class Control:
    """Which operator, given the features: learned from what solved."""

    def __init__(self) -> None:
        self.used: dict = defaultdict(int)
        self.uses: dict = defaultdict(int)
        self.problems = 0

    def learn(self, features, operators) -> None:
        self.problems += 1
        for op in set(operators):
            self.uses[op] += 1
            for one in features:
                self.used[(one, op)] += 1

    def utility(self, op, features) -> float:
        total = self.uses[op]
        score = math.log((total + 1) / (self.problems + 2))
        for one in features:
            score += math.log((self.used[(one, op)] + 1) / (total + 2))
        return score

    def order(self, operators, features, key=lambda one: one) -> list:
        if not self.problems:
            return list(operators)
        return sorted(operators,
                      key=lambda one: -self.utility(key(one), features))
