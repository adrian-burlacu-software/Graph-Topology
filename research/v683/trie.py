"""Allocation-counting predicate trie.

The trie is the paper's router: predicates are the alphabet, individuals are
the semantic goal nodes (Figure 20). Node identity is an integer, never a
string, so two branches may carry the same predicate symbol without collapsing
-- Figure 20 shows exactly that, with `Spots` appearing under `Big` and again
under the root.

Every structural change goes through `ensure`, which reports whether it
allocated. That flag is the paper's growth signal: "the topographical growth of
tries is driven by allocation" (Section 15).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Iterator

Predicate = Hashable
ROOT = 0


@dataclass(frozen=True)
class Insertion:
    """What storing one individual cost."""

    individual: Hashable
    node: int
    depth: int
    allocated: int
    reused: int


class PredicateTrie:
    """A trie whose alphabet is the predicate set of an ontology."""

    def __init__(self) -> None:
        self._children: dict[int, dict[Predicate, int]] = {ROOT: {}}
        self._parent: dict[int, int | None] = {ROOT: None}
        self._symbol: dict[int, Predicate | None] = {ROOT: None}
        self._terminal: dict[int, list[Hashable]] = {}
        self._next_id = ROOT
        self.allocated = 0
        self.reused = 0

    # -- structure -------------------------------------------------------
    def ensure(self, node: int, symbol: Predicate) -> tuple[int, bool]:
        """Return the child of `node` under `symbol`, allocating if absent."""
        existing = self._children[node].get(symbol)
        if existing is not None:
            self.reused += 1
            return existing, False
        self._next_id += 1
        child = self._next_id
        self._children[node][symbol] = child
        self._children[child] = {}
        self._parent[child] = node
        self._symbol[child] = symbol
        self.allocated += 1
        return child, True

    def insert(self, individual: Hashable, predicates: Iterable[Predicate]) -> Insertion:
        """Store one individual along an already-ordered predicate path."""
        node, allocated, reused, depth = ROOT, 0, 0, 0
        for predicate in predicates:
            node, is_new = self.ensure(node, predicate)
            allocated += is_new
            reused += not is_new
            depth += 1
        self._terminal.setdefault(node, []).append(individual)
        return Insertion(individual, node, depth, allocated, reused)

    def lookup(self, predicates: Iterable[Predicate]) -> int | None:
        """Walk an ordered predicate path; None when the route does not exist."""
        node = ROOT
        for predicate in predicates:
            node = self._children[node].get(predicate)
            if node is None:
                return None
        return node

    def individuals_at(self, node: int) -> tuple[Hashable, ...]:
        return tuple(self._terminal.get(node, ()))

    def path(self, node: int) -> tuple[Predicate, ...]:
        """The predicate sequence from the root down to `node`."""
        route: list[Predicate] = []
        while node != ROOT:
            route.append(self._symbol[node])
            node = self._parent[node]  # type: ignore[assignment]
        return tuple(reversed(route))

    def children(self, node: int) -> dict[Predicate, int]:
        return dict(self._children[node])

    # -- measurement -----------------------------------------------------
    @property
    def node_count(self) -> int:
        """Allocated nodes, excluding the root, which every trie has for free."""
        return self._next_id

    @property
    def terminal_nodes(self) -> int:
        return len(self._terminal)

    def depths(self) -> Iterator[int]:
        """Access depth for each stored individual."""
        for node, members in self._terminal.items():
            depth = len(self.path(node))
            for _ in members:
                yield depth

    def edges(self) -> Iterator[tuple[int, Predicate, int]]:
        for parent, children in self._children.items():
            for symbol, child in children.items():
                yield parent, symbol, child
