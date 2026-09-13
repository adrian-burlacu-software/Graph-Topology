"""Walks over the graph: one closure and one path, for every relation.

R1 walks `is_a` up from a concept, T2 walks `before` between occurrences, S2
walks `bigger` between individuals, and S4 walks the compass as a map. Each
had its own breadth-first search. They are these two, told how to step
(`v690/DESIGN.md` §4.5); what makes a relation walkable at all is its row in
`links.py`.
"""
from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Iterator


def levels(start: Hashable, neighbours: Callable[[Hashable], Iterable],
           max_depth: int | None = None, max_nodes: int | None = None
           ) -> Iterator[tuple[int, list]]:
    """Breadth-first from `start`: (distance, the nodes first reached at it).

    Nearest first, each node once. The walk stops when a level adds nothing,
    at `max_depth`, or once more than `max_nodes` have been seen -- before the
    level that crossed the bound is yielded, so a pathological branch cannot
    hang a caller. Lazy: a level is computed only when it is asked for.
    """
    seen = {start}
    frontier = [start]
    yield 0, [start]
    distance = 0
    while max_depth is None or distance < max_depth:
        distance += 1
        reached = []
        for node in frontier:
            for other in neighbours(node):
                if other not in seen:
                    seen.add(other)
                    reached.append(other)
        if not reached or (max_nodes is not None and len(seen) > max_nodes):
            return
        yield distance, reached
        frontier = reached


def path(start: Hashable, goal: Hashable,
         steps: Callable[[Hashable], Iterable[tuple]]) -> list | None:
    """The shortest walk from `start` to `goal`, or None when there is none.

    `steps(node)` gives (label, node reached) for every edge out of a node;
    the walk is [(label, node reached)] in order, so what each step went
    along is kept with where it went. A node is not a walk to itself.
    """
    parent: dict = {start: None}
    frontier = [start]
    while frontier:
        reached = []
        for node in frontier:
            for label, other in steps(node):
                if other in parent:
                    continue
                parent[other] = (node, label)
                if other == goal:
                    route, at = [], other
                    while parent[at] is not None:
                        before, used = parent[at]
                        route.append((used, at))
                        at = before
                    return route[::-1]
                reached.append(other)
        frontier = reached
    return None
