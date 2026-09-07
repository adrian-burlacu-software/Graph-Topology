"""Finding the route between two concepts a question names, under a budget.

"What does a dog's owner need" names two things. Answering it means showing
that a dog *has* an owner before saying anything about owners -- otherwise the
answer is about owners in general and the dog was decoration.

Finding that route is where v684's search stops working. Going up the `is_a`
DAG the branching factor is 1.02 and 15 concepts is the whole search; across
facts it is 870 from `dog` alone and 8,064 at two hops. There is nothing to
control in the first case and no way to avoid controlling the second.

So this is a bounded best-first search with two budgets, both explicit:

    depth    how many hops a route may take
    breadth  how many frontier nodes survive each round

and a score deciding what survives. The score is not learned. Three signals
that are already in the ontology do the work:

    confidence   a hop through a 0.59 fact beats one through a 0.12 fact
    hubness      `person` reaches 3,476 concepts; routing through it proves
                 nothing, because everything routes through it
    kinship      how much taxonomic ancestry a node shares with the target,
                 counted only over ancestors specific enough to matter

Every expansion is recorded, so the budget's effect is visible rather than
asserted: `search()` reports what it expanded, what it pruned and what it
never looked at.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

from .graph import FactGraph, Hop


@dataclass
class Route:
    """A path from one concept to another, and the facts that license it."""
    hops: list[Hop] = field(default_factory=list)
    cost: float = 0.0

    @property
    def target(self) -> str:
        return self.hops[-1].target if self.hops else ""

    def concepts(self) -> list[str]:
        return ([self.hops[0].source] if self.hops else []) + \
               [h.target for h in self.hops]

    def as_dict(self) -> dict[str, Any]:
        return {"hops": [h.as_dict() for h in self.hops],
                "cost": round(self.cost, 4),
                "concepts": self.concepts()}

    def __str__(self) -> str:
        if not self.hops:
            return "(empty)"
        parts = [self.hops[0].source.rsplit(".", 2)[0]]
        for hop in self.hops:
            parts.append(f"--{hop.relation}[{hop.text}]--> "
                         f"{hop.target.rsplit('.', 2)[0]}")
        return " ".join(parts)


@dataclass
class SearchReport:
    """What the budget actually did, so its cost can be read off."""
    routes: list[Route] = field(default_factory=list)
    expanded: int = 0
    generated: int = 0
    pruned: int = 0
    depth_reached: int = 0
    exhausted: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"routes": [r.as_dict() for r in self.routes],
                "expanded": self.expanded, "generated": self.generated,
                "pruned": self.pruned, "depth_reached": self.depth_reached,
                "exhausted": self.exhausted}


class Bridge:
    """Bounded best-first search over the fact graph."""

    #: Ancestors this general are shared by everything, so sharing one says
    #: nothing about kinship. `entity` is an ancestor of 82,114 concepts.
    GENERIC_ANCESTOR_DEGREE = 8000

    def __init__(self, graph: FactGraph, depth: int = 3, breadth: int = 60):
        self.graph = graph
        self.depth = depth
        self.breadth = breadth
        self._ancestors: dict[str, set[str]] = {}

    # -- the three signals -------------------------------------------------
    def ancestors(self, concept: str, limit: int = 8) -> set[str]:
        cached = self._ancestors.get(concept)
        if cached is not None:
            return cached
        seen: set[str] = set()
        frontier = [concept]
        for _ in range(limit):
            nxt: list[str] = []
            for node in frontier:
                for parent in self.graph.parents.get(node, ()):
                    if parent not in seen:
                        seen.add(parent)
                        nxt.append(parent)
            if not nxt:
                break
            frontier = nxt
        self._ancestors[concept] = seen
        return seen

    def kinship(self, concept: str, target: str) -> float:
        """Shared ancestry, counting only ancestors specific enough to mean it.

        Every pair of concepts shares `entity`; the question is whether they
        share anything narrower. Ancestors with 8,000+ descendants are dropped
        before counting, which is the same line R12 draws for the same reason.
        """
        if concept == target:
            return 1.0
        mine = self.ancestors(concept)
        theirs = self.ancestors(target) | {target}
        if not mine or not theirs:
            return 0.0
        shared = {a for a in mine & theirs
                  if len(self.graph.children.get(a, ())) < 60}
        return len(shared) / math.sqrt(len(mine) * len(theirs))

    def hop_cost(self, hop: Hop, target: str) -> float:
        """Lower is better. Confidence and hubness, before any lookahead."""
        cost = -math.log(max(hop.confidence, 0.01))
        degree = self.graph.degree(hop.target)
        if degree >= self.graph.HUB_DEGREE:
            cost += 4.0                      # routing through everything
        else:
            cost += math.log1p(degree) / 6.0
        return cost

    def priority(self, route: Route, node: str, target: str) -> float:
        """Best-first key: cost so far, less how promising the node looks."""
        return route.cost - 3.0 * self.kinship(node, target)

    # -- the search --------------------------------------------------------
    def search(self, source: str, target: str, want: int = 3) -> SearchReport:
        """Routes from `source` to `target` within the depth/breadth budget."""
        report = SearchReport()
        if source == target:
            return report
        best_cost: dict[str, float] = {source: 0.0}
        frontier: list[tuple[float, int, Route]] = [(0.0, 0, Route())]
        counter = 1
        for level in range(self.depth):
            if not frontier or len(report.routes) >= want:
                break
            report.depth_reached = level + 1
            # breadth budget: only the most promising survive to be expanded
            frontier.sort(key=lambda item: item[0])
            if len(frontier) > self.breadth:
                report.pruned += len(frontier) - self.breadth
                frontier = frontier[:self.breadth]
            nxt: list[tuple[float, int, Route]] = []
            for _, _, route in frontier:
                node = route.target or source
                report.expanded += 1
                for hop in self.graph.hops(node):
                    report.generated += 1
                    if hop.target in {h.source for h in route.hops} or \
                            hop.target == source:
                        continue                      # no going back
                    cost = route.cost + self.hop_cost(hop, target)
                    if hop.target == target:
                        report.routes.append(Route(route.hops + [hop], cost))
                        continue
                    if level + 1 >= self.depth:
                        continue
                    if cost >= best_cost.get(hop.target, float("inf")):
                        continue
                    best_cost[hop.target] = cost
                    counter += 1
                    extended = Route(route.hops + [hop], cost)
                    nxt.append((self.priority(extended, hop.target, target),
                                counter, extended))
            frontier = nxt
        report.exhausted = not frontier
        report.routes.sort(key=lambda r: (len(r.hops), r.cost))
        report.routes = report.routes[:want]
        return report
