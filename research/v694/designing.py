"""Designing a way to a goal in the open world: propose, fill, imagine,
learn.

This is v693's designer with the maths taken out, which is the point of it:
**the loop never knew it was doing mathematics.** v691's agent plans in the
world of one draft, whose facts are what the clause is like
(`goals.features`: a state, of a living thing, whose verb takes an
instrument...), and whose moves are the ways (`ways.WAYS`), each declared as
bringing about `designed d`. What a move does is fill its hole from the
store and the scene, and **imagine the result**: the steps are done in a
`W.Imagined` world from the scene, each must apply when its turn comes, and
the clause must hold at the end (`checked`). A way that finds nothing, or
whose steps do not come to the goal, leaves the draft undesigned -- v691's
surprise, and the learner's evidence.

Then **Occam**, as v693 had it, with a different count: not unknowns but
what a way relies on (`Candidate.cost`) -- nobody else, nothing supposed,
fewest steps, then the store's support. A way that supposes a knife is not
settled for while one that uses the scissors on the table is untried.

A goal of several clauses is designed clause by clause, each in the world
the ways before it leave: a knife fetched to cut the rope is held when the
string is to be cut too, and costs nothing the second time.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from research.v691 import acting, world as W
from research.v694 import ways as Ways
from research.v694.goals import Clause, Goal, features

D = "d"
DESIGNED = f"designed {D}"


def fact(name: str) -> str:
    return f"{name} {D}"


def verb(way: Ways.Way) -> str:
    return f"fit-{way.name}"


def moves(ways) -> tuple:
    return tuple(W.Action(f"{verb(way)} {D}",
                          frozenset(fact(one) for one in way.needs),
                          frozenset({DESIGNED}), frozenset())
                 for way in ways)


def checked(candidate: Ways.Candidate, facts, learner=None):
    """The facts the candidate's steps leave, done in an imagined world
    from `facts` -- or None when a step does not apply when its turn
    comes, or the clause does not hold at the end. What has been learned
    about acting (`lessons`) is folded into the steps first: a lesson that
    a broken fridge keeps nothing cold stops the step, not the store."""
    steps = list(candidate.steps)
    if learner is not None:
        steps = learner.applied(steps)
    imagined = W.Imagined(frozenset(facts))
    for step in steps:
        if not imagined.do(step):
            return None
    if candidate.clause.literal not in imagined.facts:
        return None
    return imagined.facts


def fit(way: Ways.Way, goal: Goal, clause: Clause, facts, learner=None,
        tried=None):
    """The way's cheapest candidate that checks, with the facts it leaves,
    or None. Every candidate imagined is recorded in `tried`."""
    best = None
    for candidate in way.build(goal, clause, facts):
        after = checked(candidate, facts, learner)
        if tried is not None:
            tried.append((way.name, candidate, after is not None))
        if after is None:
            continue
        if best is None or candidate.cost() < best[0].cost():
            best = (candidate, after)
    return best


class Draft(W.Imagined):
    """The world of one clause's design. Its facts are the clause's
    features; a move fits a way and, if a candidate checks, the draft is
    designed."""

    def __init__(self, goal: Goal, clause: Clause, facts, ways,
                 learner=None) -> None:
        self.goal, self.clause, self.scene = goal, clause, frozenset(facts)
        self.ways = {verb(way): way for way in ways}
        self.learner = learner
        #: (way name, candidate, whether it checked)
        self.tried: list = []
        #: way names fitted, in order, with whether each found a way
        self.fitted: list = []
        self.best = None
        super().__init__(frozenset(fact(one) for one in
                                   features(goal, clause)))

    def can(self, action: W.Action) -> bool:
        return action.holds_in(self.facts)

    def do(self, action: W.Action) -> bool:
        if not self.can(action):
            return False
        way = self.ways[action.name.split()[0]]
        found = fit(way, self.goal, self.clause, self.scene, self.learner,
                    self.tried)
        self.fitted.append((way.name, found is not None))
        self.did.append(action)
        if found is not None and (self.best is None or
                                  found[0].cost() < self.best[0].cost()):
            self.best = found
            self.facts = self.facts | {DESIGNED}
        return True


def surprised(action, expected, before, now) -> bool:
    wanted = frozenset(action.adds) - frozenset(before)
    return bool(wanted - frozenset(now))


def least(way: Ways.Way) -> tuple:
    """The least a way could cost: whoever it must rely on, nothing
    supposed, one step."""
    return (way.helpers, 0, 1)


def _simpler(draft: Draft, ways, learner) -> None:
    """Not settled for while something simpler is untried: every way
    that could cost less than the one found is fitted too, and the
    cheapest that checks is kept."""
    tried = {name for name, _ in draft.fitted}
    doubted = {way.name for way in held_back(ways, draft.facts, learner)}
    for way in ways:
        if draft.best is None:
            return
        if way.name in tried | doubted or \
                least(way) > draft.best[0].cost()[:3] or \
                not {fact(one) for one in way.needs} <= draft.facts:
            continue
        found = fit(way, draft.goal, draft.clause, draft.scene, learner,
                    draft.tried)
        draft.fitted.append((way.name, found is not None))
        if found is not None and found[0].cost() < draft.best[0].cost():
            draft.best = found
            if learner is not None:
                learner.worked(moves([way])[0], draft.facts, draft.facts)


CURIOSITY = 0.15


def held_back(ways, facts, learner) -> list:
    if learner is None:
        return []
    raw = moves(ways)
    applied = learner.applied(raw)
    return [way for way, bare, learned in zip(ways, raw, applied)
            if bare.holds_in(facts) and not learned.holds_in(facts)]


def _explore(way, draft: Draft, learner, why: str, explored: list) -> None:
    action = moves([way])[0]
    before = draft.facts
    had = draft.best
    draft.do(action)
    explored.append((way.name, why))
    if draft.best is not had and learner is not None:
        learner.worked(action, before, draft.facts)


@dataclass
class Designed:
    """One clause's design."""

    clause: Clause
    best: Ways.Candidate | None = None
    fitted: list = field(default_factory=list)
    tried: list = field(default_factory=list)
    explored: list = field(default_factory=list)


def design_clause(goal: Goal, clause: Clause, facts, learner=None,
                  order=None, curiosity: float = CURIOSITY,
                  rng=None) -> tuple:
    """(Designed, the facts its way leaves) for one clause -- v693's
    loop: one plan per run of the agent, a way that failed leaves the
    table, curiosity and a last resort for what lessons hold back, and
    Occam once something is found."""
    from research.v687.executive import Working
    ways = list(Ways.usable(features(goal, clause)))
    if order:
        rank = {name: index for index, name in enumerate(order)}
        ways.sort(key=lambda way: rank.get(way.name, len(rank)))
    draft = Draft(goal, clause, facts, ways, learner)
    target = frozenset({DESIGNED})
    explored: list = []
    rng = rng if rng is not None else random.Random(clause.literal)
    doubted = held_back(ways, draft.facts, learner)
    if doubted and rng.random() < curiosity:
        _explore(doubted[0], draft, learner, "curious", explored)
    left = [way for way in ways
            if way.name not in {name for name, _ in draft.fitted}]
    while left and draft.best is None:
        problem = W.Problem("design", draft.facts, target, moves(left))
        before = len(draft.fitted)
        report = acting.Attempt(name="design")
        agent = acting.agent(problem, draft, None, report, tries=1,
                             learner=learner, surprised=surprised)
        agent.run(Working(goal=f"design {clause.literal}"))
        if len(draft.fitted) == before:
            break
        failed = {name for name, ok in draft.fitted[before:] if not ok}
        left = [way for way in left if way.name not in failed]
    if draft.best is not None:
        _simpler(draft, ways, learner)
    if draft.best is None:
        tried = {name for name, _ in draft.fitted}
        for way in held_back(ways, draft.facts, learner):
            if way.name in tried:
                continue
            _explore(way, draft, learner, "last resort", explored)
            if draft.best is not None:
                break
    found = Designed(clause, draft.best[0] if draft.best else None,
                     draft.fitted, draft.tried, explored)
    return found, (draft.best[1] if draft.best else frozenset(facts))


@dataclass
class Design:
    goal: Goal
    parts: list = field(default_factory=list)

    @property
    def done(self) -> bool:
        return bool(self.parts) and all(one.best is not None
                                        for one in self.parts)

    @property
    def steps(self) -> list:
        return [step for one in self.parts if one.best
                for step in one.best.steps]

    def supposed(self) -> list:
        return [thing for one in self.parts if one.best
                for thing in one.best.supposed]

    def said(self) -> str:
        out = []
        for one in self.parts:
            if one.best is None:
                tried = ", ".join(dict.fromkeys(
                    name for name, _ in one.fitted)) or "nothing"
                out.append(f"I could not find a way to get "
                           f"{one.clause.said()} (tried: {tried})")
                continue
            text = one.best.text()
            if one.best.why:
                text += f" -- {one.best.why}"
            out.append(text)
        return "; ".join(out)


def design(goal: Goal, learner=None, order=None, curiosity=CURIOSITY,
           rng=None) -> Design:
    """A way to every clause of the goal, each designed in the world the
    ways before it leave."""
    facts = frozenset(goal.facts)
    out = Design(goal)
    for clause in goal.clauses:
        if clause.literal in facts:
            continue
        mine = order(goal, clause) if callable(order) else order
        found, facts = design_clause(goal, clause, facts, learner, mine,
                                     curiosity, rng)
        out.parts.append(found)
    return out
