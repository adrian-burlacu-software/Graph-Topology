"""The world, the actions that change it, and problems to solve in it.

Everything before v691 was knowledge: what is true of dogs, what a story
said, what follows from what. A world is different in one way that turns out
to matter everywhere -- **it changes, and only by acting**. `gives` in the
executive means *I now know X*; here an action's `adds` mean *the world is
now X*, and its `deletes` mean something that was true is not any more. A
knowledge base only ever grows. A world does not.

That is why the blocks world is the first thing v691 builds and why it is
built with no reader at all: it is the smallest domain where deletes are
load-bearing. Picking a block up makes it not-on-the-table; stacking it on
another makes that other one not-clear. A planner that ignores deletes will
happily produce a sequence that cannot be executed, and one that handles
them has to interleave goals rather than finish them in turn. Both show up
in four blocks, so a failure here is the architecture and not the parser.

## Facts are strings

`on a b`, `clear a`, `table a`, `held a`, `empty`. A string rather than a
tuple because a fact is going to *be a slot in working memory* (see
`acting.Situation`), and the executive sorts its slot names when it keys a
chunk -- a set mixing strings and tuples raises there rather than here.

## Two worlds, on purpose

`World` is the real one: its `do` is committed, it records an `effect` so
nothing can change it unseen, and it is the only thing an agent is ever
scored against. The model the planner searches in is a separate, cheap,
throwaway set of facts (`acting.Situation`). Keeping them apart is the whole
point of `DESIGN.md` §2: you can imagine putting the water back and you
cannot pour it back.
"""
from __future__ import annotations

import collections
import itertools
import random
from dataclasses import dataclass

from research.v687.executive import effect

#: The store name every operator that acts must declare in its `effects`.
WORLD = "world"


@dataclass(frozen=True)
class Action:
    """One ground action: what must hold, what becomes true, what stops.

    Ground, not schematic. `stack a b` and `stack a c` are two actions, and
    the domain hands over all of them. The executive's operators carry no
    variables and bind nothing, so grounding here is what lets an action be
    an `Operator` with no translation layer at all -- its `needs` are the
    preconditions and its `gives` are the adds. What has no counterpart
    there is `deletes`, and that absence is the finding of v691a.
    """

    name: str
    needs: frozenset
    adds: frozenset
    deletes: frozenset

    def holds_in(self, facts) -> bool:
        return self.needs <= facts

    def on(self, facts: frozenset) -> frozenset:
        """The state this action leaves, without asking whether it applies.

        Deletes first, then adds: `stack a b` deletes `clear b` and adds
        `clear a`, and an action that deleted after adding would undo its
        own work wherever the two sets touch.
        """
        return (facts - self.deletes) | self.adds

    def __str__(self) -> str:
        return self.name


class World:
    """The real world: a set of facts changed only by `do`.

    Every change is announced with `effect`, so an operator that acts has to
    declare `WORLD` in its `effects` or the executive refuses it -- the same
    guard v687 put on the store, now on the thing the agent moves. The
    `undo` handed to `effect` is the state this action replaced, which is
    honest here and is the exact thing that stops being honest the moment
    the world is anything but blocks (`DESIGN.md` §2).
    """

    def __init__(self, facts) -> None:
        self.facts = frozenset(facts)
        #: every action done, in order: what actually happened
        self.did: list[Action] = []

    def holds(self, fact: str) -> bool:
        return fact in self.facts

    def can(self, action: Action) -> bool:
        return action.holds_in(self.facts)

    def do(self, action: Action) -> bool:
        """Act. False, and nothing changes, when the action does not apply:
        the world refuses, rather than the agent checking first, because an
        agent that could only ever act on a world it had modelled correctly
        would never be surprised."""
        if not self.can(action):
            return False
        before = self.facts
        self.facts = action.on(self.facts)
        self.did.append(action)
        effect(WORLD, action.name, lambda: setattr(self, "facts", before))
        return True

    def solved(self, wanted) -> bool:
        return set(wanted) <= self.facts

    def towers(self) -> list:
        """The blocks as stacks, bottom first -- how a person would read it."""
        above = {}
        for fact in self.facts:
            parts = fact.split()
            if parts[0] == "on":
                above[parts[2]] = parts[1]
        out = []
        for fact in sorted(self.facts):
            parts = fact.split()
            if parts[0] == "table":
                stack, block = [parts[1]], parts[1]
                while block in above:
                    block = above[block]
                    stack.append(block)
                out.append(stack)
        return out


# -- the blocks domain -----------------------------------------------------

def start_of(towers) -> frozenset:
    """The facts of a configuration given as stacks, bottom block first."""
    facts = {"empty"}
    for stack in towers:
        facts.add(f"table {stack[0]}")
        for lower, upper in zip(stack, stack[1:]):
            facts.add(f"on {upper} {lower}")
        facts.add(f"clear {stack[-1]}")
    return frozenset(facts)


def blocks(names) -> list:
    """Every ground action over these blocks.

    Four schemas, so 2n + 2n(n-1) actions: twenty for three blocks, fifty-six
    for five. Grounding the whole domain up front is what a STRIPS planner
    of the period did and what makes the operators need no binding; it is
    also why this does not scale past a handful of blocks, which is fine,
    because nothing here is a scaling claim.
    """
    out = []
    for one in names:
        out.append(Action(f"take {one}",
                          frozenset({f"clear {one}", f"table {one}",
                                     "empty"}),
                          frozenset({f"held {one}"}),
                          frozenset({f"clear {one}", f"table {one}",
                                     "empty"})))
        out.append(Action(f"drop {one}",
                          frozenset({f"held {one}"}),
                          frozenset({f"clear {one}", f"table {one}",
                                     "empty"}),
                          frozenset({f"held {one}"})))
    for one, other in itertools.permutations(names, 2):
        out.append(Action(f"stack {one} {other}",
                          frozenset({f"held {one}", f"clear {other}"}),
                          frozenset({"empty", f"clear {one}",
                                     f"on {one} {other}"}),
                          frozenset({f"held {one}", f"clear {other}"})))
        out.append(Action(f"unstack {one} {other}",
                          frozenset({f"clear {one}", f"on {one} {other}",
                                     "empty"}),
                          frozenset({f"held {one}", f"clear {other}"}),
                          frozenset({f"clear {one}", f"on {one} {other}",
                                     "empty"})))
    return out


@dataclass
class Problem:
    """A start, a goal, and the actions available."""

    name: str
    names: tuple
    start: frozenset
    goal: frozenset

    @property
    def actions(self) -> list:
        return blocks(self.names)

    def world(self) -> World:
        return World(self.start)


def problem(name: str, names, towers, goal) -> Problem:
    return Problem(name, tuple(names), start_of(towers), frozenset(goal))


#: The fixed suite. `sussman` is first because it is the reason this file
#: exists: achieving `on a b` and then `on b c` in turn undoes the first, so
#: a planner that finishes one goal before starting the next cannot solve it
#: in fewer than the optimal six steps, and a careless one cannot solve it
#: at all. Everything after it is a size ladder, so the numbers say where
#: the method stops rather than only whether it works.
SUITE = [
    problem("one step", "ab", [["a"], ["b"]], ["on a b"]),
    problem("undo a tower", "ab", [["a", "b"]], ["on a b"]),
    problem("three in a row", "abc", [["a"], ["b"], ["c"]],
            ["on a b", "on b c"]),
    problem("sussman", "abc", [["c", "a"], ["b"]], ["on a b", "on b c"]),
    problem("invert three", "abc", [["a", "b", "c"]],
            ["on b c", "on a b"]),
    problem("four apart", "abcd", [["a"], ["b"], ["c"], ["d"]],
            ["on a b", "on b c", "on c d"]),
    problem("four inverted", "abcd", [["a", "b", "c", "d"]],
            ["on a b", "on b c", "on c d"]),
    problem("two towers", "abcd", [["a", "b"], ["c", "d"]],
            ["on b c", "on d a"]),
    problem("five apart", "abcde", [[one] for one in "abcde"],
            ["on a b", "on b c", "on c d", "on d e"]),
    problem("five inverted", "abcde", [["a", "b", "c", "d", "e"]],
            ["on a b", "on b c", "on c d", "on d e"]),
]


def sampled(count: int = 20, names: str = "abcd", seed: int = 0) -> list:
    """Random start and goal configurations, for numbers the fixed suite
    cannot give: it was chosen to be hard and so it says nothing about how
    often a method works."""
    rng = random.Random(seed)

    def configuration() -> list:
        order = list(names)
        rng.shuffle(order)
        towers, stack = [], [order[0]]
        for one in order[1:]:
            if rng.random() < 0.5:
                stack.append(one)
            else:
                towers.append(stack)
                stack = [one]
        towers.append(stack)
        return towers

    out = []
    for index in range(count):
        start, want = configuration(), configuration()
        goal = {f"on {upper} {lower}" for stack in want
                for lower, upper in zip(stack, stack[1:])}
        if not goal or start_of(start) >= goal:
            continue
        out.append(problem(f"sampled {index}", names, start, sorted(goal)))
    return out


# -- the oracle ------------------------------------------------------------

def shortest(problem: Problem, ceiling: int = 100000) -> list | None:
    """The shortest sequence of actions from start to goal, by breadth
    first search over states -- what a plan is measured against.

    Exhaustive and exponential, which is the point of an oracle: it says
    what the answer was, so the executive's plan can be called optimal,
    longer, or absent rather than merely present. Five blocks is 501 states
    and finishes instantly; it is not meant to go further.
    """
    actions = problem.actions
    start = problem.start
    if problem.goal <= start:
        return []
    seen = {start}
    queue = collections.deque([(start, [])])
    while queue and len(seen) < ceiling:
        facts, path = queue.popleft()
        for action in actions:
            if not action.holds_in(facts):
                continue
            after = action.on(facts)
            if after in seen:
                continue
            if problem.goal <= after:
                return path + [action]
            seen.add(after)
            queue.append((after, path + [action]))
    return None
