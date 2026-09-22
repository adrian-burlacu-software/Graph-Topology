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

`on a b`, `at parcel london`, `carrying spanner`: a predicate and its
arguments, separated by spaces. A string rather than a tuple because a fact
is going to *be a slot in working memory* (see `acting.Situation`), and the
executive sorts its slot names when it keys a chunk -- a set mixing strings
and tuples raises there rather than here.

## What is here, and what is not

Nothing in this file names a predicate, a kind of thing or an action. A
domain is a string read by `domains.py`; the problems over one are built in
`problems.py`; what is here is only what a world *is*.

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
from dataclasses import dataclass, field

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
    #: what must *not* hold: a negative precondition. `open door` is
    #: forbidden while `locked door` holds. Nothing in VerbNet gives these
    #: -- they are learned (`lessons.py`) -- and the planner, whose `needs`
    #: can only ask for presence, sees each as a positive `not X` slot
    #: (`acting.negated`), which is the textbook compilation.
    forbids: frozenset = frozenset()

    def holds_in(self, facts) -> bool:
        return self.needs <= facts and not (self.forbids & facts)

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
        self.announce(action, before)
        return True

    def solved(self, wanted) -> bool:
        return set(wanted) <= self.facts

    def announce(self, action: Action, before) -> None:
        effect(WORLD, action.name, lambda: setattr(self, "facts", before))

    def towers(self) -> list:
        """Facts of the form `on X Y` as chains, bottom first.

        The one thing here that knows a predicate by name, and it is only
        a convenience for reading a stack back: a domain with no `on` gets
        an empty list and says its state some other way.
        """
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


class Imagined(World):
    """A world only thought about: `what steps would it take` is answered
    by acting in one of these. It changes like the real one and announces
    nothing, because nothing real moved -- so the act that imagines it has
    no effect on the world to declare, and it does not."""

    def announce(self, action: Action, before) -> None:
        return None


@dataclass
class Problem:
    """A start, a goal, and the actions that can be taken.

    The actions are carried rather than derived, because deriving them means
    knowing the domain and a world does not. `problems.py` grounds a
    `domains.Domain` over its objects and hands the result here.
    """

    name: str
    start: frozenset
    goal: frozenset
    actions: tuple = ()
    #: what the objects are, as name -> kind, where anything cares
    objects: dict = field(default_factory=dict)

    def world(self) -> World:
        return World(self.start)


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
