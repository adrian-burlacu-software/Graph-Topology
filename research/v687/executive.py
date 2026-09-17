"""The executive: which operator runs next, and what the question came to.

Control was three cascades. v687's `ask` tried nine layers in a fixed order;
v689's `say` looked an act up in a dict; its `_ask` was 125 lines of
if-this-return. Each was written anew for each new form of question, which is
why the knowledge generalised and the control did not (`v690/DESIGN.md` §1).

This is the mechanism all three run on (§4.6): procedural memory as operators
over a working memory, one fired per cycle, as a production system does it
(Soar, ACT-R) and as the trie paper puts executive memory -- an extended
finite state machine whose next state is chosen from the current state and
its conditions, with short-term memory as the variables it reads and writes.

    working memory   `Working`: a stack of goal frames, the top one a
                     dict of slots -- what operators found
    an operator      a condition on working memory (`proposes`), an action
                     (`apply`), a utility, and the rule it carries out
    a cycle          propose every operator whose condition holds and that
                     has not fired; fire the one of highest utility
    an outcome       ANSWERED ends the goal; CONTINUE means it wrote slots
                     for others to read; DECLINED means it had nothing
    an impasse       nothing left to propose and nothing answered. An
                     operator that cannot get past something names it in
                     the slot `impasse`; if the executive has a `Subgoal` of
                     that name, it is pushed, run, and popped, what it
                     `returns` is written back, and the cycle goes on --
                     Soar's impasse, substate and result. Otherwise the
                     impasse is the caller's: ask which one, refuse by name

**Utilities are the order the cascades were written in**, as priors: the first
listed is the most useful. So converting a cascade changes nothing it
answers. What changes is that the order is a number that can be learned from
what each operator's answers came to (`reward`, ACT-R's utility learning,
off by default), and that every question leaves a trace of what fired.
"""
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field

ANSWERED, CONTINUE, DECLINED = "answered", "continue", "declined"

#: How far a reward moves a utility. Zero: the priors are the behaviour
#: until something is fitted.
RATE = 0.0


@dataclass
class Operator:
    name: str
    apply: Callable[[dict], str | None]
    proposes: Callable[[dict], bool] = lambda memory: True
    #: None: set from the operator's place in the list it was given in
    utility: float | None = None
    rule: str = ""


@dataclass
class Fired:
    operator: str
    rule: str
    outcome: str

    def as_dict(self) -> dict:
        return {"operator": self.operator, "rule": self.rule,
                "outcome": self.outcome}


@dataclass
class Trace:
    fired: list = field(default_factory=list)
    #: the operator that answered, or None at an impasse
    answered_by: str | None = None
    #: the goal the run worked on (`Working.goal`), and how many were open
    goal: str = ""
    depth: int = 1
    #: the trace of each subgoal an impasse pushed, in order
    subgoals: list = field(default_factory=list)

    @property
    def impasse(self) -> bool:
        return self.answered_by is None

    def as_dict(self) -> dict:
        out = {"fired": [one.as_dict() for one in self.fired],
               "answered_by": self.answered_by}
        if self.subgoals:
            out["subgoals"] = [{"goal": one.goal, **one.as_dict()}
                               for one in self.subgoals]
        return out


@dataclass
class Subgoal:
    """What an impasse of one name opens: a goal of its own, the operators
    that work on it, and the slots of its frame handed back to the goal
    that reached the impasse -- the substate's result."""

    goal: str
    executive: "Executive"
    returns: tuple[str, ...] = ()


class Executive:
    """Operators, and the cycle that fires them."""

    def __init__(self, operators: list[Operator], rate: float = RATE,
                 subgoals: dict[str, Subgoal] | None = None) -> None:
        self.operators = list(operators)
        #: impasse name -> the subgoal it opens
        self.subgoals = dict(subgoals or {})
        count = len(self.operators)
        for index, one in enumerate(self.operators):
            if one.utility is None:
                one.utility = float(count - index)
        #: ties go to the one listed first
        self.place = {one.name: index
                      for index, one in enumerate(self.operators)}
        self.rate = rate

    def run(self, memory: dict) -> Trace:
        trace = Trace(goal=getattr(memory, "goal", ""),
                      depth=getattr(memory, "depth", 1))
        fired: set = set()
        resolved: set = set()
        while True:
            proposed = [one for one in self.operators
                        if one.name not in fired and one.proposes(memory)]
            if not proposed:
                if self._subgoal(memory, trace, resolved):
                    continue
                return trace
            chosen = max(proposed, key=lambda one: (one.utility,
                                                    -self.place[one.name]))
            fired.add(chosen.name)
            outcome = chosen.apply(memory) or DECLINED
            trace.fired.append(Fired(chosen.name, chosen.rule, outcome))
            if outcome == ANSWERED:
                trace.answered_by = chosen.name
                return trace

    def _subgoal(self, memory, trace: Trace, resolved: set) -> bool:
        """At an impasse, push the subgoal its name opens, run it, and hand
        back its result. False when there is none to push: no `Working` to
        push on, no impasse named in this goal's own frame, no subgoal of
        that name, or one already pushed for it in this run."""
        if not isinstance(memory, Working):
            return False
        # Only an impasse this goal reached: one named in a goal beneath is
        # that goal's to resolve, and reading it through would re-open it.
        name = dict.get(memory, "impasse")
        subgoal = self.subgoals.get(name)
        if subgoal is None or name in resolved:
            return False
        resolved.add(name)
        with memory.subgoal(subgoal.goal) as result:
            trace.subgoals.append(subgoal.executive.run(memory))
        for slot in subgoal.returns:
            if slot in result:
                memory[slot] = result[slot]
        memory.setdefault("resolved", []).append(name)
        return True

    def reward(self, trace: Trace, value: float) -> None:
        """Utility learning: each operator that did something on the way
        moves toward what the answer was worth (`U += rate * (value - U)`)."""
        if not self.rate:
            return
        by_name = {one.name: one for one in self.operators}
        for step in trace.fired:
            if step.outcome == DECLINED:
                continue
            one = by_name[step.operator]
            one.utility += self.rate * (value - one.utility)


class Working(dict):
    """Working memory as a stack of goal frames (Soar's states and
    substates; §4.6).

    **It is the top frame.** As a dict it holds the slots of the goal being
    worked on now, so everything that reads and writes working memory by key
    -- `memory["answer"]`, `memory.get("goals")` -- is unchanged, and at one
    frame deep it is exactly the dict it replaces.

    **A subgoal is pushed, and popped with its result.** `push` opens a frame
    for a goal of its own. What it writes stays in it; what it reads and does
    not have, it reads from the goals below, as a Soar substate reads its
    superstate. `pop` closes it, returns what it wrote, and restores the
    frame and goal beneath. Nothing written in a subgoal reaches the goal
    that pushed it except what that goal takes from the result, so a subgoal
    that fails leaves nothing behind.

    What pushes one -- an impasse, when nothing can be proposed -- is the
    executive's (E2); this is only where the goals are kept.
    """

    def __init__(self, *args, goal: str = "", **slots) -> None:
        super().__init__(*args, **slots)
        #: what the top frame is working toward
        self.goal = goal
        #: (goal, slots) of every frame beneath the top, the bottom first
        self._beneath: list[tuple[str, dict]] = []

    # -- reading through to the goals below --------------------------------
    def __missing__(self, key):
        for _, frame in reversed(self._beneath):
            if key in frame:
                return frame[key]
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key) -> bool:
        return (dict.__contains__(self, key)
                or any(key in frame for _, frame in self._beneath))

    # -- the stack ---------------------------------------------------------
    @property
    def depth(self) -> int:
        """How many goals are open: 1 with no subgoal."""
        return len(self._beneath) + 1

    @property
    def open_goals(self) -> list[str]:
        """Every open goal, the outermost first. (Not the slot `goals`,
        which is a question's goals as `goals.py` reads them.)"""
        return [goal for goal, _ in self._beneath] + [self.goal]

    def push(self, goal: str, **slots) -> None:
        """Open a subgoal: a frame of its own on top."""
        self._beneath.append((self.goal, dict(self)))
        self.clear()
        self.update(slots)
        self.goal = goal

    def pop(self) -> dict:
        """Close the top goal: what it wrote, and the frame beneath back."""
        if not self._beneath:
            raise IndexError("the outermost goal is not a subgoal")
        result = dict(self)
        self.goal, frame = self._beneath.pop()
        self.clear()
        self.update(frame)
        return result

    @contextmanager
    def subgoal(self, goal: str, **slots):
        """`push` for as long as this lasts, and always `pop`: a subgoal that
        raises must not leave its frame as the goal beneath's working
        memory. Yields the dict `pop` will return, filled on exit."""
        result: dict = {}
        self.push(goal, **slots)
        try:
            yield result
        finally:
            result.update(self.pop())


def attempt(function: Callable[[], object | None], slot: str = "answer"
            ) -> Callable[[dict], str | None]:
    """An operator's action from a function that answers or returns None:
    what it returns goes in `slot`, and it answered if it returned anything."""
    def apply(memory: dict) -> str | None:
        found = function()
        if found is None:
            return DECLINED
        memory[slot] = found
        return ANSWERED
    return apply
