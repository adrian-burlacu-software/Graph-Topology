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
    an operator      the slots it needs and gives, a condition on working
                     memory beyond them (`proposes`), an action (`apply`),
                     a utility, and the rule it carries out
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

**What an answer came to is kept before anything learns from it** (E3). Every
run inside an `episode` -- a turn -- is recorded with the executive's name,
subgoals within the run that pushed them, and a `Ledger` credits each
operator that did something with what the answer was worth. With `RATE` at
zero that moves no utility: it says what learning would do, so it can be
looked at first.
"""
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

ANSWERED, CONTINUE, DECLINED = "answered", "continue", "declined"

#: How far a reward moves a utility. Zero: the priors are the behaviour
#: until something is fitted.
RATE = 0.0

#: The runs of the episode open now, as (executive name, trace), or None.
_EPISODE: ContextVar[list | None] = ContextVar("episode", default=None)


@contextmanager
def episode():
    """Record every run of an executive for as long as this lasts -- a turn
    -- and yield the list they go into, in the order they finished. A
    subgoal's run is not recorded apart: it is in the trace of the run that
    pushed it. Episodes nest; an inner one keeps its own runs."""
    runs: list = []
    token = _EPISODE.set(runs)
    try:
        yield runs
    finally:
        _EPISODE.reset(token)


@dataclass
class Operator:
    name: str
    apply: Callable[[dict], str | None]
    proposes: Callable[[dict], bool] = lambda memory: True
    #: None: set from the operator's place in the list it was given in
    utility: float | None = None
    rule: str = ""
    #: the slots it reads: it is proposed only once all of them are in
    #: working memory, and `proposes` is what it asks of them beyond that
    needs: tuple[str, ...] = ()
    #: the slots it writes when it goes on (CONTINUE) -- what another
    #: operator's `needs` can be met by (E4)
    gives: tuple[str, ...] = ()

    def ready(self, memory: dict) -> bool:
        return (all(slot in memory for slot in self.needs)
                and self.proposes(memory))


@dataclass
class Fired:
    operator: str
    rule: str
    outcome: str
    #: every operator proposed in the cycle this one was chosen from -- the
    #: conflict set, which is what a utility decides (E3)
    candidates: tuple = ()

    def as_dict(self) -> dict:
        out = {"operator": self.operator, "rule": self.rule,
               "outcome": self.outcome}
        if len(self.candidates) > 1:
            # Only where there was a choice: a cycle of one proposal decided
            # nothing, and every trace written before this reads the same.
            out["candidates"] = list(self.candidates)
        return out


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


class Unwired(ValueError):
    """An operator's declared needs that nothing can give."""


class Executive:
    """Operators, and the cycle that fires them.

    `given`, when passed, is the slots working memory starts with, and the
    operators' declarations are checked against it once, here: every
    operator must be reachable -- each of its needs given at the start, by
    another operator that can itself be reached, or by a subgoal's result --
    or this raises `Unwired`. A need nothing gives is an operator that can
    never fire, which is a mistake in the wiring, not a question it declined.
    """

    def __init__(self, operators: list[Operator], rate: float = RATE,
                 subgoals: dict[str, Subgoal] | None = None,
                 name: str = "", given: tuple[str, ...] | None = None
                 ) -> None:
        #: what this executive is, as a ledger names it
        self.name = name
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
        if given is not None:
            unreached = self.unreachable(given)
            if unreached:
                raise Unwired(f"{self.name or 'executive'}: " + "; ".join(
                    f"{one.name} needs {', '.join(missing)}"
                    for one, missing in unreached))

    def unreachable(self, given: tuple[str, ...] = ()
                    ) -> list[tuple[Operator, list[str]]]:
        """(operator, the needs nothing reachable gives), for every operator
        that could never be proposed from `given`: forward chaining over
        the declarations, as a planner would search them (E5). A subgoal
        gives what it returns, and `impasse` and `resolved` along with it."""
        have = set(given)
        for subgoal in self.subgoals.values():
            have.update(subgoal.returns)
        if self.subgoals:
            have.update(("impasse", "resolved"))
        waiting = list(self.operators)
        while True:
            reached = [one for one in waiting if have.issuperset(one.needs)]
            if not reached:
                break
            for one in reached:
                have.update(one.gives)
                waiting.remove(one)
        return [(one, [slot for slot in one.needs if slot not in have])
                for one in waiting]

    def run(self, memory: dict, subgoal: bool = False) -> Trace:
        trace = self._cycle(memory)
        runs = _EPISODE.get()
        if runs is not None and not subgoal:
            runs.append((self.name, trace))
        return trace

    def _cycle(self, memory: dict) -> Trace:
        trace = Trace(goal=getattr(memory, "goal", ""),
                      depth=getattr(memory, "depth", 1))
        fired: set = set()
        resolved: set = set()
        while True:
            proposed = [one for one in self.operators
                        if one.name not in fired and one.ready(memory)]
            if not proposed:
                if self._subgoal(memory, trace, resolved):
                    continue
                return trace
            chosen = max(proposed, key=lambda one: (one.utility,
                                                    -self.place[one.name]))
            fired.add(chosen.name)
            outcome = chosen.apply(memory) or DECLINED
            if outcome == CONTINUE:
                missing = [slot for slot in chosen.gives
                           if slot not in memory]
                if missing:
                    raise Unwired(f"{chosen.name} went on without giving "
                                  f"{', '.join(missing)}")
            trace.fired.append(Fired(chosen.name, chosen.rule, outcome,
                                     tuple(one.name for one in proposed)))
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
            trace.subgoals.append(subgoal.executive.run(memory,
                                                        subgoal=True))
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


class Ledger:
    """What each operator's answers came to, kept while utilities stay put.

    `credit` takes a run as a trace records it and what the answer it went
    into was worth. Every operator that did something on the way -- answered
    or wrote slots, not declined -- is credited, within the subgoals it
    pushed too, as `reward` would move its utility. ACT-R's utility, learned
    at a small rate, comes to the mean of the rewards it is given: `mean` is
    where each operator's utility would be heading.
    """

    def __init__(self) -> None:
        #: (executive, operator) -> {"count", "total", "outcomes": {...}}
        self.rows: dict[tuple[str, str], dict] = {}
        #: (executive, chosen) -> {another proposed with it: times}
        self.conflicts: dict[tuple[str, str], dict] = {}

    def credit(self, executive: str, trace: dict, value: float,
               outcome: str = "") -> None:
        for step in trace.get("fired", ()):
            others = [one for one in step.get("candidates", ())
                      if one != step["operator"]]
            if others:
                seen = self.conflicts.setdefault(
                    (executive, step["operator"]), {})
                for one in others:
                    seen[one] = seen.get(one, 0) + 1
            if step.get("outcome") == DECLINED:
                continue
            row = self.rows.setdefault((executive, step["operator"]),
                                       {"count": 0, "total": 0.0,
                                        "outcomes": {}})
            row["count"] += 1
            row["total"] += value
            if outcome:
                row["outcomes"][outcome] = row["outcomes"].get(outcome, 0) + 1
        for inner in trace.get("subgoals", ()):
            self.credit(executive, inner, value, outcome)

    def mean(self, executive: str, operator: str) -> float | None:
        row = self.rows.get((executive, operator))
        return row["total"] / row["count"] if row and row["count"] else None

    def reversals(self) -> list[tuple]:
        """(executive, chosen, its mean, passed over, its mean, times): a
        choice learning would reverse, because an operator proposed beside
        the one chosen has earned more when it did fire. Only operators with
        credit of their own are compared -- one that never fired has no mean
        -- so this is what the rewards say, not what an untried operator
        might have done.

        It assumes utilities on the reward's scale. The priors are not: they
        are the order written (`len - index`, so 12, 11, ... 1), and any
        operator that fired would be pulled from its prior toward a mean
        below one, under every operator that never fired. Putting the priors
        on the reward's scale comes before turning learning on."""
        out = []
        for (executive, chosen), others in self.conflicts.items():
            mine = self.mean(executive, chosen)
            if mine is None:
                continue
            for other, times in others.items():
                theirs = self.mean(executive, other)
                if theirs is not None and theirs > mine:
                    out.append((executive, chosen, mine, other, theirs, times))
        return sorted(out, key=lambda one: -one[5])

    def table(self) -> list[tuple]:
        """(executive, operator, count, mean, outcomes), the most credited
        first."""
        return sorted(((executive, operator, row["count"],
                        row["total"] / row["count"], dict(row["outcomes"]))
                       for (executive, operator), row in self.rows.items()),
                      key=lambda one: (-one[2], one[0], one[1]))


def record(name: str, trace: Trace) -> dict:
    """A run as a payload carries it, for whatever credits it (E3): the
    executive's name, its goal, and what fired."""
    return {"executive": name, "goal": trace.goal, **trace.as_dict()}


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
