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

    working memory   a dict of slots: the goal, and what operators found
    an operator      a condition on working memory (`proposes`), an action
                     (`apply`), a utility, and the rule it carries out
    a cycle          propose every operator whose condition holds and that
                     has not fired; fire the one of highest utility
    an outcome       ANSWERED ends the goal; CONTINUE means it wrote slots
                     for others to read; DECLINED means it had nothing
    an impasse       nothing left to propose and nothing answered: the
                     caller's to resolve -- ask which one, refuse by name

**Utilities are the order the cascades were written in**, as priors: the first
listed is the most useful. So converting a cascade changes nothing it
answers. What changes is that the order is a number that can be learned from
what each operator's answers came to (`reward`, ACT-R's utility learning,
off by default), and that every question leaves a trace of what fired.
"""
from __future__ import annotations

from collections.abc import Callable
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

    @property
    def impasse(self) -> bool:
        return self.answered_by is None

    def as_dict(self) -> dict:
        return {"fired": [one.as_dict() for one in self.fired],
                "answered_by": self.answered_by}


class Executive:
    """Operators, and the cycle that fires them."""

    def __init__(self, operators: list[Operator], rate: float = RATE) -> None:
        self.operators = list(operators)
        count = len(self.operators)
        for index, one in enumerate(self.operators):
            if one.utility is None:
                one.utility = float(count - index)
        #: ties go to the one listed first
        self.place = {one.name: index
                      for index, one in enumerate(self.operators)}
        self.rate = rate

    def run(self, memory: dict) -> Trace:
        trace = Trace()
        fired: set = set()
        while True:
            proposed = [one for one in self.operators
                        if one.name not in fired and one.proposes(memory)]
            if not proposed:
                return trace
            chosen = max(proposed, key=lambda one: (one.utility,
                                                    -self.place[one.name]))
            fired.add(chosen.name)
            outcome = chosen.apply(memory) or DECLINED
            trace.fired.append(Fired(chosen.name, chosen.rule, outcome))
            if outcome == ANSWERED:
                trace.answered_by = chosen.name
                return trace

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
