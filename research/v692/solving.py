"""Solving an equation as acting in a world: planned, done, watched.

v691 has an agent that plans means-ends over a model of a world, acts one
step at a time, looks after each, and treats a surprise as an impasse it
has to make sense of -- learning, where the evidence allows, what the
action needed (`acting.agent`, `lessons.Learner`). This gives it a world in
which the thing acted on is an equation.

**The state is what the equation is like**, worked out by the world after
every step (`features`): whether it has brackets or fractions, whether the
variable is on both sides, whether it is linear or quadratic, zero on the
right, factored, and -- for a quadratic -- whether its discriminant is a
perfect square. Each is a fact about the one thing acted on, `eq`.

**The actions are algebra's moves** (`MOVES`): expand, clear fractions,
gather the variable's terms on one side, move the constants, divide by the
coefficient, bring everything to one side, factor, split a product that is
zero, the quadratic formula. Each says what it needs and what it is for, as
a textbook would; that is the model the planner searches. What a move does
to a real equation is sympy's (`apply`), so the plan is always tried
against mathematics, not against the model.

**A surprise is a move that did not do what it was for.** Factoring
`x^2 + x - 1 = 0` does nothing over the rationals: the planner expected
`factored eq` and the world says no. That is v691's impasse, and v691's
learner compares it with the times factoring worked -- every one had a
square discriminant, this one had not -- and learns that factoring
*requires* it. The replan goes by the formula. Nothing in this file says
when factoring works.

A move only brings about the facts it is for, but a world says everything
true after it -- expanding may also leave the variable on both sides --
so what counts as a surprise here is an intended fact that did not come
about (`surprised`), not every fact the model did not predict.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sympy as S

from research.v691 import acting, world as W

EQ = "eq"


def _fact(name: str) -> str:
    return f"{name} {EQ}"


def _move(name: str, needs=(), adds=(), deletes=()) -> W.Action:
    return W.Action(f"{name} {EQ}", frozenset(_fact(one) for one in needs),
                    frozenset(_fact(one) for one in adds),
                    frozenset(_fact(one) for one in deletes))


#: Algebra's moves, as a textbook states them: what each needs, and what
#: it is for.
MOVES = (
    _move("expand", needs=("brackets",), adds=("expanded",),
          deletes=("brackets",)),
    _move("clear-fractions", needs=("fractions",), adds=("whole",),
          deletes=("fractions",)),
    _move("gather", needs=("both-sides", "expanded", "whole"),
          adds=("variable-left",), deletes=("both-sides",)),
    _move("swap", needs=("variable-right",), adds=("variable-left",),
          deletes=("variable-right",)),
    _move("move-constants", needs=("variable-left", "linear"),
          adds=("isolated",), deletes=("constant-left",)),
    _move("divide", needs=("isolated",), adds=("solved",)),
    _move("zero-right", needs=("quadratic", "expanded", "whole"),
          adds=("zero-right",)),
    _move("factor", needs=("quadratic", "zero-right"), adds=("factored",)),
    _move("split", needs=("factored",), adds=("linear", "constant-left",
                                              "variable-left"),
          deletes=("factored", "quadratic")),
    _move("formula", needs=("quadratic", "zero-right"), adds=("solved",)),
)
BY_NAME = {one.name.split()[0]: one for one in MOVES}


def _terms(side) -> list:
    return list(S.Add.make_args(side))


def _has(side, var) -> bool:
    return var in getattr(side, "free_symbols", set())


def solved(equation, var) -> bool:
    return equation.lhs == var and not _has(equation.rhs, var)


def features(equations: list, var) -> frozenset:
    """What the equations still to solve are like, as facts about `eq`."""
    open_ = [one for one in equations if not solved(one, var)]
    out = set()
    if not open_:
        return frozenset({_fact("solved")} if equations else set())
    for eq in open_:
        left, right = eq.lhs, eq.rhs
        if any(S.expand(side) != side for side in (left, right)
               if _has(side, var)):
            out.add("brackets")
        else:
            out.add("expanded")
        if any(S.fraction(S.together(side))[1] != 1
               for side in (left, right)):
            out.add("fractions")
        else:
            out.add("whole")
        on_left, on_right = _has(left, var), _has(right, var)
        if on_left and on_right:
            out.add("both-sides")
        elif on_right:
            out.add("variable-right")
        elif on_left:
            out.add("variable-left")
        if on_left and not on_right and any(
                not _has(term, var) for term in _terms(S.expand(left))):
            out.add("constant-left")
        if on_left and not on_right and len(_terms(S.expand(left))) == 1 \
                and S.degree(S.expand(left), var) == 1:
            out.add("isolated")
        try:
            degree = S.degree(S.expand(left - right), var)
        except S.PolynomialError:
            degree = 0
        if degree == 1:
            out.add("linear")
        elif degree == 2:
            out.add("quadratic")
            poly = S.Poly(S.expand(left - right), var)
            a, b, c = (poly.coeff_monomial(var ** 2),
                       poly.coeff_monomial(var), poly.coeff_monomial(1))
            discriminant = b * b - 4 * a * c
            if discriminant.is_number and discriminant >= 0 and \
                    S.sqrt(discriminant).is_rational:
                out.add("square-discriminant")
        if right == 0:
            out.add("zero-right")
            if isinstance(left, S.Mul) and sum(
                    1 for one in left.args if _has(one, var)) >= 2:
                out.add("factored")
    return frozenset(_fact(one) for one in out)


def apply(move: str, equations: list, var) -> list:
    """What a move does to every equation still to solve -- sympy's work,
    not the model's."""
    out = []
    for eq in equations:
        if solved(eq, var):
            out.append(eq)
            continue
        out.extend(_one(move, eq, var))
    return out


def _one(move: str, eq, var) -> list:
    left, right = eq.lhs, eq.rhs
    if move == "expand":
        return [S.Eq(S.expand(left), S.expand(right), evaluate=False)]
    if move == "clear-fractions":
        denominators = [S.fraction(S.together(term))[1]
                        for side in (left, right) for term in _terms(side)]
        common = S.lcm_list(denominators) if denominators else 1
        return [S.Eq(S.expand(left * common), S.expand(right * common),
                     evaluate=False)]
    if move == "gather":
        moving = sum((term for term in _terms(S.expand(right))
                      if _has(term, var)), S.Integer(0))
        return [S.Eq(S.expand(left - moving), S.expand(right - moving),
                     evaluate=False)]
    if move == "swap":
        return [S.Eq(right, left, evaluate=False)]
    if move == "move-constants":
        moving = sum((term for term in _terms(S.expand(left))
                      if not _has(term, var)), S.Integer(0))
        return [S.Eq(S.expand(left - moving), S.expand(right - moving),
                     evaluate=False)]
    if move == "divide":
        coefficient = S.expand(left).coeff(var)
        if coefficient == 0:
            return [eq]
        return [S.Eq(var, S.simplify(right / coefficient), evaluate=False)]
    if move == "zero-right":
        return [S.Eq(S.expand(left - right), 0, evaluate=False)]
    if move == "factor":
        factored = S.factor(left)
        if isinstance(factored, S.Mul) and sum(
                1 for one in factored.args if _has(one, var)) >= 2:
            return [S.Eq(factored, 0, evaluate=False)]
        # Over the rationals it does not come apart: the equation is as it
        # was, and the world says so.
        return [eq]
    if move == "split":
        if isinstance(left, S.Mul) and right == 0:
            parts = [one for one in left.args if _has(one, var)]
            return [S.Eq(one, 0, evaluate=False) for one in parts]
        return [eq]
    if move == "formula":
        roots = S.solveset(S.Eq(left, right), var, domain=S.S.Reals)
        if isinstance(roots, S.FiniteSet):
            return [S.Eq(var, one, evaluate=False) for one in
                    sorted(roots, key=S.default_sort_key)]
        return [eq]
    return [eq]


class Board(W.World):
    """The real equation, changed only by moves."""

    def __init__(self, equation, var) -> None:
        self.var = var
        self.equations = [equation]
        #: every move made, with the equations after it
        self.steps: list = []
        super().__init__(features(self.equations, var))

    def can(self, action: W.Action) -> bool:
        return action.holds_in(self.facts)

    def do(self, action: W.Action) -> bool:
        if not self.can(action):
            return False
        before = (self.facts, list(self.equations))
        self.equations = apply(action.name.split()[0], self.equations,
                               self.var)
        self.facts = features(self.equations, self.var)
        self.did.append(action)
        self.steps.append((action.name.split()[0], list(self.equations)))
        self.announce(action, before[0])
        return True


def surprised(action, expected, before, now) -> bool:
    """A move that did not bring about what it was for."""
    wanted = frozenset(action.adds) - frozenset(before)
    return bool(wanted - frozenset(now))


@dataclass
class Solution:
    equation: object
    var: object
    solved: bool = False
    #: (move, the equations after it)
    steps: list = field(default_factory=list)
    roots: list = field(default_factory=list)
    attempt: object = None


def solve(equation, var=None, learner=None, tries: int = 6,
          goal: str = "solved") -> Solution:
    """Plan, act and watch until the equation is solved -- or, with `goal`
    `factored`, factored -- or no plan is left: v691's agent, in the world
    of one equation."""
    from research.v687.executive import Working
    if not isinstance(equation, S.Equality):
        equation = S.Eq(equation, 0, evaluate=False)
    if var is None:
        free = sorted(equation.free_symbols, key=lambda one: one.name)
        var = free[0] if free else S.Symbol("x")
    board = Board(equation, var)
    problem = W.Problem(goal, board.facts, frozenset({_fact(goal)}), MOVES)
    report = acting.Attempt(name="solve")
    agent = acting.agent(problem, board, None, report, tries=tries,
                         learner=learner, surprised=surprised)
    agent.run(Working(goal=f"solve {equation}"))
    out = Solution(equation, var, solved=board.solved(problem.goal),
                   steps=board.steps, attempt=report)
    if out.solved and goal == "solved":
        out.roots = [one.rhs for one in board.equations]
    return out
