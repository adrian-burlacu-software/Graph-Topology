"""Designing towards a specification: propose a form, fit it, watch, learn.

The designer is v691's agent in a world whose one thing is a **draft**,
`d`. What the world says of it is what the specification is like
(`Spec.features`: roots given, terms of one sign, evenly stepped...) and,
once it is done, `designed d`. The actions are **design moves**, one for
each form that fits the kind (`forms.FORMS`): *fit a geometric sequence*,
*fit a factor for each root times a polynomial for the rest*. Each is
declared as bringing about `designed d`, which is the only thing any of
them is for.

What a move does is exact (`fit`): the form's unknowns are put into every
clause's residuals, sympy solves the equations, what nothing decided is
left at its default, and the result is **checked against the whole
specification** before the move counts as having worked. A form that
cannot meet the specification -- the terms are not one ratio apart, there
are more conditions than a scale can meet -- leaves the draft undesigned,
and that is v691's surprise: the move was for `designed d`, and it did not
bring it about.

The surprise is where the learning is. v691's learner compares the failure
with every time the same move worked: what held each time *fit-geometric*
worked and not the time it failed is what it requires (`lessons.Learner`),
and once learned, the planner does not try it where that is missing. The
first time through a bank of goals, the designer tries forms and is
corrected; the second time it goes straight to the one that fits. Nothing
in this file says when a form works.

**The order moves are tried in is the proposer.** Equal in the planner's
eyes, moves are tried in the order they are handed over (`acting.think`).
`design(order=...)` is where something that has learned which form suits
which specification -- the learned proposer -- puts its guess first.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sympy as S

from research.v691 import acting, world as W
from research.v693.forms import FORMS, Form
from research.v693.spec import Spec

D = "d"
DESIGNED = f"designed {D}"


def fact(name: str) -> str:
    return f"{name} {D}"


def verb(form: Form) -> str:
    return f"fit-{form.name}"


def moves(forms) -> tuple:
    """Each form as an action: built from what it needs, for `designed`."""
    return tuple(W.Action(f"{verb(form)} {D}",
                          frozenset(fact(one) for one in
                                    (form.kind,) + form.needs),
                          frozenset({DESIGNED}), frozenset())
                 for form in forms)


#: How many degrees bigger a form that grows is tried at.
GROWTH = 2


def sizes(form: Form, spec: Spec) -> range:
    """The sizes a form is tried at, smallest first, as degrees more or
    less than counting conditions suggests. Counting assumes no condition
    repeats another; a specification stated from a quadratic can give
    eight conditions a quadratic already meets, and the least design is
    the quadratic, not a polynomial of degree seven. So a form that grows
    is tried from as small as it can be, up to two degrees past the
    count -- and the first size that fits is the least design."""
    from research.v693.forms import _sized
    if not form.grows or spec.of("degree"):
        return range(0, 1)
    return range(-_sized(spec), GROWTH + 1)


def fit(form: Form, spec: Spec):
    """The form, its unknowns decided by the specification: the first
    solution that meets the whole specification, or None -- at the least
    size that fits, for a form that grows (`sizes`)."""
    for extra in sizes(form, spec):
        found = _fit(form, spec, extra)
        if found is not None:
            return found
    return None


def _fit(form: Form, spec: Spec, extra: int):
    try:
        obj, holes, defaults = form.build(spec, extra)
    except (ValueError, TypeError, S.PolynomialError):
        return None
    try:
        equations = _interleaved([[S.expand(one) for one in
                                   clause.residuals(obj, spec.var)]
                                  for clause in spec.clauses])
    except (ValueError, TypeError, S.PolynomialError):
        return None
    equations = [one for one in equations if one != 0]
    if any(one.is_number for one in equations):
        # A condition no unknown can touch, and it does not hold.
        return None
    solutions = _solved(equations, holes, bool(form.parts))
    for solution in sorted(solutions, key=_plainness):
        if not all(value.is_real is not False
                   for value in solution.values()):
            continue
        candidate = obj.subs(solution)
        free = [hole for hole in holes if candidate.has(hole)]
        # Small values are searched only to meet a clause no equation
        # says (`integer`): anything an equation could decide, it did.
        checked_only = any(not clause.conditions() and clause.kind
                           != "degree" for clause in spec.clauses)
        for choice in (_choices(free, defaults) if checked_only
                       else [{hole: defaults.get(hole, 0)
                              for hole in free}]):
            found = _tidied(candidate.subs(choice), spec)
            if _near(spec, found) and spec.holds(found):
                return found
    return None


def _interleaved(lists: list) -> list:
    """Each clause's equations in turn -- the first of every clause, then
    the second -- so that a form solved from its first few equations is
    solved from all the clauses, not from the first clause's many."""
    out = []
    for index in range(max((len(one) for one in lists), default=0)):
        out += [one[index] for one in lists if index < len(one)]
    return out


def _near(spec: Spec, obj) -> bool:
    """A cheap look before the exact check: every condition, in floating
    point, close to zero. Exact checking of a function simplifies trig and
    exponentials, and most candidates are nowhere near."""
    try:
        for value in spec.residuals(obj):
            number = complex(S.N(value, 15))
            if abs(number) > 1e-6 * (1 + abs(number)):
                return False
    except (TypeError, ValueError, AttributeError, S.PolynomialError):
        return True
    return True


#: The values an unknown nothing decided is tried at, after its default.
SMALL = (1, 2, -1, 3, -2, 4, 5, 6, 8, 10, 12)
#: At most this many choices of the free unknowns are tried in all.
CHOICES = 200


def _choices(free: list, defaults: dict):
    """Values for the unknowns no equation decided: their defaults first,
    then small whole numbers. This is what meets a clause that is only
    checked, never solved -- *whole-number coefficients* is met by a scale
    of 2 for roots 1/2 and 3, and no equation says so."""
    yield {hole: defaults.get(hole, 0) for hole in free}
    if not free:
        return
    import itertools
    tried = 0
    for values in itertools.product(SMALL, repeat=len(free)):
        tried += 1
        if tried > CHOICES:
            return
        yield dict(zip(free, values))


def usable(spec: Spec) -> list:
    """The forms of the kind that can be built from this specification."""
    return [form for form in FORMS if form.kind == spec.kind
            and set(form.needs) <= spec.features()]


def fitting(spec: Spec) -> list:
    """The names of every form that meets the specification."""
    return [form.name for form in usable(spec)
            if fit(form, spec) is not None]


def unknowns(form: Form, spec: Spec) -> int:
    """How many unknowns the form needed to meet the specification: at
    the size it fitted at, for a form that grows."""
    for extra in sizes(form, spec):
        if _fit(form, spec, extra) is not None:
            try:
                return len(form.build(spec, extra)[1])
            except (ValueError, TypeError, S.PolynomialError):
                break
    return 99


def simplest(spec: Spec, works: list) -> str | None:
    """Of the forms that fit, the one with the fewest unknowns -- the
    design that says least beyond what was asked (ties: table order)."""
    from research.v693.forms import BY_NAME
    order = [form.name for form in FORMS]
    return min(works, key=lambda name: (unknowns(BY_NAME[name], spec),
                                        order.index(name)), default=None)


#: The highest degree a composed form's system may have to be solved.
MOST_DEGREE = 4


def _solved(equations: list, holes: list, composed: bool) -> list:
    """Solutions for the unknowns, in two stages: the equations that are
    polynomials in the unknowns first -- sympy solves those -- and then
    what is left, with what they fixed put in. A value of e^(kx) at 1 is
    e^k; with k already fixed by the equation the function satisfies, it
    is a number, and the scale follows. Solved at once, sympy can search
    a system of such equations for as long as it likes.

    A composed form is solved from as many polynomial conditions as it has
    unknowns, the rest only checked: an overdetermined system of powers is
    another thing sympy will not finish."""
    def easy(one) -> bool:
        return one.is_polynomial(*holes)

    first = [one for one in equations if easy(one)]
    rest = [one for one in equations if not easy(one)]
    if composed and len(first) > len(holes):
        # The lowest powers: terms 1 to 4 of a sum of two geometric
        # sequences, not terms 6 and 7, which put sixth powers of both
        # ratios into one system.
        first = sorted(first, key=lambda one: S.Poly(
            one, *holes).total_degree())[:len(holes)]
    if composed and first and max(S.Poly(one, *holes).total_degree()
                                  for one in first) > MOST_DEGREE:
        # Beyond this, sympy's solve of a system in several unknowns can
        # run for as long as it likes (it did: an hour, on four terms
        # far apart). The form is not tried; a simpler one may fit.
        return []
    try:
        found = S.solve(first, holes, dict=True) if first else [{}]
    except (NotImplementedError, ValueError, TypeError):
        return []
    out = []
    for solution in found:
        left = [S.simplify(one.subs(solution)) for one in rest]
        left = [one for one in left if one != 0]
        free = [hole for hole in holes if hole not in solution
                and any(one.has(hole) for one in left)]
        if not left:
            out.append(solution)
            continue
        if not free:
            continue
        if any(inner.has(hole) for one in left
               for inner in one.atoms(S.sin, S.cos) for hole in free):
            # A frequency is not solved for from values -- sin(-2w) = ...
            # is a search sympy can make last as long as it likes. It
            # comes from the equation the function satisfies, or not at
            # all.
            continue
        try:
            more = S.solve(left[:len(free)], free, dict=True)
        except (NotImplementedError, ValueError, TypeError):
            continue
        out += [{**solution, **{key: value.subs(solution)
                                for key, value in one.items()}}
                for one in more]
    return out


def least(form: Form, spec: Spec) -> int:
    """The fewest unknowns the form could meet the specification with:
    its size, or for a form that grows, its smallest."""
    extra = sizes(form, spec)[0]
    try:
        return len(form.build(spec, extra)[1])
    except (ValueError, TypeError, S.PolynomialError):
        return 99


def best_fit(spec: Spec) -> str | None:
    """The simplest form that meets the specification, found without
    fitting every form: forms are tried fewest unknowns first, and once
    one fits, a form that could not have fewer is not tried. The sum of an
    exponential and an oscillation does meet `value 1 at 0, 3 at 1, slope
    2 at 0` -- after twenty seconds of sympy, as a page of sines -- and
    2x + 1 has already met it with two."""
    order = [form.name for form in FORMS]
    found, count = None, None
    for form in sorted(usable(spec), key=lambda one: (least(one, spec),
                                                      order.index(one.name))):
        if count is not None and least(form, spec) > count:
            break
        if fit(form, spec) is None:
            continue
        mine = unknowns(form, spec)
        if count is None or (mine, order.index(form.name)) < (
                count, order.index(found)):
            found, count = form.name, mine
    return found


def _plainness(solution: dict) -> tuple:
    """Plainer solutions first: real, whole, positive, small."""
    values = list(solution.values())
    return (sum(value.is_real is not True for value in values),
            sum(value.is_integer is not True for value in values),
            sum(bool(value.is_negative) for value in values),
            str(values))


def _tidied(obj, spec: Spec):
    if spec.kind == "polynomial":
        return S.expand(obj)
    return S.simplify(obj)


def shown(obj, spec: Spec) -> str:
    """A design as a person would write it: factored where that is
    shorter."""
    from research.v692.doing import written
    if spec.kind == "polynomial":
        factored = S.factor(obj)
        expanded = S.expand(obj)
        best = factored if (factored != expanded and len(str(factored))
                            <= len(str(expanded))) else expanded
        return written(best)
    return written(S.simplify(obj))


class Draft(W.Imagined):
    """The world of one design: the specification's features, and the
    draft once it is designed. Imagined, because designing moves nothing
    real (`v687.executive.effect` would call a real world Unwired)."""

    def __init__(self, spec: Spec, forms) -> None:
        self.spec = spec
        self.forms = {verb(form): form for form in forms}
        #: (form name, what fitting it came to), in the order tried
        self.tried: list = []
        self.design = None
        self.form: Form | None = None
        super().__init__(frozenset(fact(one) for one in spec.features()))

    def can(self, action: W.Action) -> bool:
        return action.holds_in(self.facts)

    def do(self, action: W.Action) -> bool:
        if not self.can(action):
            return False
        form = self.forms[action.name.split()[0]]
        found = fit(form, self.spec)
        self.tried.append((form.name, found))
        self.did.append(action)
        if found is not None:
            self.design, self.form = found, form
            self.facts = self.facts | {DESIGNED}
        # A move that did not design is still a move made: the world
        # says what came of it, and the agent sees that it was not what
        # the move was for.
        return True


def surprised(action, expected, before, now) -> bool:
    """A move that did not bring about what it was for."""
    wanted = frozenset(action.adds) - frozenset(before)
    return bool(wanted - frozenset(now))


@dataclass
class Design:
    spec: Spec
    design: object = None
    form: Form | None = None
    #: (form name, what it came to), in the order tried
    tried: list = field(default_factory=list)
    attempt: object = None
    #: forms a lesson held back that were tried anyway, and why:
    #: `curious` or `last resort`
    explored: list = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.design is not None

    def said(self) -> str:
        if self.design is None:
            names = ", ".join(name for name, _ in self.tried) or "nothing"
            return (f"I could not design {self.spec.said()} (tried: "
                    f"{names})")
        return (f"{shown(self.design, self.spec)} -- {self.form.said}")


def _simpler(draft: Draft, forms, spec: Spec, learner) -> None:
    """A design with an unknown for every condition only repeats what it
    was given -- a polynomial through five points is any five points --
    and explains nothing. So it is not settled for while a form with fewer
    unknowns is untried: those are tried, in the proposer's order, and the
    one with fewest that fits replaces it."""
    tried = {name for name, _ in draft.tried}
    # What a lesson holds back stays held back here too: curiosity is
    # where lessons are doubted, not this.
    doubted = {form.name for form in held_back(forms, draft.facts, learner)}
    best = unknowns(draft.form, spec)
    for form in forms:
        if form.name in tried | doubted or least(form, spec) >= best or                 not set(form.needs) <= spec.features():
            continue
        found = fit(form, spec)
        draft.tried.append((form.name, found))
        if found is None:
            continue
        count = unknowns(form, spec)
        if count < best:
            draft.design, draft.form, best = found, form, count
            if learner is not None:
                action = moves([form])[0]
                learner.worked(action, draft.facts, draft.facts)


#: How often a held-back form is tried first anyway, out of curiosity.
CURIOSITY = 0.15


def held_back(forms, facts, learner) -> list:
    """The forms a lesson keeps from being tried here: buildable from the
    specification, and not once what was learned is folded in."""
    if learner is None:
        return []
    raw = moves(forms)
    applied = learner.applied(raw)
    return [form for form, bare, learned in zip(forms, raw, applied)
            if bare.holds_in(facts) and not learned.holds_in(facts)]


def _explore(form: Form, draft: Draft, learner, why: str,
             explored: list) -> None:
    """Try a form a lesson held back, and tell the learner what came of
    it: a success refutes the lesson (`Learner.worked`), and a failure
    leaves it standing -- it said so."""
    action = moves([form])[0]
    before = draft.facts
    draft.do(action)
    explored.append((form.name, why))
    if draft.design is not None and learner is not None:
        learner.worked(action, before, draft.facts)


def design(spec: Spec, learner=None, order=None,
           curiosity: float = CURIOSITY, rng=None) -> Design:
    """Design towards a specification: v691's agent, planning, fitting and
    watching, in the world of one draft. `order` is the names of the forms
    to try first, in order (the proposer's guess).

    **A form that failed on this draft is not proposed for it again.** A
    failure the learner cannot yet explain teaches nothing, and an agent
    left to replan would propose the same form -- the most useful-looking
    move has not changed -- until it ran out of tries. So each run of the
    agent is one plan, and a form that failed leaves the table for this
    draft. What the failure taught stays with the learner, for every draft
    after it.

    **And a lesson is not the last word.** A lesson learned from one
    failure can be wrong, and a lesson that stops a form being tried stops
    the success that would refute it from ever happening. So a form a
    lesson holds back is tried anyway in two cases: **out of curiosity**,
    first, once in a while (`curiosity`); and **as a last resort**, before
    the designer says a goal cannot be met. A success refutes the lesson.
    A goal is only said to be impossible when every form has been tried.
    """
    import random
    from research.v687.executive import Working
    forms = [form for form in FORMS if form.kind == spec.kind]
    if order:
        rank = {name: index for index, name in enumerate(order)}
        forms.sort(key=lambda form: rank.get(form.name, len(rank)))
    draft = Draft(spec, forms)
    goal = frozenset({DESIGNED})
    reports = []
    explored: list = []
    rng = rng if rng is not None else random.Random(spec.said())
    doubted = held_back(forms, draft.facts, learner)
    if doubted and rng.random() < curiosity:
        _explore(doubted[0], draft, learner, "curious", explored)
    left = [form for form in forms
            if form.name not in {name for name, _ in draft.tried}]
    while left and draft.design is None:
        problem = W.Problem("design", draft.facts, goal, moves(left))
        before = len(draft.tried)
        # A report of its own: the agent stops once a report has had
        # more plans than it may try, and each run here is one plan.
        report = acting.Attempt(name="design")
        reports.append(report)
        agent = acting.agent(problem, draft, None, report, tries=1,
                             learner=learner, surprised=surprised)
        agent.run(Working(goal=f"design {spec.said()}"))
        failed = {name for name, found in draft.tried[before:]
                  if found is None}
        if len(draft.tried) == before:
            # No plan: what is left cannot be built from this
            # specification, or has been learned not to work for it.
            break
        left = [form for form in left if form.name not in failed]
    if draft.design is not None and draft.form is not None and unknowns(
            draft.form, spec) >= spec.conditions() >= 4:
        _simpler(draft, forms, spec, learner)
    if draft.design is None:
        tried = {name for name, _ in draft.tried}
        for form in held_back(forms, draft.facts, learner):
            if form.name in tried:
                continue
            _explore(form, draft, learner, "last resort", explored)
            if draft.design is not None:
                break
    return Design(spec, draft.design, draft.form, draft.tried, reports,
                  explored)
