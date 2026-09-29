"""Rung 2: control forms, their callbacks as holes, the holes as subgoals.

`PLAN.md`, rung 2. A form (`program.Form`) is read from the compiler: a
member whose callback is a hole with a scope. Two ways to fill it:

**Deduced** -- the form says how its output is made from the hole's, so
the spec is pushed down into the hole and the hole is solved as a spec of
its own, by a child solver sharing the parent's memory (what it learns
from a body is recognised next time, in any form):

    map over a list as long as the output        out[j] = body(x_j, j)
    map then join("") for a string output        out[j] = body(x_j, j)
    filter whose output is a subsequence         keep_j = body(x_j, j)

**Checked** -- `some`, `every`, `find`, `findIndex`, `sort`, `reduce`: the
output does not say what each call of the body gave, so small bodies are
grown in the hole's scope and every whole form is checked.

**Forward** -- form applications with small bodies are offered to the
forward trie like any expression, so a form can sit inside a composition:
`xs.filter((x, i) => (x > 0)).length`.
"""
from __future__ import annotations

import itertools
import json

from research.v696 import program as P
from research.v696.spec import Spec

#: How many sub-examples a deduced hole is solved from.
SUB_EXAMPLES = 24
#: What a child solver may spend, and how deep a body grows.
SUB_BUDGET = 6000
SUB_DEPTH = 2
#: How many bodies a checked or forward form is tried with.
BODIES = 60


def _names(form: P.Form, outer) -> list:
    """The hole's scope, renamed where it would hide an outer parameter."""
    taken = {name for name, _ in outer}
    out = []
    for name, kind in form.scope:
        new, index = name, 1
        while new in taken:
            new, index = f"{name}{index}", index + 1
        taken.add(new)
        out.append((new, kind))
    return out


def _settled(scope, body: str) -> list:
    return [(name, body if kind == "U" else kind) for name, kind in scope]


def _key(value) -> str:
    return json.dumps(value, sort_keys=True)


def _subsequence(small, big):
    """Which of `big` are kept, if `small` is `big` with some left out, in
    order; None if it is not."""
    if not isinstance(small, list) or not isinstance(big, list):
        return None
    kept, at = [], 0
    for one in big:
        if at < len(small) and _key(one) == _key(small[at]):
            kept.append(True)
            at += 1
        else:
            kept.append(False)
    return kept if at == len(small) else None


def _element_type(kind: str) -> str:
    return kind[:-2] if kind.endswith("[]") else ""


def _sub_spec(spec: Spec, scope, body_type: str, rows) -> Spec | None:
    """A spec for the hole: its parameters the scope and the outer ones,
    its examples one per element of every example."""
    examples, seen = [], set()
    for args, out in rows:
        key = _key([args, out])
        if key not in seen:
            seen.add(key)
            examples.append((args, out))
        if len(examples) >= SUB_EXAMPLES:
            break
    if len(examples) < 2:
        return None
    return Spec(f"{spec.name}/hole", list(scope) + list(spec.params),
                body_type, examples)


def _induced(solver, spec: Spec, receiver: P.Expr, values: list,
             form: P.Form, scope, result) -> P.Expr | None:
    """A fold found by induction (rung 3): where two examples' lists
    differ by one element at the end, the fold's answer for the shorter,
    that element and its place give the answer for the longer -- a row of
    the step's own spec. Two such rows and the step is a subgoal; the start
    is the answer where the list is empty, or one of the constants."""
    body_type = spec.returns
    if len(form.extra) != 1 or form.body not in (body_type, "U") \
            or form.gives not in (body_type, "U"):
        return None
    rows, empty = [], []
    for (args_a, out_a), got_a in zip(spec.examples, values):
        if got_a == []:
            empty.append(out_a)
        for (args_b, out_b), got_b in zip(spec.examples, values):
            if isinstance(got_a, list) and isinstance(got_b, list) \
                    and len(got_b) == len(got_a) + 1 \
                    and got_b[:-1] == got_a:
                rows.append(([out_a, got_b[-1], len(got_a)][:len(scope)]
                             + list(args_b), out_b))
    settled = _settled(scope, body_type)
    op = form.op(body_type)
    starts = [P.const(value, body_type) for value in empty[:1]] + [
        P.const(value, kind) for value, kind in P.CONSTANTS
        if kind == body_type]
    # The step over the loop's own scope first: the outer inputs change
    # with it from one example to the next (the longer list is often the
    # larger input), so a step read off them fits the rows and not the
    # loop. Only if that fails are they offered.
    width = len(settled)
    own = [(args[:width], out) for args, out in rows]
    for pushed in (Spec(f"{spec.name}/step", list(settled), body_type,
                        _unique(own)[:SUB_EXAMPLES]),
                   _sub_spec(spec, settled, body_type, rows)):
        if pushed is None or len(pushed.examples) < 2:
            continue
        found = _solve_hole(solver, pushed, result)
        if found is None:
            continue
        tried = [P.apply(op, [receiver, P.lambda_(settled, found), start])
                 for start in starts]
        hit = solver._check(spec, tried, result)
        if hit is not None:
            return hit
    return None


def _unique(rows) -> list:
    seen, out = set(), []
    for args, out_ in rows:
        key = _key([args, out_])
        if key not in seen:
            seen.add(key)
            out.append((args, out_))
    return out


def deduced(solver, spec: Spec, receiver: P.Expr, values: list,
            result, kinds=("map", "filter", "reduce")) -> P.Expr | None:
    """Try every deduced form over one receiver, whose values on the
    examples are `values`: a sub-spec pushed down, a child solving it."""
    lib = P.library()
    element = _element_type(receiver.type)
    outputs = spec.outputs
    for form in lib.forms:
        if form.receiver != receiver.type or form.name not in kinds:
            continue
        scope = _names(form, spec.params)
        if form.name == "reduce":
            found = _induced(solver, spec, receiver, values, form, scope,
                             result)
            if found is not None:
                return found
            continue
        if form.name == "map":
            targets = []
            if spec.returns.endswith("[]"):
                targets.append((spec.returns[:-2], False))
            if spec.returns == "string":
                targets.append(("string", True))
            for body_type, joined in targets:
                rows = []
                for (args, out), got in zip(spec.examples, values):
                    items = list(out) if joined and isinstance(out, str) \
                        else out
                    if not isinstance(items, list) or not isinstance(
                            got, list) or len(items) != len(got):
                        rows = None
                        break
                    for j, (x, y) in enumerate(zip(got, items)):
                        rows.append(([x, j][:len(scope)] + list(args), y))
                if not rows:
                    continue
                sub = _sub_spec(spec, _settled(scope, body_type), body_type,
                                rows)
                found = _solve_hole(solver, sub, result)
                if found is None:
                    continue
                op = form.op(body_type)
                expr = P.apply(op, [receiver, P.lambda_(
                    _settled(scope, body_type), found)])
                if joined:
                    join = next(one for one in lib.ops if one.key ==
                                "method:join:string[],string->string")
                    expr = P.apply(join, [expr, P.const("", "string")])
                if solver._check(spec, [expr], result) is not None:
                    return expr
        elif form.name == "filter" and spec.returns == receiver.type:
            rows = []
            for (args, out), got in zip(spec.examples, values):
                kept = _subsequence(out, got)
                if kept is None:
                    rows = None
                    break
                for j, (x, keep) in enumerate(zip(got, kept)):
                    rows.append(([x, j][:len(scope)] + list(args), keep))
            if not rows or all(keep for _, keep in rows):
                continue
            sub = _sub_spec(spec, _settled(scope, "boolean"), "boolean",
                            rows)
            found = _solve_hole(solver, sub, result)
            if found is None:
                continue
            expr = P.apply(form.op("boolean"), [receiver, P.lambda_(
                _settled(scope, "boolean"), found)])
            if solver._check(spec, [expr], result) is not None:
                return expr
    return None


def _solve_hole(solver, sub: Spec | None, result) -> P.Expr | None:
    """The hole as a subgoal: a child solver, the parent's memory."""
    if sub is None:
        return None
    from research.v696.search import Solver, Switches
    child = Solver(Switches(meet=True, coarse=True, repair=True,
                            recognition=solver.switches.recognition,
                            learned=solver.switches.learned),
                   memory=solver.memory, depth=SUB_DEPTH,
                   budget=SUB_BUDGET)
    got = child.solve(sub)
    result.evaluated += got.evaluated
    result.subgoals = getattr(result, "subgoals", 0) + 1
    return got.program


def bodies(scope, outer, body_type: str, literals=()) -> list:
    """Small bodies of a type in a hole's scope: its parameters, the outer
    ones, constants, and one operator over them."""
    pool = [P.param(name, kind) for name, kind in scope] + \
        [P.param(name, kind) for name, kind in outer] + \
        [P.const(value, kind) for value, kind in P.CONSTANTS] + \
        list(literals)
    lib = P.library()
    out = [one for one in pool if one.type == body_type]
    scoped = {name for name, _ in scope}
    for op in lib.ops:
        if op.gives != body_type:
            continue
        choices = [[one for one in pool if one.type == need]
                   for need in op.needs]
        if not all(choices):
            continue
        for args in itertools.product(*choices):
            # A body that does not read its scope is a constant.
            if not any(arg.kind == "param" and arg.name in scoped
                       for arg in args):
                continue
            out.append(P.apply(op, args))
            if len(out) >= BODIES * 4:
                break
    return [one for one in out
            if one.kind != "const"][:BODIES * 4]


def applied(spec: Spec, receiver: P.Expr, literals=()) -> list:
    """Every checked form over a receiver, with small bodies: candidates to
    be checked whole, or offered to the forward trie."""
    lib = P.library()
    out = []
    for form in lib.forms:
        if form.receiver != receiver.type:
            continue
        types = [spec.returns] if form.free else [form.body]
        if form.free and spec.returns not in P.TYPES:
            continue
        for body_type in types:
            op = form.op(body_type)
            scope = _settled(_names(form, spec.params), body_type)
            extra_choices = []
            for need in op.needs[2:]:
                extra_choices.append([P.const(value, kind) for value, kind
                                      in P.CONSTANTS if kind == need][:2])
            if not all(extra_choices):
                continue
            for body in bodies(scope, spec.params, body_type,
                               literals)[:BODIES]:
                for extra in itertools.product(*extra_choices):
                    out.append(P.apply(op, [receiver, P.lambda_(scope,
                                                                body),
                                            *extra]))
    return out
