"""Rung 4's edits in Python: every single edit of a program, off its syntax.

`editing.py` finds its sites in the search's tree, which keeps where each
node of a TypeScript source came from; Python's reader (`pyparse.py`)
matches templates and keeps no spans. Python's own `ast` does -- each node's
line and column, start and end -- so the sites are its nodes, and each edit
is the same kind of edit as TypeScript's, made in the text:

    constant   a number nudged by one, negated, 0, 1, 2, -1 or another the
               program says; a truth flipped; a string emptied
    name       another name the function reads (a parameter or a local)
    operator   another of its family (`<` for `<=`, `-` for `+`, `or` for
               `and`), or a `not` / unary minus removed
    member     another method or builtin with the same needs and gives, as
               Python's library has them (`max` for `min`, `rfind` for
               `find`)
    swap       a comparison's or an operation's two sides exchanged
    unwrap     an expression replaced by one of its parts
    wrap       an operation made absolute or negated, a test negated
"""
from __future__ import annotations

import ast
import json
import re

from research.v696.editing import Edit, _numbers

#: operators one may have been meant for another, by family
FAMILIES = (
    ("+", "-", "*", "/", "//", "%", "**"),
    ("<", "<=", ">", ">=", "==", "!="),
    ("and", "or"),
    ("<<", ">>", "&", "|", "^"),
    ("in", "not in"), ("is", "is not"),
)
BINARY = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
          ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**",
          ast.LShift: "<<", ast.RShift: ">>", ast.BitAnd: "&",
          ast.BitOr: "|", ast.BitXor: "^"}
COMPARE = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
           ast.Eq: "==", ast.NotEq: "!=", ast.In: "in", ast.NotIn: "not in",
           ast.Is: "is", ast.IsNot: "is not"}
#: an operation the same whichever side is which: not swapped
COMMUTES = ("+", "*", "==", "!=", "and", "or", "&", "|", "^")


def _family(said: str) -> tuple:
    return next((one for one in FAMILIES if said in one), ())


def _offsets(source: str) -> list:
    """Where each line starts in the text."""
    out, at = [0], 0
    for line in source.splitlines(keepends=True):
        at += len(line)
        out.append(at)
    return out


def _members() -> dict:
    """{a method's or builtin's name: the others with its needs and gives},
    from Python's library (`pylibrary.py`)."""
    from research.v696 import program as P
    lib = P.library(language="python")
    groups: dict = {}
    for op in lib.ops + lib.read_only:
        if not op.py or op.kind not in ("method", "function", "property"):
            continue
        found = re.search(r"\.(\w+)\(", op.py) or \
            re.match(r"^([A-Za-z_][\w.]*)\(", op.py)
        if not found:
            continue
        name = found.group(1).split(".")[-1]
        groups.setdefault((op.kind, tuple(op.needs), op.gives),
                          set()).add(name)
    out: dict = {}
    for names in groups.values():
        for name in names:
            out.setdefault(name, set()).update(names - {name})
    return out


_MEMBERS: dict = {}


def edits(source: str, entry: str) -> list:
    """Every single edit of `source`'s functions -- the entry and those it
    has beside it -- most of a kind together, in the order of the text."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    if not _MEMBERS:
        _MEMBERS.update(_members())
    starts = _offsets(source)

    def span(node) -> tuple | None:
        if getattr(node, "end_lineno", None) is None:
            return None
        return (starts[node.lineno - 1] + node.col_offset,
                starts[node.end_lineno - 1] + node.end_col_offset)

    def text(node) -> str:
        found = span(node)
        return source[found[0]:found[1]] if found else ""

    functions = [one for one in ast.walk(tree)
                 if isinstance(one, (ast.FunctionDef, ast.AsyncFunctionDef))]
    out: list = []
    for function in functions:
        names = {arg.arg for arg in function.args.args}
        names |= {one.id for one in ast.walk(function)
                  if isinstance(one, ast.Name)
                  and isinstance(one.ctx, ast.Store)}
        numbers = sorted({one.value for one in ast.walk(function)
                          if isinstance(one, ast.Constant)
                          and isinstance(one.value, (int, float))
                          and not isinstance(one.value, bool)})
        # its annotations say nothing it does: no edit is made in them
        typed = {id(one) for note in [function.returns]
                 + [arg.annotation for arg in function.args.args]
                 if note is not None for one in ast.walk(note)}
        for node in ast.walk(function):
            where = span(node)
            if where is None or id(node) in typed:
                continue
            start, end = where
            out += _of(node, start, end, source, text, span, names, numbers)
    unique, seen = [], set()
    for one in out:
        key = (one.start, one.end, one.text)
        if key in seen or one.apply(source) == source:
            continue
        seen.add(key)
        unique.append(one)
    unique.sort(key=lambda one: one.start)
    return unique


def _between(source: str, left_end: int, right_start: int,
             said: str) -> tuple | None:
    """Where an operator is said between two sides."""
    found = source.find(said, left_end, right_start)
    if found < 0:
        return None
    return found, found + len(said)


def _of(node, start, end, source, text, span, names, numbers) -> list:
    out = []
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):
            out.append(Edit(start, end, repr(not value), "constant"))
        elif isinstance(value, (int, float)):
            out += [Edit(start, end, json.dumps(one), "constant")
                    for one in _numbers(value, numbers)]
        elif isinstance(value, str) and value:
            out.append(Edit(start, end, "''", "constant"))
        return out
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) \
            and node.id in names:
        return [Edit(start, end, other, "name")
                for other in sorted(names) if other != node.id]
    if isinstance(node, ast.BinOp):
        said = BINARY.get(type(node.op))
        left, right = span(node.left), span(node.right)
        at = _between(source, left[1], right[0], said) if said and left \
            and right else None
        if at:
            out += [Edit(at[0], at[1], other, "operator")
                    for other in _family(said) if other != said]
            if said not in COMMUTES:
                out.append(Edit(left[0], right[1], text(node.right)
                                + source[left[1]:right[0]] + text(node.left),
                                "swap"))
        out += [Edit(start, end, f"abs({text(node)})", "wrap"),
                Edit(start, end, f"-({text(node)})", "wrap")]
    elif isinstance(node, ast.Compare) and len(node.ops) == 1:
        said = COMPARE.get(type(node.ops[0]))
        left, right = span(node.left), span(node.comparators[0])
        at = _between(source, left[1], right[0], said) if said and left \
            and right else None
        if at:
            out += [Edit(at[0], at[1], other, "operator")
                    for other in _family(said) if other != said]
            if said not in COMMUTES:
                out.append(Edit(left[0], right[1], text(node.comparators[0])
                                + source[left[1]:right[0]] + text(node.left),
                                "swap"))
        out.append(Edit(start, end, f"not ({text(node)})", "wrap"))
    elif isinstance(node, ast.BoolOp) and len(node.values) == 2:
        said = "and" if isinstance(node.op, ast.And) else "or"
        left, right = span(node.values[0]), span(node.values[1])
        at = _between(source, left[1], right[0], said) if left and right \
            else None
        if at:
            out.append(Edit(at[0], at[1], "or" if said == "and" else "and",
                            "operator"))
        out.append(Edit(start, end, f"not ({text(node)})", "wrap"))
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op,
                                                      (ast.Not, ast.USub)):
        inner = span(node.operand)
        if inner:
            out.append(Edit(start, inner[0], "", "operator"))
    elif isinstance(node, ast.Call):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else \
            func.id if isinstance(func, ast.Name) else None
        if name in _MEMBERS:
            where = span(func)
            if where:
                a = where[1] - len(name)
                out += [Edit(a, where[1], other, "member")
                        for other in sorted(_MEMBERS[name])]
    for part in ast.iter_child_nodes(node):
        if isinstance(part, ast.expr) and isinstance(node, ast.expr) \
                and not isinstance(node, (ast.Name, ast.Constant)):
            found = span(part)
            if found and found != (start, end):
                out.append(Edit(start, end, text(part), "unwrap"))
    return out
