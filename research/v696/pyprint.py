"""A program tree written out as Python.

The trees are the engine's, whatever the language (`program.py`); this is
how Python writes one. An operator of Python's library carries its own
template (`Op.py`: `"{0}.upper()"`, `"{1}.join({0})"`); what is the
engine's own -- a constant, a callback, a conditional, an element, a range,
a form with its callback -- is written here, the way people write Python:

    c ? a : b               (a if c else b)
    xs.map((x) => x * 2)    [x * 2 for x in xs]
    xs.filter((x) => p)     [x for x in xs if p]
    xs.reduce(f, init)      functools.reduce(lambda acc, x: ..., xs, init)
    xs.some / every         any(... for x in xs) / all(...)

What TypeScript writes as a closure on the spot (a value bound lazily, a
loop as an expression, a change made on a copy) Python writes as a call of
a small runtime the prelude declares (`RUNTIME`).
"""
from __future__ import annotations

import json
import re

from research.v696 import program as P

#: the engine's operator symbols, as Python writes them
OPERATORS = {"===": "==", "!==": "!=", "&&": "and", "||": "or", "!": "not",
             "==": "==", "!=": "!="}

#: what the closures TypeScript writes on the spot are, in Python: a value
#: worked out once when first read, a loop as an expression, a change made
#: on a copy
RUNTIME = '''def _lazy(make):
    box = []
    def get():
        if not box:
            box.append(make())
        return box[0]
    return get


def _while(state, test, update):
    while test(state):
        state = update(state)
    return state


def _changed(container, method, *args):
    import copy
    made = copy.copy(container)
    getattr(made, method)(*args)
    return made


def _set(container, key, value):
    import copy
    made = copy.copy(container)
    made[key] = value
    return made
'''


def literal(value) -> str:
    """A JSON value as a Python literal: `True`, `None`, a set."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if value == "Infinity":
        return "float('inf')"
    if value == "-Infinity":
        return "float('-inf')"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(literal(one) for one in value) + "]"
    if isinstance(value, dict):
        if set(value) == {"$set"}:
            inner = ", ".join(literal(one) for one in value["$set"])
            return "{" + inner + "}" if inner else "set()"
        if set(value) == {"$map"}:
            return "{" + ", ".join(f"{_key(k)}: {literal(v)}"
                                   for k, v in value["$map"]) + "}"
        return "{" + ", ".join(f"{literal(k)}: {literal(v)}"
                               for k, v in value.items()) + "}"
    return json.dumps(value)


def _key(value) -> str:
    if isinstance(value, list):
        return "(" + "".join(literal(one) + ", " for one in value) + ")"
    return literal(value)


def _uses(name: str, text: str) -> bool:
    return re.search(rf"(?<![\w.]){re.escape(name)}(?!\w)", text) is not None


def text(expr: P.Expr) -> str:
    """The tree as Python writes it."""
    if expr.kind == "param":
        return expr.name
    if expr.kind == "const":
        return literal(expr.value)
    if expr.kind == "hole":
        return "..."
    if expr.kind == "lambda":
        names = expr.name.replace(" ", "")
        return f"(lambda {names}: {text(expr.args[0])})"
    return _apply(expr.op, list(expr.args))


def _lambda(expr: P.Expr):
    """(parameter names, body text) of a callback."""
    if expr.kind != "lambda":
        return None, text(expr)
    return ([one.strip() for one in expr.name.split(",") if one.strip()],
            text(expr.args[0]))


def _apply(op: P.Op, args: list) -> str:
    said = [text(one) for one in args]
    if op.py:
        return op.py.format(*said)
    kind = op.kind
    if kind == "operator":
        symbol = OPERATORS.get(op.name, op.name)
        if len(said) == 1:
            return f"({symbol} {said[0]})" if symbol.isalpha() else \
                f"({symbol}{said[0]})"
        return f"({said[0]} {symbol} {said[1]})"
    if kind == "ternary":
        return f"({said[1]} if {said[0]} else {said[2]})"
    if kind == "index":
        return f"{said[0]}[{said[1]}]"
    if kind == "range":
        return f"list(range({said[0]}, {said[1]}))"
    if kind == "append":
        return f"[*{said[0]}, {said[1]}]"
    if kind == "tuple":
        return "(" + "".join(one + ", " for one in said) + ")"
    if kind == "let":
        return f"({said[1]})(_lazy(lambda: {said[0]}))"
    if kind == "force":
        return f"{said[0]}()"
    if kind == "while":
        return f"_while({said[0]}, {said[1]}, {said[2]})"
    if kind == "effect":
        rest = "".join(", " + one for one in said[1:])
        return f"_changed({said[0]}, {op.name!r}{rest})"
    if kind == "setitem":
        return f"_set({said[0]}, {said[1]}, {said[2]})"
    if kind == "form":
        return _form(op, args, said)
    if kind in ("helper", "recurse"):
        return f"{op.name}({', '.join(said)})"
    if kind == "property":
        return f"{said[0]}.{op.name}"
    if kind == "method":
        return f"{said[0]}.{op.name}({', '.join(said[1:])})"
    return f"{op.name}({', '.join(said)})"


def _chars(expr: P.Expr):
    """The string a `list(s)` is made of, or None: a loop goes over the
    string itself."""
    if expr.kind == "apply" and expr.op.py == "list({0})" and \
            expr.args[0].type == "string":
        return expr.args[0]
    return None


def _form(op: P.Op, args: list, said: list) -> str:
    """A member with a callback, as Python writes it: a comprehension, a
    reduction, a test of any or all."""
    receiver = said[0]
    string = _chars(args[0])
    if string is not None:
        receiver = text(string)
    names, body = _lambda(args[1])
    names = names or ["x"]
    element = names[0]
    indexed = len(names) > 1 and _uses(names[1], body)

    def over(target: str = element) -> str:
        if indexed:
            return f"for {names[1]}, {target} in enumerate({receiver})"
        return f"for {target} in {receiver}"

    name = op.name
    if name == "map":
        inner = args[0]
        if inner.kind == "apply" and inner.op.kind == "form" and \
                inner.op.name == "filter" and not indexed:
            # a map of a filter: one comprehension, its test after
            kept, test = _lambda(inner.args[1])
            kept_names = kept or ["x"]
            if not (len(kept_names) > 1 and _uses(kept_names[1], test)):
                source = _chars(inner.args[0])
                source = text(source) if source is not None else \
                    text(inner.args[0])
                if kept_names[0] != element:
                    body = f"(lambda {element}: {body})({kept_names[0]})"
                return (f"[{body} for {kept_names[0]} in {source} "
                        f"if {test}]")
        return f"[{body} {over()}]"
    if name == "filter":
        return f"[{element} {over()} if {body}]"
    if name in ("some", "any"):
        return f"any({body} {over()})"
    if name in ("every", "all"):
        return f"all({body} {over()})"
    if name == "find":
        return f"next(({element} {over()} if {body}), None)"
    if name == "findIndex":
        index = names[1] if len(names) > 1 else "i"
        return (f"next(({index} for {index}, {element} in "
                f"enumerate({receiver}) if {body}), -1)")
    if name == "flatMap":
        return f"[y {over()} for y in {body}]"
    if name == "sumOf":
        return f"sum({body} {over()})"
    if name == "sortedBy":
        return f"sorted({receiver}, key=lambda {element}: {body})"
    if name in ("maxBy", "minBy"):
        return (f"{name[:3]}({receiver}, key=lambda {element}: {body})")
    if name == "reduce":
        # (acc, x[, i]) => body, from `init`
        acc = names[0]
        item = names[1] if len(names) > 1 else "x"
        start = said[2] if len(said) > 2 else None
        if len(names) > 2 and _uses(names[2], body):
            pairs = f"enumerate({receiver})"
            step = (f"lambda {acc}, _p: (lambda {names[2]}, {item}: "
                    f"{body})(*_p)")
            return (f"functools.reduce({step}, {pairs}"
                    + (f", {start})" if start is not None else ")"))
        return (f"functools.reduce(lambda {acc}, {item}: {body}, {receiver}"
                + (f", {start})" if start is not None else ")"))
    rest = "".join(", " + one for one in said[1:])
    return f"{receiver}.{name}({rest[2:]})"


def prelude(exprs) -> str:
    """The runtime the trees need, then their helpers as `def`s."""
    needs = False
    for expr in exprs:
        for op in expr.ops():
            if op.kind in ("let", "while", "effect", "setitem"):
                needs = True
    out = RUNTIME + "\n\n" if needs else ""
    for expr in exprs:
        # a tree that calls itself: the function it is, defined by it, so
        # that the call has something to call
        recursive = next((op for op in expr.ops() if op.kind == "recurse"),
                         None)
        if recursive is not None and recursive.params:
            from research.v696 import pytypes
            said = ", ".join(recursive.params)
            out += (f"def {recursive.name}({said}):\n"
                    f"    return {text(expr)}\n\n\n")
    for op in P.helpers(exprs):
        from research.v696 import pytypes
        said = ", ".join(f"{name}: {pytypes.written(kind)}"
                         for name, kind in zip(op.params, op.needs))
        out += (f"def {op.name}({said}) -> {pytypes.written(op.gives)}:\n"
                f"    return {text(op.body)}\n\n\n")
    return out
