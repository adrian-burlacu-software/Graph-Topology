"""Code read back into the search's own trees.

Whatever writes TypeScript -- the decoder, a person, a file in a project --
is only trusted once it is a tree the search could have built: every node
one of the library's operators or forms (`program.library`), with the type
the compiler gives it. `parse` returns that tree, printed the search's way
from then on, or None where the code says something the library cannot.

    parse("function f(xs: number[]): number { return Math.max(...xs); }",
          "f", [("xs", "number[]")])  ->  Expr  (`Math.max(...xs)`)
"""
from __future__ import annotations

from research.v696 import program as P
from research.v696.meaning import interface, word


class Unread(Exception):
    """A node the library has no operator for."""


def _lookup() -> dict:
    lib = P.library()
    table: dict = {}
    for op in lib.ops:
        table.setdefault((word(op), op.kind), []).append(op)
    forms: dict = {}
    for form in lib.forms:
        forms.setdefault((f"{interface(form.receiver)}.{form.name}",
                          form.receiver), []).append(form)
    return {"ops": table, "forms": forms}


_TABLE: dict = {}


def _table() -> dict:
    if not _TABLE:
        _TABLE.update(_lookup())
    return _TABLE


def _op(member: str, kinds, needs: tuple, gives: str,
        spread: bool = False) -> P.Op:
    for kind in kinds:
        for op in _table()["ops"].get((member, kind), ()):
            if op.needs == needs and op.gives == gives \
                    and op.spread == spread:
                return op
    raise Unread(f"{member}{needs}->{gives}")


def _typed(node: dict) -> dict:
    """The compiler's type as the library says it: `number | undefined`
    (what `pop()` gives) is a number, as `program._plain` reads it."""
    if "type" in node and "=>" not in node["type"]:
        node["type"] = P._plain(node["type"]) or node["type"]
    return node


def _read(node: dict, scope: dict) -> P.Expr:
    node = _typed(node)
    kind = node["k"]
    if kind == "id":
        if node["name"] not in scope:
            raise Unread(node["name"])
        name, type_ = scope[node["name"]]
        return P.param(name, type_)
    if kind == "lit":
        if node["type"] not in P.SCALARS:
            raise Unread(str(node["value"]))
        return P.const(node["value"], node["type"])
    if kind in ("bin", "pre"):
        args = [_read(one, scope) for one in node["args"]]
        return P.apply(_op(node["op"], ("operator",),
                           tuple(one.type for one in args), node["type"]),
                       args)
    if kind == "cond":
        args = [_read(one, scope) for one in node["args"]]
        return P.apply(_op("?:", ("ternary",),
                           tuple(one.type for one in args), node["type"]),
                       args)
    if kind == "prop":
        receiver = _read(node["recv"], scope)
        return P.apply(_op(node["member"], ("property",),
                           (receiver.type,), node["type"]), [receiver])
    if kind == "call":
        return _call(node, scope)
    raise Unread(node.get("text", kind))


def _call(node: dict, scope: dict) -> P.Expr:
    node = _typed(node)
    member = node["member"]
    if node["recv"] is None:
        spread = bool(node["args"]) and node["args"][-1]["k"] == "spread"
        args = [_read(one["args"][0] if one["k"] == "spread" else one,
                      scope) for one in node["args"]]
        return P.apply(_op(member, ("function",),
                           tuple(one.type for one in args), node["type"],
                           spread), args)
    receiver = _read(node["recv"], scope)
    if node["args"] and node["args"][0]["k"] == "arrow":
        return _form(node, receiver, scope)
    spread = bool(node["args"]) and node["args"][-1]["k"] == "spread"
    args = [_read(one["args"][0] if one["k"] == "spread" else one, scope)
            for one in node["args"]]
    return P.apply(_op(member, ("method",),
                       (receiver.type, *(one.type for one in args)),
                       node["type"], spread), [receiver, *args])


def _form(node: dict, receiver: P.Expr, scope: dict) -> P.Expr:
    """A member called with a callback: the callback's parameters renamed
    to the form's own scope names, its body read in that scope."""
    from research.v696.forms import _names, _settled
    arrow = node["args"][0]
    outer = [value for value in scope.values()]
    for form in _table()["forms"].get((node["member"], receiver.type), ()):
        names = _names(form, outer)
        if len(arrow["params"]) > len(names):
            continue
        try:
            inner = dict(scope)
            body_scope = []
            for at, (name, kind) in enumerate(names):
                body_scope.append((name, kind))
            # The body's type settles a free `U`; read it with the scope's
            # declared types, trying the body as the compiler typed it.
            body_type = _typed(arrow["body"])["type"]
            settled = _settled(body_scope, body_type)
            for given, (name, kind) in zip(arrow["params"], settled):
                inner[given] = (name, kind)
            body = _read(arrow["body"], inner)
            op = form.op(body.type)
            extra = [_read(one, scope) for one in node["args"][1:]]
            if tuple(one.type for one in extra) != op.needs[2:] \
                    or op.gives != node["type"]:
                continue
            return P.apply(op, [receiver, P.lambda_(settled, body),
                                *extra])
        except Unread:
            continue
    raise Unread(node["member"])


def parse(source: str, entry: str, params) -> P.Expr | None:
    """`entry`'s returned expression as the search's tree, or None."""
    from research.v696.checker import CheckerError, checker
    try:
        node = checker().tree(source, entry)
    except CheckerError:
        return None
    if node is None:
        return None
    scope = {name: (name, kind) for name, kind in params}
    try:
        return _read(node, scope)
    except (Unread, KeyError, TypeError, IndexError):
        return None
