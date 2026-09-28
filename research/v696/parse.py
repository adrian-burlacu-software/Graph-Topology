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


class Module:
    """A file read as the compiler has it (`tscheck.js` `tree`): each
    function its parameters, the names it binds once, its return. A
    binding is a node shared wherever the name is read; a helper is an
    operator carrying its own body (`Op.body`), read once."""

    def __init__(self, functions: dict) -> None:
        self.functions = functions
        self.helpers: dict = {}
        self.reading: set = set()

    def function(self, name: str, params=None) -> P.Expr:
        """What `name` returns, as a tree over its parameters."""
        read = self.functions.get(name)
        if read is None or "unread" in read:
            raise Unread(read["unread"] if read else name)
        if params is None:
            params = [(one, _typed({"type": kind})["type"])
                      for one, kind in read["params"]]
        scope = {one: P.param(one, kind) for one, kind in params}
        for bound, node in read["steps"]:
            scope[bound] = self.read(node, scope)
        return self.read(read["ret"], scope)

    def helper(self, name: str) -> P.Op:
        if name in self.helpers:
            return self.helpers[name]
        if name in self.reading:
            # calling itself: recursion is not a tree (rung 3b's loops)
            raise Unread(f"{name} calls itself")
        self.reading.add(name)
        read = self.functions[name]
        params = [(one, _typed({"type": kind})["type"])
                  for one, kind in read.get("params", ())]
        body = self.function(name, params)
        returns = _typed({"type": read["returns"]})["type"]
        if body.type != returns:
            raise Unread(f"{name} returns {body.type}, said {returns}")
        op = P.Op(name, "helper", tuple(kind for _, kind in params),
                  returns, params=tuple(one for one, _ in params),
                  body=body)
        self.reading.discard(name)
        self.helpers[name] = op
        return op

    def read(self, node: dict, scope: dict) -> P.Expr:
        node = _typed(node)
        kind = node["k"]
        if kind == "id":
            if node["name"] not in scope:
                raise Unread(node["name"])
            return scope[node["name"]]
        if kind == "lit":
            if node["value"] == "Infinity":
                return P.const("Infinity", "number")
            if node["value"] == [] and node["type"].endswith("[]"):
                return P.const([], node["type"])
            if node["type"] not in P.SCALARS:
                raise Unread(str(node["value"]))
            return P.const(node["value"], node["type"])
        if kind in ("bin", "pre"):
            args = [self.read(one, scope) for one in node["args"]]
            return P.apply(_op(node["op"], ("operator",),
                               tuple(one.type for one in args),
                               node["type"]), args)
        if kind == "cond":
            args = [self.read(one, scope) for one in node["args"]]
            return P.apply(_op("?:", ("ternary",),
                               tuple(one.type for one in args),
                               node["type"]), args)
        if kind == "prop":
            receiver = self.read(node["recv"], scope)
            return P.apply(_op(node["member"], ("property",),
                               (receiver.type,), node["type"]), [receiver])
        if kind == "call":
            return self._call(node, scope)
        if kind == "append":
            args = [self.read(one, scope) for one in node["args"]]
            return P.apply(_op("[...]", ("append",),
                               tuple(one.type for one in args),
                               node["type"]), args)
        if kind == "index":
            args = [self.read(one, scope) for one in node["args"]]
            return P.apply(_op("[i]", ("index",),
                               tuple(one.type for one in args),
                               node["type"]), args)
        if kind == "range":
            args = [self.read(one, scope) for one in node["args"]]
            if tuple(one.type for one in args) != P.RANGE.needs:
                raise Unread("a range of what is not a number")
            return P.apply(P.RANGE, args)
        raise Unread(node.get("text", kind))

    def _call(self, node: dict, scope: dict) -> P.Expr:
        member = node["member"]
        if node["recv"] is None:
            spread = bool(node["args"]) and node["args"][-1]["k"] == "spread"
            args = [self.read(one["args"][0] if one["k"] == "spread"
                              else one, scope) for one in node["args"]]
            if member in self.functions:
                op = self.helper(member)
                if tuple(one.type for one in args) != op.needs:
                    raise Unread(member)
                return P.apply(op, args)
            return P.apply(_op(member, ("function",),
                               tuple(one.type for one in args),
                               node["type"], spread), args)
        receiver = self.read(node["recv"], scope)
        if node["args"] and node["args"][0]["k"] == "arrow":
            return self._form(node, receiver, scope)
        spread = bool(node["args"]) and node["args"][-1]["k"] == "spread"
        args = [self.read(one["args"][0] if one["k"] == "spread" else one,
                          scope) for one in node["args"]]
        return P.apply(_op(member, ("method",),
                           (receiver.type, *(one.type for one in args)),
                           node["type"], spread), [receiver, *args])

    def _form(self, node: dict, receiver: P.Expr, scope: dict) -> P.Expr:
        """A member called with a callback: the callback's parameters
        renamed to the form's own scope names, its body read in that
        scope."""
        from research.v696.forms import _names, _settled
        arrow = node["args"][0]
        outer = [(one.name, one.type) for one in scope.values()
                 if one.kind == "param"]
        for form in _table()["forms"].get((node["member"], receiver.type),
                                          ()):
            names = _names(form, outer)
            if len(arrow["params"]) > len(names):
                continue
            try:
                # The body's type settles a free `U`.
                body_type = _typed(arrow["body"])["type"]
                settled = _settled(names, body_type)
                inner = dict(scope)
                for given, (name, kind) in zip(arrow["params"], settled):
                    inner[given] = P.param(name, kind)
                body = self.read(arrow["body"], inner)
                op = form.op(body.type)
                extra = [self.read(one, scope) for one in node["args"][1:]]
                if tuple(one.type for one in extra) != op.needs[2:] \
                        or op.gives != node["type"]:
                    continue
                return P.apply(op, [receiver, P.lambda_(settled, body),
                                    *extra])
            except Unread:
                continue
        raise Unread(node["member"])


def module(source: str) -> Module | None:
    from research.v696.checker import CheckerError, checker
    try:
        return Module(checker().tree(source))
    except CheckerError:
        return None


def parse(source: str, entry: str, params) -> P.Expr | None:
    """`entry`'s return as the search's tree -- its steps shared, its
    helpers operators -- or None where it is more than that."""
    read = module(source)
    if read is None:
        return None
    try:
        return read.function(entry, list(params))
    except (Unread, KeyError, TypeError, IndexError):
        return None


def why(source: str, entry: str, params) -> str | None:
    """What stopped `parse`, or None if nothing did."""
    read = module(source)
    if read is None:
        return "the checker"
    try:
        read.function(entry, list(params))
    except Unread as reason:
        return str(reason)
    except (KeyError, TypeError, IndexError) as error:
        return type(error).__name__
    return None
