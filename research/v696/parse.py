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

import dataclasses
import json

from research.v696 import program as P
from research.v696.meaning import interface, word


class Unread(Exception):
    """A node the library has no operator for."""


def _lookup() -> dict:
    lib = P.library()
    table: dict = {}
    for op in lib.ops + lib.read_only:
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


def _vague(type_: str) -> bool:
    """A type the compiler could not say: `any`, or nothing yet."""
    return type_ is None or "any" in type_ or "never" in type_ \
        or "unknown" in type_


def _op(member: str, kinds, needs: tuple, gives: str,
        spread: bool = False, made: str | None = None) -> P.Op:
    """The library's operator for this, or -- where reading needs one
    the search does not grow (`made`, the kind) -- one made here with the
    compiler's types: it prints the same code, so it runs the same."""
    for kind in kinds:
        for op in _table()["ops"].get((member, kind), ()):
            if op.needs == needs and op.spread == spread and (
                    op.gives == gives or _vague(gives)):
                return op
    if made is None or member is None:
        raise Unread(f"{member}{needs}->{gives}")
    name = member.split(".")[-1] if made in ("method", "property") \
        else member
    return P.Op(name, made, needs, gives, spread)




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

    def helper(self, name: str, given: tuple) -> P.Op:
        """A helper function as an operator, for the types it is called
        with: a parameter the file leaves untyped (JavaScript) takes the
        type of what it is given. Calling itself, it is a call to the
        function being read (recursion)."""
        read = self.functions[name]
        declared = [(one, _typed({"type": kind})["type"])
                    for one, kind in read.get("params", ())]
        if len(declared) != len(given):
            raise Unread(f"{name} called with {len(given)} of "
                         f"{len(declared)}")
        params = [(one, kind if not _vague(kind) else other)
                  for (one, kind), other in zip(declared, given)]
        key = (name, tuple(kind for _, kind in params))
        if key in self.helpers:
            return self.helpers[key]
        returns = _typed({"type": read["returns"]})["type"]
        if key in self.reading:
            return P.Op(name, "recurse", key[1], returns)
        self.reading.add(key)
        try:
            body = self.function(name, params)
        finally:
            self.reading.discard(key)
        if body.type != returns and not _vague(returns) \
                and not _vague(body.type):
            raise Unread(f"{name} returns {body.type}, said {returns}")
        op = P.Op(name, "helper", key[1],
                  body.type if _vague(returns) else returns,
                  params=tuple(one for one, _ in params), body=body)
        self.helpers[key] = op
        return op

    def read(self, node: dict, scope: dict) -> P.Expr:
        expr = self._read(node, scope)
        if "span" in node and expr.where is None:
            where = {key: tuple(node[key]) for key in ("span", "opspan",
                                                       "namespan")
                     if key in node}
            if "said" in node:
                where["said"] = node["said"]
            expr = dataclasses.replace(expr, where=where)
        return expr

    def _read(self, node: dict, scope: dict) -> P.Expr:
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
                               node["type"], made="operator"), args)
        if kind == "cond":
            args = [self.read(one, scope) for one in node["args"]]
            gives = node["type"] if not _vague(node["type"]) \
                else args[1].type
            return P.apply(_op("?:", ("ternary",),
                               tuple(one.type for one in args), gives,
                               made="ternary"), args)
        if kind == "prop":
            receiver = self.read(node["recv"], scope)
            return P.apply(_op(node["member"], ("property",),
                               (receiver.type,), node["type"],
                               made="property"), [receiver])
        if kind == "opaque":
            # what the tree does not model: its own text, its holes read
            text, holes = node["text"], sorted(node["holes"],
                                               key=lambda one: one["at"][0])
            parts, at = [], 0
            for hole in holes:
                parts.append(text[at:hole["at"][0]])
                at = hole["at"][1]
            parts.append(text[at:])
            args = [self.read(hole["value"], scope) for hole in holes]
            return P.apply(P.Op(json.dumps(parts), "opaque",
                                tuple(one.type for one in args),
                                node["type"]), args)
        if kind == "let":
            value = self.read(node["value"], scope)
            inner = dict(scope)
            inner[node["key"]] = P.param(node["key"], f"lazy:{value.type}")
            body = self.read(node["body"], inner)
            return P.apply(P.let_op(value.type, body.type),
                           [value, P.lambda_([(node["key"],
                                               f"lazy:{value.type}")],
                                             body)])
        if kind == "ref":
            if node["key"] not in scope:
                raise Unread("a shared value read outside where it is bound")
            binding = scope[node["key"]]
            return P.apply(P.force_op(binding.type[len("lazy:"):]),
                           [binding])
        if kind == "inline":
            # a helper made inside the function: its body, its parameters
            # what it was given
            fn = node["fn"]
            args = [self.read(one, scope) for one in node["args"]]
            if len(args) != len(fn["params"]):
                raise Unread("a helper called with fewer or more")
            inner = dict(scope)
            inner.update(zip(fn["params"], args))
            return self.read(fn["body"], inner)
        if kind in ("effect", "setitem"):
            args = [self.read(one, scope) for one in node["args"]]
            name = node.get("member", "[]=")
            return P.apply(P.Op(name, kind, tuple(one.type for one in args),
                                args[0].type), args)
        if kind == "call":
            return self._call(node, scope)
        if kind == "append":
            args = [self.read(one, scope) for one in node["args"]]
            return P.apply(_op("[...]", ("append",),
                               tuple(one.type for one in args),
                               node["type"]), args)
        if kind == "tuple":
            args = [self.read(one, scope) for one in node["args"]]
            return P.apply(P.tuple_op([one.type for one in args]), args)
        if kind == "while":
            start = self.read(node["args"][0], scope)
            inner = dict(scope)
            inner[node["state"]] = P.param(node["state"], start.type)
            state = [(node["state"], start.type)]
            test = self.read(node["args"][1], inner)
            update = self.read(node["args"][2], inner)
            if test.type != "boolean" or update.type != start.type:
                raise Unread("a loop whose state changes its type")
            return P.apply(P.while_op(start.type),
                           [start, P.lambda_(state, test),
                            P.lambda_(state, update)])
        if kind == "index" and node["args"][0]["k"] != "lit" and \
                P.elements(_typed(node["args"][0]).get("type", "")):
            tuple_, at = [self.read(one, scope) for one in node["args"]]
            if at.kind != "const" or not P.elements(tuple_.type):
                raise Unread("a tuple read at a place not fixed")
            return P.apply(P.element_op(tuple_.type, at.value), [tuple_, at])
        if kind == "index":
            args = [self.read(one, scope) for one in node["args"]]
            return P.apply(_op("[i]", ("index",),
                               tuple(one.type for one in args),
                               node["type"], made="index"), args)
        if kind == "arrow":
            # a callback given where no form takes it (`sort`'s comparator
            # in a change made in place): as it was written
            return self._lambda(node, scope)
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
                op = self.helper(member, tuple(one.type for one in args))
                return P.apply(op, args)
            made = "function" if node.get("global") \
                or (member or "").startswith("Math.") else None
            return P.apply(_op(member, ("function",),
                               tuple(one.type for one in args),
                               node["type"], spread, made), args)
        receiver = self.read(node["recv"], scope)
        if node["args"] and node["args"][0]["k"] == "arrow":
            return self._form(node, receiver, scope)
        spread = bool(node["args"]) and node["args"][-1]["k"] == "spread"
        args = [self.read(one["args"][0] if one["k"] == "spread" else one,
                          scope) for one in node["args"]]
        return P.apply(_op(member, ("method",),
                           (receiver.type, *(one.type for one in args)),
                           node["type"], spread, made="method"),
                       [receiver, *args])

    def _form(self, node: dict, receiver: P.Expr, scope: dict) -> P.Expr:
        """A member called with a callback: the callback's parameters
        renamed to the form's own scope names, its body read in that
        scope."""
        from research.v696.forms import _names, _settled
        arrow = node["args"][0]
        outer = [(one.name, one.type) for one in scope.values()
                 if one.kind == "param"]
        inside = "no form for this receiver"
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
                        or (op.gives != node["type"]
                            and not _vague(node["type"])):
                    continue
                return P.apply(op, [receiver, P.lambda_(settled, body),
                                    *extra])
            except Unread as reason:
                inside = str(reason)
                continue
        if node["member"] is not None:
            # No form of the library's: the callback as it was written, its
            # parameters its own and typed as the compiler has them.
            try:
                return self._callback(node, receiver, scope, arrow)
            except Unread as reason:
                inside = f"{inside}; as written: {reason}"
        # what stopped it inside, not only which member it was
        raise Unread(f"{node['member']} on {receiver.type}: {inside}")

    def _lambda(self, arrow: dict, scope: dict) -> P.Expr:
        """A callback as it was written: its own parameters, typed as the
        compiler has them, its body read with them."""
        said = arrow["type"]
        inside = said[said.index("(") + 1:said.index(")")] \
            if "(" in said and ")" in said else ""
        kinds = [part.partition(":")[2].strip()
                 for part in P._split_top(inside)] if inside.strip() else []
        if len(kinds) < len(arrow["params"]):
            raise Unread("a callback's parameters untyped")
        own = [(name, _typed({"type": kind})["type"])
               for name, kind in zip(arrow["params"], kinds)]
        inner = dict(scope)
        inner.update({name: P.param(name, kind) for name, kind in own})
        return P.lambda_(own, self.read(arrow["body"], inner))

    def _callback(self, node, receiver, scope, arrow) -> P.Expr:
        callback = self._lambda(arrow, scope)
        extra = [self.read(one, scope) for one in node["args"][1:]]
        op = P.Op(node["member"].split(".")[-1], "method",
                  (receiver.type, callback.type,
                   *(one.type for one in extra)), node["type"])
        return P.apply(op, [receiver, callback, *extra])


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
