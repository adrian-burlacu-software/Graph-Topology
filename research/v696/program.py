"""Programs as typed trees with holes, and the library as operators.

A program is never a string template (`PLAN.md`, the ladder): it is a
tree whose every node has a type, printed to TypeScript only to be
checked. That is what lets the same moves -- compose, wrap in a form,
extract a helper, edit -- apply from a one-line body to a project.

    Expr      a parameter, a constant, a hole, or an operator applied
    Op        what one library member does, as the compiler declares it:
              the types it needs, the type it gives -- an action whose
              preconditions are its argument types (`v691.world.Action`)
    library   every Op, read from TypeScript's own declarations
              (`checker.signatures`), for the types a task mentions

Rung 1 (expressions) keeps what needs no function argument: `map` and
`sort` with a comparator take a callback, and a callback is a form with a
hole, rung 2's.
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field

#: The value types rung 1 reads and composes.
SCALARS = ("number", "string", "boolean")
TYPES = SCALARS + tuple(f"{one}[]" for one in SCALARS)

#: What is on hand in any program, beside the task's own values: the
#: small constants every program is written with.
CONSTANTS = ((0, "number"), (1, "number"), (2, "number"), ("", "string"),
             (" ", "string"), (",", "string"), (True, "boolean"),
             (False, "boolean"))

#: The operators the language has that are not library members, as
#: (symbol, argument types, result type).
OPERATORS = (
    ("+", ("number", "number"), "number"),
    ("-", ("number", "number"), "number"),
    ("*", ("number", "number"), "number"),
    ("/", ("number", "number"), "number"),
    ("%", ("number", "number"), "number"),
    ("+", ("string", "string"), "string"),
    ("===", ("number", "number"), "boolean"),
    ("===", ("string", "string"), "boolean"),
    ("<", ("number", "number"), "boolean"),
    (">", ("number", "number"), "boolean"),
    ("<=", ("number", "number"), "boolean"),
    (">=", ("number", "number"), "boolean"),
    ("&&", ("boolean", "boolean"), "boolean"),
    ("||", ("boolean", "boolean"), "boolean"),
    ("!", ("boolean",), "boolean"),
    ("-", ("number",), "number"),
)

GLOBALS = ("parseInt", "parseFloat", "String", "Number", "Boolean", "Math")


@dataclass(frozen=True)
class Op:
    """One thing the language can do: needs these types, gives that one."""

    name: str
    #: method | property | function | operator
    kind: str
    needs: tuple
    gives: str
    #: a method's or property's receiver is `needs[0]`
    #: a rest parameter is spread from an array: `Math.max(...xs)`
    spread: bool = False

    def said(self, args: list) -> str:
        if self.kind == "operator":
            if len(args) == 1:
                return f"({self.name}{args[0]})"
            return f"({args[0]} {self.name} {args[1]})"
        if self.kind == "property":
            return f"{args[0]}.{self.name}"
        rest = list(args[1:] if self.kind == "method" else args)
        if self.spread and rest:
            rest[-1] = f"...{rest[-1]}"
        call = f"{self.name}({', '.join(rest)})"
        return f"{args[0]}.{call}" if self.kind == "method" else call

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.name}:{','.join(self.needs)}->{self.gives}"


@dataclass(frozen=True)
class Expr:
    """A node of a program: its type, and how it is built."""

    type: str
    #: param | const | hole | apply
    kind: str
    name: str = ""
    value: object = None
    op: Op | None = None
    args: tuple = ()

    def source(self) -> str:
        return self.text

    @functools.cached_property
    def text(self) -> str:
        """The node printed, once: a search asks it of the same node many
        times."""
        if self.kind == "param":
            return self.name
        if self.kind == "const":
            return json.dumps(self.value)
        if self.kind == "hole":
            return f"/*?{self.type}*/"
        return self.op.said([one.source() for one in self.args])

    @property
    def size(self) -> int:
        return 1 + sum(one.size for one in self.args)

    @property
    def depth(self) -> int:
        return 1 + max((one.depth for one in self.args), default=0)

    def ops(self) -> list:
        out = [] if self.op is None else [self.op]
        for one in self.args:
            out += one.ops()
        return out


def param(name: str, kind: str) -> Expr:
    return Expr(kind, "param", name=name)


def const(value, kind: str) -> Expr:
    return Expr(kind, "const", value=value)


def apply(op: Op, args) -> Expr:
    return Expr(op.gives, "apply", op=op, args=tuple(args))


def _plain(type_: str) -> str:
    """`number | undefined` is a number that may be left out."""
    parts = [one.strip() for one in type_.split("|")]
    parts = [one for one in parts if one not in ("undefined", "RegExp")]
    return parts[0] if len(parts) == 1 else ""


@dataclass
class Library:
    ops: list = field(default_factory=list)

    def giving(self, type_: str) -> list:
        return [one for one in self.ops if one.gives == type_]

    def needing(self, type_: str) -> list:
        return [one for one in self.ops if type_ in one.needs]


_LIBRARY: dict = {}


def library(types=TYPES) -> Library:
    """Every rung-1 operator over these types: the language's operators,
    and each library member whose parameters and result are among them --
    optional parameters left out, each as the compiler declares it."""
    key = tuple(sorted(types))
    if key in _LIBRARY:
        return _LIBRARY[key]
    from research.v696.checker import checker
    wanted = set(types)
    ops, seen = [], set()

    def add(op: Op) -> None:
        # A call with nothing to work on is a constant, or random
        # (`Math.random()`): not an operator.
        if op.key not in seen and op.needs \
                and all(one in wanted for one in op.needs) \
                and op.gives in wanted:
            seen.add(op.key)
            ops.append(op)

    for symbol, needs, gives in OPERATORS:
        add(Op(symbol, "operator", needs, gives))
    receivers = [one for one in types]
    for signature in checker().signatures(receivers, GLOBALS):
        # Generic members take a callback (rung 2); deprecated ones are
        # what the language itself says not to write.
        if signature["generic"] or signature.get("deprecated"):
            continue
        gives = _plain(signature["returns"])
        receiver = signature["receiver"]
        if not gives:
            continue
        if signature["property"]:
            add(Op(signature["name"], "property", (receiver,), gives))
            continue
        # Each prefix of the optional parameters is its own operator:
        # `slice()`, `slice(start)`, `slice(start, end)`.
        params = signature["params"]
        required = sum(1 for one in params if not one["optional"])
        for count in range(required, len(params) + 1):
            needs, spread, usable = [], False, True
            for one in params[:count]:
                kind = _plain(one["type"])
                if one["rest"]:
                    spread = True
                if not kind or "=>" in kind:
                    usable = False
                    break
                needs.append(kind)
            if not usable:
                break
            if receiver is None:
                add(Op(signature["name"], "function", tuple(needs), gives,
                       spread))
            else:
                add(Op(signature["name"], "method", (receiver, *needs),
                       gives, spread))
    found = Library(ops)
    _LIBRARY[key] = found
    return found
