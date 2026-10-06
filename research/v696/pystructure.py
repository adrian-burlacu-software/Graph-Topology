"""What a Python program is made of, and what its risks are made of -- off
its syntax, in the words TypeScript's are said in.

The reader of meaning is taught both languages (`PLAN.md` v699: one model
per role): what a program uses is said in one vocabulary, so a Python `==`
is TypeScript's `===`, `s.upper()` is `String.toUpperCase`, a comprehension
is `Array.map` (and `Array.filter`), `for x in xs` is `for of`, `for i in
range(n)` is `for`. A node is named by the library operator whose template
it matches (`pyparse`), so what the reader expects and what the search
grows are named alike (`meaning.word`); what no operator writes is named
as Python says it (`len`, `str.isnumeric`).

    structure(source, entry) -> {"uses": {word: count}, "root": word}
    qualities(source, entry) -> {names, decisions, unbounded, mutates,
                                 partial, loops, found}   (`risk.py`)
"""
from __future__ import annotations

import ast
from collections import Counter

from research.v696 import program as P

#: the types every operator is read over: the library whole
ALL = P.TYPES

#: what a method name tells of the receiver it is called on, where the
#: receiver is not read (a list's `append`, a string's `split`)
RECEIVERS = {
    "String": ("upper", "lower", "strip", "lstrip", "rstrip", "split",
               "join", "replace", "startswith", "endswith", "find", "title",
               "capitalize", "swapcase", "isdigit", "isalpha", "isalnum",
               "isupper", "islower", "isspace", "zfill", "format",
               "rsplit", "splitlines", "center", "ljust", "rjust",
               "isnumeric", "isdecimal", "casefold", "partition", "encode"),
    "Array": ("append", "extend", "insert", "pop", "remove", "sort",
              "reverse", "copy", "clear", "index"),
    "Map": ("keys", "values", "items", "get", "setdefault", "update",
            "popitem"),
    "Set": ("add", "discard", "union", "intersection", "difference",
            "issubset", "issuperset", "symmetric_difference"),
}
#: what changes what it is called on
MUTATING = {"append", "extend", "insert", "pop", "remove", "sort", "reverse",
            "clear", "update", "add", "discard", "setdefault", "popitem"}
#: what fails on input it cannot read
PARSING = {"int", "float", "eval", "index", "next"}

COMPARE = {ast.Eq: "===", ast.NotEq: "!==", ast.Lt: "<", ast.Gt: ">",
           ast.LtE: "<=", ast.GtE: ">=", ast.In: "in", ast.NotIn: "not in",
           ast.Is: "is", ast.IsNot: "is not"}
BINARY = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
          ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**", ast.BitXor: "^",
          ast.BitAnd: "&", ast.BitOr: "|", ast.LShift: "<<",
          ast.RShift: ">>", ast.MatMult: "@"}


def _function(tree, entry):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                node.name == entry:
            return node
    return None


def _template(node) -> str | None:
    """The word of the library operator whose template this node is."""
    from research.v696 import meaning as M
    from research.v696 import pyparse
    for op, pattern in pyparse._index(tuple(sorted(ALL))):
        if pyparse._match(pattern, node, {}):
            return M.word(op)
    return None


def _said(node, entry: str) -> str | None:
    """A node's word, as the compiler's structure reading would say it."""
    if isinstance(node, ast.Call):
        found = _template(node)
        if found:
            return found
        callee = node.func
        if isinstance(callee, ast.Name):
            if callee.id == entry:
                return "recursion"
            return callee.id
        if isinstance(callee, ast.Attribute):
            if isinstance(callee.value, ast.Name) and callee.value.id in (
                    "math", "re", "itertools", "functools", "collections",
                    "heapq", "bisect", "statistics", "operator"):
                return f"{callee.value.id}.{callee.attr}"
            for interface, names in RECEIVERS.items():
                if callee.attr in names:
                    return f"{interface}.{callee.attr}"
            return f"?.{callee.attr}"
        return "call"
    if isinstance(node, ast.BinOp):
        return _template(node) or BINARY.get(type(node.op), "?")
    if isinstance(node, ast.Compare):
        return COMPARE.get(type(node.ops[0]), "?")
    if isinstance(node, ast.BoolOp):
        return "&&" if isinstance(node.op, ast.And) else "||"
    if isinstance(node, ast.UnaryOp):
        return {ast.Not: "!", ast.USub: "-", ast.Invert: "~",
                ast.UAdd: "+"}.get(type(node.op), "?")
    if isinstance(node, ast.IfExp):
        return "?:"
    if isinstance(node, ast.Subscript):
        return "slice" if isinstance(node.slice, ast.Slice) else "[i]"
    if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
        return "Array.filter" if node.generators[0].ifs else "Array.map"
    if isinstance(node, (ast.SetComp, ast.DictComp)):
        return "comprehension"
    if isinstance(node, ast.Lambda):
        return "=>"
    if isinstance(node, ast.JoinedStr):
        return "template"
    if isinstance(node, ast.Name):
        return "a name"
    if isinstance(node, ast.Constant):
        return "a literal"
    if isinstance(node, ast.List):
        return "[]"
    if isinstance(node, ast.Dict):
        return "{}"
    if isinstance(node, ast.Tuple):
        return "[,]"
    if isinstance(node, ast.Set):
        return "new Set"
    return None


def structure(source: str, entry: str) -> dict:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"uses": {}, "root": None}
    function = _function(tree, entry)
    if function is None:
        return {"uses": {}, "root": None}
    uses: Counter = Counter()
    root = None

    def walk(node, inside: bool) -> None:
        nonlocal root
        if isinstance(node, ast.For):
            over = node.iter
            counting = isinstance(over, ast.Call) and isinstance(
                over.func, ast.Name) and over.func.id == "range"
            uses["for" if counting else "for of"] += 1
        elif isinstance(node, ast.While):
            uses["while"] += 1
        elif isinstance(node, ast.If):
            uses["if"] += 1
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            uses["let"] += 1
        elif isinstance(node, ast.Match):
            uses["switch"] += 1
        elif isinstance(node, (ast.Break, ast.Continue)):
            uses[type(node).__name__.lower()] += 1
        elif isinstance(node, ast.Starred):
            uses["..."] += 1
        if isinstance(node, (ast.Call, ast.BinOp, ast.Compare, ast.BoolOp,
                             ast.UnaryOp, ast.IfExp, ast.Subscript,
                             ast.ListComp, ast.GeneratorExp, ast.SetComp,
                             ast.DictComp, ast.Lambda, ast.JoinedStr,
                             ast.List, ast.Dict, ast.Set)):
            word = _said(node, entry)
            if word:
                uses[word] += 1
        if isinstance(node, ast.Return) and node.value is not None and \
                not inside:
            root = _said(node.value, entry)
        nested = inside or isinstance(node, (ast.FunctionDef, ast.Lambda,
                                             ast.AsyncFunctionDef))
        for child in ast.iter_child_nodes(node):
            walk(child, nested)

    for statement in function.body:
        walk(statement, False)
    return {"uses": dict(uses), "root": root}


def qualities(source: str, entry: str) -> dict:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"names": 0, "decisions": 0, "unbounded": 0, "mutates": 0,
                "partial": 0, "loops": 0, "found": False}
    function = _function(tree, entry)
    params = {one.arg for one in function.args.args} if function else set()
    names, decisions, unbounded, mutates, partial, loops = (set(), 0, 0, 0,
                                                            0, 0)
    helpers = {node.name for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

    def root_of(node):
        while isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    def walk(node, owner) -> None:
        nonlocal decisions, unbounded, mutates, partial, loops
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name != entry:
                names.add(node.name)
            for one in node.args.args:
                names.add(one.arg)
            for child in ast.iter_child_nodes(node):
                walk(child, node.name)
            return
        if isinstance(node, ast.Lambda):
            for one in node.args.args:
                names.add(one.arg)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else \
                [node.target]
            for target in targets:
                for one in ast.walk(target):
                    if isinstance(one, ast.Name):
                        names.add(one.id)
                if isinstance(target, (ast.Subscript, ast.Attribute)) and \
                        root_of(target) in params:
                    mutates += 1
        if isinstance(node, (ast.For, ast.comprehension)):
            for one in ast.walk(node.target):
                if isinstance(one, ast.Name):
                    names.add(one.id)
        if isinstance(node, (ast.If, ast.IfExp, ast.match_case, ast.Break,
                             ast.Continue)):
            decisions += 1
        if isinstance(node, ast.BoolOp):
            decisions += len(node.values) - 1
        if isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            partial += 1
        if isinstance(node, (ast.Subscript, ast.Raise, ast.Assert)):
            partial += 1
        if isinstance(node, ast.While):
            unbounded += 1
        if isinstance(node, (ast.For, ast.While)):
            loops += 1
        if isinstance(node, ast.Call):
            callee = node.func
            if isinstance(callee, ast.Name):
                if callee.id == owner and owner in helpers:
                    unbounded += 1
                if callee.id in PARSING:
                    partial += 1
            elif isinstance(callee, ast.Attribute):
                if callee.attr in PARSING:
                    partial += 1
                if callee.attr in MUTATING and root_of(callee.value) in params:
                    mutates += 1
        for child in ast.iter_child_nodes(node):
            walk(child, owner)

    walk(tree, None)
    return {"names": len(names), "decisions": decisions,
            "unbounded": unbounded, "mutates": mutates, "partial": partial,
            "loops": loops, "found": function is not None}
