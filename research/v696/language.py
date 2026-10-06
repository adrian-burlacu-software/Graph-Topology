"""The languages code is written in: everything that is a language's own.

The search, its trees and its judgements are the same whatever the
language (`search.py`, `cognition.py`, the risk matrix); what a language
decides is how a tree is written out, how a function is signed, what runs
and checks it (its checker: `tscheck.js` for TypeScript, `pycheck.py` for
Python, one JSON protocol), and how it is named on the page.

    of("python").values(names, cases, trees)   -> each tree's values
    of(spec.language).function(...)            -> the program whole

A tree's `source()` stays its identity everywhere (as TypeScript writes it):
what a program *is* does not change with the language it is shown in.
"""
from __future__ import annotations

from research.v696 import program as P

DEFAULT = "typescript"


class Language:
    name = ""
    #: how it is labelled on a code block, its file extension, the fences
    #: (```ts) it is written in
    label = ""
    extension = ""
    fences: tuple = ()

    def checker(self):
        raise NotImplementedError

    def text(self, expr: P.Expr) -> str:
        """The expression as this language writes it."""
        raise NotImplementedError

    def prelude(self, exprs) -> str:
        """The helpers the expressions call, declared."""
        raise NotImplementedError

    def signature(self, entry: str, params, returns: str,
                  written=None) -> str:
        """`written`: the types as the person wrote them, by parameter and
        "return", where the language has words of its own for them."""
        raise NotImplementedError

    def function(self, entry: str, params, returns: str, body: P.Expr,
                 written=None) -> str:
        """The program whole: the helpers it calls, then the function."""
        raise NotImplementedError

    def parse(self, source: str, entry: str, params):
        """`entry` of `source` read into the search's tree, or None."""
        raise NotImplementedError

    def saying(self, functions: bool) -> str:
        """What a writer is told it writes."""
        raise NotImplementedError

    def example(self, entry: str, args: list, value) -> str:
        """An example as a comment in this language: `f(1) == 2`."""
        raise NotImplementedError

    def wrapped(self, text: str, signature: str) -> str:
        """What a writer wrote, a whole function: a bare expression is the
        body of the function signed so."""
        raise NotImplementedError

    def values(self, names, cases, exprs, timeout: int = 50,
               types=None) -> list:
        """For each tree, its value on each case: {"value"} or {"error"}.
        `types`: the parameters' engine types, where values must be made
        what they say (a tuple, a set)."""
        exprs = list(exprs)
        if not exprs:
            return []
        return self.checker().values(names, cases,
                                     [self.text(one) for one in exprs],
                                     timeout=timeout,
                                     prelude=self.prelude(exprs))


class TypeScript(Language):
    name, label, extension = "typescript", "TS", ".ts"
    fences = ("ts", "typescript", "js", "javascript")

    def saying(self, functions: bool) -> str:
        from research.v696.sketcher import SAYING, SAYING_FUNCTIONS
        return SAYING_FUNCTIONS if functions else SAYING

    def example(self, entry: str, args: list, value) -> str:
        import json
        return f"// {entry}({json.dumps(args)[1:-1]}) === {json.dumps(value)}"

    def checker(self):
        from research.v696.checker import checker
        return checker("typescript")

    def text(self, expr: P.Expr) -> str:
        return expr.source()

    def prelude(self, exprs) -> str:
        return P.prelude(list(exprs))

    def parse(self, source: str, entry: str, params):
        from research.v696.parse import parse
        return parse(source, entry, params)

    def wrapped(self, text: str, signature: str) -> str:
        if "function " in text:
            return text
        text = text.strip().rstrip(";")
        if text.startswith("return "):
            text = text[len("return "):]
        return f"{signature} {{\n  return {text};\n}}\n"

    def signature(self, entry: str, params, returns: str,
                  written=None) -> str:
        said = ", ".join(f"{name}: {kind}" for name, kind in params)
        return f"function {entry}({said}): {returns}"

    def function(self, entry: str, params, returns: str, body: P.Expr,
                 written=None) -> str:
        return (self.prelude([body])
                + f"{self.signature(entry, params, returns)} {{\n  return "
                  f"{self.text(body)};\n}}\n")


class Python(Language):
    name, label, extension = "python", "PY", ".py"
    fences = ("py", "python", "python3")

    def saying(self, functions: bool) -> str:
        """What a writer is told: the language is in it (`PLAN.md` v699:
        one writer for both, the language in its prompt)."""
        if functions:
            return "You write Python: the whole function asked for."
        return ("You write Python: one expression that the function "
                "returns, in the library's own words.")

    def example(self, entry: str, args: list, value) -> str:
        from research.v696.pyprint import literal
        said = ", ".join(literal(one) for one in args)
        return f"# {entry}({said}) == {literal(value)}"

    def checker(self):
        from research.v696.checker import checker
        return checker("python")

    def text(self, expr: P.Expr) -> str:
        from research.v696 import pyprint
        return pyprint.text(expr)

    def prelude(self, exprs) -> str:
        from research.v696 import pyprint
        return pyprint.prelude(list(exprs))

    def parse(self, source: str, entry: str, params):
        from research.v696.pyparse import parse
        return parse(source, entry, params)

    def wrapped(self, text: str, signature: str) -> str:
        if "def " in text:
            return self.function_only(text)
        text = text.strip()
        if text.startswith("return "):
            text = text[len("return "):]
        return f"{signature}\n    return {text}\n"

    @staticmethod
    def function_only(text: str) -> str:
        """What a writer wrote, as its program: the imports and the
        functions, without what it shows after them (`day(0) == 'Sun'`, a
        print of a call)."""
        import ast
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return text
        kept = [one for one in tree.body if isinstance(
            one, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                  ast.AsyncFunctionDef, ast.ClassDef))]
        if not kept or len(kept) == len(tree.body):
            return text
        lines = text.splitlines()
        return "\n\n".join("\n".join(lines[one.lineno - 1:one.end_lineno])
                           for one in kept) + "\n"

    def signature(self, entry: str, params, returns: str,
                  written=None) -> str:
        from research.v696 import pytypes
        written = written or {}
        said = ", ".join(f"{name}: {pytypes.written(kind, written.get(name))}"
                         for name, kind in params)
        back = pytypes.written(returns, written.get("return"))
        return f"def {entry}({said}) -> {back}:"

    def function(self, entry: str, params, returns: str, body: P.Expr,
                 written=None) -> str:
        return (self.prelude([body])
                + f"{self.signature(entry, params, returns, written)}\n"
                  f"    return {self.text(body)}\n")

    def values(self, names, cases, exprs, timeout: int = 50,
               types=None) -> list:
        exprs = list(exprs)
        if not exprs:
            return []
        return self.checker().values(names, cases,
                                     [self.text(one) for one in exprs],
                                     timeout=timeout,
                                     prelude=self.prelude(exprs),
                                     types=types)


LANGUAGES = {one.name: one for one in (TypeScript(), Python())}


def of(name: str | None) -> Language:
    """The language by name; TypeScript where none is said."""
    return LANGUAGES[name or DEFAULT]
