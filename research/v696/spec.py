"""What a function must do: its signature and its examples.

    Spec.params     [(name, type)]
    Spec.returns    the type it gives
    Spec.examples   [(args, output)] the search may look at
    Spec.hidden     [(args, output)] it may not: whether what was found
                    generalises, or only fits what it was shown

A spec's **features** are what recognition walks (`PLAN.md` 2a): its
types, and what the examples show of how the output stands to the inputs
-- the same length, a part of it, a count, one of its elements. They are
read off the values, not off the task's name or words.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from research.v696 import program as P


@dataclass
class Spec:
    name: str
    params: list
    returns: str
    examples: list
    hidden: list = field(default_factory=list)
    #: the program that is known to meet it, for a generated one
    answer: object = None
    #: MultiPL-E's own tests, for an independent one
    tests: str = ""
    entry: str = "f"
    #: what the request says in English, where it says anything
    english: str = ""
    #: what the reader of meaning (`reader.py`) expects of the program:
    #: {"uses": {word: p}, "behaviour": {predicate: p}}
    expected: dict | None = None
    #: programs the decoder proposed, read back into trees (`sketcher.py`)
    proposals: list = field(default_factory=list)
    #: operators beyond the language's: a project's own functions (rung 5),
    #: grown by the search as the library's members are
    library: list = field(default_factory=list)

    @property
    def names(self) -> list:
        return [name for name, _ in self.params]

    @property
    def cases(self) -> list:
        return [args for args, _ in self.examples]

    @property
    def outputs(self) -> list:
        return [output for _, output in self.examples]

    def inputs(self) -> list:
        return [P.param(name, kind) for name, kind in self.params]

    def signature(self) -> str:
        said = ", ".join(f"{name}: {kind}" for name, kind in self.params)
        return f"function {self.entry}({said}): {self.returns}"

    def function(self, body: P.Expr) -> str:
        """The program whole: the helpers it calls, then the function."""
        return (P.prelude([body]) + f"{self.signature()} {{\n  return "
                f"{body.source()};\n}}\n")

    def features(self) -> frozenset:
        """What the examples show, as predicates for the recognition trie:
        the types, then how output stands to input."""
        out = {f"returns {self.returns}"}
        out.update(f"takes {kind}" for _, kind in self.params)
        outputs = self.outputs
        if outputs and all(one == outputs[0] for one in outputs):
            out.add("constant")
        for index, (name, kind) in enumerate(self.params):
            values = [args[index] for args in self.cases]
            pairs = list(zip(values, outputs))
            if all(_length(a) is not None and _length(a) == _length(b)
                   for a, b in pairs):
                out.add(f"same length as {kind}")
            if all(_length(a) is not None and isinstance(b, (int, float))
                   and b == _length(a) for a, b in pairs):
                out.add(f"is the length of {kind}")
            if all(_part(b, a) for a, b in pairs):
                out.add(f"part of {kind}")
            if all(isinstance(a, list) and b in a for a, b in pairs):
                out.add(f"an element of {kind}")
            if all(_same_items(a, b) for a, b in pairs):
                out.add(f"same items as {kind}")
            if all(isinstance(a, str) and isinstance(b, str)
                   and a.lower() == b.lower() for a, b in pairs):
                out.add(f"same letters as {kind}")
        if self.returns == "boolean" and len(set(map(json.dumps,
                                                     outputs))) > 1:
            out.add("both answers seen")
        return frozenset(out)


def _length(value):
    return len(value) if isinstance(value, (str, list)) else None


def _part(small, whole) -> bool:
    if isinstance(small, str) and isinstance(whole, str):
        return small in whole
    if isinstance(small, list) and isinstance(whole, list):
        return all(one in whole for one in small)
    return False


def _same_items(a, b) -> bool:
    return (isinstance(a, list) and isinstance(b, list)
            and sorted(map(json.dumps, a)) == sorted(map(json.dumps, b)))
