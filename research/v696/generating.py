"""Generated tasks: a composition of the library, kept as the known answer.

Unlimited and graded: a seed and a depth give one task, the composition
that made it is its answer, and its examples are what that composition
does on random inputs -- five to search with, five hidden. Dev seeds and
held seeds never meet (`DEV`, `HELD`), so nothing tuned on the one says
anything about the other.

A composition is kept only if it is a task worth setting: it uses every
parameter, runs without error on every input, does not give the same
answer to all of them, and is not simply one of its inputs.
"""
from __future__ import annotations

import json
import random

from research.v696 import program as P
from research.v696.checker import checker
from research.v696.spec import Spec

#: Where dev and held seeds start: they never meet.
DEV = 0
HELD = 1_000_000

#: The signatures tasks are set over.
SHAPES = (
    (("s", "string"),),
    (("xs", "number[]"),),
    (("words", "string[]"),),
    (("n", "number"),),
    (("s", "string"), ("t", "string")),
    (("xs", "number[]"), ("n", "number")),
    (("s", "string"), ("n", "number")),
)

SHOWN = 5
HIDDEN = 5
LETTERS = "abcdefghij"


def _value(kind: str, rng: random.Random):
    if kind == "number":
        return rng.randint(-5, 20)
    if kind == "string":
        size = rng.randint(1, 7)
        word = "".join(rng.choice(LETTERS) for _ in range(size))
        if rng.random() < 0.3:
            cut = rng.randint(1, size)
            word = word[:cut] + " " + word[cut:]
        return word
    if kind == "boolean":
        return rng.random() < 0.5
    inner = kind[:-2]
    return [_value(inner, rng) for _ in range(rng.randint(1, 6))]


def _grow_typed(rng, pool, ops, depth: int, kind: str) -> P.Expr | None:
    if depth == 0:
        fitting = [one for one in pool if one.type == kind]
        return rng.choice(fitting) if fitting else None
    giving = [one for one in ops if one.gives == kind]
    rng.shuffle(giving)
    for op in giving[:20]:
        args = []
        deep = rng.randrange(len(op.needs))
        for index, need in enumerate(op.needs):
            found = _grow_typed(rng, pool, ops,
                                depth - 1 if index == deep else 0, need)
            if found is None:
                break
            args.append(found)
        else:
            return P.apply(op, args)
    return None


def _uses(expr: P.Expr, names) -> bool:
    found = set()

    def walk(one):
        if one.kind == "param":
            found.add(one.name)
        for arg in one.args:
            walk(arg)
    walk(expr)
    return set(names) <= found


def task(seed: int, depth: int) -> Spec | None:
    """One task, or None when this seed's composition is not worth one."""
    rng = random.Random(seed * 7919 + depth)
    shape = rng.choice(SHAPES)
    names = [name for name, _ in shape]
    lib = P.library()
    pool = [P.param(name, kind) for name, kind in shape] + [
        P.const(value, kind) for value, kind in P.CONSTANTS]
    ops = lib.ops
    for _ in range(30):
        answer = _grow_typed(rng, pool, ops, depth,
                             rng.choice(P.TYPES))
        if answer is None or not _uses(answer, names):
            continue
        cases = [[_value(kind, rng) for _, kind in shape]
                 for _ in range(SHOWN + HIDDEN)]
        row = checker().values(names, cases, [answer.source()])[0]
        if any("error" in one for one in row):
            continue
        outputs = [one["value"] for one in row]
        if len({json.dumps(one) for one in outputs}) < 2:
            continue
        if any(all(json.dumps(case[index]) == json.dumps(out)
                   for case, out in zip(cases, outputs))
               for index in range(len(names))):
            continue
        pairs = list(zip(cases, outputs))
        return Spec(f"gen-{seed}-{depth}", list(shape), answer.type,
                    pairs[:SHOWN], pairs[SHOWN:], answer=answer)
    return None


def tasks(count: int, depth: int, held: bool = False) -> list:
    """`count` tasks at one depth, from dev seeds or held ones."""
    out, seed = [], HELD if held else DEV
    while len(out) < count:
        found = task(seed, depth)
        if found is not None:
            out.append(found)
        seed += 1
    return out
