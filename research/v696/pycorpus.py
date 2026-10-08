"""The code-meaning corpora in Python, beside TypeScript's (`teach_meaning`).

The models of the code world are taught both languages at once
(`research/v699/PLAN.md`: one model per role, the language in its
prompt). What TypeScript's corpora are made of, Python's are made of too:

    docs-py.jsonl            each member of Python's library (`pylibrary`)
                             with its own docstring, run on generated values
    solutions-py.jsonl       MBPP's own solutions (no teacher needed: they
                             are people's), kept where they pass the tests;
    solutions-held-py.jsonl  HumanEval's canonical ones, to measure with
    described-py.jsonl       generated Python programs said in English by
                             the teacher, kept by a round trip: the English
                             rewritten as Python by the teacher does what
                             the program does on every input

Every record is in `teach_meaning`'s form with `"language": "python"`; the
uses and root of a program are said in the shared vocabulary
(`pystructure.py`), so the reader learns one meaning for both languages.

    python -m research.v696.pycorpus docs | solutions | described
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import random
import re
import sys
import time

from research.v696 import generating as G
from research.v696 import language as L
from research.v696 import meaning as M
from research.v696 import program as P
from research.v696 import pylibrary
from research.v696.checker import CheckerError, checker
from research.v696.teach_meaning import (DATA, TEACH, Teacher, _append,
                                         _english, _kept, split)

DOCS = DATA / "docs-py.jsonl"
SOLUTIONS = DATA / "solutions-py.jsonl"
SOLUTIONS_HELD = DATA / "solutions-held-py.jsonl"
DESCRIBED = DATA / "described-py.jsonl"
PY = L.of("python")
FENCE = re.compile(r"```(?:py|python|python3)?\s*\n(.*?)```", re.S)


def code_in(reply: str, entry: str) -> str | None:
    """The function named `entry` in a reply, fenced or not -- and what it
    imports before it."""
    found = FENCE.findall(reply)
    text = found[0] if found else reply
    at = text.find(f"def {entry}")
    if at < 0:
        return None
    head = "".join(line + "\n" for line in text[:at].splitlines()
                   if line.startswith(("import ", "from ")))
    return head + text[at:].strip() + "\n"


# -- the library's members, with their own docstrings ------------------------------

#: a receiver's Python type, for its methods' docstrings
OBJECTS = {"string": str, "number[]": list, "string[]": list,
           "boolean[]": list}


def _doc_of(op: P.Op) -> str | None:
    """The first sentences of the docstring of what a template calls."""
    try:
        node = ast.parse(op.py.format(*[f"_a{at}" for at in
                                        range(len(op.needs))]),
                         mode="eval").body
    except SyntaxError:
        return None
    # one call, of its arguments as given: where a template is more (a
    # `reverse=True`, a call wrapped in another, an element taken), the
    # callee's docstring is not what it does
    if not isinstance(node, ast.Call) or node.keywords or not all(
            isinstance(one, ast.Name) and one.id.startswith("_a")
            for one in node.args):
        return None
    callee = node.func
    target = None
    if isinstance(callee, ast.Attribute):
        if isinstance(callee.value, ast.Name) and callee.value.id in (
                "math", "statistics"):
            target = getattr(__import__(callee.value.id), callee.attr, None)
        elif isinstance(callee.value, ast.Name) and \
                callee.value.id.startswith("_a"):
            owner = OBJECTS.get(op.needs[int(callee.value.id[2:])])
            target = getattr(owner, callee.attr, None) if owner else None
        elif isinstance(callee.value, ast.Constant):
            target = getattr(type(callee.value.value), callee.attr, None)
    elif isinstance(callee, ast.Name):
        import builtins
        target = getattr(builtins, callee.id, None)
    doc = inspect.getdoc(target) if target is not None else None
    if not doc:
        return None
    # the prose, not the signature line some builtins open with
    lines = [one for one in doc.splitlines()
             if one.strip() and not re.match(r"^\w+\(.*\)( -> .*)?$", one)]
    text = " ".join(lines).split(">>>")[0]
    return " ".join(re.split(r"(?<=\.)\s", text)[:2]).strip() or None


def docs() -> int:
    lib = pylibrary.library()
    rng = random.Random(699)
    rows = []
    for op in lib.ops:
        if not op.py:
            continue
        doc = _doc_of(op)
        if not doc:
            continue
        takes = op.needs
        names = ["a", "b", "c"][:len(takes)]
        if len(takes) > 3:
            continue
        cases = [[G._value(kind, rng) for kind in takes] for _ in range(8)]
        body = P.apply(op, [P.param(n, k) for n, k in zip(names, takes)])
        try:
            row = PY.values(names, cases, [body])[0]
        except CheckerError:
            continue
        pairs = [(case, one["value"]) for case, one in zip(cases, row)
                 if "error" not in one]
        params = list(zip(names, takes))
        signature = PY.signature("f", params, op.gives)
        rows.append({
            "name": f"doc-py:{M.word(op)}:{','.join(takes)}:{op.py}",
            "source": "docs-py", "language": "python", "split": "train",
            "english": doc, "signature": signature, "entry": "f",
            "examples": pairs[:3], "body": PY.function("f", params,
                                                       op.gives, body),
            "meaning": M.Meaning(tuple(takes), op.gives, M.behaviour(pairs),
                                 frozenset([M.word(op)]),
                                 M.word(op)).json()})
    DOCS.parent.mkdir(parents=True, exist_ok=True)
    with DOCS.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    print(f"{len(rows)} Python members with their English -> {DOCS}")
    return len(rows)


# -- solutions: people's own, kept by the tests -----------------------------------

def solutions() -> dict:
    from research.v696 import tasks
    counts = {}
    for config, path in (("mbpp-py", SOLUTIONS),
                         ("humaneval-py", SOLUTIONS_HELD)):
        kept = 0
        with path.open("w", encoding="utf-8") as out:
            for task in tasks.load(config):
                code = task.solution or None
                if code and checker("python").tests(
                        code + "\n\n" + task.tests, timeout=10000) is not None:
                    code = None
                kept += code is not None
                out.write(json.dumps({"name": task.name, "code": code})
                          + "\n")
        counts[config] = kept
    print(counts)
    return counts


# -- generated programs, said in English, kept by a round trip ---------------------

DESCRIBE = ("Here is a Python function:\n\n```python\n{code}```\n\n"
            "Write the one-sentence request someone would make for this "
            "function, starting \"Write a function that\". Say what it "
            "computes from its inputs, not how: do not name the methods or "
            "operators it uses. Reply with the sentence only.")
REWRITE = ("{english}\n\nComplete this Python function. Reply with only "
           "the whole function in one ```python block.\n\n```python\n"
           "{signature}\n```")


def _programs(count: int) -> list:
    """Generated Python programs at rungs 1 and 2, from the teaching seeds
    (TypeScript's own: the same seeds, Python's library)."""
    out, seed = [], TEACH
    while len(out) < count:
        rung = seed % 4
        found = (G.task2(seed, 1, "python") if rung == 3
                 else G.task(seed, 1 + rung, "python"))
        if found is not None:
            out.append(found)
        seed += 1
    return out


def described(count: int = 3000, batch: int = 16, samples: int = 2) -> None:
    kept = _kept(DESCRIBED)
    todo = [one for one in _programs(count) if one.name not in kept]
    print(f"{len(todo)} generated Python programs to describe", flush=True)
    teacher = Teacher()
    agreed = 0
    started = time.time()
    for at in range(0, len(todo), batch):
        chunk = todo[at:at + batch]
        codes = [one.function(one.answer) for one in chunk]
        sentences = teacher.write([DESCRIBE.format(code=code)
                                   for code in codes], longest=60,
                                  samples=samples, temperature=0.8)
        asks, whose = [], []
        for index, (spec, said) in enumerate(zip(chunk, sentences)):
            for sentence in {one.split("\n")[0].strip() for one in said}:
                if sentence:
                    asks.append(REWRITE.format(english=sentence,
                                               signature=spec.signature()))
                    whose.append((index, sentence))
        rewritten = teacher.write(asks, longest=200) if asks else []
        good = {}
        for (index, sentence), reply in zip(whose, rewritten):
            spec = chunk[index]
            code = code_in(reply[0], spec.entry)
            if code is None or index in good:
                continue
            cases = [args for args, _ in spec.examples + spec.hidden]
            wanted = [out for _, out in spec.examples + spec.hidden]
            got = checker("python").run(code, spec.entry, cases)
            if all("value" in one and json.dumps(one["value"])
                   == json.dumps(out) for one, out in zip(got, wanted)):
                good[index] = sentence
        for index, (spec, code) in enumerate(zip(chunk, codes)):
            agreed += index in good
            _append(DESCRIBED, {"name": spec.name, "code": code,
                                "english": good.get(index),
                                "said": sentences[index]})
        print(f"  {at + len(chunk)}/{len(todo)}: {agreed} agreed "
              f"({time.time() - started:.0f}s)", flush=True)


# -- the records, for the corpus ------------------------------------------------------

def _task_records(config: str, solved: dict) -> list:
    from research.v696 import tasks
    rows = []
    for task in tasks.load(config):
        pairs = M.values_of(M.test_pairs_python(task.tests), "python")
        if not pairs:
            continue
        code = solved.get(task.name)
        uses = root = None
        if code:
            uses, root = M.structure(code, task.entry, "python")
        spec_signature = PY.signature(task.entry, task.params, task.returns,
                                      task.written)
        rows.append({
            "name": task.name, "source": config, "language": "python",
            "split": split(task.name.replace("-py", "")),
            "english": _english(task.prompt), "signature": spec_signature,
            "entry": task.entry, "examples": pairs[:3], "body": code,
            "meaning": M.Meaning(tuple(kind for _, kind in task.params),
                                 task.returns, M.behaviour(pairs), uses,
                                 root).json()})
    return rows


def _described_records() -> list:
    if not DESCRIBED.exists():
        return []
    written = [json.loads(line) for line in DESCRIBED.open(encoding="utf-8")]
    programs = {one.name: one for one in _programs(len(written))}
    rows = []
    for row in written:
        spec = programs.get(row["name"])
        if spec is None:
            continue
        pairs = spec.examples + spec.hidden
        uses, root = M.structure(row["code"], spec.entry, "python")
        rows.append({
            "name": row["name"], "source": "generated-py",
            "language": "python", "split": "train",
            "english": row["english"], "signature": spec.signature(),
            "entry": spec.entry, "examples": spec.examples[:3],
            "body": row["code"],
            "meaning": M.Meaning(tuple(kind for _, kind in spec.params),
                                 spec.returns, M.behaviour(pairs), uses,
                                 root).json()})
    return rows


def records() -> list:
    """Every Python record, for `teach_meaning.corpus`."""
    solved = {}
    for path in (SOLUTIONS, SOLUTIONS_HELD):
        if path.exists():
            for line in path.open(encoding="utf-8"):
                row = json.loads(line)
                if row["code"]:
                    solved[row["name"]] = row["code"]
    rows = []
    if DOCS.exists():
        rows += [json.loads(line) for line in DOCS.open(encoding="utf-8")]
    rows += _task_records("mbpp-py", solved)
    rows += _task_records("humaneval-py", solved)
    rows += _described_records()
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job", choices=("docs", "solutions", "described"))
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args(argv)
    if args.job == "docs":
        docs()
    elif args.job == "solutions":
        solutions()
    else:
        described(args.count, batch=args.batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
