"""Teaching the decoder: a request and its meaning, and the program as a
tree the search could have built.

The decoder (`sketcher.py`) writes TypeScript that is read back into the
search's trees (`parse.py`) and checked like any candidate; what it is
taught is therefore only ever programs in that language, printed the
search's way, each verified:

    generated   the programs of `teach_meaning`, with their English where
                a round trip kept one
    docs        each library member said by its JSDoc
    mbpp        MBPP-TS solved by SmolLM3 as one `return` expression, kept
                when the task's tests pass and it parses into a tree

The request is what the reader of meaning reads (`reader.request`) and the
meaning is what it reads there -- from a view of the record chosen afresh,
so the decoder learns a meaning as the reader gives it, not a perfect one.

    python -m research.v696.teach_sketch expressions   SmolLM3 on MBPP
    python -m research.v696.teach_sketch corpus
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

from research.v696 import teach_meaning as T
from research.v696.checker import CheckerError, checker
from research.v696.parse import parse

EXPRESSIONS = T.DATA / "expressions.jsonl"
SKETCHES = T.DATA / "sketches.jsonl"
#: Rung 3: whole functions as people write them -- steps, loops, helpers --
#: each read into the tree (`parse.py`) and so kept only when it reads.
FUNCTIONS = T.DATA / "sketches-functions.jsonl"

ONE = ("Complete this TypeScript function. Its body must be a single "
       "`return` of one expression: use array and string methods (map, "
       "filter, reduce, split, join, slice, ...), arrow functions and the "
       "ternary; no loops, no local variables, no statements. Reply with "
       "only the whole function in one ```ts block.\n\n```ts\n{prompt}```")


def _params(signature: str) -> list:
    inside = signature[signature.index("(") + 1:signature.rindex(")")]
    from research.v696.tasks import _params as split
    return split(inside) if inside.strip() else []


def expressions(batch: int = 8, samples: int = 6) -> None:
    """MBPP-TS (train and dev) solved in the tree language."""
    from research.v696 import tasks
    kept_names = T._kept(EXPRESSIONS)
    todo = [one for one in tasks.load("mbpp-ts")
            if one.name not in kept_names]
    print(f"{len(todo)} MBPP-TS tasks to solve in one expression",
          flush=True)
    teacher = T.Teacher()
    solved, started = 0, time.time()
    for at in range(0, len(todo), batch):
        chunk = todo[at:at + batch]
        replies = teacher.write([ONE.format(prompt=one.prompt)
                                 for one in chunk], longest=256,
                                samples=samples, temperature=0.8)
        for task, written in zip(chunk, replies):
            kept = None
            for reply in written:
                code = T.code_in(reply, task.entry)
                if not code:
                    continue
                tree = parse(code, task.entry, task.params)
                if tree is None:
                    continue
                try:
                    passed = checker().tests(code + "\n" + task.tests) \
                        is None
                except CheckerError:
                    checker().restart()
                    passed = False
                if passed:
                    kept = tree.source()
                    break
            solved += kept is not None
            T._append(EXPRESSIONS, {"name": task.name, "expression": kept})
        print(f"  {at + len(chunk)}/{len(todo)}: {solved} in the tree "
              f"language ({time.time() - started:.0f}s)", flush=True)


def said_meaning(probs: dict, uses_floor: float = 0.3,
                 behaviour_floor: float = 0.5) -> str:
    """A reading as the decoder is told it: one line."""
    returns = max(probs["returns"], key=probs["returns"].get)
    root = max(probs["root"], key=probs["root"].get)
    uses = sorted((one for one, p in probs["uses"].items()
                   if p >= uses_floor), key=lambda one: -probs["uses"][one])
    behaviour = sorted(one for one, p in probs["behaviour"].items()
                       if p >= behaviour_floor)
    return (f"returns {returns}; made by {root}; uses {', '.join(uses)}; "
            f"{'; '.join(behaviour)}")


def prompt(english: str, code: str, meaning: str | None) -> str:
    """What the decoder is given: the request as the reader read it, and
    the meaning it read -- or no meaning, for the decoder measured without
    one."""
    lines = []
    if english:
        lines.append(english)
    lines.append(code)
    if meaning is not None:
        lines.append(f"meaning: {meaning}")
    return "\n".join(lines)


def _function(signature: str, expression: str) -> str:
    return f"{signature} {{\n  return {expression};\n}}"


def _python_params(signature: str) -> list:
    """(name, engine type) of a Python signature's parameters."""
    import ast

    from research.v696 import pytypes
    try:
        tree = ast.parse(signature.rstrip(":") + ":\n    pass\n")
    except SyntaxError:
        return []
    function = tree.body[0]
    return [(one.arg, pytypes.engine(ast.unparse(one.annotation)
                                     if one.annotation else None))
            for one in function.args.args]


def _python_targets(record: dict, functions: bool, written: dict) -> list:
    """A Python record's programs, as a writer is taught to write them: the
    solution people wrote, where it reads into the tree; a generated
    program or a member, printed from its tree."""
    from research.v696 import language as L
    from research.v696 import pyparse, pyprint
    entry = record.get("entry") or "f"
    params = _python_params(record["signature"])
    out = []
    if record["source"] == "mbpp-py":
        code = written.get(record["name"])
        if code and pyparse.parse(code, entry, params) is not None:
            out.append(code.strip())
    elif record["source"] in ("generated-py", "docs-py") and record["body"]:
        tree = pyparse.parse(record["body"], entry, params)
        if tree is not None:
            returns = record["meaning"]["returns"]
            out.append(L.of("python").function(entry, params, returns,
                                               tree).strip()
                       if functions else pyprint.text(tree))
    return out


def corpus(model: str = "meaning-unixcoder", seed: int = 696,
           functions: bool = False, out: Path | None = None) -> None:
    """The decoder's records. With `functions`, each target is a whole
    function: the verified program as it was written where it reads into
    the tree (steps, loops, helpers), else the tree printed as one.

    What reads grows with the reader, and a decoder is taught from what
    read when it was taught: a corpus made after the reader has widened
    goes to its own file (`out`), for a decoder of its own."""
    out = out or (FUNCTIONS if functions else SKETCHES)
    from research.v696 import reader as R
    rng = random.Random(seed)
    records = R.load()
    expressions = {}
    if EXPRESSIONS.exists():
        for line in EXPRESSIONS.open(encoding="utf-8"):
            row = json.loads(line)
            if row["expression"]:
                expressions[row["name"]] = row["expression"]
    written = {}
    if functions and T.SOLUTIONS.exists():
        for line in T.SOLUTIONS.open(encoding="utf-8"):
            row = json.loads(line)
            if row["code"]:
                written[row["name"]] = row["code"]
    from research.v696 import pycorpus
    if functions and pycorpus.SOLUTIONS.exists():
        for line in pycorpus.SOLUTIONS.open(encoding="utf-8"):
            row = json.loads(line)
            if row["code"]:
                written[row["name"]] = row["code"]
    rows = []
    for record in records:
        if record.get("language") == "python":
            for target in dict.fromkeys(_python_targets(record, functions,
                                                        written)):
                rows.append((record, target))
            continue
        targets = []
        entry = re.search(r"function\s+(\w+)", record["signature"]).group(1) \
            if record["signature"] else None
        if record["source"] == "mbpp-ts":
            if record["name"] in expressions:
                targets.append(_function(record["signature"],
                                         expressions[record["name"]])
                               if functions
                               else expressions[record["name"]])
            code = written.get(record["name"])
            if code and parse(code, entry, _params(record["signature"])):
                targets.append(code.strip())
        elif record["source"] in ("generated", "docs") and record["body"]:
            tree = parse(record["body"], entry, _params(record["signature"]))
            if tree is not None:
                targets.append(_function(record["signature"], tree.source())
                               if functions else tree.source())
        for target in dict.fromkeys(targets):
            rows.append((record, target))
    print(f"{len(rows)} records with a program in the tree language",
          flush=True)
    reader = R.Reader.load(R.LLM / model)
    # the reading each is given: from a view chosen afresh, the full
    # request for dev and held (as at run time)
    views, pairs = [], []
    for record, _ in rows:
        if record["split"] == "train":
            names = [one for one in R.views(record) if one not in (
                "body", "english+body")]
            view = rng.choice(names) if names else \
                "english+signature+examples"
        else:
            view = "english+signature+examples"
        views.append(view)
        pairs.append(R.said(record, R.VIEWS[view]))
    readings = reader.read(pairs)
    with out.open("w", encoding="utf-8") as file:
        for (record, target), probs in zip(rows, readings):
            english, code = R.said(record, R.VIEWS[
                "english+signature+examples" if record["english"]
                else "signature+examples"])
            file.write(json.dumps({
                "name": record["name"], "source": record["source"],
                "split": record["split"], "english": english, "code": code,
                "meaning": said_meaning(probs), "target": target,
                "language": record.get("language", "typescript")}) + "\n")
    count: dict = {}
    for record, _ in rows:
        key = (record["source"], record["split"])
        count[key] = count.get(key, 0) + 1
    for key in sorted(count):
        print(f"  {key[0]:13} {key[1]:5} {count[key]}")
    print(f"-> {out}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job", choices=("expressions", "corpus"))
    parser.add_argument("--functions", action="store_true",
                        help="whole functions as written (rung 3)")
    parser.add_argument("--out", default="",
                        help="the corpus file's name in data/code-meaning "
                             "(a corpus made with a wider reader)")
    parser.add_argument("--reader", default="meaning-unixcoder",
                        help="the reader of meaning whose readings are the "
                             "meaning lines (v699: the bilingual one)")
    args = parser.parse_args(argv)
    if args.job == "expressions":
        expressions()
    else:
        corpus(model=args.reader, functions=args.functions,
               out=T.DATA / args.out if args.out else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
