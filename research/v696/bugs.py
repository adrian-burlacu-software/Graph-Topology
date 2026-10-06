"""Rung 4's benchmarks: programs that exist and are wrong.

    humanevalfix   164 bugs people wrote into HumanEval's solutions
                   (HumanEvalPack, JavaScript), each body typed by
                   MultiPL-E's TypeScript signature for the same task, the
                   cases MultiPL-E's tests, judged in the end by both test
                   files. Independent, and held.
    generated      a verified program that reads, made wrong by one edit
                   (`editing.edits`, at random) that fails a test: from
                   MBPP for dev, from HumanEval for held.

    python -m research.v696.bugs fetch        data/humanevalfix/js.jsonl
"""
from __future__ import annotations

import json
import random
import re
import sys
import urllib.request
from pathlib import Path

from research.v696 import meaning as M
from research.v696 import tasks
from research.v696.editing import Bug, edits, _passes
from research.v696.parse import parse

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "humanevalfix"
ROWS = ("https://datasets-server.huggingface.co/rows?dataset=bigcode/"
        "humanevalpack&config=js&split=test&offset={offset}&length=100")
COUNT = 164

#: `console.assert` only logs; the tests are judged by one that throws.
ASSERTING = ("const console = { log: () => {}, assert: (holds, ...said) => "
             "{ if (!holds) throw new Error(\"assertion failed\"); } };\n")


def fetch(config: str = "js") -> int:
    """HumanEvalPack's bugs in a language (`js`, `python`)."""
    rows = []
    url = ROWS.replace("config=js", f"config={config}")
    for offset in range(0, COUNT, 100):
        with urllib.request.urlopen(url.format(offset=offset)) as reply:
            rows += [one["row"] for one in json.load(reply)["rows"]]
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{config}.jsonl"
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    print(f"{len(rows)} bugs -> {path}")
    return len(rows)


def humanevalfix_python() -> list:
    """Python's bugs (v699), each its declaration -- typed as HumanEval's
    own -- and the body written wrong, judged in the end by its own
    `check` and the task's tests; the cases those the task's tests say."""
    typed = {int(task.name.split("_")[1]): task
             for task in tasks.load("humaneval-py")}
    out = []
    for line in (DATA / "python.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        number = int(row["task_id"].split("/")[1])
        task = typed.get(number)
        if task is None:
            continue
        entry = row["entry_point"]
        source = row["declaration"] + row["buggy_solution"]
        pairs = M.values_of(M.test_pairs_python(task.tests), "python")
        if not pairs:
            continue
        tests = row["test"] + f"\ncheck({entry})\n"
        out.append(Bug(f"fixpy-{number}-{row['bug_type'].replace(' ', '-')}",
                       source, entry, list(task.params), task.returns,
                       [list(args) for args, _ in pairs],
                       [want for _, want in pairs], tests,
                       language="python"))
    return out


def humanevalfix() -> list:
    """Each bug typed by MultiPL-E's signature for its task, when there is
    one and the parameters agree."""
    typed = {}
    for task in tasks.load("humaneval-ts"):
        number = int(task.name.split("_")[1])
        typed[number] = task
    out = []
    for line in (DATA / "js.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        number = int(row["task_id"].split("/")[1])
        task = typed.get(number)
        if task is None:
            continue
        found = re.search(r"\(([^)]*)\)", row["declaration"])
        names = [one.strip() for one in found.group(1).split(",")
                 if one.strip()] if found else []
        if len(names) != len(task.params):
            continue
        params = [(name, kind) for name, (_, kind) in zip(names,
                                                           task.params)]
        entry = row["entry_point"]
        said = ", ".join(f"{name}: {kind}" for name, kind in params)
        # what the task declares before the function (a helper it calls)
        # is part of the program
        before = row["declaration"]
        helpers = before[:max(before.rfind(f"const {entry} "), 0)].strip()
        source = ((helpers + "\n\n" if helpers else "")
                  + f"function {entry}({said}): {task.returns} {{\n"
                  + row["buggy_solution"])
        pairs = M.values_of(M.test_pairs(task.tests))
        if not pairs:
            continue
        theirs = task.tests.replace(f"let candidate = {task.entry};",
                                    f"let candidate = {entry};")
        tests = ASSERTING + row["test"] + "\n" + theirs
        out.append(Bug(f"fix-{number}-{row['bug_type'].replace(' ', '-')}",
                       source, entry, params, task.returns,
                       [list(args) for args, _ in pairs],
                       [want for _, want in pairs], tests))
    return out


FROZEN = ROOT / "data" / "code-meaning"


def generated(held: bool, count: int = 40, seed: int = 696) -> list:
    """Verified programs made wrong by one edit: MBPP's for dev,
    HumanEval's for held. Made once and kept (`bugs-dev.jsonl`,
    `bugs-held.jsonl`): the edits may grow, the benchmark does not move
    with them."""
    path = FROZEN / f"bugs-{'held' if held else 'dev'}.jsonl"
    if path.exists():
        return [Bug(**json.loads(line))
                for line in path.open(encoding="utf-8")]
    made = _generated(held, count, seed)
    with path.open("w", encoding="utf-8") as out:
        for bug in made:
            out.write(json.dumps(vars(bug)) + "\n")
    return made


def _generated(held: bool, count: int, seed: int) -> list:
    path = ROOT / "data" / "code-meaning" / (
        "solutions-held.jsonl" if held else "solutions.jsonl")
    config = "humaneval-ts" if held else "mbpp-ts"
    by = {one.name: one for one in tasks.load(config)}
    rng = random.Random(seed + (1 if held else 0))
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    rng.shuffle(rows)
    out = []
    for row in rows:
        if len(out) >= count:
            break
        if not row["code"]:
            continue
        task = by[row["name"]]
        tree = parse(row["code"], task.entry, task.params)
        if tree is None:
            continue
        pairs = M.values_of(M.test_pairs(task.tests))
        if not pairs:
            continue
        base = Bug(task.name, row["code"], task.entry, list(task.params),
                   task.returns, [list(args) for args, _ in pairs],
                   [want for _, want in pairs])
        choices = edits(tree, row["code"], task.entry)
        rng.shuffle(choices)
        for one in choices[:12]:
            broken = one.apply(row["code"])
            ok = _passes(base, broken)
            if not all(ok) and parse(broken, task.entry,
                                     task.params) is not None:
                out.append(Bug(f"mut-{task.name}-{one.kind}", broken,
                               task.entry, list(task.params), task.returns,
                               base.cases, base.wanted,
                               original=row["code"]))
                break
    return out


def main(argv=None) -> int:
    said = argv or sys.argv[1:]
    if said[:1] == ["fetch"]:
        fetch(said[1] if len(said) > 1 else "js")
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
