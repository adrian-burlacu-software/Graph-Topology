"""Rung 5's benchmark: small projects, built from verified programs, frozen.

A project is several modules, each exporting one verified function (the
teacher's MBPP solutions for dev, HumanEval's for held -- `teach_meaning`),
a `main.ts` that uses them, and `main.test.ts`, its tests. Three kinds of
task, each made once and kept (`data/code-meaning/projects-*.jsonl`):

    use      a function `main` whose answer composes two of the project's
             functions (`g(f(x))`): given its examples, write it with what
             the project has (5b)
    bug      `main` as written, one of its helpers made wrong by one edit
             (`editing.edits`): `main`'s tests fail, the fault in another
             file (5c)
    change   a helper's API changed -- renamed, or its two parameters
             swapped -- with `main` left calling it the old way: make the
             project compile and its tests pass again (5d)

Every case a task shows is what the original program did on its own
test inputs; nothing is written by hand.

    python -m research.v696.projects         build (if missing) and count
"""
from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from research.v696 import meaning as M
from research.v696 import tasks
from research.v696.checker import CheckerError, checker
from research.v696.parse import parse

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "code-meaning"
TYPES = ("number", "string", "boolean", "number[]", "string[]", "boolean[]")


@dataclass
class Task:
    name: str
    #: use | bug | change
    kind: str
    files: dict
    #: the file and function the task is about
    file: str
    entry: str
    params: list
    returns: str
    cases: list
    wanted: list
    #: the test file that judges it
    tests: str = "/p/main.test.ts"
    #: what it was before it was made wrong (bug, change)
    original: dict = field(default_factory=dict)
    #: for `use`: the composition that answers it, as written
    answer: str = ""


def _verified(held: bool) -> list:
    """(task, code) for every verified program of one parameter of the
    library's types that reads into the tree."""
    path = DATA / ("solutions-held.jsonl" if held else "solutions.jsonl")
    config = "humaneval-ts" if held else "mbpp-ts"
    by = {one.name: one for one in tasks.load(config)}
    out = []
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if not row["code"]:
            continue
        task = by[row["name"]]
        if not all(kind in TYPES for _, kind in task.params) \
                or task.returns not in TYPES:
            continue
        if parse(row["code"], task.entry, task.params) is None:
            continue
        pairs = M.values_of(M.test_pairs(task.tests))
        if pairs:
            out.append((task, row["code"], pairs))
    return out


def _exported(code: str, entry: str) -> str:
    """The function as a module exports it."""
    return re.sub(rf"(^|\n)function {re.escape(entry)}\b",
                  rf"\1export function {entry}", code, count=1)


def _test_file(entry: str, cases: list, wanted: list) -> str:
    lines = [f'import {{ {entry} }} from "./main";',
             'declare var require: any;',
             'const assert = require("node:assert");']
    for args, want in zip(cases, wanted):
        said = ", ".join(json.dumps(one) for one in args)
        lines.append(f"assert.deepEqual({entry}({said}), {json.dumps(want)});")
    return "\n".join(lines) + "\n"


def _run(files: dict, file: str, entry: str, cases: list) -> list | None:
    """What `entry` of `file` gives on each case, the project as it is."""
    source = "".join(text for name, text in files.items()
                     if name != file) + "\n" + files[file]
    # imports and exports mean nothing once the files are one source
    source = re.sub(r"^import .*$", "", source, flags=re.M)
    source = source.replace("export function", "function")
    try:
        got = checker().run(source, entry, cases)
    except CheckerError:
        checker().restart()
        return None
    if any("value" not in one for one in got):
        return None
    return [one["value"] for one in got]


def _use(rng, pool, number: int) -> Task | None:
    """`main(x) = g(f(x))` for two functions whose types meet."""
    single = [one for one in pool if len(one[0].params) == 1]
    rng.shuffle(single)
    for task_f, code_f, pairs_f in single[:60]:
        for task_g, code_g, _ in single:
            if task_g is task_f or task_g.params[0][1] != task_f.returns:
                continue
            (name, kind), = task_f.params
            said = f"{name}: {kind}"
            main = (f'import {{ {task_f.entry} }} from "./{task_f.entry}";\n'
                    f'import {{ {task_g.entry} }} from "./{task_g.entry}";\n'
                    f"export function main({said}): {task_g.returns} {{\n"
                    f"  return {task_g.entry}({task_f.entry}({name}));\n}}\n")
            files = {f"/p/{task_f.entry}.ts": _exported(code_f, task_f.entry),
                     f"/p/{task_g.entry}.ts": _exported(code_g, task_g.entry),
                     "/p/main.ts": main}
            cases = [list(args) for args, _ in pairs_f]
            wanted = _run(files, "/p/main.ts", "main", cases)
            if wanted is None or len({json.dumps(one) for one in wanted}) < 2:
                continue
            files["/p/main.test.ts"] = _test_file("main", cases, wanted)
            if checker().project(files, "/p/main.test.ts") is not None:
                continue
            answer = f"{task_g.entry}({task_f.entry}({name}))"
            # what is asked: main's signature and examples, its body not
            asked = dict(files)
            asked["/p/main.ts"] = main.replace(f"  return {answer};",
                                               "  return undefined as any;")
            return Task(f"use-{number}", "use", asked, "/p/main.ts", "main",
                        list(task_f.params), task_g.returns, cases, wanted,
                        answer=answer)
    return None


def _bug(rng, pool, number: int) -> Task | None:
    """A `use` project as written, one helper made wrong."""
    from research.v696.editing import edits
    made = _use(rng, pool, number)
    if made is None:
        return None
    files = dict(made.files)
    files["/p/main.ts"] = made.files["/p/main.ts"].replace(
        "  return undefined as any;", f"  return {made.answer};")
    helpers = [name for name in files if name not in ("/p/main.ts",
                                                      "/p/main.test.ts")]
    rng.shuffle(helpers)
    for helper in helpers:
        entry = Path(helper).stem
        from research.v696.parse import project as read_project
        module = read_project({helper: files[helper]})
        if module is None:
            continue
        key = f"{helper}#{entry}"
        declared = module.functions.get(key)
        if not declared or "unread" in declared:
            continue
        params = [(one, kind) for one, kind in declared["params"]]
        tree = module.function(key, params)
        choices = edits(tree, files[helper], entry)
        rng.shuffle(choices)
        for one in choices[:12]:
            broken = dict(files)
            broken[helper] = one.apply(files[helper])
            if checker().diagnose(broken):
                continue
            if checker().project(broken, "/p/main.test.ts") is None:
                continue
            return Task(f"bug-{number}", "bug", broken, "/p/main.ts", "main",
                        made.params, made.returns, made.cases, made.wanted,
                        original=files)
    return None


def _change(rng, pool, number: int) -> Task | None:
    """A helper renamed, or its two parameters swapped, `main` left
    calling it as it did -- the compiler says where."""
    two = [one for one in pool if len(one[0].params) == 2]
    rng.shuffle(two)
    for task_h, code_h, pairs_h in two[:40]:
        (a, kind_a), (b, kind_b) = task_h.params
        entry = task_h.entry
        # two functions call the helper, so a change has more than one place
        main = (f'import {{ {entry} }} from "./{entry}";\n'
                f"export function main({a}: {kind_a}, {b}: {kind_b}): "
                f"{task_h.returns} {{\n"
                f"  return {entry}({a}, {b});\n}}\n"
                f"export function same({a}: {kind_a}, {b}: {kind_b}): "
                f"boolean {{\n"
                f"  return {entry}({a}, {b}) === main({a}, {b});\n}}\n")
        files = {f"/p/{entry}.ts": _exported(code_h, entry),
                 "/p/main.ts": main}
        cases = [list(args) for args, _ in pairs_h]
        wanted = _run(files, "/p/main.ts", "main", cases)
        if wanted is None:
            continue
        files["/p/main.test.ts"] = _test_file("main", cases, wanted)
        if checker().project(files, "/p/main.test.ts") is not None:
            continue
        changed = dict(files)
        if kind_a != kind_b and rng.random() < 0.5:
            # the two parameters swapped in the declaration
            head = re.search(rf"export function {re.escape(entry)}\s*\(([^)]*)\)",
                             changed[f"/p/{entry}.ts"])
            if head is None:
                continue
            parts = [one.strip() for one in head.group(1).split(",")]
            if len(parts) != 2:
                continue
            changed[f"/p/{entry}.ts"] = changed[f"/p/{entry}.ts"].replace(
                head.group(0), f"export function {entry}({parts[1]}, "
                               f"{parts[0]})", 1)
            how = "swapped"
        else:
            renamed = f"{entry}_v2"
            changed[f"/p/{entry}.ts"] = re.sub(
                rf"\bexport function {re.escape(entry)}\b",
                f"export function {renamed}", changed[f"/p/{entry}.ts"])
            how = "renamed"
        if not checker().diagnose(changed):
            continue
        return Task(f"change-{number}-{how}", "change", changed,
                    "/p/main.ts", "main", [[a, kind_a], [b, kind_b]],
                    task_h.returns, cases, wanted, original=files)
    return None


def build(held: bool, each: int = 15, seed: int = 695) -> list:
    import time
    rng = random.Random(seed + (1 if held else 0))
    pool = _verified(held)
    out, number = [], 0
    started = time.time()
    for kind, make in (("use", _use), ("bug", _bug), ("change", _change)):
        made = 0
        tries = 0
        while made < each and tries < each * 4:
            tries += 1
            number += 1
            task = make(rng, pool, number)
            if task is not None and task.name not in {one.name
                                                      for one in out}:
                out.append(task)
                made += 1
            # one line an attempt, as it goes: a long build shows where
            print(f"  {kind}: {made}/{each} made, try {tries} "
                  f"({time.time() - started:.0f}s)", flush=True)
    return out


def load(held: bool) -> list:
    """The frozen tasks: made once, then read."""
    path = DATA / f"projects-{'held' if held else 'dev'}.jsonl"
    if not path.exists():
        # built whole, then written: a build that stops leaves nothing, not
        # a file that reads as fewer tasks
        made = build(held)
        with path.open("w", encoding="utf-8") as out:
            for task in made:
                out.write(json.dumps(asdict(task)) + "\n")
    return [Task(**json.loads(line)) for line in path.open(encoding="utf-8")]


def main() -> None:
    from collections import Counter
    for held in (False, True):
        found = load(held)
        print("held" if held else "dev", len(found),
              dict(Counter(one.kind for one in found)))


if __name__ == "__main__":
    main()
