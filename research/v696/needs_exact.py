"""Is reading exact? Every verified program that reads into a tree
(`parse.py`) is printed back from the tree and run on its task's own tests.
And every program that is wrong (HumanEvalFix's bugs): its tree, printed
back, must do on the bug's cases what the source does -- the same value, or
fail where it fails. A reader that read a wrong program as a right one
would have nothing to repair.

    python -m research.v696.needs_exact
"""
from __future__ import annotations

import json

from research.v696 import tasks
from research.v696.checker import checker
from research.v696.parse import parse
from research.v696.spec import Spec

SOURCES = (("data/code-meaning/solutions.jsonl", "mbpp-ts"),
           ("data/code-meaning/solutions-held.jsonl", "humaneval-ts"))


def verified() -> None:
    for path, config in SOURCES:
        by = {one.name: one for one in tasks.load(config)}
        passed, differ, programs = 0, [], 0
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            if not row["code"]:
                continue
            task = by[row["name"]]
            programs += 1
            tree = parse(row["code"], task.entry, task.params)
            if tree is None:
                continue
            spec = Spec(task.name, task.params, task.returns, [],
                        entry=task.entry)
            if checker().tests(spec.function(tree) + "\n"
                               + task.tests) is None:
                passed += 1
            elif row["code"].count("{") == row["code"].count("}") and \
                    checker().tests(row["code"] + "\n" + task.tests) is None:
                # (a program cut off before its end, or one that fails its
                # own tests, says nothing here)
                differ.append(task.name)
        print(f"{config}: of {programs} programs, {passed} read trees pass "
              f"the tests, {len(differ)} differ {differ[:5]}")


def _said(row: dict) -> str:
    return json.dumps(row["value"], sort_keys=True) if "value" in row \
        else "fails"


def wrong() -> None:
    from research.v696 import bugs
    if not (bugs.DATA / "js.jsonl").exists():
        return
    found = bugs.humanevalfix()
    read, differ = 0, []
    for bug in found:
        tree = parse(bug.source, bug.entry, bug.params)
        if tree is None:
            continue
        read += 1
        spec = Spec(bug.name, bug.params, bug.returns, [], entry=bug.entry)
        theirs = checker().run(bug.source, bug.entry, bug.cases)
        ours = checker().run(spec.function(tree), bug.entry, bug.cases)
        if [_said(one) for one in theirs] != [_said(one) for one in ours]:
            differ.append(bug.name)
    print(f"humanevalfix: of {len(found)} bugs, {read} read, "
          f"{len(differ)} differ from their source {differ[:5]}")


def main() -> None:
    verified()
    wrong()


if __name__ == "__main__":
    main()
