"""Is reading exact? Every verified program that reads into a tree
(`parse.py`) is printed back from the tree and run on its task's own tests.

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


def main() -> None:
    for path, config in SOURCES:
        by = {one.name: one for one in tasks.load(config)}
        passed, differ = 0, []
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            if not row["code"]:
                continue
            task = by[row["name"]]
            tree = parse(row["code"], task.entry, task.params)
            if tree is None:
                continue
            spec = Spec(task.name, task.params, task.returns, [],
                        entry=task.entry)
            if checker().tests(spec.function(tree) + "\n"
                               + task.tests) is None:
                passed += 1
            else:
                differ.append(task.name)
        print(f"{config}: {passed} read trees pass the tests, "
              f"{len(differ)} differ {differ[:5]}")


if __name__ == "__main__":
    main()
