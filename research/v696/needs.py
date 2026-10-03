"""What verified programs are made of -- what rung 3 had to read.

The teacher's verified solutions (`data/code-meaning/solutions*.jsonl`,
measured with, never taught from), each counted once per kind of thing it
uses, and whether it reads into the search's tree (`parse.py`).

    python -m research.v696.needs
"""
from __future__ import annotations

import json
import re
from collections import Counter

from research.v696 import meaning as M
from research.v696 import tasks
from research.v696.parse import parse

SOURCES = (("data/code-meaning/solutions.jsonl", "mbpp-ts"),
           ("data/code-meaning/solutions-held.jsonl", "humaneval-ts"))
LIBRARY_TYPES = ("number", "string", "boolean", "number[]", "string[]",
                 "boolean[]")


def kinds(code: str, task) -> set:
    functions = len(re.findall(r"\bfunction\s+\w+", code)) + len(
        re.findall(r"const\s+\w+\s*=\s*\(", code))
    uses, _ = M.structure(code, task.entry)
    out = set()
    if parse(code, task.entry, task.params) is not None:
        out.add("one expression in the library")
    if functions > 1:
        out.add("helper functions")
    if uses & {"for", "for of", "for in", "while"}:
        out.add("a loop")
    if "recursion" in uses:
        out.add("recursion")
    if "let" in uses:
        out.add("local variables")
    if uses & {"new Set", "new Map", "{}", "Object.keys", "Object.entries"}:
        out.add("sets/maps/objects")
    if "Array.from" in uses or "new Array" in uses:
        out.add("Array.from / new Array")
    if not all(kind in LIBRARY_TYPES for _, kind in task.params) \
            or task.returns not in LIBRARY_TYPES:
        out.add("types beyond the library's")
    return out


def main() -> None:
    for path, config in SOURCES:
        by = {one.name: one for one in tasks.load(config)}
        counts, verified = Counter(), 0
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            if not row["code"]:
                continue
            verified += 1
            counts.update(kinds(row["code"], by[row["name"]]))
        print(config, verified, "verified")
        for kind, count in counts.most_common():
            print(f"   {count:4} {kind}")


if __name__ == "__main__":
    main()
