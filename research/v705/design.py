"""New design as steps (v705): a change its history made by inventing a
helper -- `determines`, "the output is fixed by this value" -- written,
and called from the function that needed it. One edit cannot hold that;
three steps can, each its own: **propose** the helper (a name, its
parameters, what it gives: a def line and a docstring), **write** it, and
**wire** it in (a local change, once it exists).

    python -m research.v705.design mine     state/v705-designs.jsonl

**Mined** (`mine`): of each commit, each function it added to a source
file that a function it changed in the same file calls in the commit's
version and did not before -- {message, file, helper (its def line,
docstring and source), caller, caller before, caller after}. What the
proposer is taught (goal + caller before -> the helper's def line and
docstring) and what the steps are measured on.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

from research.v705 import commits

ROOT = Path(__file__).resolve().parents[2]
DESIGNS = ROOT / "state" / "v705-designs.jsonl"


def _calls(source: str) -> set:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                out.add(func.id)
            elif isinstance(func, ast.Attribute):
                out.add(func.attr)
    return out


def _head(source: str) -> str:
    """A function's def line, on one line."""
    return " ".join(source.split(":\n")[0].split()) + ":"


def mine() -> dict:
    shas = commits._git("rev-list", "HEAD", "--", "research").split()
    rows = []
    for sha in shas:
        message = commits._git("log", "-1", "--format=%s", sha).strip()
        files = [one for one in commits._git(
            "diff-tree", "--no-commit-id", "--name-only", "-r", sha).split()
            if one.startswith("research/") and one.endswith(".py")
            and not commits._test(one)]
        for path in files:
            before = commits._show(f"{sha}^", path)
            after = commits._show(sha, path)
            if before is None or after is None:
                continue
            old, new = commits.functions(before), commits.functions(after)
            added = {name: source for name, source in new.items()
                     if name not in old}
            if not added:
                continue
            for caller, source in new.items():
                if caller not in old or old[caller] == source:
                    continue
                gained = _calls(source) - _calls(old[caller])
                for helper, body in added.items():
                    if helper.split(".")[-1] not in gained:
                        continue
                    tree = ast.parse(body.strip()) if body else None
                    doc = ast.get_docstring(tree.body[0]) if tree and \
                        tree.body and isinstance(
                            tree.body[0], (ast.FunctionDef,
                                           ast.AsyncFunctionDef)) else None
                    rows.append({"sha": sha, "message": message,
                                 "file": path, "helper": helper,
                                 "head": _head(body), "doc": doc or "",
                                 "helper source": body, "caller": caller,
                                 "before": old[caller], "after": source})
    DESIGNS.parent.mkdir(parents=True, exist_ok=True)
    with DESIGNS.open("w", encoding="utf-8") as out:
        for one in rows:
            out.write(json.dumps(one) + "\n")
    found = {"designs": len(rows), "commits": len({one["sha"]
                                                   for one in rows}),
             "with a docstring": sum(bool(one["doc"]) for one in rows)}
    print(json.dumps(found))
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("mine",))
    parser.parse_args(argv)
    mine()
    return 0


if __name__ == "__main__":
    sys.exit(main())
