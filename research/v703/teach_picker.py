"""Teaching the architecture to choose where a change goes: a request,
and each function (or file) of the project it could touch -- changed or not.

    python -m research.v703.teach_picker pairs    llm/plan-data/picks-*.jsonl
    python -m research.v703.teach_picker train    llm/picker

A plan written by the planner (360M) named files well and what to do in
them badly: `graph_size` in the server, a function it has not. Planning is
choosing (Adrian, 2026-10-09): what the project has is put before a judge,
one at a time, and the judge says whether the request changes it -- it can
choose only what is there.

**Pairs, by construction**: this project's commits -- every function of
the files changed (and of those joined to them), as the project was before,
changed where the diff touches its lines; CommitChronicle's commits of
Python projects -- a function a diff changes is named by its hunk's header
(`@@ -12,7 +12,9 @@ def health(self):`), the others of its project are the
ones it did not. **The judge**: UniXcoder taught as `change-judge2` was
(`v700/teach_judge`), the request beside a function said as
`path :: name(params) -- what its docstring says first`.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys

from research.v703 import teach_plans as P

PAIRS = P.DATA / "picks-{}.jsonl"
PICKER = P.LLM / "picker"
SEED = 704
#: unchanged functions beside each changed one
NEGATIVES = 4


def unit(path: str, function: dict | None = None) -> str:
    """A function as the judge reads it: where, what, and what its doc says
    first -- a file whole where it has none (a document, a setting)."""
    if function is None:
        return f"{path} :: the file"
    params = ", ".join(one[0] for one in function.get("params") or ()
                       if one[0] not in ("self", "cls"))
    doc = (function.get("doc") or "").strip().split("\n")[0][:120]
    return f"{path} :: {function['name']}({params})" + \
        (f" -- {doc}" if doc else "")


def _old_lines(diff: str) -> list:
    """The lines of the file before, that a diff takes out or changes."""
    out, at = [], 0
    for line in diff.splitlines():
        found = re.match(r"@@ -(\d+)(?:,(\d+))? \+\d+", line)
        if found:
            at = int(found.group(1))
            continue
        if line.startswith("-") and not line.startswith("---"):
            out.append(at)
            at += 1
        elif line.startswith("+") and not line.startswith("+++"):
            out.append(at)
        elif not line.startswith("\\"):
            at += 1
    return out


def _ours(rng) -> list:
    """(request, unit, label, key) of this project's commits."""
    from research.v703 import structure
    rows = [json.loads(line) for line in P.COMMITS.open(encoding="utf-8")]
    ours = [one for one in P._git("ls-files").split() if P.CODE.search(one)]
    present = structure.subset({
        path: (P.ROOT / path).read_text(encoding="utf-8", errors="replace")
        for path in ours if path.endswith(".py") and (P.ROOT / path).exists()})
    out = []
    for row in rows:
        if not row["key"].startswith("ours:"):
            continue
        commit = row["key"].split(":", 1)[1]
        request = row["message"].splitlines()[0][:300]
        changed = {mod["path"]: mod for mod in row["mods"]
                   if mod["change"] != "A"}
        if not changed:
            continue
        paths = list(changed) + structure.linked(present, list(changed), 3)
        files = {}
        for path in paths:
            text = P._git("show", f"{commit}^:{path}")
            if text:
                files[path] = text
        held = structure.subset(files)
        for path in files:
            functions = (held.outline().get(path) or {}).get("functions", ())
            if not path.endswith(".py") or not functions:
                out.append((request, unit(path), float(path in changed),
                            row["key"]))
                continue
            lines = set(_old_lines(changed[path]["diff"])) \
                if path in changed else set()
            for one in functions:
                touched = any(one["start"] <= line <= one["end"]
                              for line in lines)
                out.append((request, unit(path, one), float(touched),
                            row["key"]))
    return out


def _chronicle(rng, most: int) -> list:
    """(request, unit, label, key) of CommitChronicle's Python commits:
    each function a hunk's header names, changed; others of its project
    (named by other commits' headers), not."""
    import pyarrow.parquet as pq
    commits, by_repo = [], {}
    for shard in sorted(P.CHRONICLE.glob("*.parquet")):
        try:
            table = pq.read_table(shard, columns=["hash", "repo", "message",
                                                  "mods", "language"])
        except Exception:                           # noqa: BLE001
            continue
        for row in table.to_pylist():
            if row["language"] != "Python":
                continue
            mods = [one for one in row["mods"] or []
                    if (one["new_path"] or "").endswith(".py") and
                    one["change_type"] == "MODIFY"]
            if not (P.LEAST_FILES <= len(row["mods"] or []) <= P.MOST_FILES) \
                    or len((row["message"] or "").split()) < 4:
                continue
            named = set()
            for mod in mods:
                for header in re.findall(r"^@@[^@]+@@ (.+)$", mod["diff"],
                                         re.M):
                    found = re.match(r"\s*(?:async\s+)?def\s+(\w+)\s*\(([^)]*)",
                                     header)
                    if found:
                        params = [one.split(":")[0].split("=")[0].strip()
                                  for one in found.group(2).split(",")
                                  if one.strip()]
                        named.add((mod["new_path"], found.group(1),
                                   tuple(params)))
            if not named:
                continue
            key = f"cc:{row['repo']}:{row['hash']}"
            commits.append((key, row["repo"], row["message"].splitlines()[0]
                            [:300], named))
            by_repo.setdefault(row["repo"], set()).update(named)
            if len(commits) >= most:
                break
    out = []
    for key, repo, request, named in commits:
        for path, name, params in named:
            out.append((request, unit(path, {"name": name, "params": [
                [one] for one in params]}), 1.0, key))
        others = sorted(by_repo[repo] - named)
        for path, name, params in rng.sample(others, min(
                len(others), NEGATIVES * len(named))):
            out.append((request, unit(path, {"name": name, "params": [
                [one] for one in params]}), 0.0, key))
    return out


def pairs_job(most: int = 9000) -> dict:
    rng = random.Random(SEED)
    found = _ours(rng) + _chronicle(rng, most)
    counts = {}
    for split in ("train", "dev", "test"):
        rows = [one for one in found if P.split_of(one[3]) == split]
        rng.shuffle(rows)
        with open(str(PAIRS).format(split), "w", encoding="utf-8") as out:
            for one in rows:
                out.write(json.dumps(one) + "\n")
        counts[split] = {"pairs": len(rows),
                         "changed": sum(one[2] for one in rows)}
    print(json.dumps(counts))
    return counts


def pairs(split: str, seed: int = SEED) -> list:
    """As `teach_judge.pairs` gives them: (request, unit, label, kind)."""
    return [tuple(json.loads(line)) for line in
            open(str(PAIRS).format(split), encoding="utf-8")]


def train(epochs: int = 2) -> None:
    from research.v700 import teach_judge as J
    J.pairs = pairs
    J.train(PICKER, epochs=epochs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("pairs", "train"))
    parser.add_argument("--most", type=int, default=9000)
    options = parser.parse_args(argv)
    if options.job == "pairs":
        pairs_job(options.most)
    else:
        train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
