"""Code tasks: a signature, examples, and a check that decides.

Two kinds, as `PLAN.md` says:

    independent   MultiPL-E's TypeScript translations of HumanEval (159)
                  and MBPP: a signature, examples in its comment, hidden
                  tests. Nobody here wrote them; mostly beyond small
                  composition, and reported as they come.
    generated     compositions of the library at a chosen depth, from a
                  seed, the composition kept as a known answer
                  (`generating.py`, Phase 1). Dev and held seeds never
                  meet.

    python -m research.v696.tasks fetch      data/multipl-e/*.jsonl
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "multipl-e"
CONFIGS = ("humaneval-ts", "mbpp-ts")
ROWS = "https://datasets-server.huggingface.co/rows?dataset=nuprl/MultiPL-E"

#: `function name(a: T, b: U): R {` -- the signature a prompt ends with.
SIGNATURE = re.compile(
    r"function\s+(\w+)\s*\(([^)]*)\)\s*:\s*([^{]+?)\s*\{\s*$", re.S)
#: `// >>> name(args)` then `// value`: an example in the prompt.
EXAMPLE = re.compile(r"//\s*>>>\s*(\w+)\((.*)\)\s*\n//\s*(.+)")


@dataclass
class Task:
    name: str
    #: the function's name and its parameters as (name, type)
    entry: str
    params: list
    returns: str
    #: the prompt as given: comment, examples, signature
    prompt: str
    #: examples readable from the prompt: (args as TS text, value as text)
    examples: list = field(default_factory=list)
    #: the hidden tests, run after a candidate
    tests: str = ""
    source: str = ""


def _params(text: str) -> list:
    out, depth, part = [], 0, ""
    for char in text:
        if char in "<([{":
            depth += 1
        elif char in ">)]}":
            depth -= 1
        if char == "," and depth == 0:
            out.append(part)
            part = ""
        else:
            part += char
    if part.strip():
        out.append(part)
    params = []
    for one in out:
        name, _, kind = one.partition(":")
        params.append((name.strip(), kind.strip()))
    return params


def parse(row: dict, config: str) -> Task | None:
    found = SIGNATURE.search(row["prompt"])
    if found is None:
        return None
    entry, params, returns = found.group(1), found.group(2), found.group(3)
    examples = [(args, value.strip()) for name, args, value in
                EXAMPLE.findall(row["prompt"]) if name == entry]
    return Task(row["name"], entry, _params(params), returns.strip(),
                row["prompt"], examples, row["tests"], config)


def fetch() -> dict:
    """MultiPL-E's TypeScript configs, a page of 100 rows at a time."""
    DATA.mkdir(parents=True, exist_ok=True)
    counts = {}
    for config in CONFIGS:
        rows, offset = [], 0
        while True:
            url = (f"{ROWS}&config={config}&split=test&offset={offset}"
                   f"&length=100")
            request = urllib.request.Request(url,
                                             headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(request, timeout=120) as response:
                page = json.loads(response.read().decode("utf-8"))
            rows += [one["row"] for one in page["rows"]]
            offset += 100
            if offset >= page["num_rows_total"]:
                break
        with (DATA / f"{config}.jsonl").open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row) + "\n")
        counts[config] = len(rows)
    return counts


def load(config: str) -> list:
    tasks = []
    with (DATA / f"{config}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            task = parse(json.loads(line), config)
            if task is not None:
                tasks.append(task)
    return tasks


def main(argv=None) -> int:
    args = (argv if argv is not None else sys.argv[1:]) or ["fetch"]
    if args[0] == "fetch":
        print(fetch())
    return 0


if __name__ == "__main__":
    sys.exit(main())
