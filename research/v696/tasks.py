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
    #: the language it is written in (`language.py`)
    language: str = "typescript"
    #: a solution known to pass the tests, where the dataset has one
    #: (Python's: MBPP's own and HumanEval's canonical ones)
    solution: str = ""
    #: the types as the prompt writes them (`int`, `list[str]`), where
    #: they are not the engine's (Python's): by parameter, and "return"
    written: dict = field(default_factory=dict)


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
    reading = parse_python if config.endswith("-py") else parse
    with (DATA / f"{config}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            task = reading(json.loads(line), config)
            if task is not None:
                tasks.append(task)
    return tasks


# -- Python: the originals MultiPL-E translated from ----------------------------

#: The same tasks in Python: MultiPL-E's typed originals (a signature, its
#: docstring, the tests) -- the very files its TypeScript was translated
#: from, so a task has one name in both languages and the splits are the
#: same; the solutions from MBPP itself and HumanEval's canonical ones.
PY_CONFIGS = ("humaneval-py", "mbpp-py")
ORIGINALS = "https://raw.githubusercontent.com/nuprl/MultiPL-E/main/datasets"
FOLDERS = {"mbpp-py": "mbpp-typed",
           "humaneval-py": "originals-with-cleaned-doctests"}
MBPP_ROWS = ("https://datasets-server.huggingface.co/rows?dataset="
             "google-research-datasets/mbpp&config=full")
HUMANEVAL_ROWS = ("https://datasets-server.huggingface.co/rows?dataset="
                  "openai/openai_humaneval&config=openai_humaneval")
SOLUTION_MARK = "### Canonical solution below ###"
TESTS_MARK = "### Unit tests below ###"


def _get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8")


def _rows(base: str, split: str) -> list:
    rows, offset = [], 0
    while True:
        page = json.loads(_get(f"{base}&split={split}&offset={offset}"
                               f"&length=100"))
        rows += [one["row"] for one in page["rows"]]
        offset += 100
        if offset >= page["num_rows_total"]:
            return rows


def _names(config: str) -> list:
    """The task names, as the TypeScript set has them -- and HumanEval's
    every one (the TypeScript set lost five in translation)."""
    if config == "humaneval-py":
        return [f"HumanEval_{row['task_id'].split('/')[1]}_"
                f"{row['entry_point']}" for row in _rows(HUMANEVAL_ROWS,
                                                         "test")]
    path = DATA / "mbpp-ts.jsonl"
    return [json.loads(line)["name"] for line in path.open(encoding="utf-8")]


def fetch_python() -> dict:
    """Each task's typed original, and the solution known to pass it."""
    DATA.mkdir(parents=True, exist_ok=True)
    mbpp = {}
    for split in ("train", "test", "validation", "prompt"):
        for row in _rows(MBPP_ROWS, split):
            mbpp[int(row["task_id"])] = row["code"].replace("\r\n", "\n")
    counts = {}
    for config in PY_CONFIGS:
        rows = []
        for name in _names(config):
            try:
                text = _get(f"{ORIGINALS}/{FOLDERS[config]}/{name}.py")
            except OSError:
                continue
            head, _, rest = text.partition(SOLUTION_MARK)
            solution, _, tests = rest.partition(TESTS_MARK)
            prompt = head.rstrip()
            if prompt.endswith("#"):
                prompt = prompt.rstrip("#").rstrip()
            if config == "mbpp-py":
                number = int(name.split("_")[1])
                solution = mbpp.get(number, "")
            else:
                solution = prompt + "\n" + solution.rstrip()
            rows.append({"name": name, "language": "py", "prompt": prompt,
                         "solution": solution, "tests": tests.strip(),
                         "original": text})
        with (DATA / f"{config}.jsonl").open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row) + "\n")
        counts[config] = len(rows)
    return counts


def parse_python(row: dict, config: str) -> Task | None:
    """A typed original as a task: its function's signature (types read
    into the engine's, and kept as written), the doctest examples its
    docstring shows, its tests as a file that runs them."""
    import ast

    from research.v696 import pytypes
    try:
        tree = ast.parse(row["prompt"] + "\n    pass\n")
    except SyntaxError:
        return None
    function = next((node for node in tree.body
                     if isinstance(node, ast.FunctionDef)), None)
    if function is None:
        return None
    entry = function.name
    params, written = [], {}
    for one in function.args.args:
        said = ast.unparse(one.annotation) if one.annotation else None
        params.append((one.arg, pytypes.engine(said)))
        if said:
            written[one.arg] = said
    said = ast.unparse(function.returns) if function.returns else None
    if said:
        written["return"] = said
    examples = []
    doc = ast.get_docstring(function) or ""
    lines = doc.splitlines()
    for at, line in enumerate(lines):
        line = line.strip()
        if line.startswith(">>>") and at + 1 < len(lines):
            call, value = line[3:].strip(), lines[at + 1].strip()
            if call.startswith(entry + "(") and call.endswith(")") and \
                    value and not value.startswith(">>>"):
                examples.append((call[len(entry) + 1:-1], value))
    tests = row["tests"]
    if "def test_check" in tests:
        tests = tests.split("def test_check")[0].rstrip()
    tests += f"\n\ncheck({entry})\n"
    return Task(row["name"], entry, params, pytypes.engine(said),
                row["prompt"], examples, tests, config, language="python",
                solution=row.get("solution", ""), written=written)


def main(argv=None) -> int:
    args = (argv if argv is not None else sys.argv[1:]) or ["fetch"]
    if args[0] == "fetch":
        print(fetch())
    if args[0] in ("fetch", "fetch-python"):
        print(fetch_python())
    return 0


if __name__ == "__main__":
    sys.exit(main())
