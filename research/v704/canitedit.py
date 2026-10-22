"""CanItEdit as conversations: a change asked of a project, checked by tests
it never sees (2026-10-22, the core measured beyond its own test).

    python -m research.v704.canitedit fetch     data/canitedit/test.jsonl
    python -m research.v704.canitedit projects  a project of each problem
    python -m research.v704.converse --suite canitedit ...   the architecture
    python -m research.v704.canitedit alone     the code model alone

CanItEdit (nuprl/CanItEdit, Cassano et al. 2023, "Can It Edit? Evaluating
the Ability of Large Language Models to Follow Code Editing Instructions"):
105 Python files, each with a change asked of it twice -- descriptive
(`Add a function called header which returns the first row ...`) and lazy
(`Add a method called header which returns the header`) -- and the tests
the file as changed must pass. Each problem is a project of one file
(`data/canitedit/projects/<name>/<name>.py`), each instruction a
conversation of one turn: `code-change`, checked by the problem's tests run
with the file as the conversation left it. The tests are kept out of the
project (`data/canitedit/tests/`): what the architecture runs is what it
writes itself.

Evaluation only -- 105 problems are too few to teach from and measure on
both -- and read as totals, as the garden fixture is: a failure is fixed
as the class it is, never as its problem.

**The code model alone** (`alone`): the same file and instruction to the
3B code model the architecture uses, one answer (greedy), the file it
writes run against the same tests -- what the architecture must do better
than (Adrian, 2026-10-21: by breaking the request down and reading it, not
by leaning on the model). Both said as right, unsure and wrong: the
architecture may say it is unsure, the model alone never does.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "canitedit"
ROWS = ("https://datasets-server.huggingface.co/rows?dataset=nuprl/CanItEdit"
        "&config=default&split=test&offset={offset}&length=100")
COUNT = 105
#: how long a problem's tests may run
SECONDS = 60


def fetch() -> int:
    """The 105 problems, as the datasets server gives them."""
    rows = []
    for offset in range(0, COUNT, 100):
        request = urllib.request.Request(ROWS.format(offset=offset),
                                         headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(request, timeout=300) as reply:
            rows += [one["row"] for one in json.load(reply)["rows"]]
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "test.jsonl").open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    print(f"{len(rows)} problems -> {DATA / 'test.jsonl'}")
    return len(rows)


def problems() -> list:
    return [json.loads(line) for line in
            (DATA / "test.jsonl").open(encoding="utf-8")]


def _file(row: dict) -> str:
    return f"{row['name']}.py"


def projects() -> int:
    """Each problem's project (its file as it was), and its tests apart --
    and which problems can be measured here: the change CanItEdit gives
    passes its tests, the file as it was does not (2026-10-22: 14 of 105
    need a library this Python has not -- pandas, z3, flask, autograd --
    and are left out, said so in `runnable.json`)."""
    runnable, left = [], {}
    for row in problems():
        if not passes(row["after"], row):
            left[row["full_name"]] = "its own change fails its tests here"
        elif passes(row["before"], row):
            left[row["full_name"]] = "the file passes its tests unchanged"
        else:
            runnable.append(row["full_name"])
    (DATA / "runnable.json").write_text(json.dumps(
        {"runnable": runnable, "left": left}, indent=1), encoding="utf-8")
    for row in problems():
        folder = DATA / "projects" / row["full_name"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / _file(row)).write_text(row["before"], encoding="utf-8")
        tests = DATA / "tests" / f"{row['full_name']}.py"
        tests.parent.mkdir(parents=True, exist_ok=True)
        tests.write_text(row["tests"], encoding="utf-8")
    print(f"{len(problems())} projects -> {DATA / 'projects'}; "
          f"{len(runnable)} runnable, {len(left)} left out")
    return len(runnable)


def runnable() -> list:
    """The problems measured here (`projects` says which)."""
    kept = set(json.loads((DATA / "runnable.json").read_text(
        encoding="utf-8"))["runnable"])
    return [row for row in problems() if row["full_name"] in kept]


def passes(text: str, row: dict, cwd: Path | None = None) -> bool:
    """Whether a file as changed passes the problem's tests: the file and
    its tests run as one module, as CanItEdit runs them."""
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        program = Path(folder) / "check.py"
        program.write_text(text + "\n\n" + row["tests"], encoding="utf-8")
        try:
            done = subprocess.run([sys.executable, str(program)],
                                  cwd=cwd or folder, capture_output=True,
                                  text=True, timeout=SECONDS)
        except subprocess.TimeoutExpired:
            return False
    return done.returncode == 0


def _check(row: dict):
    """The conversation's check: the file as it left it, tested."""
    def check(turn, root):
        if root is None:
            return False
        path = Path(root) / _file(row)
        return path.exists() and passes(path.read_text(encoding="utf-8"),
                                        row, Path(root))
    check.about = f"passes CanItEdit's tests of {row['full_name']}"
    check.changes = True
    return check


def suite() -> list:
    """Two conversations a problem -- the descriptive instruction, then the
    lazy -- each of the problem's project."""
    from research.v704.suite import Conversation
    out = []
    for row in runnable():
        fixture = str(DATA / "projects" / row["full_name"])
        for kind in ("descriptive", "lazy"):
            out.append(Conversation("code-change",
                                    [row[f"instruction_{kind}"]],
                                    _check(row), True, fixture))
    return out


#: what the code model alone is asked: the file, the instruction, the file
#: back whole
ALONE = ("Here is a Python file, {name}:\n```python\n{before}\n```\n"
         "Change it as asked: \"{instruction}\"\nWrite the whole file as "
         "changed in one ```python block, nothing else.")


def alone(label: str = "alone") -> dict:
    """The code model alone on every instruction: one answer each, run
    against the problem's tests."""
    from research.v705 import coder, commits
    model = coder.big()
    out = DATA / f"{label}.jsonl"
    done = {}
    if out.exists():
        done = {(one["problem"], one["kind"]): one for one in
                map(json.loads, out.open(encoding="utf-8"))}
    with out.open("a", encoding="utf-8") as stream:
        for row in runnable():
            for kind in ("descriptive", "lazy"):
                if (row["full_name"], kind) in done:
                    continue
                started = time.time()
                asked = ALONE.format(name=_file(row), before=row["before"],
                                     instruction=row[f"instruction_{kind}"])
                # one answer: greedy
                reply = model.write([asked], longest=1500)[0]
                reply = reply[0] if isinstance(reply, list) else reply
                code = commits._code_of(reply) or ""
                one = {"problem": row["full_name"], "kind": kind,
                       "verdict": "right" if code and passes(code, row)
                       else "wrong",
                       "seconds": round(time.time() - started, 1)}
                done[(row["full_name"], kind)] = one
                stream.write(json.dumps(one) + "\n")
                stream.flush()
    rows = list(done.values())
    totals = {kind: {"right": sum(one["verdict"] == "right" for one in rows
                                  if one["kind"] == kind),
                     "of": sum(one["kind"] == kind for one in rows)}
              for kind in ("descriptive", "lazy")}
    print(json.dumps(totals))
    return totals


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("fetch", "projects", "alone"))
    parser.add_argument("--label", default="alone")
    options = parser.parse_args(argv)
    if options.job == "alone":
        alone(options.label)
    else:
        {"fetch": fetch, "projects": projects}[options.job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
