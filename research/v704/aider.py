"""Aider's editing benchmark as conversations: an exercise's instructions
asked of a project holding its stub, checked by tests it never sees
(2026-10-25, the core's code-change measured at the size of a program).

    python -m research.v704.aider fetch      data/exercism/ (exercism/python)
    python -m research.v704.aider projects   a project of each exercise
    python -m research.v704.converse --suite aider ...   the architecture
    python -m research.v704.aider alone      the code model alone

Aider's code editing benchmark (Gauthier 2023, aider.chat/docs/benchmarks)
asks a model to write Exercism's Python practice exercises: each a stub
(`two_fer.py`: `def two_fer(name): pass`), the exercise's instructions,
and Exercism's unit tests the file as written must pass. Here the
exercises are exercism/python's at a pinned commit (`COMMIT`, MIT):
142, of which those whose own example passes its tests and whose stub
does not are measured (`runnable.json`). Each is a project of its stub
(`data/exercism/projects/<slug>/`), the request a conversation of one
turn, as Aider words it -- the instructions, then `Use the above
instructions to modify the supplied files` -- checked by the exercise's
tests run with the file as the conversation left it. The tests are kept
out of the project (`data/exercism/tests/<slug>/`).

Aider gives a model a second try, shown the tests' failures; a
conversation's turns are fixed before it starts, so what is measured is
the first try, for the architecture and the model alone both.

Evaluation only, read as totals: a failure is fixed as the class it is,
never as its exercise. Qwen2.5-Coder has likely seen Exercism's
solutions; where the 3B writes the code, that inflates both columns.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "exercism"
COMMIT = "6105f6d9d3ef66ce3d5c8f529d2882af3d76b706"
TARBALL = f"https://codeload.github.com/exercism/python/tar.gz/{COMMIT}"
PRACTICE = DATA / "src" / "exercises" / "practice"
#: how long an exercise's tests may run
SECONDS = 60
#: what Aider says after the instructions (benchmark/benchmark.py)
ASKED = ("\n\n####\n\nUse the above instructions to modify the supplied "
         "files: {files}\nDon't change the names of existing functions or "
         "classes, as they may be referenced from other code like unit "
         "tests, etc.\nOnly use standard python libraries, don't suggest "
         "installing any packages.")


def fetch() -> int:
    """exercism/python's practice exercises at `COMMIT`."""
    DATA.mkdir(parents=True, exist_ok=True)
    archive = DATA / "python.tar.gz"
    if not archive.exists():
        request = urllib.request.Request(TARBALL,
                                         headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(request, timeout=600) as reply:
            archive.write_bytes(reply.read())
    top = f"python-{COMMIT}/"
    with tarfile.open(archive) as tar:
        kept = []
        for one in tar.getmembers():
            name = one.name[len(top):] if one.name.startswith(top) else ""
            if not (name == "LICENSE" or
                    name.startswith("exercises/practice/")) or \
                    ".." in Path(name).parts or not (one.isfile() or
                                                     one.isdir()):
                continue
            one.name = name
            kept.append(one)
        tar.extractall(DATA / "src", members=kept)
    count = sum(1 for one in PRACTICE.iterdir() if one.is_dir())
    print(f"{count} exercises -> {PRACTICE}")
    return count


def exercises() -> list:
    """Each exercise: its slug, files, and what is asked."""
    out = []
    for folder in sorted(PRACTICE.iterdir()):
        config = folder / ".meta" / "config.json"
        if not config.exists():
            continue
        files = json.loads(config.read_text(encoding="utf-8"))["files"]
        docs = folder / ".docs"
        said = "\n\n".join(
            (docs / name).read_text(encoding="utf-8").strip()
            for name in ("introduction.md", "instructions.md",
                         "instructions.append.md")
            if (docs / name).exists())
        out.append({"slug": folder.name, "folder": folder,
                    "solution": files["solution"], "test": files["test"],
                    "example": files["example"],
                    "asked": said + ASKED.format(
                        files=" ".join(files["solution"]))})
    return out


def passes(texts: dict, one: dict) -> bool:
    """Whether the solution files as written ({name: text}) pass the
    exercise's tests: Exercism's unittest modules, run beside them."""
    with tempfile.TemporaryDirectory() as folder:
        for name, text in texts.items():
            (Path(folder) / name).write_text(text, encoding="utf-8")
        tests = DATA / "tests" / one["slug"]
        for name in one["test"]:
            shutil.copy(tests / name if tests.exists()
                        else one["folder"] / name, Path(folder) / name)
        modules = [Path(name).stem for name in one["test"]
                   if name.endswith("_test.py")]
        try:
            done = subprocess.run(
                [sys.executable, "-m", "unittest", *modules], cwd=folder,
                capture_output=True, text=True, timeout=SECONDS)
        except subprocess.TimeoutExpired:
            return False
    return done.returncode == 0


def projects() -> int:
    """Each exercise's project (its stub), its tests apart, and which are
    measured: the example passes, the stub does not."""
    runnable, left = [], {}
    for one in exercises():
        folder = one["folder"]
        example = {name: (folder / path).read_text(encoding="utf-8")
                   for name, path in zip(one["solution"], one["example"])}
        stub = {name: (folder / name).read_text(encoding="utf-8")
                for name in one["solution"]}
        if not passes(example, one):
            left[one["slug"]] = "its own example fails its tests here"
        elif passes(stub, one):
            left[one["slug"]] = "the stub passes its tests"
        else:
            runnable.append(one["slug"])
        project = DATA / "projects" / one["slug"]
        project.mkdir(parents=True, exist_ok=True)
        for name, text in stub.items():
            (project / name).write_text(text, encoding="utf-8")
        tests = DATA / "tests" / one["slug"]
        tests.mkdir(parents=True, exist_ok=True)
        for name in one["test"]:
            shutil.copy(folder / name, tests / name)
    (DATA / "runnable.json").write_text(json.dumps(
        {"commit": COMMIT, "runnable": runnable, "left": left}, indent=1),
        encoding="utf-8")
    print(f"{len(runnable)} runnable, {len(left)} left out: {left}")
    return len(runnable)


def runnable() -> list:
    kept = set(json.loads((DATA / "runnable.json").read_text(
        encoding="utf-8"))["runnable"])
    return [one for one in exercises() if one["slug"] in kept]


def _check(one: dict):
    """The conversation's check: the files as it left them, tested."""
    def check(turn, root):
        if root is None:
            return False
        paths = [Path(root) / name for name in one["solution"]]
        return all(path.exists() for path in paths) and passes(
            {path.name: path.read_text(encoding="utf-8") for path in paths},
            one)
    check.about = f"passes Exercism's tests of {one['slug']}"
    check.changes = True
    return check


def suite() -> list:
    from research.v704.suite import Conversation
    return [Conversation("code-change", [one["asked"]], _check(one), True,
                         str(DATA / "projects" / one["slug"]))
            for one in runnable()]


#: what the code model alone is given: the files, then what Aider asks
ALONE = ("{files}\n\n{asked}\n\nWrite each file whole, its name on the line "
         "before its ```python block, nothing else.")


def alone(label: str = "alone") -> dict:
    """The code model alone on every exercise: one answer (greedy), the
    file it writes run against the tests."""
    from research.v705 import coder, commits
    model = coder.big()
    out = DATA / f"{label}.jsonl"
    done = {}
    if out.exists():
        done = {one["slug"]: one for one in
                map(json.loads, out.open(encoding="utf-8"))}
    with out.open("a", encoding="utf-8") as stream:
        for one in runnable():
            if one["slug"] in done:
                continue
            started = time.time()
            print(one["slug"], flush=True)
            stubs = {name: (DATA / "projects" / one["slug"] / name)
                     .read_text(encoding="utf-8") for name in one["solution"]}
            asked = ALONE.format(files="\n\n".join(
                f"{name}\n```python\n{text}\n```"
                for name, text in stubs.items()), asked=one["asked"])
            try:
                reply = model.write([asked], longest=1500)[0]
                reply = reply[0] if isinstance(reply, list) else reply
                code = commits._code_of(reply) or ""
            except MemoryError:
                code = ""
            name = one["solution"][0]
            row = {"slug": one["slug"],
                   "verdict": "right" if code and len(one["solution"]) == 1
                   and passes({name: code}, one) else "wrong",
                   "seconds": round(time.time() - started, 1)}
            done[one["slug"]] = row
            stream.write(json.dumps(row) + "\n")
            stream.flush()
    rows = list(done.values())
    totals = {"right": sum(row["verdict"] == "right" for row in rows),
              "of": len(rows)}
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
