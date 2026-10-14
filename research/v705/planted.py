"""Bugs planted in its own code (v705): each function its tests run, made
wrong by one edit -- a constant, a comparison, a name, an `if` taken out
-- kept where a test that ran it now fails. The fix is the edit undone,
known exactly: what the editor is taught the local changes of its own
code from, where its history's changes were features grown with what
their commit added (`commits`), which no teacher could write.

    python -m research.v705.planted cover      state/v705-covered.json
    python -m research.v705.planted plant      state/v705-planted.jsonl
    python -m research.v705.planted corpus     llm/history-data/own-*.jsonl
    python -m research.v705.planted train      llm/editor-own
    python -m research.v705.planted measure    held-out planted bugs

**Covered** (`cover`): every test of `research/` run once in this process,
each class set up once, with the functions of its own code each runs
(`commits.ran`) and how long it took; a test that fails as the code is,
or takes longer than `QUICK`, is not used. **Planted** (`plant`): of each
function a quick test runs, up to `EACH` edits of kinds not yet taken
(`pyediting.edits`, and an `if` block taken out), each put in for the
while (`commits.Put`) and its quick tests run -- kept where one fails.
**Held out**: a tenth of the functions, by the hash of their names.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import textwrap
import time
from pathlib import Path

from research.v705 import commits

ROOT = Path(__file__).resolve().parents[2]
COVERED = ROOT / "state" / "v705-covered.json"
PLANTED = ROOT / "state" / "v705-planted.jsonl"
CORPUS = commits.HISTORY / "own-corpus.jsonl"
EDITOR = ROOT / "llm" / "editor-own"
#: a test used to find bugs runs in at most this long (seconds)
QUICK = 3.0
#: the edits planted in each function at most, and the tests run of each
EACH, TESTS = 4, 4
#: the share of functions held out
HELD = 0.1
SEED = 7060


def held(file: str, function: str) -> bool:
    at = hashlib.sha1(f"{file}#{function}".encode()).digest()[0] / 256
    return at < HELD


def _tests() -> list:
    """Every test of `research/`: {file, test} -- the classes of each
    module that are unittest's."""
    import unittest
    out = []
    for path in sorted((ROOT / "research").rglob("test*.py")):
        file = path.relative_to(ROOT).as_posix()
        try:
            module = commits._module(file)
        except Exception:                           # noqa: BLE001
            continue
        for name in dir(module):
            cls = getattr(module, name)
            if isinstance(cls, type) and issubclass(cls, unittest.TestCase) \
                    and cls.__module__ == module.__name__:
                out += [{"file": file, "test": f"{name}.{one}"}
                        for one in sorted(dir(cls))
                        if one.startswith("test")]
    return out


def cover() -> dict:
    """Each quick, passing test and the functions it runs; by function,
    the tests that run it, quickest first."""
    from research.v705 import thinking
    found = {}
    if COVERED.exists():
        found = json.loads(COVERED.read_text(encoding="utf-8"))
    tests = _tests()
    if commits.RUNNING.exists():
        # the test that never came back last time: not one to use
        stuck = json.loads(commits.RUNNING.read_text(encoding="utf-8"))
        if "test" in stuck and "file" in stuck:
            found.setdefault("tests", {})[
                f"{stuck['file']}::{stuck['test']}"] = {
                "ok": False, "seconds": commits.LONGEST, "ran": []}
        commits.RUNNING.unlink()
    for at, test in enumerate(tests):
        key = f"{test['file']}::{test['test']}"
        if key in found.get("tests", {}):
            continue
        commits.RUNNING.write_text(json.dumps(test), encoding="utf-8")
        seen: dict = {}
        started = time.time()
        timer = commits._watched(commits.LONGEST)
        ran = commits.ran(lambda: seen.update(ok=commits.passes(test)))
        timer.cancel()
        seconds = time.time() - started
        found.setdefault("tests", {})[key] = {
            "ok": bool(seen.get("ok")), "seconds": round(seconds, 2),
            "ran": sorted(f"{file}#{name}" for file, name in ran
                          if file not in thinking.JUDGING_FILES)}
        if at % 50 == 0:
            COVERED.write_text(json.dumps(found), encoding="utf-8")
            print(f"{at} of {len(tests)}", flush=True)
    by: dict = {}
    for key, one in found["tests"].items():
        if not one["ok"] or one["seconds"] > QUICK:
            continue
        file, test = key.split("::")
        for place in one["ran"]:
            by.setdefault(place, []).append((one["seconds"], file, test))
    found["functions"] = {place: [{"file": file, "test": test}
                                  for _, file, test in sorted(tests)]
                          for place, tests in by.items()}
    COVERED.write_text(json.dumps(found), encoding="utf-8")
    if commits.RUNNING.exists():
        commits.RUNNING.unlink()
    counts = {"tests": len(found["tests"]),
              "quick and passing": sum(one["ok"] and one["seconds"] <= QUICK
                                       for one in found["tests"].values()),
              "functions": len(found["functions"])}
    print(json.dumps(counts))
    return counts


def _edited(source: str, name: str) -> list:
    """[(kind, the function as edited)]: v696's single edits, and each
    `if` block taken out (the fix puts code in)."""
    from research.v696.pyediting import edits
    from research.v705.teach_debugging import _dropped
    out = []
    for one in edits(source, name):
        try:
            out.append((one.kind, one.apply(source)))
        except Exception:                           # noqa: BLE001
            continue
    for one in _dropped(source, "python"):
        out.append(("missing", one.apply(source)))
    return [(kind, made) for kind, made in out if made != source]


def plant(most: int = 0) -> dict:
    """Each function its quick tests run, up to `EACH` edits of kinds not
    yet taken, kept where a test fails -- with how it fails."""
    from research.v705 import inside
    covered = json.loads(COVERED.read_text(encoding="utf-8"))["functions"]
    done = set()
    if PLANTED.exists():
        done = {one["place"] for one in map(json.loads,
                                            PLANTED.open(encoding="utf-8"))}
    rng = random.Random(SEED)
    places = sorted(covered)
    if most:
        rng.shuffle(places)
        places = places[:most]
    counts = {"functions": 0, "bugs": 0}
    if commits.RUNNING.exists():
        # the function a test never came back from last time: left
        stuck = json.loads(commits.RUNNING.read_text(encoding="utf-8"))
        if stuck.get("place"):
            done.add(stuck["place"])
            with PLANTED.open("a", encoding="utf-8") as out:
                out.write(json.dumps({"place": stuck["place"], "file": "",
                                      "function": "", "right": "",
                                      "held": False, "bugs": [],
                                      "slow": True}) + "\n")
        commits.RUNNING.unlink()
    for place in places:
        if place in done or "<" in place:
            continue
        commits.RUNNING.write_text(json.dumps({"place": place}),
                                   encoding="utf-8")
        file, name = place.split("#")
        text = (ROOT / file).read_text(encoding="utf-8")
        source = commits.functions(text).get(name)
        if not source:
            continue
        case = {"file": file, "function": name}
        try:
            function = commits._current(case)
        except Exception:                           # noqa: BLE001
            continue
        at = text.index(source)
        indent = text[text.rfind("\n", 0, at) + 1:at]
        flat = textwrap.dedent(indent + source) if indent else source
        edits = _edited(flat, name.split(".")[-1])
        rng.shuffle(edits)
        kinds, kept = set(), []
        tests = covered[place][:TESTS]
        for kind, made in edits:
            if len(kept) >= EACH or kind in kinds:
                continue
            placed = textwrap.indent(made, indent).lstrip() if indent \
                else made
            wrong = text.replace(source, placed.rstrip("\n"))
            try:
                changed = inside.changed_function(function, wrong)
            except Exception:                       # noqa: BLE001
                continue
            if changed is None:
                continue
            fails = []
            with commits.Put(function, changed):
                for test in tests:
                    timer = commits._watched(commits.LONGEST)
                    try:
                        ok = commits.passes(test)
                    except BaseException:           # noqa: BLE001
                        ok = False
                    timer.cancel()
                    if not ok:
                        fails.append({**test, "failure": commits.failure(
                            test)})
                        break
            if fails:
                kinds.add(kind)
                kept.append({"kind": kind, "wrong": placed.rstrip("\n"),
                             "fails": fails})
        counts["functions"] += 1
        counts["bugs"] += len(kept)
        with PLANTED.open("a", encoding="utf-8") as out:
            out.write(json.dumps({"place": place, "file": file,
                                  "function": name, "right": source,
                                  "held": held(file, name),
                                  "bugs": kept}) + "\n")
        print(f"{name}: {len(kept)} of {len(edits)} edits", flush=True)
    if commits.RUNNING.exists():
        commits.RUNNING.unlink()
    print(json.dumps(counts))
    return counts


#: what a test failing of the world, not of the edit, says
NOISE = ("sqlite3.", "subprocess.TimeoutExpired", "OSError",
         "FileNotFoundError", "PermissionError", "MemoryError",
         "CUDA", "unable to open database")
#: what a test failing by a crash says (`NameError: name 'word' is not
#: defined` names its fix: an edit undone)
CRASHES = ("NameError", "UnboundLocalError", "AttributeError", "TypeError",
           "KeyError", "IndexError", "ValueError", "ZeroDivisionError",
           "RecursionError", "ImportError")


def failing(bug: dict) -> str:
    """How a planted bug shows: `noise` (the world failed, not the edit),
    `crash` (an exception names it), else `wrong` (it runs and says the
    wrong thing -- what self-improvement meets: `_whose` giving nothing).
    Of the first 351 planted, 69% were crashes."""
    said = bug["fails"][0].get("failure", "")
    last = said.split(" / ")[-1]
    if any(one in said for one in NOISE):
        return "noise"
    if "AssertionError" in last or not any(one in last for one in CRASHES):
        return "wrong"
    return "crash"


def _taught(row: dict, bug: dict) -> bool:
    """Whether a planted bug is one to teach or measure: not noise, and of
    the missing kind only where a whole `if` was taken out (taking out a
    declaration -- `cache = {}` -- is a crash, not a missing case)."""
    if failing(bug) == "noise":
        return False
    if bug["kind"] == "missing":
        gone = [one for one in row["right"].splitlines()
                if one not in bug["wrong"].splitlines()]
        return bool(gone) and gone[0].strip().startswith(("if ", "elif "))
    return True


def statement(row: dict, bug: dict) -> str:
    """What the editor is told of a planted bug: as of its history's
    (`commits.statement`), the goal the failing test."""
    test = bug["fails"][0]
    return commits.statement(
        {"message": f"make {test['test']} pass", "function": row["function"],
         "file": row["file"], "tests": [test]}, test["failure"])


def corpus() -> dict:
    """Each planted bug not held out as the editor's row -- the bug's
    function, the edit undone as blocks -- and as many of its own."""
    from research.v700 import teach_editor as T
    rows = []
    for row in map(json.loads, PLANTED.open(encoding="utf-8")):
        if row["held"] or not row["bugs"]:
            continue
        text = (ROOT / row["file"]).read_text(encoding="utf-8")
        at = text.find(row["right"])
        indent = text[text.rfind("\n", 0, at) + 1:at] if at >= 0 else ""
        right = (indent + row["right"]).splitlines(keepends=True)
        for bug in row["bugs"]:
            if not _taught(row, bug):
                continue
            wrong = (indent + bug["wrong"]).splitlines(keepends=True)
            blocks = T.blocks_of(wrong, right, 0, len(wrong))
            if not blocks:
                continue
            rows.append({"split": "train", "language": "python",
                         "statement": statement(row, bug),
                         "path": row["file"], "part": "".join(wrong),
                         "start": 1, "target": T.said(blocks),
                         "kind": bug["kind"], "failing": failing(bug)})
    rng = random.Random(SEED)
    rng.shuffle(rows)
    # wrong answers, all of them; crashes no more than as many -- taught
    # as they come, the editor would learn to undo typos a traceback names
    wrong = [one for one in rows if one["failing"] == "wrong"]
    rows = wrong + [one for one in rows
                    if one["failing"] == "crash"][:len(wrong)]
    planted = len(rows)
    rng.shuffle(rows)
    for one in rows[:len(rows) // 20]:
        one["split"] = "dev"
    own = [json.loads(line) for line in (ROOT / "llm" / "debug-data2" /
                                         "fixes.jsonl").open(
        encoding="utf-8")]
    rng.shuffle(own)
    rows += [one for one in own if one["split"] == "train"][:planted]
    rng.shuffle(rows)
    commits.HISTORY.mkdir(parents=True, exist_ok=True)
    with CORPUS.open("w", encoding="utf-8") as out:
        for one in rows:
            out.write(json.dumps(one) + "\n")
    found = {"rows": len(rows), "planted": planted,
             "wrong answers": len(wrong)}
    print(json.dumps(found))
    return found


def train(epochs: int = 2) -> None:
    """The debugging editor taught its own code's planted bugs."""
    from research.v700 import teach_editor as T
    from research.v705.teach_debugging import SAYING
    T.SAYING = SAYING
    T.train(EDITOR, epochs=epochs, corpus=CORPUS, longest=1024,
            base=ROOT / "llm" / "editor-debug2", rate=5e-5)


#: the held-out bugs an editor is measured on at most, and its rounds
MEASURED, ROUNDS = 60, 3


def measure(editor: str = "editor-debug2") -> dict:
    """An editor told a held-out planted bug -- the place, the failing
    test and how it fails -- fixing it: changed where the tests that ran
    it pass, those beside them too."""
    from research.v698.asking import Subject
    from research.v698.project import Project
    from research.v700 import fixing
    from research.v705 import inside
    from research.v705.teach_debugging import SAYING
    fixing.VERIFIED_ROUNDS = ROUNDS
    covered = json.loads(COVERED.read_text(encoding="utf-8"))["functions"]
    bugs = [(row, bug) for row in map(json.loads,
                                      PLANTED.open(encoding="utf-8"))
            if row["held"] for bug in row["bugs"] if _taught(row, bug)]
    random.Random(SEED).shuffle(bugs)
    # as many wrong answers as crashes, each counted apart (`by failing`):
    # a gain on crashes says nothing of what self-improvement meets
    wrong = [one for one in bugs if failing(one[1]) == "wrong"]
    crash = [one for one in bugs if failing(one[1]) == "crash"]
    bugs = wrong[:MEASURED // 2] + crash[:MEASURED - min(len(wrong),
                                                         MEASURED // 2)]
    fixed, by_kind, out = 0, {}, []
    by_failing: dict = {}
    for row, bug in bugs[:MEASURED]:
        case = {"file": row["file"], "function": row["function"],
                "tests": [{"file": one["file"], "test": one["test"]}
                          for one in bug["fails"]],
                "guards": [one for one in covered[row["place"]][:TESTS]
                           if one["test"] not in {f["test"] for f in
                                                  bug["fails"]}]}
        text = (ROOT / row["file"]).read_text(encoding="utf-8")
        wrong_text = text.replace(row["right"], bug["wrong"])
        function = commits._current(case)
        name = str(ROOT / row["file"]) + commits.PLANTED
        import linecache
        linecache.cache[name] = (len(wrong_text), None,
                                 wrong_text.splitlines(True), name)
        try:
            made = inside.changed_function(function, wrong_text,
                                           filename=name)
            held_project = Project("measure")
            held_project.put({row["file"]: wrong_text})
            with commits.Put(function, made):
                now = commits._current(case)
                context = {"editor": editor, "saying": SAYING,
                           "verify": commits.verifier(case, now),
                           "imports": inside.imports_of(now)}
                found = fixing.change(
                    statement(row, bug), held_project,
                    Subject("function", row["function"], row["file"]),
                    "measure", context)
        except Exception as bad:                    # noqa: BLE001
            found = {"status": f"failed: {type(bad).__name__}: {bad}"[:120]}
        ok = found.get("status") in ("changed", "partly")
        fixed += ok
        kind = by_kind.setdefault(bug["kind"], [0, 0])
        kind[0] += ok
        kind[1] += 1
        shows = by_failing.setdefault(failing(bug), [0, 0])
        shows[0] += ok
        shows[1] += 1
        out.append({"place": row["place"], "kind": bug["kind"], "fixed": ok,
                    "status": found.get("status")})
        print(f"{row['function']} ({bug['kind']}): {found.get('status')}",
              flush=True)
    result = {"editor": editor, "bugs": len(out), "fixed": fixed,
              "by failing": by_failing, "by kind": by_kind}
    (ROOT / "state" / f"v705-planted-measure-{editor}.json").write_text(
        json.dumps({**result, "each": out}, indent=1), encoding="utf-8")
    print(json.dumps(result))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("cover", "plant", "corpus", "train",
                                        "measure"))
    parser.add_argument("--most", type=int, default=0)
    parser.add_argument("--editor", default="editor-debug2")
    options = parser.parse_args(argv)
    if options.job == "plant":
        plant(options.most)
    elif options.job == "measure":
        measure(options.editor)
    else:
        {"cover": cover, "corpus": corpus, "train": train}[options.job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
