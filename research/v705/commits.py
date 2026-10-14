"""Its own history as cases to think about (v705): each commit that fixed
one function of its own code, with the tests that came with it, is a bug
it once had -- the function put back as it was before (in memory, never
on disk), the tests fail; the thinking is set to it, and the tests run
again say whether what it changed answers it. Where it finds nothing, the
commit shows where it was and what was done there (a chain in hindsight).

    python -m research.v705.commits mine       state/v705-commits.jsonl
    python -m research.v705.commits check      which still make a bug
    python -m research.v705.commits think      the thinking on each

**Mined** (`mine`): a commit whose source files (not tests) differ in one
function only -- nothing else in any of them -- and which changed or added
tests; kept where that function is, at HEAD, as the commit left it, so
putting back its parent's version undoes that commit alone. **Checked**
(`check`): the commit's tests that exist now pass as the code is, and some
fail with the old function -- those are the case's checks.
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MINED = ROOT / "state" / "v705-commits.jsonl"
CHECKED = ROOT / "state" / "v705-commits-checked.jsonl"
#: a test's longest run, checking (seconds)
LONGEST = 120


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8",
                          errors="replace").stdout


def _show(sha: str, path: str) -> str | None:
    out = subprocess.run(["git", "show", f"{sha}:{path}"], cwd=ROOT,
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    return out.stdout if out.returncode == 0 else None


def functions(text: str) -> dict:
    """{qualname: source} of a module's functions and methods (one level
    of class); {} where it does not parse."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = ast.get_source_segment(text, node) or ""
        elif isinstance(node, ast.ClassDef):
            for one in node.body:
                if isinstance(one, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f"{node.name}.{one.name}"] = \
                        ast.get_source_segment(text, one) or ""
    return out


def _test(path: str) -> bool:
    return Path(path).name.startswith("test")


def mine() -> dict:
    """Each function a commit changed that is, at HEAD, as the commit left
    it -- one case each, its old version put back alone -- of commits that
    changed or added tests."""
    shas = _git("rev-list", "HEAD", "--", "research").split()
    kept, seen = [], 0
    now: dict = {}
    for sha in shas:
        files = [one for one in _git("diff-tree", "--no-commit-id",
                                     "--name-only", "-r", sha).split()
                 if one.startswith("research/") and one.endswith(".py")]
        source = [one for one in files if not _test(one)]
        tests = [one for one in files if _test(one)]
        if not source or not tests:
            continue
        seen += 1
        added = []
        for test in tests:
            was = functions(_show(f"{sha}^", test) or "")
            is_ = functions(_show(sha, test) or "")
            if test not in now:
                now[test] = functions((ROOT / test).read_text(
                    encoding="utf-8")) if (ROOT / test).exists() else {}
            added += [{"file": test, "test": one} for one in is_
                      if "." in one and one.split(".")[1].startswith("test")
                      and is_[one] != was.get(one) and one in now[test]]
        if not added:
            continue
        message = _git("log", "-1", "--format=%s", sha).strip()
        for path in source:
            before, after = _show(f"{sha}^", path), _show(sha, path)
            if before is None or after is None:
                continue
            old, new = functions(before), functions(after)
            if path not in now:
                now[path] = functions((ROOT / path).read_text(
                    encoding="utf-8")) if (ROOT / path).exists() else {}
            for name in new:
                if name in old and old[name] != new[name] and \
                        now[path].get(name) == new[name]:
                    kept.append({"sha": sha, "message": message,
                                 "file": path, "function": name,
                                 "before": old[name], "tests": added})
    MINED.parent.mkdir(parents=True, exist_ok=True)
    with MINED.open("w", encoding="utf-8") as out:
        for one in kept:
            out.write(json.dumps(one) + "\n")
    found = {"commits": len(shas), "with tests": seen,
             "functions kept": len(kept),
             "commits kept": len({one["sha"] for one in kept})}
    print(json.dumps(found))
    return found


#: the case being checked, kept in case a test never comes back
RUNNING = ROOT / "state" / "v705-commits-running.json"
_SET_UP: set = set()


def _module(path: str):
    import importlib
    return importlib.import_module(".".join(Path(path).with_suffix("")
                                            .parts))


def _current(case: dict):
    """The function a case is of, as it runs now."""
    found = _module(case["file"])
    for part in case["function"].split("."):
        found = getattr(found, part)
    return found


def passes(test: dict) -> bool:
    """Whether one test passes: its class set up once, the test run alone
    (`setUpClass` is what costs: spaCy, the store)."""
    import unittest
    holder, method = test["test"].split(".")
    cls = getattr(_module(test["file"]), holder)
    if cls not in _SET_UP:
        _SET_UP.add(cls)
        cls.setUpClass()
    result = unittest.TestResult()
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        cls(method).run(result)
    return result.wasSuccessful()


def planted(case: dict):
    """The function as it was before the commit, made in its module as it
    is now -- None where it does not compile there."""
    from research.v705 import inside
    function = _current(case)
    text = (ROOT / case["file"]).read_text(encoding="utf-8")
    now = functions(text)[case["function"]]
    old = text.replace(now, case["before"])
    # a file of its own, its lines the old text's: its source read back
    # (`inspect`, through linecache) is what runs -- the file's own entry
    # set to the old text shifted every other function of it
    import linecache
    name = str(ROOT / case["file"]) + PLANTED
    linecache.cache[name] = (len(old), None, old.splitlines(True), name)
    try:
        return inside.changed_function(function, old, filename=name)
    except Exception:                               # noqa: BLE001
        return None


#: what a planted function's file is named by: its own, beside the real one
PLANTED = ".planted"


class Put:
    """A function put in for the while, and put back."""

    def __init__(self, function, made) -> None:
        from research.v705 import inside
        self.owner = inside._owner(function)
        self.name = function.__name__
        self.made = made

    def __enter__(self):
        self.kept = vars(self.owner)[self.name]
        made = self.made
        if isinstance(self.kept, (staticmethod, classmethod)):
            made = type(self.kept)(made)
        setattr(self.owner, self.name, made)
        return self

    def __exit__(self, *_):
        setattr(self.owner, self.name, self.kept)


def _watched(seconds: float):
    """Ends the process where a test runs past `seconds`: a test cannot be
    stopped in it, and the next run takes up from the next case."""
    import os
    import threading
    timer = threading.Timer(seconds, lambda: os._exit(3))
    timer.daemon = True
    timer.start()
    return timer


def check() -> dict:
    """Each mined case: its tests that pass now, then run with the old
    function put back -- those that fail are its checks."""
    import unittest  # noqa: F401
    cases = [json.loads(line) for line in MINED.open(encoding="utf-8")]
    done = set()
    if CHECKED.exists():
        done = {(one["sha"], one["function"]) for one in
                map(json.loads, CHECKED.open(encoding="utf-8"))}
    if RUNNING.exists():
        # what never came back last time: slow, not a case
        stuck = json.loads(RUNNING.read_text(encoding="utf-8"))
        if (stuck["sha"], stuck["function"]) not in done:
            with CHECKED.open("a", encoding="utf-8") as out:
                out.write(json.dumps({**stuck, "status": "slow"}) + "\n")
            done.add((stuck["sha"], stuck["function"]))
        RUNNING.unlink()
    now: dict = {}
    counts: dict = {}
    for case in cases:
        key = (case["sha"], case["function"])
        if key in done:
            continue
        RUNNING.write_text(json.dumps({"sha": case["sha"],
                                       "function": case["function"],
                                       "file": case["file"]}),
                           encoding="utf-8")
        started = time.time()
        status, fails = "", []
        try:
            made = planted(case)
            if made is None:
                status = "does not compile now"
            else:
                tests = []
                for test in case["tests"]:
                    name = (test["file"], test["test"])
                    if name not in now:
                        timer = _watched(LONGEST)
                        try:
                            now[name] = passes(test)
                        except Exception:           # noqa: BLE001
                            now[name] = False
                        timer.cancel()
                    if now[name]:
                        tests.append(test)
                with Put(_current(case), made):
                    for test in tests:
                        timer = _watched(LONGEST)
                        try:
                            ok = passes(test)
                        except Exception:           # noqa: BLE001
                            ok = False
                        timer.cancel()
                        if not ok:
                            fails.append(test)
                status = "bug" if fails else ("no test passes now"
                                              if not tests else
                                              "tests still pass")
        except Exception as bad:                    # noqa: BLE001
            status = f"failed: {type(bad).__name__}: {bad}"[:160]
        counts[status.split(":")[0]] = counts.get(status.split(":")[0],
                                                  0) + 1
        with CHECKED.open("a", encoding="utf-8") as out:
            out.write(json.dumps({"sha": case["sha"],
                                  "function": case["function"],
                                  "file": case["file"], "status": status,
                                  "fails": fails, "seconds": round(
                                      time.time() - started, 1)}) + "\n")
        print(f"{case['sha'][:7]} {case['function']}: {status} "
              f"({len(fails)})", flush=True)
    if RUNNING.exists():
        RUNNING.unlink()
    print(json.dumps(counts))
    return counts


# -- thinking about them --------------------------------------------------------

def ran(run) -> set:
    """The functions of its own code (not tests) `run()` calls:
    {(file, qualname)}, by `sys.monitoring` -- each reported once."""
    from research.v705 import inside
    monitoring = sys.monitoring
    tool = monitoring.PROFILER_ID
    seen: set = set()
    places: dict = {}

    def start(code, _offset):
        name = code.co_filename.removesuffix(PLANTED)
        path = places.get(name, False)
        if path is False:
            path = inside._relative(name) if "research" in name else None
            if path and ("/test" in path or not path.startswith(
                    "research/")):
                path = None
            places[name] = path
        if path:
            seen.add((path, getattr(code, "co_qualname", code.co_name)))
        return monitoring.DISABLE
    monitoring.use_tool_id(tool, "v705-commits")
    try:
        monitoring.register_callback(tool, monitoring.events.PY_START, start)
        monitoring.set_events(tool, monitoring.events.PY_START)
        monitoring.restart_events()
        try:
            run()
        except Exception:                           # noqa: BLE001
            pass
    finally:
        monitoring.set_events(tool, 0)
        monitoring.register_callback(tool, monitoring.events.PY_START, None)
        monitoring.free_tool_id(tool)
    return seen


#: the passing tests of a failing test's class a spectrum is taken over
PASSING = 12
#: how long the thinking is given each bug (seconds)
SECONDS = 240


def localize(case: dict, most: int = 8) -> list:
    """Where the failing tests go wrong, by spectrum: the functions each
    failing test runs, and the passing tests of the same classes, scored
    by Ochiai -- [{function, file, score}], most suspected first."""
    import math
    failing = [ran(lambda test=test: passes(test)) for test in case["tests"]]
    passing = []
    for holder in {(one["file"], one["test"].split(".")[0])
                   for one in case["tests"]}:
        cls = getattr(_module(holder[0]), holder[1])
        names = [one for one in dir(cls) if one.startswith("test")]
        failed = {one["test"] for one in case["tests"]}
        for name in names:
            test = {"file": holder[0], "test": f"{holder[1]}.{name}"}
            if test["test"] in failed or len(passing) >= PASSING:
                continue
            seen: dict = {}
            covered = ran(lambda test=test: seen.update(ok=passes(test)))
            if seen.get("ok"):
                passing.append(covered)
    scores = []
    for place in set().union(*failing) if failing else set():
        ef = sum(place in one for one in failing)
        ep = sum(place in one for one in passing)
        score = ef / math.sqrt(len(failing) * (ef + ep)) if ef else 0.0
        scores.append({"file": place[0], "function": place[1],
                       "score": round(score, 3)})
    scores.sort(key=lambda one: (-one["score"], one["function"]))
    return [one for one in scores if "<" not in one["function"]][:most]


def verifier(case: dict, function):
    """A changed file's text tried: the function as it has it put in, the
    failing tests run -- None where all pass, else which does not."""
    from research.v705 import inside

    def verify(text: str) -> str | None:
        try:
            made = inside.changed_function(function, text)
        except Exception as bad:                    # noqa: BLE001
            return f"it does not compile: {type(bad).__name__}"
        if made is None:
            return f"the change does not define {function.__name__}"
        if "guards" not in case:
            # what passes beside them, as the code is: kept passing --
            # taking out a line the failing test trips on passed it twice
            # (`self.calls += 1`; the loop counting a saying's words)
            case["guards"] = _beside(case)
        with Put(function, made):
            for test in case["tests"] + case["guards"]:
                timer = _watched(LONGEST)
                try:
                    ok = passes(test)
                except BaseException:               # noqa: BLE001
                    ok = False
                timer.cancel()
                if not ok:
                    return (f"{test['test']} still fails"
                            if test in case["tests"] else
                            f"{test['test']}, which passed, now fails")
        return None
    return verify


def _beside(case: dict) -> list:
    """The tests of the failing tests' classes that pass as the code is
    now (`PASSING` at most)."""
    out = []
    failed = {one["test"] for one in case["tests"]}
    for holder in sorted({(one["file"], one["test"].split(".")[0])
                          for one in case["tests"]}):
        cls = getattr(_module(holder[0]), holder[1])
        for name in sorted(one for one in dir(cls)
                           if one.startswith("test")):
            test = {"file": holder[0], "test": f"{holder[1]}.{name}"}
            if test["test"] in failed or len(out) >= PASSING:
                continue
            timer = _watched(LONGEST)
            try:
                ok = passes(test)
            except BaseException:                   # noqa: BLE001
                ok = False
            timer.cancel()
            if ok:
                out.append(test)
    return out


def think(most: int = 0) -> dict:
    """The thinking set to each bug its history holds: the function put
    back as it was (in the project held and as it runs), the commit's
    message the goal, its failing tests the case; the chain kept, and
    with it where the commit says the fix was."""
    import random
    from research.v705 import thinking
    mined = {(one["sha"], one["function"]): one for one in
             map(json.loads, MINED.open(encoding="utf-8"))}
    bugs = [one for one in map(json.loads, CHECKED.open(encoding="utf-8"))
            if one["status"] == "bug"]
    done = set()
    if thinking.CHAINS.exists():
        done = {(one.get("commit") or {}).get("sha", "") + "/" +
                (one.get("commit") or {}).get("function", "")
                for one in map(json.loads,
                               thinking.CHAINS.open(encoding="utf-8"))}
    mind = thinking.Mind()
    counts = {"fixed": 0, "place found": 0, "thought": 0}
    for bug in bugs[:most or None]:
        case = {**mined[(bug["sha"], bug["function"])], "fails": bug["fails"]}
        if f"{case['sha']}/{case['function']}" in done:
            continue
        function = _current(case)
        made = planted(case)
        text = mind.held.files[case["file"]]
        old = text.replace(functions(text)[case["function"]], case["before"])
        mind.held.files[case["file"]] = old
        try:
            with Put(function, made):
                pad = thinking.think_case(
                    case["message"], {"said": case["message"],
                                      "tests": case["fails"],
                                      "commit": case}, mind,
                    random.Random(thinking.SEED), seconds=SECONDS)
        finally:
            mind.held.files[case["file"]] = text
        places = [one["function"] for one in pad.places]
        counts["thought"] += 1
        counts["fixed"] += pad.reward >= 1
        counts["place found"] += case["function"] in places
        print(f"{case['sha'][:7]} {case['function']}: reward {pad.reward}, "
              f"place {'found at ' + str(places.index(case['function'])) if case['function'] in places else 'missed'}",
              flush=True)
    print(json.dumps(counts))
    return counts


# -- teaching the editor from them ---------------------------------------------

HISTORY = ROOT / "llm" / "history-data"
EDITS = HISTORY / "edits.jsonl"
CORPUS = HISTORY / "corpus.jsonl"
EDITOR = ROOT / "llm" / "editor-history"
#: a function taught at most this many lines long, a change at most this
#: many lines changed
LINES, CHANGED = 80, 24
#: the editor's own rows kept beside them, against forgetting, per row
KEPT = 1.0


def edits() -> dict:
    """Every function its history changed -- the commit's message, the
    function before, the change as the editor's blocks -- of commits that
    hold none of the bugs it is measured on (`CHECKED`)."""
    import difflib
    from research.v700 import teach_editor as T
    held = {one["sha"] for one in map(json.loads,
                                      CHECKED.open(encoding="utf-8"))
            if one["status"] == "bug"}
    shas = _git("rev-list", "HEAD", "--", "research").split()
    rows = []
    for sha in shas:
        if sha in held:
            continue
        message = _git("log", "-1", "--format=%s", sha).strip()
        files = [one for one in _git("diff-tree", "--no-commit-id",
                                     "--name-only", "-r", sha).split()
                 if one.startswith("research/") and one.endswith(".py")
                 and not _test(one)]
        for path in files:
            before, after = _show(f"{sha}^", path), _show(sha, path)
            if before is None or after is None:
                continue
            old, new = functions(before), functions(after)
            for name in new:
                if name not in old or old[name] == new[name]:
                    continue
                was = old[name].splitlines(keepends=True)
                is_ = new[name].splitlines(keepends=True)
                if len(was) > LINES:
                    continue
                changed = sum(1 for one in difflib.ndiff(was, is_)
                              if one[:1] in "+-")
                if not changed or changed > CHANGED:
                    continue
                blocks = T.blocks_of(was, is_, 0, len(was))
                if not blocks:
                    continue
                rows.append({"sha": sha, "language": "python",
                             "statement": f"{message}\nChange {name} "
                                          f"in {path} for it.",
                             "path": path, "part": old[name], "start": 1,
                             "target": T.said(blocks)})
    HISTORY.mkdir(parents=True, exist_ok=True)
    with EDITS.open("w", encoding="utf-8") as out:
        for one in rows:
            out.write(json.dumps(one) + "\n")
    found = {"edits": len(rows), "commits": len({one["sha"] for one in rows})}
    print(json.dumps(found))
    return found


def corpus() -> dict:
    """The history's edits (a twentieth kept apart, by commit) and as many
    of the debugging editor's own rows."""
    import hashlib
    rows = [json.loads(line) for line in EDITS.open(encoding="utf-8")]
    for one in rows:
        at = hashlib.sha1(one["sha"].encode()).digest()[0] / 256
        one["split"] = "dev" if at < 0.05 else "train"
    own = [json.loads(line) for line in (ROOT / "llm" / "debug-data2" /
                                         "fixes.jsonl").open(
        encoding="utf-8")]
    rng = random.Random(7058)
    rng.shuffle(own)
    rows += [one for one in own if one["split"] == "train"][
        :int(len(rows) * KEPT)]
    rng.shuffle(rows)
    with CORPUS.open("w", encoding="utf-8") as out:
        for one in rows:
            out.write(json.dumps(one) + "\n")
    found = {"rows": len(rows), "history": sum("sha" in one for one in rows)}
    print(json.dumps(found))
    return found


def train(epochs: int = 2) -> None:
    """The debugging editor taught its own history's changes."""
    from research.v700 import teach_editor as T
    from research.v705.teach_debugging import SAYING
    T.SAYING = SAYING
    T.train(EDITOR, epochs=epochs, corpus=CORPUS, longest=1024,
            base=ROOT / "llm" / "editor-debug2", rate=5e-5)


#: the bugs an editor is measured on, and the rounds of answers it is given
MEASURED, ROUNDS = 40, 3
#: the first bug measured (a run taken up again)
FROM = 0


def measure(editor: str = "editor-debug2") -> dict:
    """An editor told the goal and the place -- the commit's message and
    its function, put back as it was -- fixing it: changed where the
    failing tests then pass. Of the first `MEASURED` bugs (none of whose
    commits it was taught)."""
    from research.v698.asking import Subject
    from research.v698.project import Project
    from research.v700 import fixing
    from research.v705 import inside
    from research.v705.teach_debugging import SAYING
    fixing.VERIFIED_ROUNDS = ROUNDS
    mined = {(one["sha"], one["function"]): one for one in
             map(json.loads, MINED.open(encoding="utf-8"))}
    bugs = [one for one in map(json.loads, CHECKED.open(encoding="utf-8"))
            if one["status"] == "bug"][:MEASURED]
    fixed, out = 0, []
    for bug in bugs:
        case = {**mined[(bug["sha"], bug["function"])], "tests": bug["fails"]}
        function = _current(case)
        made = planted(case)
        disk = (ROOT / case["file"]).read_text(encoding="utf-8")
        old = disk.replace(functions(disk)[case["function"]], case["before"])
        held = Project("measure")
        held.put({case["file"]: old})
        try:
            with Put(function, made):
                planted_now = _current(case)
                context = {"editor": editor, "saying": SAYING,
                           "verify": verifier(case, planted_now),
                           "imports": inside.imports_of(planted_now)}
                found = fixing.change(
                    f"{case['message']}\nChange {case['function']} in "
                    f"{case['file']} for it.", held,
                    Subject("function", case["function"], case["file"]),
                    "measure", context)
        except Exception as bad:                    # noqa: BLE001
            import traceback
            traceback.print_exc()
            found = {"status": f"failed: {type(bad).__name__}"}
        ok = found.get("status") in ("changed", "partly")
        fixed += ok
        out.append({"sha": case["sha"], "function": case["function"],
                    "fixed": ok, "status": found.get("status")})
        print(f"{case['sha'][:7]} {case['function']}: {found.get('status')}",
              flush=True)
    result = {"editor": editor, "bugs": len(bugs), "fixed": fixed}
    (ROOT / "state" / f"v705-commits-measure-{editor}.json").write_text(
        json.dumps({**result, "each": out}, indent=1), encoding="utf-8")
    print(json.dumps(result))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("mine", "check", "think", "edits",
                                           "corpus", "train", "measure"))
    parser.add_argument("--editor", default="editor-debug2")
    parser.add_argument("--most", type=int, default=0)
    options = parser.parse_args(argv)
    if options.job == "think":
        think(options.most)
    elif options.job == "measure":
        measure(options.editor)
    else:
        {"mine": mine, "check": check, "edits": edits, "corpus": corpus,
         "train": train}[options.job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
