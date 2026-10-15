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
    import threading
    # a thread the test starts that dies of an error is the test failing:
    # unittest never hears of it, and `Service.run` held with its waiting
    # readers ending on `return answer` unbound, `calls == ['same']` true
    before = {one.ident for one in threading.enumerate()}
    died = []
    hook = threading.excepthook

    def heard(args):
        if args.thread is None or args.thread.ident not in before:
            died.append(f"{args.exc_type.__name__}: {args.exc_value}")
        else:
            hook(args)
    threading.excepthook = heard
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            cls(method).run(result)
    finally:
        threading.excepthook = hook
    LAST["failure"] = next((text for _, text in result.failures +
                            result.errors), "")
    if died and not LAST["failure"]:
        LAST["failure"] = f"a thread it started died: {died[0]}"
    return result.wasSuccessful() and not died


#: what the last test run said where it failed (`passes`)
LAST: dict = {}


def failure(test: dict) -> str:
    """How a test fails, as its traceback ends: the line that failed and
    what was raised (`AssertionError: 'x' != 'y'`)."""
    passes(test)
    lines = [one for one in LAST.get("failure", "").splitlines()
             if one.strip()]
    said = [one.strip() for one in lines if one.strip().startswith(
        ("self.assert", "assert"))][-1:] + lines[-1:]
    return " / ".join(said)[:400]


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
    """A test that runs past `seconds` fails: a TimeoutError raised in the
    thread running it (a change that loops forever is a wrong change, and
    ending the process on it ended a whole run at `read`); and where it
    is still stuck as long again (waiting in C, where nothing is raised)
    the process ends, and the next run takes up from the next case."""
    import ctypes
    import os
    import threading
    stuck = threading.get_ident()

    class Watch:
        def __init__(self):
            self.raised = threading.Timer(seconds, self.stop)
            self.ended = threading.Timer(2 * seconds, self.end)
            for one in (self.raised, self.ended):
                one.daemon = True
                one.start()

        def end(self):
            # where it was stuck, said before it goes
            import faulthandler
            import sys
            faulthandler.dump_traceback(sys.__stderr__, all_threads=True)
            os._exit(3)

        def stop(self):
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(stuck), ctypes.py_object(TimeoutError))

        def cancel(self):
            self.raised.cancel()
            self.ended.cancel()
            # raised just as it finished: taken back before it lands
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(stuck), None)
    return Watch()


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
    # functions only: a class's body runs too where its module is imported
    # (`Expr`, `Form`), and was taken for a place nothing could change
    from research.v705 import inside
    out = []
    for one in scores:
        if "<" in one["function"]:
            continue
        try:
            found = inside.function_at(one["file"], one["function"])
        except Exception:                           # noqa: BLE001
            found = None
        if found is not None:
            out.append(one)
        if len(out) >= most:
            break
    return out


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
        copied = copies_test(case, function, text)
        if copied:
            return f"it writes the test's {copied!r} into the code"
        loose = reads_unbound(function, text)
        if loose:
            return f"it reads {loose} where a way there leaves it unset"
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


def _strings(source: str) -> set:
    """The string constants of some code, three characters or more."""
    import textwrap
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return set()
    return {one.value for one in ast.walk(tree) if isinstance(
        one, ast.Constant) and isinstance(one.value, str) and
        len(one.value.strip()) >= 3}


def copies_test(case: dict, function, text: str) -> str | None:
    """A string a change puts in the function that its failing tests say
    -- the case written into the code, not fixed (told what a check
    should get, the code model wrote `'for x in xs if (x % 2 == 0):'`, the
    test's own). None where it writes none."""
    import inspect
    try:
        before = _strings(inspect.getsource(function))
    except (OSError, TypeError):
        before = set()
    name = function.__qualname__
    after = _strings(functions(text).get(name, "") or functions(text).get(
        name.split(".")[-1], ""))
    said = " ".join(case.get("wanted values") or [])
    for test in case.get("tests") or []:
        said += functions((ROOT / test["file"]).read_text(
            encoding="utf-8")).get(test["test"], "")
    # what a check wanted, as a value: written inside a new string too
    # (`' for x in xs if '` in `'for x in xs if (x % 2 == 0):'`)
    wanted = []
    for one in case.get("wanted values") or []:
        try:
            value = ast.literal_eval(one)
        except (ValueError, SyntaxError):
            value = one
        if isinstance(value, str) and len(value.strip()) >= 3:
            wanted.append(value.strip())
    import keyword
    import re
    for one in sorted(after - before, key=len, reverse=True):
        core = one.strip().strip(":;,.")
        # glue of the language's own words and marks (`'def '`, `'() ->
        # '`) is how code is written, not what one test says
        words = re.findall(r"[A-Za-z_]\w*", core)
        if not any(not keyword.iskeyword(word) for word in words):
            continue
        # one name -- a key, a field, a kind (`'language'`, `'python'`) --
        # is what the code and its test share by design, not a case
        if core.isidentifier():
            continue
        if len(core) >= 3 and core in said:
            return core[:60]
        if any(value in one for value in wanted):
            return core[:60]
    return None


def reads_unbound(function, text: str) -> str | None:
    """A local name a change reads where some way to that read never sets
    it, and the function as it was had no such read -- passing tests that
    never take that way say nothing of it (`return answer` in the branch
    that found the answer cached). None where it adds none."""
    import inspect
    try:
        was = unbound(inspect.getsource(function))
    except (OSError, TypeError):
        was = set()
    name = function.__qualname__
    found = functions(text)
    now = found.get(name, "") or found.get(name.split(".")[-1], "")
    new = sorted(unbound(now) - was) if now else []
    return new[0] if new else None


def unbound(source: str) -> set:
    """The local names a function reads somewhere a way to it leaves them
    unset: each statement taken in order with what is surely set by then
    -- both arms of an `if` for what follows, none of a loop's body, a
    `try`'s handlers from before it; a way that returns, raises, breaks or
    goes on sets everything (nothing after it on that way runs)."""
    import textwrap
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return set()
    function = next((one for one in tree.body if isinstance(
        one, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if function is None:
        return set()
    arguments = function.args
    given = {one.arg for one in arguments.posonlyargs + arguments.args +
             arguments.kwonlyargs} | {one.arg for one in (arguments.vararg,
                                                          arguments.kwarg)
                                      if one}
    outer = set()
    local = set(given)
    for node in _own(function.body):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            outer |= set(node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (
                ast.Store, ast.Del)):
            local.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            local.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            local |= {(one.asname or one.name).split(".")[0]
                      for one in node.names}
        elif isinstance(node, ast.ExceptHandler) and node.name:
            local.add(node.name)
    local -= outer
    flagged: set = set()

    def read(node, bound):
        for one in _own([node], expression=True):
            if isinstance(one, ast.Name) and isinstance(one.ctx, ast.Load) \
                    and one.id in local and one.id not in bound:
                flagged.add(one.id)
        for one in _own([node], expression=True):
            if isinstance(one, ast.NamedExpr):
                bound.add(one.target.id)

    def sets(target) -> set:
        return {one.id for one in ast.walk(target)
                if isinstance(one, ast.Name)} | {
            getattr(one, "name", None) or getattr(one, "rest", None)
            for one in ast.walk(target) if isinstance(one, (
                ast.MatchAs, ast.MatchStar, ast.MatchMapping))} - {None}

    def walk(body, bound):
        """What is surely set after `body`, or None where it never ends
        by running off its end."""
        bound = set(bound)
        for statement in body:
            if isinstance(statement, (ast.Return, ast.Raise)):
                for part in ast.iter_child_nodes(statement):
                    read(part, bound)
                return None
            if isinstance(statement, (ast.Break, ast.Continue)):
                return None
            if isinstance(statement, ast.If):
                read(statement.test, bound)
                arms = [walk(statement.body, bound),
                        walk(statement.orelse, bound)]
                ended = [one for one in arms if one is not None]
                if not ended:
                    return None
                bound = set.intersection(*ended)
            elif isinstance(statement, (ast.For, ast.AsyncFor)):
                read(statement.iter, bound)
                walk(statement.body, bound | sets(statement.target))
                after = walk(statement.orelse, bound)
                bound = after if after is not None else bound
            elif isinstance(statement, ast.While):
                read(statement.test, bound)
                walk(statement.body, bound)
                after = walk(statement.orelse, bound)
                bound = after if after is not None else bound
            elif isinstance(statement, (ast.With, ast.AsyncWith)):
                for item in statement.items:
                    read(item.context_expr, bound)
                    if item.optional_vars is not None:
                        bound |= sets(item.optional_vars)
                after = walk(statement.body, bound)
                if after is None:
                    return None
                bound = after
            elif isinstance(statement, ast.Try) or type(
                    statement).__name__ == "TryStar":
                tried = walk(statement.body, bound)
                tried = walk(statement.orelse, tried) if tried is not None \
                    else None
                caught = [walk(handler.body, bound | ({handler.name} if
                                                      handler.name else
                                                      set()))
                          for handler in statement.handlers]
                ended = [one for one in [tried] + caught if one is not None]
                after = walk(statement.finalbody, set.intersection(*ended)
                             if ended else bound)
                if not ended or after is None:
                    return None
                bound = after
            elif isinstance(statement, ast.Match):
                read(statement.subject, bound)
                arms = [walk(case.body, bound | sets(case.pattern))
                        for case in statement.cases] + [bound]
                bound = set.intersection(*[one for one in arms
                                           if one is not None])
            elif isinstance(statement, (ast.FunctionDef,
                                        ast.AsyncFunctionDef,
                                        ast.ClassDef)):
                for part in statement.decorator_list:
                    read(part, bound)
                bound.add(statement.name)
            elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                bound |= {(one.asname or one.name).split(".")[0]
                          for one in statement.names}
            elif isinstance(statement, ast.AugAssign):
                read(statement.value, bound)
                read(statement.target, bound)
                if isinstance(statement.target, ast.Name) and \
                        statement.target.id in local and \
                        statement.target.id not in bound:
                    flagged.add(statement.target.id)
            elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
                if statement.value is not None:
                    read(statement.value, bound)
                targets = statement.targets if isinstance(
                    statement, ast.Assign) else [statement.target]
                for target in targets:
                    if not isinstance(target, ast.Name):
                        read(target, bound)
                    bound |= {one.id for one in ast.walk(target) if
                              isinstance(one, ast.Name) and
                              isinstance(one.ctx, ast.Store)}
            elif isinstance(statement, ast.Delete):
                for target in statement.targets:
                    read(target, bound)
                    if isinstance(target, ast.Name):
                        bound.discard(target.id)
            else:
                for part in ast.iter_child_nodes(statement):
                    read(part, bound)
        return bound
    walk(function.body, given)
    return flagged


def _own(nodes: list, expression: bool = False):
    """The nodes under these in the same scope: not inside a function,
    lambda, class or comprehension of their own (a comprehension's reads
    of its own names are not the function's)."""
    scoped = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
              ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp,
              ast.GeneratorExp)
    stack = list(nodes)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, scoped):
            if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                                 ast.GeneratorExp)):
                # the first iterable is read in the function's own scope
                stack.append(node.generators[0].iter)
            elif not expression and isinstance(node, (
                    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            continue
        stack.extend(ast.iter_child_nodes(node))


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
    # and the tests anywhere that run the function (`planted.cover`): its
    # class alone let a `read` through that set `yours` on every request,
    # and broke a test of the ways of writing two versions on
    seen = failed | {one["test"] for one in out}
    for test in _covering(case)[:COVERING * 3]:
        if len(out) >= PASSING + COVERING:
            break
        if test["test"] in seen:
            continue
        seen.add(test["test"])
        timer = _watched(LONGEST)
        try:
            ok = passes(test)
        except BaseException:                       # noqa: BLE001
            ok = False
        timer.cancel()
        if ok:
            out.append(test)
    return out


#: the tests beyond its class a change is judged by, at most
COVERING = 12


def _covering(case: dict) -> list:
    """The quick tests that run the case's function, quickest first, as
    `planted.cover` found them -- none where it was never gathered."""
    path = ROOT / "state" / "v705-covered.json"
    if not path.exists():
        return []
    if "covered" not in _COVERED:
        _COVERED["covered"] = json.loads(path.read_text(
            encoding="utf-8")).get("functions", {})
    return list(_COVERED["covered"].get(
        f"{case['file']}#{case['function']}", []))


_COVERED: dict = {}


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
                    statement(case, failure(case["tests"][0])), held,
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


# -- the teacher's fixes, kept where they hold -----------------------------------

TAUGHT = HISTORY / "teacher-fixes.jsonl"
TAUGHT_CORPUS = HISTORY / "taught-corpus.jsonl"
TAUGHT_EDITOR = ROOT / "llm" / "editor-taught"
#: the fixes the teacher writes of each bug
TRIES = 8

FIX = ("A function in a Python project has a bug. The commit that fixed it "
       "said: \"{message}\"\nIn {path}, the function as it is:\n```python\n"
       "{before}\n```\nThis test fails:\n```python\n{test}\n```\nIt fails "
       "with: {failure}\nIn the same file, beside it:\n{reach}\n"
       "Write the whole corrected function -- the same "
       "name and parameters -- in one ```python block, nothing else.")


#: the functions beside it a fix is told of
REACH = 10


def reach(case: dict, about: str = "") -> list:
    """What the function could call in its own file, as a programmer would
    look: each other function's def line and the first line of what it
    says, the nearest it in the file first -- a helper is written beside
    what it helps. The fixes of its history call what their commit added
    beside them (`determines`, nine lines below `relevance`): told nothing
    of it, the teacher held 0 of 24; ranked by the commit's message
    (`rung 2: control forms`), it was not among the ten."""
    text = (ROOT / case["file"]).read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    own = case["function"].split(".")[-1]
    at = next((node.lineno for node in ast.walk(tree) if isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == own),
        0)
    said = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                node.name != own and not node.name.startswith("test"):
            source = ast.get_source_segment(text, node) or ""
            head = " ".join(source.split(":\n")[0].split()) + ":"
            doc = (ast.get_docstring(node) or "").split("\n")[0]
            said.append((abs(node.lineno - at),
                         f"{head}  # {doc}" if doc else head))
    return [one for _, one in sorted(said)][:REACH]


def statement(case: dict, failed: str) -> str:
    """What the editor is told of a bug, taught and asked alike: the goal
    (the commit's message), the place, how its test fails, and what it
    could call beside it."""
    test = case["tests"][0]["test"]
    near = reach(case)
    return (f"{case['message']}\nChange {case['function']} in "
            f"{case['file']} for it.\n{test} fails: {failed}"
            + ("\nBeside it:\n" + "\n".join(near) if near else ""))


def _code_of(reply: str) -> str | None:
    import re
    found = re.search(r"```(?:python)?\n(.*?)```", reply, re.S)
    return found.group(1) if found else None


def _placed(fixed: str, before: str, text: str) -> str | None:
    """The file's text with the function as the teacher wrote it, indented
    as the old one stood; None where it is not one function of its name."""
    import textwrap
    code = textwrap.dedent(fixed).strip("\n")
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    name = before.split("(")[0].split()[-1]
    # the function of its name, among any others the answer wrote (a
    # helper beside it, an import): those are not put in
    # a method written in its class is the method (84 of 293 of the code
    # model's answers were thrown away for the class around them)
    found = next((one for one in ast.walk(tree) if isinstance(
        one, (ast.FunctionDef, ast.AsyncFunctionDef)) and one.name == name),
        None)
    if found is None:
        return None
    # as if written at the margin: a method's lines after its first keep
    # the class's indent
    code = textwrap.dedent(" " * found.col_offset +
                           ast.get_source_segment(code, found))
    at = text.index(before)
    indent = text[text.rfind("\n", 0, at) + 1:at]
    lines = code.splitlines()
    placed = "\n".join([lines[0]] + [indent + one if one.strip() else one
                                     for one in lines[1:]])
    return text[:at] + placed + text[at + len(before):]


def _held_out() -> set:
    """The commits of the bugs editors are measured on (`measure`)."""
    bugs = [one for one in map(json.loads, CHECKED.open(encoding="utf-8"))
            if one["status"] == "bug"]
    return {one["sha"] for one in bugs[:MEASURED]}


def teach() -> dict:
    """The teacher (SmolLM3, offline) told each bug not measured on --
    the commit's message, the function, a failing test and how it fails
    -- writing `TRIES` fixes; each run with the tests and those beside
    them, kept where all pass."""
    from research.v696.teach_meaning import Teacher
    mined = {(one["sha"], one["function"]): one for one in
             map(json.loads, MINED.open(encoding="utf-8"))}
    held = _held_out()
    bugs = [one for one in map(json.loads, CHECKED.open(encoding="utf-8"))
            if one["status"] == "bug" and one["sha"] not in held]
    done = set()
    if TAUGHT.exists():
        done = {(one["sha"], one["function"]) for one in
                map(json.loads, TAUGHT.open(encoding="utf-8"))}
    teacher = Teacher()
    HISTORY.mkdir(parents=True, exist_ok=True)
    counts = {"bugs": 0, "fixed": 0, "fixes": 0}
    for bug in bugs:
        if (bug["sha"], bug["function"]) in done:
            continue
        case = {**mined[(bug["sha"], bug["function"])], "tests": bug["fails"]}
        function = _current(case)
        made = planted(case)
        disk = (ROOT / case["file"]).read_text(encoding="utf-8")
        old = disk.replace(functions(disk)[case["function"]], case["before"])
        test = case["tests"][0]
        source = functions((ROOT / test["file"]).read_text(
            encoding="utf-8")).get(test["test"], "")
        kept, why = [], []
        replies = []
        with Put(function, made):
            failed = failure(test)
            replies = teacher.write([FIX.format(
                message=case["message"], path=case["file"],
                before=case["before"], test=source, failure=failed,
                reach="\n".join(reach(case, "")))],
                longest=900, samples=TRIES, temperature=0.8)[0]
            verify = verifier(case, _current(case))
            seen = set()
            why = []
            for reply in replies:
                code = _code_of(reply)
                text = _placed(code, case["before"], old) if code else None
                if text is None:
                    why.append("no code block" if not code else
                               "not one function of its name")
                    continue
                if text in seen:
                    why.append("the same again")
                    continue
                seen.add(text)
                if text == old:
                    why.append("unchanged")
                    continue
                said = verify(text)
                why.append(said or "holds")
                if said is None:
                    kept.append(functions(text)[case["function"]])
        counts["bugs"] += 1
        counts["fixed"] += bool(kept)
        counts["fixes"] += len(kept)
        with TAUGHT.open("a", encoding="utf-8") as out:
            out.write(json.dumps({"sha": case["sha"],
                                  "function": case["function"],
                                  "file": case["file"],
                                  "statement": statement(case, failed),
                                  "before": case["before"],
                                  "fixes": kept, "why": why,
                                  "replies": replies}) + "\n")
        print(f"{case['sha'][:7]} {case['function']}: {len(kept)} of "
              f"{len(replies)} hold", flush=True)
    print(json.dumps(counts))
    return counts


def taught_corpus() -> dict:
    """The fixes that held as the editor's rows -- told what it will be
    told, answered with the change as blocks -- and three times as many
    of its own (against forgetting)."""
    from research.v700 import teach_editor as T
    rows = []
    for one in map(json.loads, TAUGHT.open(encoding="utf-8")):
        file_text = (ROOT / one["file"]).read_text(encoding="utf-8")
        at = file_text.find(one["before"].split("\n")[0])
        indent = file_text[file_text.rfind("\n", 0, at) + 1:at] \
            if at >= 0 else ""
        was = (indent + one["before"]).splitlines(keepends=True)
        for fixed in one["fixes"]:
            is_ = (indent + fixed).splitlines(keepends=True)
            blocks = T.blocks_of(was, is_, 0, len(was))
            if not blocks:
                continue
            rows.append({"split": "train", "language": "python",
                         "statement": one["statement"], "path": one["file"],
                         "part": "".join(was), "start": 1,
                         "target": T.said(blocks), "sha": one["sha"]})
    taught = len(rows)
    own = [json.loads(line) for line in (ROOT / "llm" / "debug-data2" /
                                         "fixes.jsonl").open(
        encoding="utf-8")]
    rng = random.Random(7059)
    rng.shuffle(own)
    rows += [one for one in own if one["split"] == "train"][
        :max(taught * 3, 200)]
    rows += [one for one in own if one["split"] == "dev"][:50]
    rng.shuffle(rows)
    with TAUGHT_CORPUS.open("w", encoding="utf-8") as out:
        for one in rows:
            out.write(json.dumps(one) + "\n")
    found = {"rows": len(rows), "taught": taught}
    print(json.dumps(found))
    return found


def train_taught(epochs: int = 3) -> None:
    """The debugging editor taught the teacher's fixes that held."""
    from research.v700 import teach_editor as T
    from research.v705.teach_debugging import SAYING
    T.SAYING = SAYING
    T.train(TAUGHT_EDITOR, epochs=epochs, corpus=TAUGHT_CORPUS,
            longest=1024, base=ROOT / "llm" / "editor-debug2", rate=5e-5)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("mine", "check", "think", "edits",
                                           "corpus", "train", "measure", "teach",
                                           "taught-corpus", "train-taught"))
    parser.add_argument("--editor", default="editor-debug2")
    parser.add_argument("--most", type=int, default=0)
    options = parser.parse_args(argv)
    if options.job == "think":
        think(options.most)
    elif options.job == "measure":
        measure(options.editor)
    else:
        {"mine": mine, "check": check, "edits": edits, "corpus": corpus,
         "train": train, "teach": teach, "taught-corpus": taught_corpus,
         "train-taught": train_taught}[options.job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
