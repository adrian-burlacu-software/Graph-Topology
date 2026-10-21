"""A change of a project's Python checked by what was asked (2026-10-20,
the core made reliable).

A change was written where the editors' answers agreed and the compiler
found nothing more wrong -- not where it did what was asked: `rename
parse_isbn to clean_isbn everywhere` changed two lines of one file, 7 of 7
agreeing; `handle should answer /api/count` was "changed" and answered
nothing new; and a right change one answer wrote was refused, no two
agreeing (`parse_isbn should strip dots too`: 3 of 19 passed the checks).

Here what was asked is made a check that runs, as CodeT pairs code with
tests both written by a model: the code model writes short tests of the
request against the project as it is (`TESTS` of them), and a test is
kept only where the project as it is fails it -- a test it already passes
says nothing of the change. A change is verified where, the project
copied with it:

- at least half the kept tests pass (a test can be wrong: none is asked
  to hold alone), and
- every test file of the project's own that passed before still passes.

Where no test written tells the project as it is from a change, there is
no check, and the change is written as before (`fixing`'s agreement).
Every run is apart (`python -I`, a time limit), in a copy.
"""
from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

#: tests the code model writes of a request
TESTS = 10
#: how long one test may run
SECONDS = 20

ASK = ("A programmer asked for this change to their Python project: "
       "\"{statement}\"\nThe project's files: {files}.\nThe file "
       "{path}, as it is now (module {module}):\n```python\n{text}\n```\n"
       "{now}"
       "Write one short test of the change: import what you need from the "
       "project (from {module} import ...), then one to three lines `assert "
       "call(literal inputs) == literal output` that hold once the change "
       "is made and not before -- work each output out from the code as it "
       "will be, step by step, before you write it. "
       "Plain assert statements at the top level: no pytest, no unittest, "
       "no mocks, no network, no files. Use only what the request and the "
       "code say. One ```python block, nothing else.")

#: what runs a test: its statements, then each `test...` function of it
#: that takes nothing (the code model writes pytest's form as often as
#: not: functions never called, run as a script)
RUN = """import sys
sys.path.insert(0, '.')
_test_space = {{"__name__": "__test__"}}
# the changed module's names in scope: a test that forgot to import Book
# is still a test of it
if {module!r}:
    try:
        exec("from " + {module!r} + " import *", _test_space)
    except BaseException:
        pass
exec(compile({code!r}, "<test>", "exec"), _test_space)
for _name, _one in list(_test_space.items()):
    if _name.startswith("test") and callable(_one):
        try:
            _one()
        except TypeError as _bad:
            if "required positional argument" not in str(_bad):
                raise
"""


def _module(path: str) -> str:
    return path[:-3].replace("/", ".").removesuffix(".__init__")


def _code_of(reply: str) -> str:
    found = re.search(r"```(?:python)?\n(.*?)```", reply, re.S)
    return (found.group(1) if found else reply).strip()


def _copy(held, root: Path, changed: dict | None = None) -> None:
    for path, text in held.files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((changed or {}).get(path, text), encoding="utf-8")


def _runs(root: Path, code: str, module: str = "") -> tuple:
    """(holds, why) of a test run in a project's copy -- `import pytest`
    left out (what it is for is not used here)."""
    code = "\n".join(line for line in code.splitlines()
                     if not re.match(r"\s*(import|from)\s+pytest\b", line))
    try:
        done = subprocess.run(
            [sys.executable, "-I", "-c", RUN.format(code=code,
                                                    module=module)],
            cwd=root,
            capture_output=True, text=True, timeout=SECONDS)
    except subprocess.TimeoutExpired:
        return False, f"ran past {SECONDS} s"
    last = (done.stderr.strip().splitlines() or [""])[-1]
    return done.returncode == 0, last[:200]


def _of_the_code(why: str, held) -> bool:
    """Whether a test's failure is of the project's code: what it asserts
    is not so, a call it makes does not fit, a name of the project it asks
    for is not there yet (`cannot import name 'newest'`) -- not a package
    missing, nor a name the test never defined."""
    if why.startswith(("AssertionError", "TypeError", "ValueError",
                       "AttributeError", "KeyError", "IndexError",
                       "ZeroDivisionError")):
        return True
    if why.startswith("ImportError"):
        return "cannot import name" in why
    if why.startswith("ModuleNotFoundError"):
        found = re.search(r"No module named '([\w.]+)'", why)
        top = found.group(1).split(".")[0] if found else ""
        return any(path.split("/")[0] == top for path in held.files)
    return False


def _calls(tests: list, held, path: str) -> list:
    """Every call of a function of `path` with literal arguments the
    written tests make (`parse_isbn("978.0-14")`), each once: what changes
    are run on to tell apart those passing the same tests."""
    from research.v698.project import Project
    names = {one.name.split(".")[-1] for one in held.index().symbols.values()
             if one.file == path and one.kind in ("function", "method")}
    module = _module(path)
    out = []
    for code in tests:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else \
                func.attr if isinstance(func, ast.Attribute) else None
            if name not in names:
                continue
            # as written, of the function by its own name: run with the
            # module's names in scope (`find_by_author([Book(1, 'a',
            # 'URSULA', 1, 1)], 'ursula')` -- arguments of the project's
            # own kinds, not literals alone)
            called = ast.Call(func=ast.Name(id=name, ctx=ast.Load()),
                              args=node.args, keywords=node.keywords)
            said = ast.unparse(called)
            if len(said) <= 300 and said not in out:
                out.append(said)
    return out[:16]


GIVES = """import sys, json
sys.path.insert(0, '.')
space = {{}}
exec("from " + {module!r} + " import *", space)
out = []
for call in {calls!r}:
    try:
        out.append(repr(eval(call, space))[:200])
    except BaseException as bad:
        out.append(type(bad).__name__)
print(json.dumps(out))
"""


def _gives(root: Path, calls: list, module: str = "") -> tuple:
    """What each call gives in a project's copy: (repr or the error's
    kind, ...)."""
    if not calls:
        return ()
    import json
    try:
        done = subprocess.run(
            [sys.executable, "-I", "-c", GIVES.format(calls=calls,
                                                      module=module)],
            cwd=root, capture_output=True, text=True, timeout=SECONDS)
        return tuple(json.loads(done.stdout.strip().splitlines()[-1]))
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        return ("ran past",)


def _own_tests(held) -> list:
    """The project's own test files."""
    from research.v698.symbols import Index
    return sorted(path for path in held.files if path.endswith(".py") and
                  Index.is_test(path) and "def test" in held.files[path])


def _test_file(root: Path, path: str) -> bool:
    try:
        done = subprocess.run(
            # (not -I: it leaves the project's folder off the path)
            [sys.executable, "-E", "-s", "-m", "unittest", path], cwd=root,
            capture_output=True, text=True, timeout=SECONDS * 3)
    except subprocess.TimeoutExpired:
        return False
    return done.returncode == 0


def _now(held, path: str) -> list:
    """What the functions of `path` give now for the calls the project's
    own tests make of them (`parse_isbn('978-0 14') == '978014'`): the
    code model writes what a change gives from what the code gives, not
    from nothing (its tests of `strip dots too` expected the dashes kept)."""
    own = [held.files[one] for one in _own_tests(held)]
    calls = _calls(own, held, path)
    if not calls:
        return []
    root = Path(tempfile.mkdtemp(prefix="requested-"))
    try:
        _copy(held, root)
        gives = _gives(root, calls, _module(path))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    out = []
    for call, given in zip(calls, gives):
        shown = call
        # a value given, not an error's name
        if given and not given[0].isupper() or \
                given.startswith(("True", "False", "None")):
            out.append(f"{shown} == {given}")
    return out[:8]


#: what the code model is asked of inputs a change concerns
INPUTS = ("For example, asked \"count_words should ignore commas\" of\n"
          "```python\ndef count_words(text):\n    return len(text.split())\n"
          "```\nthe calls whose results the change makes different are\n"
          "```python\ncount_words('a , b')\ncount_words('one, two ,')\n"
          "count_words(', ,')\n```\n-- each has what the request is about "
          "(commas) in its input.\n\n"
          "A programmer asked for this change to their Python project: "
          "\"{statement}\"\nThe file {path}, as it is now:\n```python\n"
          "{text}\n```\n{now}Write 6 calls of its functions, with literal "
          "inputs that have what the request is about in them, whose "
          "results the change asked makes different -- only the calls, one "
          "per line, no results, no asserts, in one ```python block.")
#: inputs asked for, in samples
INPUT_SAMPLES = 3


def inputs(statement: str, held, path: str) -> list:
    """Calls of `path`'s functions the code model chooses as what the
    request concerns, inputs only: [code]."""
    from research.v705 import coder
    now = _now(held, path)
    asked = INPUTS.format(statement=statement, path=path,
                          text=held.files[path][:6000],
                          now=("Run as it is now:\n" + "\n".join(now) + "\n")
                          if now else "")
    held_ = coder.big()
    held_.torch.manual_seed(zlib.crc32(asked.encode()))
    replies = held_.write([asked], longest=200, samples=INPUT_SAMPLES,
                          temperature=0.7)[0]
    return [_code_of(reply) for reply in replies]


def tests(statement: str, held, path: str) -> list:
    """The code model's tests of the request, each once: [code]."""
    from research.v705 import coder
    files = ", ".join(sorted(one for one in held.files
                             if one.endswith(".py"))[:40])
    now = _now(held, path)
    asked = ASK.format(statement=statement, files=files, path=path,
                       module=_module(path), text=held.files[path][:6000],
                       now=("Run as it is now, it gives:\n" + "\n".join(now)
                            + "\nAssert calls whose output the change "
                            "makes different.\n") if now else "")
    held_ = coder.big()
    # sampled the same for the same request: the same tests each time
    held_.torch.manual_seed(zlib.crc32(asked.encode()))
    replies = held_.write([asked], longest=300, samples=TESTS,
                                temperature=0.7)[0]
    out = []
    for reply in replies:
        code = _code_of(reply)
        if "assert" in code and code not in out:
            out.append(code)
    return out


#: what the code model is asked to write of the change itself
REWRITE = ("A programmer asked for this change to their Python project: "
           "\"{statement}\"\nIn {path}, {what} as it is now:\n```python\n"
           "{before}\n```\nWrite {whole} changed as asked -- nothing else "
           "changed -- in one ```python block, nothing else.")
#: answers the code model writes of a change
WRITTEN = 12


def rewritten(statement: str, held, path: str, name: str | None) -> list:
    """The code model's changes: the function `name` written whole again
    as asked, put where it stands -- or, of a file, the file whole:
    [changed text of `path`]. The editors write a line or two of it (19
    answers of `parse_isbn should strip dots too`, none adding a
    `.replace(".", "")`)."""
    from research.v705 import coder, commits
    text = held.files[path]
    if name:
        found = [one for one in held.index().find(name)
                 if one.file == path and one.kind in ("function", "method")]
        if not found:
            return []
        lines = text.splitlines(keepends=True)
        before = "".join(lines[found[0].start - 1:found[0].end])
        before = before.strip("\n").lstrip()
        asked = REWRITE.format(statement=statement, path=path,
                               what=f"the function {name}", before=before,
                               whole="the whole function")
    else:
        if len(text) > 6000:
            return []
        asked = REWRITE.format(statement=statement, path=path,
                               what="the file", before=text.rstrip("\n"),
                               whole="the whole file")
    held_ = coder.big()
    held_.torch.manual_seed(zlib.crc32(asked.encode()))
    # in two halves: twelve answers of 900 at once ran the card out
    replies = []
    for _ in range(2):
        replies += held_.write([asked], longest=900, samples=WRITTEN // 2,
                               temperature=0.7)[0]
    out = []
    for reply in replies:
        code = commits._code_of(reply)
        if not code:
            continue
        if name:
            changed = commits._placed(code, before, text)
        else:
            changed = code.rstrip("\n") + "\n"
            try:
                compile(changed, path, "exec")
            except SyntaxError:
                changed = None
        # each answer kept, the same as another too: answers written apart
        # writing the same change is what chooses it (`fixing`)
        if changed and changed != text:
            out.append(changed)
    return out


def checker(statement: str, held, path: str) -> dict:
    """The check of a change of `path`: {verify, tests, kept, own} --
    `verify(changed text)` says '' where the change holds, else why; None
    where no test tells the project as it is from a change."""
    if not path.endswith(".py") or path not in held.files:
        return {"verify": None, "why": "not Python"}
    try:
        written = tests(statement, held, path)
    except Exception as bad:                    # noqa: BLE001
        return {"verify": None, "why": f"the code model: {bad}"}
    root = Path(tempfile.mkdtemp(prefix="requested-"))
    try:
        _copy(held, root)
        kept, left = [], []
        for code in written:
            holds, why = _runs(root, code, _module(path))
            # a test that cannot be read is no test; one the project
            # already passes says nothing of the change
            if "SyntaxError" in why or "IndentationError" in why:
                left.append({"test": code, "why": "not Python"})
            elif holds:
                left.append({"test": code, "why": "holds already"})
            elif not _of_the_code(why, held):
                # it fails for itself, not for the project: a package not
                # here (`requests`), a name it never imported
                left.append({"test": code, "why": why})
            else:
                kept.append(code)
        own = [one for one in _own_tests(held) if _test_file(root, one)]
        # the calls a change is run on: those the tests make, and inputs
        # the code model chooses as what the request concerns -- inputs,
        # not outputs: a small model chooses them well and works out what
        # a change gives badly (every test it wrote of `strip dots too`
        # held already, or expected the dashes kept)
        try:
            chosen = inputs(statement, held, path)
        except Exception:                       # noqa: BLE001
            chosen = []
        calls = _calls(written + chosen, held, path)
        original = _gives(root, calls, _module(path))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    need = max(1, (len(kept) + 1) // 2)

    def verify(changed: str) -> str:
        return verify_files({path: changed})

    def score(changed: str) -> tuple:
        """(the project's own tests still pass, the kept tests it passes
        -- by their place --, what it gives for each call the tests make)
        -- for choosing among changes as CodeT does
        (`fixing._agreed_by_tests`): a test is as often wrong as right, and
        changes passing the same weak tests may still differ."""
        place = Path(tempfile.mkdtemp(prefix="requested-"))
        try:
            _copy(held, place, {path: changed})
            own_ok = all(_test_file(place, one) for one in own)
            # (tuples: grouped by, and said as they are)
            passed = tuple(at for at, code in enumerate(kept)
                           if _runs(place, code, _module(path))[0]) if own_ok else ()
            gives = _gives(place, calls, _module(path)) if own_ok else ()
        finally:
            shutil.rmtree(place, ignore_errors=True)
        return own_ok, passed, gives

    def verify_files(changes: dict, least: int | None = None) -> str:
        """'' where the project, these files so changed, holds: its own
        tests, and `least` of the kept (half of them where not said)."""
        least = need if least is None else least
        place = Path(tempfile.mkdtemp(prefix="requested-"))
        try:
            _copy(held, place, changes)
            broken = [one for one in own if not _test_file(place, one)]
            if broken:
                return f"it breaks the project's tests: {', '.join(broken)}"
            passed = sum(1 for code in kept if _runs(place, code, _module(path))[0])
        finally:
            shutil.rmtree(place, ignore_errors=True)
        if passed >= least:
            return ""
        return (f"{passed} of {len(kept)} tests of what was asked hold "
                f"(at least {least} must)")
    # (a change made, not guessed -- a rename -- is checked by the project's
    # own tests and, where any test of the request is kept, one of them)
    if not kept and not calls:
        return {"verify": None, "score": None, "verify files": verify_files,
                "tests": written, "kept": [], "left": left, "own": own,
                "need": need, "calls": [], "original": (),
                "why": "no test tells the project from the change"}
    return {"verify": verify if kept else None,
            "verify files": verify_files, "score": score,
            "tests": written, "kept": kept, "left": left, "own": own,
            "need": need, "calls": calls, "original": original}


def code_shaped(name: str) -> bool:
    """Whether a name is written as code writes names and words are not:
    `clean_isbn`, `cleanIsbn`, `v2` -- not `to`, `everywhere`."""
    return "_" in name.strip("_") or any(one.isdigit() for one in name) or         bool(re.search(r"[a-z][A-Z]", name))


def renames(statement: str, held) -> list:
    """The renames a request may ask, by what the project has: a name of
    its (`parse_isbn`) said, and after it a name it has not
    (`clean_isbn`) -- each made in every file that uses it
    (`symbols.renamed`): [(old, new, {path: text})]. Which, if any, is
    asked is what the request's tests say."""
    from research.v698.symbols import renamed
    index = held.index()
    tokens = re.findall(r"[A-Za-z_]\w*", statement)
    out = []
    for at, old in enumerate(tokens):
        if not [one for one in index.find(old) if "." not in one.name]:
            continue
        for new in tokens[at + 1:]:
            if new == old or index.find(new) or not new.isidentifier():
                continue
            changed = renamed(held.files, index, old, new)
            if changed:
                out.append((old, new, changed))
    return out
