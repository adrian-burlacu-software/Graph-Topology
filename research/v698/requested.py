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

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: tests the code model writes of a request
TESTS = 6
#: how long one test may run
SECONDS = 20

ASK = ("A programmer asked for this change to their Python project: "
       "\"{statement}\"\nThe project's files: {files}.\nThe file "
       "{path}, as it is now (module {module}):\n```python\n{text}\n```\n"
       "Write one short test of the change, as Python statements: import "
       "what you need from the project (from {module} import ...), then "
       "assert what must hold once the change is made -- and not before. "
       "Use only what the request and the code say. One ```python block, "
       "nothing else.")


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


def _runs(root: Path, code: str) -> tuple:
    """(holds, why) of Python run in a project's copy."""
    try:
        done = subprocess.run(
            [sys.executable, "-I", "-c",
             "import sys; sys.path.insert(0, '.')\n" + code], cwd=root,
            capture_output=True, text=True, timeout=SECONDS)
    except subprocess.TimeoutExpired:
        return False, f"ran past {SECONDS} s"
    last = (done.stderr.strip().splitlines() or [""])[-1]
    return done.returncode == 0, last[:200]


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


def tests(statement: str, held, path: str) -> list:
    """The code model's tests of the request, each once: [code]."""
    from research.v705 import coder
    files = ", ".join(sorted(one for one in held.files
                             if one.endswith(".py"))[:40])
    asked = ASK.format(statement=statement, files=files, path=path,
                       module=_module(path), text=held.files[path][:6000])
    replies = coder.get().write([asked], longest=300, samples=TESTS,
                                temperature=0.7)[0]
    out = []
    for reply in replies:
        code = _code_of(reply)
        if "assert" in code and code not in out:
            out.append(code)
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
            holds, why = _runs(root, code)
            # a test that cannot be read is no test; one the project
            # already passes says nothing of the change
            if "SyntaxError" in why or "IndentationError" in why:
                left.append({"test": code, "why": "not Python"})
            elif holds:
                left.append({"test": code, "why": "holds already"})
            else:
                kept.append(code)
        own = [one for one in _own_tests(held) if _test_file(root, one)]
    finally:
        shutil.rmtree(root, ignore_errors=True)
    if not kept:
        return {"verify": None, "tests": written, "left": left,
                "why": "no test tells the project from the change"}
    need = max(1, (len(kept) + 1) // 2)

    def verify(changed: str) -> str:
        return verify_files({path: changed})

    def verify_files(changes: dict) -> str:
        """'' where the project, these files so changed, holds."""
        place = Path(tempfile.mkdtemp(prefix="requested-"))
        try:
            _copy(held, place, changes)
            broken = [one for one in own if not _test_file(place, one)]
            if broken:
                return f"it breaks the project's tests: {', '.join(broken)}"
            passed = sum(1 for code in kept if _runs(place, code)[0])
        finally:
            shutil.rmtree(place, ignore_errors=True)
        if passed >= need:
            return ""
        return (f"{passed} of {len(kept)} tests of what was asked hold "
                f"(at least {need} must)")
    return {"verify": verify, "verify files": verify_files,
            "tests": written, "kept": kept, "left": left, "own": own,
            "need": need}


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
