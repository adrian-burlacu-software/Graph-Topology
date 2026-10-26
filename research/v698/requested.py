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


class _Probed(ast.NodeTransformer):
    """A test's asserts made what they look at: `assert q.skipped == 1` ->
    `_seen(lambda: q.skipped)` -- the value its code gives, not the one the
    test expects."""

    def visit_Assert(self, node):
        test = node.test
        looked = test.left if isinstance(test, ast.Compare) and len(
            test.ops) == 1 and isinstance(test.ops[0], (ast.Eq, ast.Is)) \
            else test
        call = ast.Call(func=ast.Name(id="_seen", ctx=ast.Load()),
                        args=[ast.Lambda(args=ast.arguments(
                            posonlyargs=[], args=[], kwonlyargs=[],
                            kw_defaults=[], defaults=[]), body=looked)],
                        keywords=[])
        return ast.copy_location(ast.Expr(value=call), node)


def _probe(code: str) -> str | None:
    """A written test as a probe of what code gives (`_Probed`), or None
    where it does not parse."""
    code = "\n".join(line for line in code.splitlines()
                     if not re.match(r"\s*(import|from)\s+pytest\b", line))
    try:
        tree = _Probed().visit(ast.parse(code))
    except SyntaxError:
        return None
    return ast.unparse(ast.fix_missing_locations(tree))


PROBED = """import sys, json
sys.path.insert(0, '.')
_all = []
for _code in {codes!r}:
    _out = []
    def _seen(look, _out=_out):
        try:
            _out.append(repr(look())[:200])
        except BaseException as bad:
            _out.append(type(bad).__name__)
    _space = {{"__name__": "__test__", "_seen": _seen}}
    try:
        exec("from " + {module!r} + " import *", _space)
    except BaseException:
        pass
    try:
        exec(compile(_code, "<probe>", "exec"), _space)
        for _name, _one in list(_space.items()):
            if _name.startswith("test") and callable(_one):
                try:
                    _one()
                except BaseException as bad:
                    _out.append(type(bad).__name__)
    except BaseException as bad:
        _out.append(type(bad).__name__)
    _all.append("; ".join(_out))
print(json.dumps(_all))
"""


def _probed(root: Path, probes: list, module: str = "") -> tuple:
    """What a project's copy gives for each probe: (what each looked at,
    said, ...) -- the tests' inputs and the objects they build, run, their
    expected outputs never asked: the code model chooses inputs well and
    works out outputs badly."""
    if not probes:
        return ()
    import json
    try:
        done = subprocess.run(
            [sys.executable, "-I", "-c", PROBED.format(codes=probes,
                                                       module=module)],
            cwd=root, capture_output=True, text=True,
            timeout=SECONDS * 2)
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
#: of a change keeping what the code gives: calls that use the code well
#: -- every case of it, not those the change makes different (none is)
KEPT_INPUTS = ("A programmer asked for this change to their Python project, "
               "one that keeps what the code gives: \"{statement}\"\nThe file "
               "{path}, as it is now:\n```python\n{text}\n```\nWrite 8 calls "
               "of its functions with literal inputs that use every case of "
               "the code -- usual inputs, edge cases, small and larger ones "
               "-- only the calls, one per line, no results, no asserts, in "
               "one ```python block.")


def inputs(statement: str, held, path: str, keeps: bool = False) -> list:
    """Calls of `path`'s functions the code model chooses as what the
    request concerns -- or, where the change keeps what the code gives,
    that use the code well: inputs only, [code]."""
    from research.v705 import coder
    now = _now(held, path)
    asked = (KEPT_INPUTS if keeps else INPUTS).format(
        statement=statement, path=path, text=held.files[path][:6000],
        now=("Run as it is now:\n" + "\n".join(now) + "\n") if now else "")
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
    # in two halves: twelve answers of 900 at once ran the card out -- and
    # of a file whole, one half: a file is longer to write than a function,
    # and eleven of CanItEdit's 91 changes ran past the turn's 300 s
    replies = []
    # as long as what is written again may be (a file of 3700 characters
    # cut off at 900 tokens: every answer of it unread)
    longest = 900 if name else min(1800, len(text) // 3 + 300)
    # as many at once as fit: six of 900 tokens did, six of 1800 ran the
    # card out (CUDA out of memory, the turn lost)
    samples = max(2, min(WRITTEN // 2, WRITTEN // 2 * 900 // longest))
    for _ in range(2 if name else 1):
        try:
            replies += held_.write([asked], longest=longest,
                                   samples=samples, temperature=0.7)[0]
        except held_.torch.cuda.OutOfMemoryError:
            held_.torch.cuda.empty_cache()
            break
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


#: what a request writes as code: in backquotes
QUOTED = re.compile(r"`([^`]+)`")
#: a signature written out: `next_n_fibs(n: int)`, `powermod(base : int,
#: exp : int, mod : int) -> int`
SIGNED = re.compile(r"^(?:[\w.]+\.)?([A-Za-z_]\w*)\s*\(([^()]*)\)")


def interface(statement: str, held) -> dict:
    """What a request says the code will have, written as code (in
    backquotes): {name: the number of parameters it writes, or None} --
    each name it adds that the project has nowhere, and each signature it
    writes (`parse(content: str)`). Read off the request, not guessed: the
    code model's tests guessed signatures (`Store.buy_yarn() takes 2
    positional arguments`) and outputs, and the change CanItEdit gives
    passed 14 of 123 of them. (A name only mentioned -- `the heuristic
    h(n)` -- is asked of no code.)"""
    import builtins
    import keyword
    index = held.index()
    text = "\n".join(held.files.values())
    out: dict = {}
    for quoted in QUOTED.findall(statement):
        quoted = quoted.strip()
        name = re.match(r"(?:[\w]+\.)*([A-Za-z_]\w*)", quoted)
        if name is None or not re.fullmatch(r"[\w.]+(\s*\(.*\).*)?",
                                            quoted, re.S):
            continue
        name = name.group(1)
        if keyword.iskeyword(name) or hasattr(builtins, name):
            continue
        signed = SIGNED.match(quoted)
        if signed is None:
            # a bare name: asked of the code where the project has it
            # nowhere (`a field skipped`)
            if not (index.find(name) or re.search(
                    rf"\b{re.escape(name)}\b", text)):
                out[name] = None
            continue
        params = signed.group(2)
        said = [one for one in (part.strip() for part in params.split(","))
                if one and one != "self"]
        # a call with values in it (`format(1, 2)`) is no signature: each
        # parameter of one is a name, typed or not
        if all(re.fullmatch(r"[A-Za-z_]\w*\s*(:.*)?(=.*)?", one)
               for one in said):
            out[name] = len(said)
    return out


#: a block of code in a request, fenced
FENCED = re.compile(r"```[\w+-]*\n(.*?)```", re.S)


def raised(statement: str, held=None) -> list:
    """The errors a request writes out the code will raise, as Python
    writes them -- `raise ValueError("input base must be >= 2")` in a block
    of its code: the messages, each a string the change will have (Exercism
    says them, and its tests check them). Code read, not words: a block
    that is not Python says none, and a message the project has already
    is asked of no change."""
    text = "\n".join(held.files.values()) if held is not None else ""
    out = []
    for block in FENCED.findall(statement):
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or not isinstance(
                    node.exc, ast.Call):
                continue
            for one in node.exc.args[:1]:
                if isinstance(one, ast.Constant) and isinstance(
                        one.value, str) and one.value.strip() and \
                        one.value not in out and one.value not in text:
                    out.append(one.value)
    return out


def _unraised(text: str, messages: list) -> list:
    """What of the messages asked code never says: each a string of it."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    said = {node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    return [f"it never says {one!r}" for one in messages if one not in said]


def stated(statement: str, names) -> list:
    """The examples a request says of what the code will give, as tests:
    `heronArea(3, 4, 5) should return 6.0` -> a test of heronArea -- read
    by the request reader (v697 `coding.read`), of a function `names` has.
    Said by the person, they hold: the code model's tests guess outputs
    (CanItEdit's own changes pass 14 of 123 of them)."""
    from research.v697 import coding
    try:
        read = coding.read(statement, "python")
    except Exception:                           # noqa: BLE001
        return []
    entry = read.get("entry")
    if not entry or entry not in names:
        return []
    out = []
    for example in read.get("examples") or ():
        value = example.get("value")
        call = f"{entry}({', '.join(map(repr, example.get('args') or ()))})"
        if isinstance(value, (int, float)) and not isinstance(value, bool) \
                and (isinstance(value, float) or re.search(
                    rf"\b{int(value)}\.\d", str(example.get("said", "")))):
            # a number said to as many places as it is said (`14859.47`) --
            # a number alone: a sentence's full stop made `'ih'` one, and
            # every change right refused
            places = re.search(r"\d\.(\d+)", str(example.get("said", "")))
            room = 0.5 * 10 ** -len(places.group(1)) if places else 1e-9
            out.append(f"_said = {call}\nassert abs(_said - {value!r}) <= "
                       f"{room!r} + 1e-9 * abs({value!r})")
        else:
            out.append(f"assert {call} == {value!r}")
    return out


def _defines(text: str, asked: dict) -> str:
    """'' where code defines every name asked, each function with the
    parameters asked; else what it lacks."""
    lacks = _lacking(text, asked)
    return lacks[0] if lacks else ""


def _lacking(text: str, asked: dict) -> list:
    """What of the names asked code lacks, each said: [] where it defines
    them all, each function with the parameters asked."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ["it does not parse"]
    defined: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            positional = [one.arg for one in node.args.posonlyargs +
                          node.args.args if one.arg not in ("self", "cls")]
            required = sum(default is None
                           for default in node.args.kw_defaults)
            whole = len(positional) + required
            needed = len(positional) - len(node.args.defaults) + required
            # as many as it needs, up to as many as it takes (any, where it
            # takes *args)
            defined.setdefault(node.name, set()).add(
                (needed, 10 ** 6 if node.args.vararg else whole))
        elif isinstance(node, ast.ClassDef):
            defined.setdefault(node.name, set())
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for target in (node.targets if isinstance(node, ast.Assign)
                           else [node.target]):
                for one in ast.walk(target):
                    if isinstance(one, ast.Attribute):
                        defined.setdefault(one.attr, set())
                    elif isinstance(one, ast.Name):
                        defined.setdefault(one.id, set())
    out = []
    for name, count in asked.items():
        if name not in defined:
            out.append(f"it has no {name}")
        elif count is not None and defined[name] and not any(
                low <= count <= high for low, high in defined[name]):
            out.append(f"its {name} does not take {count}")
    return out


def checker(statement: str, held, path: str, keeps: bool = False) -> dict:
    """The check of a change of `path`: {verify, tests, kept, own} --
    `verify(changed text)` says '' where the change holds, else why; None
    where no test tells the project as it is from a change. `keeps`: the
    change keeps what the code gives (the reader's aspect `same`) -- the
    code as it is the oracle, on calls that use it well."""
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
        # the examples the request says, first: true where the code model's
        # are guesses -- kept where the project as it is fails them, and
        # asked of every change whether it does or not
        asked = interface(statement, held)
        messages = raised(statement, held)
        names = {one.name.split(".")[-1] for one in
                 held.index().symbols.values() if one.file == path} | \
            set(asked)
        said = stated(statement, names)
        told = [code for code in said
                if not _runs(root, code, _module(path))[0]]
        kept = told + [code for code in kept if code not in told]
        own = [one for one in _own_tests(held) if _test_file(root, one)]
        # the calls a change is run on: those the tests make, and inputs
        # the code model chooses as what the request concerns -- inputs,
        # not outputs: a small model chooses them well and works out what
        # a change gives badly (every test it wrote of `strip dots too`
        # held already, or expected the dashes kept)
        try:
            chosen = inputs(statement, held, path, keeps)
        except Exception:                       # noqa: BLE001
            chosen = []
        calls = _calls(written + chosen, held, path)
        # (every written test as a probe of what code gives -- `_probe` --
        # grouped changes finer and chose none better, 2026-10-22: kept
        # for what may use it, not used to choose)
        probes: list = []
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
            # (what the request writes out the code will have: `lacks`,
            # weighed against the other changes)
            own_ok = all(_test_file(place, one) for one in own) and \
                all(_runs(place, code, _module(path))[0] for code in said)
            # (tuples: grouped by, and said as they are)
            passed = tuple(at for at, code in enumerate(kept)
                           if _runs(place, code, _module(path))[0]) if own_ok else ()
            gives = _gives(place, calls, _module(path)) if own_ok else ()
        finally:
            shutil.rmtree(place, ignore_errors=True)
        return own_ok, passed, gives

    def lacks(changed: str) -> list:
        """What of what the request writes out the code will have
        (`interface`) a change lacks: no change asked where another change
        has it -- a name no change has was a word of the request ("raise an
        `exception`"), not code asked."""
        return _lacking(changed, asked) + _unraised(changed, messages)

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
    if not kept and not calls and not probes:
        return {"verify": None, "score": None, "verify files": verify_files,
                "tests": written, "kept": [], "left": left, "own": own,
                "need": need, "calls": [], "original": (),
                "why": "no test tells the project from the change"}
    return {"verify": verify if kept else None,
            "verify files": verify_files, "score": score, "lacks": lacks,
            "tests": written, "kept": kept, "left": left, "own": own,
            "need": need, "calls": calls, "original": original,
            "interface": asked, "stated": len(told), "said": said,
            "keeps": keeps}


def code_shaped(name: str) -> bool:
    """Whether a name is written as code writes names and words are not:
    `clean_isbn`, `cleanIsbn`, `v2` -- not `to`, `everywhere`."""
    return "_" in name.strip("_") or any(one.isdigit() for one in name) or         bool(re.search(r"[a-z][A-Z]", name))


#: how many words after a name of the project a new name is said, to be
#: that name's new name
RENAMED_WITHIN = 3


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
        # the new name said with the old, a word or two after it (`parse_
        # isbn to clean_isbn`) -- not anywhere after it (`Replace the kl_div
        # function body ... num_samples` was renamed `num_samples`)
        for new in tokens[at + 1:at + 1 + RENAMED_WITHIN]:
            if new == old or index.find(new) or not new.isidentifier():
                continue
            changed = renamed(held.files, index, old, new)
            if changed:
                out.append((old, new, changed))
    return out
