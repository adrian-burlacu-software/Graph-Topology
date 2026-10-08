"""The exact checks of Python code: one persistent process, `tscheck.js`'s
protocol (a JSON request a line, a JSON reply a line).

    run       a function on cases: {"value"} or {"error"}, what it printed
    values    expressions over parameters, on cases (the search's loop)
    tests     a whole file -- a candidate and its asserts -- run through
    project   a project's file run, its own imports among its files
    check     mypy's errors in one source
    diagnose  mypy's errors in a project, strict too (and pyflakes': names
              never used)
    shape     how a program is written, off its syntax (v698 `ways.py`)
    outline   a project's functions, imports, docs, top-level calls

Values travel as `tscheck.js` sends them: sets `{"$set": [...]}`, maps with
keys that are not strings `{"$map": [[k, v], ...]}`, infinities as
`"Infinity"`; a float that is whole is the int it equals (`2.0 == 2` in
Python, and one `number` in the engine). Inputs are made what the
function's annotations say (a list is a tuple where a tuple is declared).

Code runs here with a watchdog: past its time it is interrupted; a process
that stops answering is restarted by its client (`pychecker.py`). As
`tscheck.js`'s `vm`, this keeps a search's candidates from hanging it -- it
is not a sandbox against code meant to do harm.
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import _thread
import time
import traceback
import typing

REPLY = sys.stdout
LARGEST_VALUE = 100000
#: what code here may use without importing it: the modules a short
#: function reaches for
GIVEN = ("math", "re", "itertools", "functools", "collections", "heapq",
         "bisect", "string", "operator", "statistics")
STATE = os.path.join(tempfile.gettempdir(), "graph-topology-pycheck")
sys.setrecursionlimit(4000)


class Stopped(Exception):
    pass


# -- values ------------------------------------------------------------------

def plain(value, depth: int = 0):
    """A value as JSON carries it, the way `tscheck.js` sends one."""
    if depth > 40:
        raise ValueError("a value nested too deep")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        # one number in the engine, as JavaScript has it: past 2^53 a
        # float, past a float's range infinite (Python's are unbounded)
        if abs(value) <= 2 ** 53:
            return value
        try:
            return float(value)
        except OverflowError:
            return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return int(value) if value.is_integer() and abs(value) < 2 ** 53 \
            else value
    if isinstance(value, str):
        if len(value) > LARGEST_VALUE:
            raise ValueError("a value too large")
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > LARGEST_VALUE:
            raise ValueError("a value too large")
        return [plain(one, depth + 1) for one in value]
    if isinstance(value, (set, frozenset)):
        items = [plain(one, depth + 1) for one in value]
        return {"$set": sorted(items, key=lambda one: json.dumps(
            one, sort_keys=True))}
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: plain(one, depth + 1) for key, one in value.items()}
        return {"$map": [[plain(key, depth + 1), plain(one, depth + 1)]
                         for key, one in value.items()]}
    if isinstance(value, (range, map, filter, zip, enumerate, reversed)) or \
            hasattr(value, "__next__"):
        raise ValueError(f"gives an iterator ({type(value).__name__}), "
                         f"not a value")
    raise ValueError(f"gives a {type(value).__name__}")


def given(value):
    """A JSON value as Python has it: sets and maps decoded."""
    if isinstance(value, list):
        return [given(one) for one in value]
    if isinstance(value, dict):
        if set(value) == {"$set"}:
            return set(given(one) for one in value["$set"])
        if set(value) == {"$map"}:
            return {_key(given(k)): given(v) for k, v in value["$map"]}
        return {key: given(one) for key, one in value.items()}
    return value


def _key(value):
    return tuple(_key(one) for one in value) if isinstance(value, list) \
        else value


def shaped(value, kind):
    """A decoded value made what an annotation says: tuples, sets, dicts,
    frozensets -- nested."""
    origin = typing.get_origin(kind)
    args = typing.get_args(kind)
    if kind is tuple or origin is tuple:
        if isinstance(value, (list, tuple)):
            if args and not (len(args) == 2 and args[1] is Ellipsis):
                return tuple(shaped(one, args[at] if at < len(args) else None)
                             for at, one in enumerate(value))
            inner = args[0] if args else None
            return tuple(shaped(one, inner) for one in value)
        return value
    if kind is list or origin is list:
        if isinstance(value, (list, tuple)):
            inner = args[0] if args else None
            return [shaped(one, inner) for one in value]
        return value
    if kind in (set, frozenset) or origin in (set, frozenset):
        if isinstance(value, (list, tuple, set)):
            inner = args[0] if args else None
            made = {shaped(one, inner) for one in value}
            return frozenset(made) if (kind is frozenset
                                       or origin is frozenset) else made
        return value
    if kind is dict or origin is dict:
        if isinstance(value, dict):
            kk, vk = (args + (None, None))[:2]
            return {_hashable(shaped(k, kk)): shaped(v, vk)
                    for k, v in value.items()}
        return value
    if origin in (typing.Union, getattr(__import__("types"), "UnionType",
                                        None)):
        for one in args:
            if one is type(None):
                continue
            made = shaped(value, one)
            if made is not value:
                return made
        return value
    return value


def _hashable(value):
    if isinstance(value, list):
        return tuple(_hashable(one) for one in value)
    return value


def _encoded(result) -> dict:
    try:
        return {"value": plain(result)}
    except ValueError as bad:
        return {"error": str(bad)}
    except RecursionError:
        return {"error": "a value nested too deep"}


# -- running, with a watchdog -------------------------------------------------

class _Watch:
    """Interrupts the main thread when time is up."""

    def __init__(self, ms: int) -> None:
        self.ms, self.fired = ms, False
        self.done = threading.Event()

    def __enter__(self):
        self.thread = threading.Thread(target=self._wait, daemon=True)
        self.thread.start()
        return self

    def _wait(self) -> None:
        if not self.done.wait(self.ms / 1000.0):
            self.fired = True
            _thread.interrupt_main()

    def __exit__(self, kind, value, trace):
        self.done.set()
        if kind is KeyboardInterrupt and self.fired:
            raise Stopped(f"timed out after {self.ms} ms")
        if self.fired and kind is None:
            # time ran out as the candidate returned: the interrupt is on
            # its way, and would land in whatever runs next -- taken here
            try:
                time.sleep(0.05)
            except KeyboardInterrupt:
                pass
        return False


# -- arithmetic that cannot be interrupted, refused past a size ------------------
#
# The watchdog interrupts the main thread between bytecodes; a power of
# Python's unbounded ints (`pow(37, int(str(n) * n))`, 37 ** 88888888) is one
# C call that runs for minutes, the whole batch waiting on it. TypeScript's
# number is Infinity there at once. What would be that large is refused as
# an error -- the code is the same as printed, only run guarded.

#: the most bits a power, a shift or a product may have
LARGEST_BITS = 100_000
#: the most items a repetition (`"ab" * n`, `[0] * n`) may have
LONGEST_REPEAT = 10_000_000


def _g_pow(base, exponent, modulus=None):
    if modulus is not None:
        return builtins.pow(base, exponent, modulus)
    if isinstance(base, int) and isinstance(exponent, int) \
            and exponent > 0 and abs(base) > 1 \
            and exponent * abs(base).bit_length() > LARGEST_BITS:
        raise OverflowError("too large a power")
    return builtins.pow(base, exponent)


def _g_shift(value, by):
    if isinstance(value, int) and isinstance(by, int) and by > LARGEST_BITS:
        raise OverflowError("too large a shift")
    return value << by


def _g_times(left, right):
    for many, count in ((left, right), (right, left)):
        if isinstance(count, int) and not isinstance(count, bool) \
                and isinstance(many, (str, bytes, list, tuple)) \
                and len(many) * count > LONGEST_REPEAT:
            raise MemoryError("too long a repetition")
    if isinstance(left, int) and isinstance(right, int) \
            and left.bit_length() + right.bit_length() > LARGEST_BITS * 10:
        raise OverflowError("too large a product")
    return left * right


class _Guard(ast.NodeTransformer):
    """`a ** b`, `a << b`, `a * b` (and `x **= b` ...) as the guarded calls."""

    CALLS = {ast.Pow: "_g_pow", ast.LShift: "_g_shift", ast.Mult: "_g_times"}

    def _call(self, op, left, right, at):
        return ast.copy_location(ast.Call(
            func=ast.Name(id=self.CALLS[type(op)], ctx=ast.Load()),
            args=[left, right], keywords=[]), at)

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if type(node.op) in self.CALLS:
            return self._call(node.op, node.left, node.right, node)
        return node

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        # a name only: `xs[f()] **= 2` would say its target twice
        if type(node.op) in self.CALLS and isinstance(node.target, ast.Name):
            read = ast.Name(id=node.target.id, ctx=ast.Load())
            return ast.copy_location(ast.Assign(
                targets=[node.target],
                value=self._call(node.op, read, node.value, node)), node)
        return node


def _compiled(source: str, mode: str):
    tree = ast.fix_missing_locations(_Guard().visit(ast.parse(source,
                                                              mode=mode)))
    return compile(tree, "<candidate>", mode)


def _guarded_math():
    import types
    out = types.ModuleType("math")
    out.__dict__.update(math.__dict__)

    def factorial(n):
        if isinstance(n, int) and n > 20_000:
            raise OverflowError("too large a factorial")
        return math.factorial(n)

    def comb(n, k):
        if isinstance(n, int) and n > 100_000:
            raise OverflowError("too large a binomial")
        return math.comb(n, k)

    def perm(n, k=None):
        if isinstance(n, int) and n > 20_000:
            raise OverflowError("too large a permutation count")
        return math.perm(n, k)

    out.factorial, out.comb, out.perm = factorial, comb, perm
    out.pow = math.pow                      # floats: overflows at once
    return out


GUARDED_MATH = _guarded_math()


def _g_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "math" and not level:
        return GUARDED_MATH
    return builtins.__import__(name, globals, locals, fromlist, level)


#: the builtins candidates run with: the guarded `pow` and `__import__`
GUARDED_BUILTINS = dict(vars(builtins), pow=_g_pow, __import__=_g_import,
                        _g_pow=_g_pow, _g_shift=_g_shift,
                        _g_times=_g_times)


def _namespace() -> dict:
    space = {"__name__": "__candidate__", "__builtins__": GUARDED_BUILTINS}
    for name in GIVEN:
        space[name] = GUARDED_MATH if name == "math" else __import__(name)
    # what `from typing import ...` would give: annotations name them
    for name in ("List", "Dict", "Tuple", "Set", "Optional", "Union", "Any",
                 "Callable", "Iterable", "Sequence"):
        space[name] = getattr(typing, name)
    return space


def _failed(bad: BaseException) -> str:
    name = type(bad).__name__
    said = str(bad)
    return f"{name}: {said}" if said else name


def _call(function, args, ms: int) -> dict:
    printed = io.StringIO()
    try:
        with _Watch(ms), contextlib.redirect_stdout(printed):
            result = function(*args)
        out = _encoded(result)
    except Stopped as bad:
        out = {"error": str(bad)}
    except RecursionError:
        out = {"error": "RecursionError: maximum recursion depth exceeded"}
    except KeyboardInterrupt:
        out = {"error": "interrupted"}
    except SystemExit:
        out = {"error": "SystemExit"}
    except BaseException as bad:                    # noqa: BLE001
        out = {"error": _failed(bad)}
    lines = printed.getvalue().splitlines()
    if lines:
        out["printed"] = lines[:200]
    return out


def _exec(source: str, space: dict, ms: int) -> str | None:
    """Run a module's text into `space`; None, or what stopped it."""
    printed = io.StringIO()
    try:
        code = _compiled(source, "exec")
        with _Watch(ms), contextlib.redirect_stdout(printed):
            exec(code, space)                       # noqa: S102
    except Stopped as bad:
        return str(bad)
    except SyntaxError as bad:
        return f"SyntaxError: {bad.msg} (line {bad.lineno})"
    except KeyboardInterrupt:
        return "interrupted"
    except SystemExit:
        return "SystemExit"
    except BaseException as bad:                    # noqa: BLE001
        line = _line_of(bad)
        return _failed(bad) + (f" (line {line})" if line else "")
    space["__printed__"] = printed.getvalue().splitlines()
    return None


def _line_of(bad: BaseException) -> int | None:
    for frame in reversed(traceback.extract_tb(bad.__traceback__)):
        if frame.filename == "<candidate>":
            return frame.lineno
    return None


def _hints(function) -> list:
    try:
        hints = typing.get_type_hints(function)
    except Exception:                               # noqa: BLE001
        hints = getattr(function, "__annotations__", {}) or {}
    try:
        import inspect
        names = list(inspect.signature(function).parameters)
    except (TypeError, ValueError):
        names = []
    return [hints.get(name) for name in names]


def run(source: str, entry: str, cases: list, timeout: int) -> dict:
    space = _namespace()
    stopped = _exec(source, space, max(timeout * 5, 1000))
    if stopped:
        return {"ok": False, "error": stopped}
    function = space.get(entry)
    if not callable(function):
        return {"ok": False, "error": f"no function {entry}"}
    kinds = _hints(function)
    outputs = []
    for case in cases:
        args = [given(one) for one in case]
        args = [shaped(one, kinds[at]) if at < len(kinds) and kinds[at]
                else one for at, one in enumerate(args)]
        outputs.append(_call(function, args, timeout))
    return {"ok": True, "outputs": outputs}


#: engine types a value is made to be, for `values` (no annotations there)
def _engine_shaped(value, kind: str | None):
    if not kind:
        return value
    kind = kind.strip()
    if kind.startswith("[") and kind.endswith("]") and \
            isinstance(value, list):
        from_parts = _parts(kind[1:-1])
        return tuple(_engine_shaped(one, from_parts[at] if at < len(
            from_parts) else None) for at, one in enumerate(value))
    if kind.endswith("[]") and isinstance(value, list):
        inner = kind[:-2].strip("()")
        return [_engine_shaped(one, inner) for one in value]
    if kind.startswith("Set<") and isinstance(value, (list, set)):
        return set(value)
    return value


def _parts(text: str) -> list:
    out, depth, start = [], 0, 0
    for at, char in enumerate(text):
        if char in "[<(":
            depth += 1
        elif char in "]>)":
            depth -= 1
        elif char == "," and depth == 0:
            out.append(text[start:at].strip())
            start = at + 1
    if text[start:].strip():
        out.append(text[start:].strip())
    return out


def values(params, cases, expressions, timeout: int, prelude: str,
           types=None) -> dict:
    space = _namespace()
    if prelude:
        stopped = _exec(prelude, space, 2000)
        if stopped:
            return {"ok": False, "error": f"the prelude: {stopped}"}
    decoded = []
    for case in cases:
        args = [given(one) for one in case]
        if types:
            args = [_engine_shaped(one, types[at] if at < len(types)
                                   else None) for at, one in enumerate(args)]
        decoded.append(args)
    said = ", ".join(params)
    out = []
    for text in expressions:
        try:
            function = eval(_compiled(f"lambda {said}: ({text})",  # noqa: S307
                                      "eval"), space)
        except SyntaxError as bad:
            out.append([{"error": f"SyntaxError: {bad.msg}"}
                        for _ in cases])
            continue
        except BaseException as bad:                # noqa: BLE001
            out.append([{"error": _failed(bad)} for _ in cases])
            continue
        out.append([_call(function, args, timeout) for args in decoded])
    return {"ok": True, "values": out}


def tests(source: str, timeout: int) -> dict:
    space = _namespace()
    space["__name__"] = "__main__"
    stopped = _exec(source, space, timeout)
    return {"ok": True} if stopped is None else {"ok": False,
                                                 "error": stopped}


def project(files: dict, main: str, timeout: int) -> dict:
    """Run `main` among `files` (paths as given: `/ws/pkg/mod.py`), its
    imports resolved in the project: None or what stopped it."""
    import runpy
    root = tempfile.mkdtemp(prefix="pyproject-")
    try:
        common = _common(list(files))
        for path, text in files.items():
            if text is None:
                continue
            where = os.path.join(root, os.path.relpath(path, common))
            os.makedirs(os.path.dirname(where), exist_ok=True)
            with open(where, "w", encoding="utf-8") as out:
                out.write(text)
        before = set(sys.modules)
        sys.path.insert(0, root)
        printed = io.StringIO()
        try:
            with _Watch(timeout), contextlib.redirect_stdout(printed):
                runpy.run_path(os.path.join(root, os.path.relpath(
                    main, common)), run_name="__main__")
            return {"ok": True}
        except Stopped as bad:
            return {"ok": False, "error": str(bad)}
        except SystemExit as done:
            return {"ok": not done.code, "error": f"exit {done.code}"}
        except BaseException as bad:                # noqa: BLE001
            return {"ok": False, "error": _failed(bad)}
        finally:
            sys.path.remove(root)
            for name in set(sys.modules) - before:
                sys.modules.pop(name, None)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _common(paths: list) -> str:
    folders = [os.path.dirname(one) for one in paths] or ["/"]
    try:
        return os.path.commonpath(folders)
    except ValueError:
        return "/"


# -- mypy --------------------------------------------------------------------

MYPY_LINE = re.compile(r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+):"
                       r"(?:(?P<eline>\d+):(?P<ecol>\d+):)?\s*"
                       r"(?P<level>error|note|warning):\s*(?P<message>.*?)"
                       r"(?:\s+\[(?P<code>[\w-]+)\])?$")


def _offset(text: str, line: int, col: int) -> int:
    lines = text.splitlines(keepends=True)
    return sum(len(one) for one in lines[:max(line - 1, 0)]) + \
        max(col - 1, 0)


def diagnose(files: dict, strict: bool = False) -> dict:
    """mypy's errors (strict: `--strict`, and pyflakes' names never used),
    with spans as offsets in the files as given."""
    from mypy import api
    holder = tempfile.mkdtemp(prefix="pymypy_")
    # under a folder whose name is a package's: a project with an
    # `__init__.py` at its top makes its folder one (`pymypy-...` is not a
    # name, and mypy stops there)
    root = os.path.join(holder, "project")
    os.makedirs(root)
    where = {}
    try:
        common = _common(list(files))
        for path, text in files.items():
            if text is None or not path.endswith((".py", ".pyi")):
                continue
            local = os.path.join(root, os.path.relpath(path, common))
            os.makedirs(os.path.dirname(local), exist_ok=True)
            with open(local, "w", encoding="utf-8") as out:
                out.write(text)
            where[os.path.normcase(os.path.abspath(local))] = path
        if not where:
            return {"ok": True, "errors": []}
        os.makedirs(STATE, exist_ok=True)
        flags = ["--show-column-numbers", "--show-error-end",
                 "--show-error-codes", "--no-error-summary",
                 "--no-color-output", "--hide-error-context",
                 "--ignore-missing-imports", "--follow-imports=silent",
                 "--cache-dir", os.path.join(STATE, "mypy-cache"),
                 "--python-version", "3.12", "--no-site-packages",
                 # each module named by where it is under the project's
                 # root (`research/v698/__main__.py` is
                 # `research.v698.__main__`): two `__main__.py` were one
                 # module, and mypy checked nothing at all
                 "--explicit-package-bases"]
        if strict:
            flags += ["--strict", "--warn-unreachable",
                      "--extra-checks"]
        # run from the root, the files by their paths under it: the bases
        # the modules are named from (this worker runs one thing at once)
        here = os.getcwd()
        os.chdir(root)
        try:
            stdout, stderr, _ = api.run(
                flags + [os.path.relpath(one, root) for one in where])
        finally:
            os.chdir(here)
        errors = []
        for line in stdout.splitlines():
            found = MYPY_LINE.match(line.strip())
            if not found and re.search(r":\s*error:", line):
                # an error of the run, not of a line (a module twice, a
                # file it cannot read): nothing was checked -- said, not
                # taken for a project with nothing wrong
                return {"ok": False, "error": "mypy checked nothing: "
                        + line.strip().split("error:", 1)[-1].strip()}
            if not found or found["level"] == "note":
                continue
            # mypy's cache says a module's errors at the path it had when
            # they were found -- another run's holder: the file is known
            # by where it is in the project, not by its holder
            inside = re.split(r"pymypy_[^\\/]+[\\/]project[\\/]",
                              found["file"])[-1]
            local = os.path.normcase(os.path.abspath(
                os.path.join(root, inside)
                if not os.path.isabs(inside) else inside))
            path = where.get(local)
            if path is None:
                continue
            text = files[path]
            start = _offset(text, int(found["line"]), int(found["col"]))
            end = _offset(text, int(found["eline"] or found["line"]),
                          int(found["ecol"] or found["col"]) + 1)
            errors.append({"file": path, "start": start, "end": end,
                           "line": int(found["line"]),
                           "message": found["message"],
                           "code": found["code"] or "error"})
        if strict:
            errors += _unused(files)
        return {"ok": True, "errors": errors}
    finally:
        shutil.rmtree(holder, ignore_errors=True)


def _unused(files: dict) -> list:
    """pyflakes' warnings: names imported or assigned and never used,
    names that are not defined."""
    try:
        from pyflakes import api as flakes
        from pyflakes import reporter
    except ImportError:
        return []
    out = []
    for path, text in files.items():
        if text is None or not path.endswith(".py"):
            continue

        class Kept(reporter.Reporter):
            def __init__(self):
                super().__init__(io.StringIO(), io.StringIO())

            def flake(self, message):
                line = message.lineno
                col = getattr(message, "col", 0) + 1
                start = _offset(text, line, col)
                out.append({"file": path, "start": start, "end": start + 1,
                            "line": line,
                            "message": message.message % message.message_args,
                            "code": type(message).__name__})

        flakes.check(text, path, Kept())
    return out


def check(source: str) -> dict:
    found = diagnose({"/check/candidate.py": source})
    return {"errors": [{"message": one["message"], "start": one["start"],
                        "code": one["code"]} for one in found["errors"]]}


# -- syntax ------------------------------------------------------------------

LOOPS = {ast.For: "for", ast.AsyncFor: "for", ast.While: "while"}


def _defined(tree) -> dict:
    """Every function by name: `def`s, methods (`Class.name`), lambdas bound
    to a name."""
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.setdefault(node.name, node)
        elif isinstance(node, ast.Assign) and isinstance(node.value,
                                                         ast.Lambda):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found.setdefault(target.id, node.value)
    return found


def shape(source: str, entry: str) -> dict:
    """How a program is written, off its syntax, every function in it:
    {uses, recursive, entry (function / lambda / method / None),
    statements, lines}."""
    try:
        tree = ast.parse(source)
    except SyntaxError as bad:
        return {"ok": False, "error": f"SyntaxError: {bad.msg}"}
    uses = set()
    recursive = False
    for node in ast.walk(tree):
        kind = type(node)
        if kind in LOOPS:
            uses.add(LOOPS[kind])
            if kind in (ast.For, ast.AsyncFor) and \
                    isinstance(node.iter, ast.Call) and \
                    isinstance(node.iter.func, ast.Name):
                uses.add(f"for in {node.iter.func.id}")
        words = {ast.If: "if", ast.IfExp: "?:", ast.Match: "match",
                 ast.ListComp: "list comprehension",
                 ast.SetComp: "set comprehension",
                 ast.DictComp: "dict comprehension",
                 ast.GeneratorExp: "generator expression",
                 ast.Lambda: "lambda", ast.JoinedStr: "f-string",
                 ast.With: "with", ast.Try: "try", ast.ClassDef: "class",
                 ast.Yield: "yield", ast.YieldFrom: "yield",
                 ast.Await: "await", ast.AsyncFunctionDef: "async",
                 ast.Assert: "assert", ast.Global: "global",
                 ast.Nonlocal: "nonlocal", ast.Starred: "*",
                 ast.Set: "set literal", ast.Dict: "dict literal",
                 ast.NamedExpr: ":=", ast.Slice: "slice"}
        if kind in words:
            uses.add(words[kind])
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                             ast.GeneratorExp)) and \
                any(one.ifs for one in node.generators):
            uses.add("comprehension with if")
        if isinstance(node, (ast.Assign, ast.For, ast.comprehension)):
            targets = node.targets if isinstance(node, ast.Assign) else \
                [node.target]
            if any(isinstance(one, (ast.Tuple, ast.List)) for one in targets):
                uses.add("unpacking")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.decorator_list:
                uses.add("decorator")
            if node.returns is not None or any(
                    one.annotation is not None for one in node.args.args):
                uses.add("annotations")
        if isinstance(node, ast.Call):
            callee = node.func
            if isinstance(callee, ast.Name):
                uses.add(callee.id + "()")
            elif isinstance(callee, ast.Attribute):
                uses.add("." + callee.attr)
                if isinstance(callee.value, ast.Name):
                    uses.add(f"{callee.value.id}.{callee.attr}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                uses.add("import " + (getattr(node, "module", None)
                                      or alias.name))
    functions = _defined(tree)
    for name, node in functions.items():
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and isinstance(inner.func,
                                                          ast.Name) and \
                    inner.func.id == name:
                recursive = True
    target = functions.get(entry)
    kind = statements = lines = None
    if target is not None:
        if isinstance(target, ast.Lambda):
            kind, statements = "lambda", 1
        else:
            kind = "method" if any(
                isinstance(parent, ast.ClassDef) and target in parent.body
                for parent in ast.walk(tree)) else "function"
            body = [one for one in target.body
                    if not (isinstance(one, ast.Expr) and isinstance(
                        one.value, ast.Constant) and isinstance(
                        one.value.value, str))]
            statements = 1 if (len(body) == 1 and isinstance(
                body[0], ast.Return)) else max(len(body), 2)
        lines = (getattr(target, "end_lineno", target.lineno)
                 - target.lineno + 1)
    return {"ok": True, "shape": {"uses": sorted(uses),
                                  "recursive": recursive, "entry": kind,
                                  "statements": statements or 0,
                                  "lines": lines or 0,
                                  "language": "python"}}


def _chain(statements: list):
    """[(test, value)...], otherwise of a body that is a chain: guarded
    returns (`if t: return a` ... `return b`), an if/elif/else of returns,
    or one `return a if t else b if u else c` -- else None."""
    def from_expression(node):
        arms = []
        while isinstance(node, ast.IfExp):
            arms.append((node.test, node.body))
            node = node.orelse
        return (arms, node) if arms else None

    if len(statements) == 1 and isinstance(statements[0], ast.Return) \
            and statements[0].value is not None:
        return from_expression(statements[0].value)
    arms, at = [], 0
    while at < len(statements):
        one = statements[at]
        if isinstance(one, ast.If) and len(one.body) == 1 and \
                isinstance(one.body[0], ast.Return) and one.body[0].value:
            arms.append((one.test, one.body[0].value))
            if one.orelse:
                if len(one.orelse) == 1 and isinstance(one.orelse[0], ast.If):
                    statements = statements[:at + 1] + one.orelse + \
                        statements[at + 1:]
                elif len(one.orelse) == 1 and isinstance(
                        one.orelse[0], ast.Return) and one.orelse[0].value:
                    return (arms, one.orelse[0].value) \
                        if at == len(statements) - 1 else None
                else:
                    return None
            at += 1
            continue
        break
    if arms and at == len(statements) - 1 and isinstance(
            statements[at], ast.Return) and statements[at].value:
        return arms, statements[at].value
    if arms and at == len(statements):
        return arms, None
    return None


def _compared(test):
    """(what is compared, [values]) of `x == 1`, `1 == x`, `x == 1 or x ==
    2`, `x in (1, 2)` -- else None."""
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
        parts = [_compared(one) for one in test.values]
        if all(parts) and len({ast.dump(one[0]) for one in parts}) == 1:
            return parts[0][0], [v for one in parts for v in one[1]]
        return None
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        left, right = test.left, test.comparators[0]
        if isinstance(test.ops[0], ast.Eq):
            if isinstance(right, ast.Constant) and not isinstance(
                    left, ast.Constant):
                return left, [right]
            if isinstance(left, ast.Constant) and not isinstance(
                    right, ast.Constant):
                return right, [left]
        if isinstance(test.ops[0], ast.In) and isinstance(
                right, (ast.Tuple, ast.List, ast.Set)) and all(
                isinstance(one, ast.Constant) for one in right.elts):
            return left, list(right.elts)
    return None


def _indented(text: str, by: str) -> str:
    return "\n".join(by + line if line else line
                     for line in text.splitlines())


def restyle(source: str, entry: str, way: str):
    """The same program written `way`, where that is syntax alone: a def
    as a lambda and back, a chain of tests on one value as a `match`, a
    `match` or a conditional as ifs, ifs as one conditional, a loop that
    appends as a comprehension and back. None where it cannot be had so."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    target = None
    for at, node in enumerate(tree.body):
        if isinstance(node, ast.FunctionDef) and node.name == entry:
            target = (at, node)
        if isinstance(node, ast.Assign) and isinstance(node.value,
                                                       ast.Lambda) and \
                isinstance(node.targets[0], ast.Name) and \
                node.targets[0].id == entry:
            target = (at, node)
    if target is None:
        return None
    at, node = target

    def body_of(function):
        return [one for one in function.body if not (
            isinstance(one, ast.Expr) and isinstance(one.value, ast.Constant)
            and isinstance(one.value.value, str))]

    def replaced(new_function) -> str:
        tree.body[at] = new_function
        return ast.unparse(ast.fix_missing_locations(tree)) + "\n"

    if way == "arrow":
        if not isinstance(node, ast.FunctionDef):
            return None
        body = body_of(node)
        if len(body) != 1 or not isinstance(body[0], ast.Return) or \
                body[0].value is None:
            return None
        args = ast.arguments(posonlyargs=[], args=[ast.arg(arg=one.arg)
                                                   for one in node.args.args],
                             kwonlyargs=[], kw_defaults=[], defaults=[])
        return replaced(ast.Assign(targets=[ast.Name(id=entry,
                                                     ctx=ast.Store())],
                                   value=ast.Lambda(args=args,
                                                    body=body[0].value),
                                   lineno=node.lineno))
    if way == "declaration":
        if not isinstance(node, ast.Assign):
            return None
        lam = node.value
        return replaced(ast.FunctionDef(
            name=entry, args=lam.args, body=[ast.Return(value=lam.body)],
            decorator_list=[], returns=None, type_params=[],
            lineno=node.lineno))
    if not isinstance(node, ast.FunctionDef):
        return None
    body = body_of(node)
    if way == "ifs":
        found = next((one for one in body if isinstance(one, ast.Match)),
                     None)
        if found is not None:
            subject = ast.unparse(found.subject)
            lines, otherwise = [], None
            for case in found.cases:
                pattern = case.pattern
                if isinstance(pattern, ast.MatchAs) and pattern.pattern is None \
                        and pattern.name is None:
                    otherwise = case.body
                    continue
                values = pattern.patterns if isinstance(
                    pattern, ast.MatchOr) else [pattern]
                if not all(isinstance(one, ast.MatchValue) for one in values):
                    return None
                test = " or ".join(f"{subject} == {ast.unparse(one.value)}"
                                   for one in values)
                inner = "\n".join(ast.unparse(one) for one in case.body)
                lines.append(f"{'if' if not lines else 'elif'} {test}:\n"
                             + _indented(inner, "    "))
            if otherwise is not None:
                inner = "\n".join(ast.unparse(one) for one in otherwise)
                lines.append("else:\n" + _indented(inner, "    ")
                             if lines else inner)
            new = ast.parse("\n".join(lines)).body
            index = node.body.index(found)
            node.body[index:index + 1] = new
            return replaced(node)
        chain = _chain(body)
        if chain is None:
            return None
        arms, otherwise = chain
        text = "\n".join(f"{'if' if not k else 'elif'} {ast.unparse(test)}:"
                         f"\n    return {ast.unparse(value)}"
                         for k, (test, value) in enumerate(arms))
        if otherwise is not None:
            text += f"\nreturn {ast.unparse(otherwise)}"
        node.body = [one for one in node.body if one not in body] + \
            ast.parse(text).body
        return replaced(node)
    if way == "ternary":
        chain = _chain(body)
        if chain is None or chain[1] is None:
            return None
        arms, otherwise = chain
        expr = otherwise
        for test, value in reversed(arms):
            expr = ast.IfExp(test=test, body=value, orelse=expr)
        node.body = [one for one in node.body if one not in body] + \
            [ast.Return(value=expr)]
        return replaced(node)
    if way == "switch":
        chain = _chain(body)
        if chain is None:
            return None
        arms, otherwise = chain
        compared = [_compared(test) for test, _ in arms]
        if not all(compared) or len({ast.dump(one[0])
                                     for one in compared}) != 1:
            return None
        subject = ast.unparse(compared[0][0])
        cases = []
        for (values, value) in [(one[1], arm[1])
                                for one, arm in zip(compared, arms)]:
            pattern = " | ".join(ast.unparse(one) for one in values)
            cases.append(f"    case {pattern}:\n        return "
                         f"{ast.unparse(value)}")
        if otherwise is not None:
            cases.append(f"    case _:\n        return "
                         f"{ast.unparse(otherwise)}")
        text = f"match {subject}:\n" + "\n".join(cases)
        node.body = [one for one in node.body if one not in body] + \
            ast.parse(text).body
        return replaced(node)
    if way == "comprehension":
        # out = []; for x in xs: [if c:] out.append(e); return out
        for k in range(len(body) - 2):
            start, loop, back = body[k], body[k + 1], body[k + 2]
            if not (isinstance(start, ast.Assign) and isinstance(
                    start.value, ast.List) and not start.value.elts
                    and isinstance(start.targets[0], ast.Name)
                    and isinstance(loop, ast.For) and not loop.orelse
                    and isinstance(back, ast.Return)
                    and isinstance(back.value, ast.Name)
                    and back.value.id == start.targets[0].id):
                continue
            name = start.targets[0].id
            inner, test = loop.body, None
            if len(inner) == 1 and isinstance(inner[0], ast.If) and \
                    not inner[0].orelse and len(inner[0].body) == 1:
                test, inner = inner[0].test, inner[0].body
            if len(inner) != 1 or not isinstance(inner[0], ast.Expr):
                return None
            call = inner[0].value
            if not (isinstance(call, ast.Call) and isinstance(
                    call.func, ast.Attribute) and call.func.attr == "append"
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == name and len(call.args) == 1):
                return None
            comp = ast.ListComp(elt=call.args[0], generators=[
                ast.comprehension(target=loop.target, iter=loop.iter,
                                  ifs=[test] if test is not None else [],
                                  is_async=0)])
            node.body = body[:k] + [ast.Return(value=comp)] + body[k + 3:]
            return replaced(node)
        return None
    if way == "no-comprehension":
        last = body[-1] if body else None
        if not (isinstance(last, ast.Return) and isinstance(
                last.value, ast.ListComp) and
                len(last.value.generators) == 1):
            return None
        comp = last.value
        loop = comp.generators[0]
        append = f"result.append({ast.unparse(comp.elt)})"
        step = append
        for test in loop.ifs:
            step = f"if {ast.unparse(test)}:\n" + _indented(step, "    ")
        text = (f"result = []\nfor {ast.unparse(loop.target)} in "
                f"{ast.unparse(loop.iter)}:\n" + _indented(step, "    ")
                + "\nreturn result")
        node.body = body[:-1] + ast.parse(text).body
        return replaced(node)
    return None


def _annotation(node) -> str | None:
    return ast.unparse(node) if node is not None else None


def _doc_comment(lines: list, start: int) -> str | None:
    """The `#` comments just above a line, as one text."""
    said = []
    at = start - 2
    while at >= 0 and lines[at].strip().startswith("#"):
        said.insert(0, lines[at].strip().lstrip("#").strip())
        at -= 1
    return " ".join(said) or None


def outline(files: dict) -> dict:
    out = {}
    for path, text in files.items():
        if text is None or not path.endswith(".py"):
            continue
        lines = text.splitlines()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            out[path] = {"functions": [], "imports": [], "about": None,
                         "runs": [], "lines": len(lines) + 1,
                         "unread": "a syntax error"}
            continue
        functions = []

        def add(node, owner):
            calls = sorted({one.func.id if isinstance(one.func, ast.Name)
                            else one.func.attr for one in ast.walk(node)
                            if isinstance(one, ast.Call) and isinstance(
                                one.func, (ast.Name, ast.Attribute))})
            args = [one for one in node.args.args
                    if not (owner and one.arg in ("self", "cls"))]
            doc = ast.get_docstring(node) if not isinstance(
                node, ast.Lambda) else None
            functions.append({
                "name": f"{owner}.{node.name}" if owner else node.name,
                "kind": "method" if owner else "function",
                "class": owner, "exported": not node.name.startswith("_"),
                "doc": (doc or _doc_comment(lines, node.lineno) or None),
                "start": node.lineno,
                "end": getattr(node, "end_lineno", node.lineno),
                "params": [[one.arg, _annotation(one.annotation)]
                           for one in args],
                "returns": _annotation(node.returns),
                "async": isinstance(node, ast.AsyncFunctionDef),
                "calls": calls})

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add(node, None)
            elif isinstance(node, ast.ClassDef):
                for inner in node.body:
                    if isinstance(inner, (ast.FunctionDef,
                                          ast.AsyncFunctionDef)):
                        add(inner, node.name)
            elif isinstance(node, ast.Assign) and isinstance(
                    node.value, ast.Lambda) and isinstance(
                    node.targets[0], ast.Name):
                lam = node.value
                functions.append({
                    "name": node.targets[0].id, "kind": "function",
                    "class": None,
                    "exported": not node.targets[0].id.startswith("_"),
                    "doc": _doc_comment(lines, node.lineno),
                    "start": node.lineno,
                    "end": getattr(node, "end_lineno", node.lineno),
                    "params": [[one.arg, None] for one in lam.args.args],
                    "returns": None, "async": False,
                    "calls": sorted({one.func.id for one in ast.walk(lam)
                                     if isinstance(one, ast.Call)
                                     and isinstance(one.func, ast.Name)})})
        imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({"from": alias.name,
                                    "names": [alias.asname or alias.name],
                                    "line": node.lineno})
            elif isinstance(node, ast.ImportFrom):
                imports.append({"from": "." * node.level + (node.module
                                                            or ""),
                                "names": [one.asname or one.name
                                          for one in node.names],
                                "line": node.lineno})
        about = ast.get_docstring(tree)
        if about is None and lines and lines[0].strip().startswith("#") \
                and not lines[0].startswith("#!"):
            about = _doc_comment(lines + [""], 1 + next(
                (at for at, line in enumerate(lines)
                 if not line.strip().startswith("#")), len(lines)))
        runs = []
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value,
                                                         ast.Call):
                callee = node.value.func
                name = callee.id if isinstance(callee, ast.Name) else \
                    getattr(callee, "attr", None)
                if name:
                    runs.append({"call": name, "line": node.lineno})
            if isinstance(node, ast.If) and "__main__" in ast.unparse(
                    node.test):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call) and isinstance(
                            inner.func, ast.Name):
                        runs.append({"call": inner.func.id,
                                     "line": inner.lineno})
        out[path] = {"functions": functions, "imports": imports,
                     "about": about, "runs": runs, "lines": len(lines) + 1}
    return out


# -- the loop ------------------------------------------------------------------

def answer(request: dict) -> dict:
    op = request.get("op")
    if op == "run":
        return run(request["source"], request["entry"], request["cases"],
                   request.get("timeout", 200))
    if op == "values":
        return values(request["params"], request["cases"],
                      request["expressions"], request.get("timeout", 50),
                      request.get("prelude", ""), request.get("types"))
    if op == "tests":
        return tests(request["source"], request.get("timeout", 2000))
    if op == "project":
        return project(request["files"], request["main"],
                       request.get("timeout", 2000))
    if op == "check":
        return check(request["source"])
    if op == "diagnose":
        return diagnose(request["files"], request.get("strict", False))
    if op == "shape":
        return shape(request["source"], request["entry"])
    if op == "restyle":
        return {"ok": True, "source": restyle(request["source"],
                                              request["entry"],
                                              request["way"])}
    if op == "outline":
        return {"ok": True, "outline": outline(request["files"])}
    if op == "ping":
        return {"ok": True}
    return {"ok": False, "error": f"no op {op}"}


#: the most memory the worker may hold: a candidate that builds a billion
#: of something (`list(range(a, 2 ** 30))`) fails at it, a MemoryError,
#: and does not swap the machine for minutes in one call
LARGEST_MEMORY = 3 * 1024 ** 3


def _capped() -> None:
    """The worker's memory capped (`LARGEST_MEMORY`): a job object on
    Windows, the address space elsewhere. Where it cannot be, it is not."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class Basic(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                            ("PerJobUserTimeLimit", ctypes.c_int64),
                            ("LimitFlags", wintypes.DWORD),
                            ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t),
                            ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.c_size_t),
                            ("PriorityClass", wintypes.DWORD),
                            ("SchedulingClass", wintypes.DWORD)]

            class Counters(ctypes.Structure):
                _fields_ = [(name, ctypes.c_uint64) for name in (
                    "ReadOperationCount", "WriteOperationCount",
                    "OtherOperationCount", "ReadTransferCount",
                    "WriteTransferCount", "OtherTransferCount")]

            class Extended(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", Basic),
                            ("IoInfo", Counters),
                            ("ProcessMemoryLimit", ctypes.c_size_t),
                            ("JobMemoryLimit", ctypes.c_size_t),
                            ("PeakProcessMemoryUsed", ctypes.c_size_t),
                            ("PeakJobMemoryUsed", ctypes.c_size_t)]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateJobObjectW.restype = wintypes.HANDLE
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            job = kernel.CreateJobObjectW(None, None)
            limits = Extended()
            limits.BasicLimitInformation.LimitFlags = 0x100  # process memory
            limits.ProcessMemoryLimit = LARGEST_MEMORY
            kernel.SetInformationJobObject(
                wintypes.HANDLE(job), 9, ctypes.byref(limits),
                ctypes.sizeof(limits))
            kernel.AssignProcessToJobObject(
                wintypes.HANDLE(job), wintypes.HANDLE(kernel.GetCurrentProcess()))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_AS,
                               (LARGEST_MEMORY, LARGEST_MEMORY))
    except Exception:                               # noqa: BLE001
        pass


def main() -> None:
    _capped()
    sys.stdout = io.StringIO()       # nothing a candidate prints reaches the
    for line in sys.stdin:           # protocol
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        try:
            reply = answer(request)
        except Stopped as bad:
            reply = {"ok": False, "error": str(bad)}
        except KeyboardInterrupt:
            reply = {"ok": False, "error": "interrupted"}
        except BaseException as bad:                # noqa: BLE001
            reply = {"ok": False, "error": _failed(bad)}
        reply["id"] = request.get("id")
        REPLY.write(json.dumps(reply) + "\n")
        REPLY.flush()
        sys.stdout = io.StringIO()


if __name__ == "__main__":
    main()
