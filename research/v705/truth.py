"""Whether a held change is the fix (v705) -- for measuring only, never
for choosing: the true function is known of a planted bug (the code as it
is) and of a history bug (the code the commit made, as it is now), and a
change the tests pass may still not be it (`language` always `'python'`,
a parameter taken and never read).

    real(case, true, text)   {calls, agree, tests, passed, real}

**Calls**: every call of the true function the tests that run it make,
its arguments copied as it was called and what it gave back -- then the
change given the same copies: what it gives back, and what it leaves the
arguments as, alike. **Tests**: every quick test that runs it, not the
dozen the change was judged by. Real where both agree throughout.
"""
from __future__ import annotations

import copy
import inspect
import sys

from research.v705 import commits

#: the calls recorded of each test, and in all, at most
EACH, MOST = 60, 1500
#: how deep two values are compared
DEPTH = 6


def _alike(one, other, depth: int = DEPTH) -> bool:
    """Two values the same in what they hold: equal where they say, else
    field by field (an object without `==` of its own is not unlike a copy
    of itself)."""
    if one is other:
        return True
    if depth <= 0:
        return True
    if type(one) is not type(other):
        return False
    if inspect.isroutine(one) or inspect.ismodule(one):
        # what it is given to call (`lemma`, a bound method copied twice
        # over is two), not what it holds
        return True
    if isinstance(one, dict):
        return one.keys() == other.keys() and all(
            _alike(one[key], other[key], depth - 1) for key in one)
    if isinstance(one, (list, tuple)):
        return len(one) == len(other) and all(
            _alike(a, b, depth - 1) for a, b in zip(one, other))
    if isinstance(one, (str, bytes, int, float, bool, type(None), set,
                        frozenset)):
        try:
            return bool(one == other)
        except Exception:                           # noqa: BLE001
            return False
    if type(one).__eq__ is not object.__eq__:
        try:
            return bool(one == other)
        except Exception:                           # noqa: BLE001
            pass
    if hasattr(one, "__dict__") and hasattr(other, "__dict__"):
        return _alike(vars(one), vars(other), depth - 1)
    return True


def _recorded(function, tests: list) -> list:
    """[(arguments copied as called, what it gave back)] of the function
    while the tests run."""
    code = function.__code__
    out: list = []
    stack: list = []

    def profiler(frame, event, arg):
        if frame.f_code is not code:
            return
        if event == "call":
            try:
                stack.append(copy.deepcopy(dict(frame.f_locals)))
            except Exception:                       # noqa: BLE001
                stack.append(None)
        elif event == "return" and stack:
            held = stack.pop()
            if held is not None and len(out) < MOST and \
                    counted[0] < EACH:
                try:
                    out.append((held, copy.deepcopy(arg)))
                    counted[0] += 1
                except Exception:                   # noqa: BLE001
                    pass
    counted = [0]
    for test in tests:
        counted[0] = 0
        stack.clear()
        previous = sys.getprofile()
        sys.setprofile(profiler)
        timer = commits._watched(commits.LONGEST)
        try:
            commits.passes(test)
        except Exception:                           # noqa: BLE001
            pass
        finally:
            timer.cancel()
            sys.setprofile(previous)
    return out


def _called(function, held: dict):
    """The function called with these arguments, by its parameters."""
    signature = inspect.signature(function)
    args, kwargs = [], {}
    for name, one in signature.parameters.items():
        if name not in held:
            continue
        if one.kind is one.VAR_POSITIONAL:
            args += list(held[name])
        elif one.kind is one.VAR_KEYWORD:
            kwargs.update(held[name])
        elif one.kind is one.POSITIONAL_ONLY or one.kind is \
                one.POSITIONAL_OR_KEYWORD and not kwargs:
            args.append(held[name])
        else:
            kwargs[name] = held[name]
    return function(*args, **kwargs)


def _covering(case: dict) -> list:
    commits._covering(case)
    return [one for one in commits._COVERED.get("covered", {}).get(
        f"{case['file']}#{case['function']}", [])]


def real(case: dict, true, text: str) -> dict:
    """The change (the file's text with it in) against the true function:
    every call the tests make of it, and every quick test that runs it."""
    from research.v705 import inside
    try:
        made = inside.changed_function(true, text)
    except Exception as bad:                        # noqa: BLE001
        return {"real": False, "why": f"{type(bad).__name__}"}
    # the true function itself, written out again: real whatever the calls
    # (Service.run's own fix had no call a copy could replay: threads)
    found = commits.functions(text)
    written = found.get(case["function"]) or found.get(
        case["function"].split(".")[-1], "")
    if written and _same_code(written, inspect.getsource(true)):
        return {"real": True, "exact": True}
    tests = list(case["tests"]) + [
        one for one in _covering(case)
        if one["test"] not in {test["test"] for test in case["tests"]}]
    recorded = _recorded(true, tests)
    agree, calls = 0, 0
    unlike = None
    for held, _ in recorded:
        # each replayed as it was called, out of its test: kept from the
        # world (a file written where the test's folder has gone), and the
        # true one replayed alike -- a call it cannot make again is none
        theirs, again = copy.deepcopy(held), copy.deepcopy(held)
        try:
            gave = _sealed(lambda: _called(true, theirs))
            twice = _sealed(lambda: _called(true, again))
        except Exception:                           # noqa: BLE001
            continue
        if not (_alike(gave, twice) and _alike(theirs, again)):
            # not the same of itself twice over: no measure of another
            continue
        mine = copy.deepcopy(held)
        try:
            got = _sealed(lambda: _called(made, mine))
        except Exception as bad:                    # noqa: BLE001
            got = bad
        calls += 1
        if _alike(got, gave) and _alike(mine, theirs):
            agree += 1
        elif unlike is None:
            unlike = f"gives {repr(got)[:80]} where it gave {repr(gave)[:80]}"
    passed = 0
    with commits.Put(true, made):
        for test in tests:
            timer = commits._watched(commits.LONGEST)
            try:
                passed += bool(commits.passes(test))
            except Exception:                       # noqa: BLE001
                pass
            timer.cancel()
    return {"calls": calls, "agree": agree, "tests": len(tests),
            "passed": passed, "unlike": unlike,
            # no call to compare is no evidence it is the same
            "real": 0 < calls == agree and passed == len(tests)}


def _same_code(one: str, other: str) -> bool:
    """Two sources the same code: alike once parsed (comments and layout
    aside)."""
    import ast
    import textwrap
    try:
        return ast.dump(ast.parse(textwrap.dedent(one))) == ast.dump(
            ast.parse(textwrap.dedent(other)))
    except SyntaxError:
        return False


def _sealed(run, seconds: float = 0):
    """`run()` with the world shut: what deduction's programs may not do
    (`deduce.WORLD`), it may not either -- and stopped past `seconds`
    (`REPLAY`): a change that loops without end, replayed out of its
    tests' watch, ran two hours and took 19 GB."""
    from research.v705 import deduce
    # the watch first: it loads what it needs, which a sealed run may not
    timer = commits._watched(seconds or REPLAY)
    deduce._PURE.on = True
    try:
        return run()
    finally:
        deduce._PURE.on = False
        timer.cancel()


#: how long one call replayed may take (seconds)
REPLAY = 10
