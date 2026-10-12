"""Debugging the architecture itself (v705): a failing conversation run
again, apart, with one of its own functions watched.

The look found "what is my dog called?" read as ask_name and taken as
generic -- lost in v689's `reader.build`. Asked to change build so that
it is read as ask_name, the editor wrote seven answers and no two agreed:
nothing said where in build it was lost. Here the conversation is said
again to a **scratch session** (an example: it learns nothing into what
the architecture knows), and each **call** of the function is kept: what
it was given, each line it took with what changed, and what it gave back
-- the `return None` it gave up at, and why there.
"""
from __future__ import annotations

import inspect
import sys

from research.v705.debugging import SAID

#: the most calls kept, and the most steps of each
CALLS, STEPS = 12, 200
SHOWN = 80


def _shown(value) -> str:
    try:
        said = repr(value)
    except Exception:                               # noqa: BLE001
        said = f"<{type(value).__name__}>"
    return said if len(said) <= SHOWN else said[:SHOWN] + "..."


def watch(function, run) -> tuple:
    """`run()` with each call of `function` kept: (what run gave, [{args,
    steps, returns | raises}])."""
    code = getattr(function, "__code__", None)
    calls: list = []
    frames: dict = {}

    def local(frame, event, arg):
        call = frames.get(id(frame))
        if call is None:
            return None
        if event == "line" and len(call["steps"]) < STEPS:
            now = {key: _shown(value) for key, value in
                   frame.f_locals.items()}
            before = call["seen"]
            call["steps"].append({"line": frame.f_lineno, "values": {
                key: value for key, value in now.items()
                if before.get(key) != value}})
            call["seen"] = now
        elif event == "return":
            call["returns"] = _shown(arg)
            call["line"] = frame.f_lineno
            # what its names were bound to, for a function called through
            # one to be followed (`act`, or a handler held in a variable)
            call["scope"] = ({key: value for key, value in
                              frame.f_locals.items() if callable(value)},
                             frame.f_globals)
            frames.pop(id(frame), None)
        elif event == "exception":
            call["raises"] = f"{type(arg[1]).__name__}: {arg[1]}"
        return local

    def tracer(frame, event, arg):
        if event == "call" and frame.f_code is code and \
                len(calls) < CALLS:
            call = {"args": {key: _shown(value) for key, value in
                             frame.f_locals.items()},
                    "steps": [], "seen": {}}
            calls.append(call)
            frames[id(frame)] = call
            return local
        return None
    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        got = run()
    finally:
        sys.settrace(previous)
    for call in calls:
        call.pop("seen", None)
    return got, calls


def replay(session, turns: list, function) -> tuple:
    """The turns said again to a scratch session like `session` -- the
    last with `function` watched: (the last turn, its calls)."""
    import contextvars
    from research.v689.session import Session

    def apart():
        scratch = Session(session.asker, None, "v705-debug", True,
                          getattr(session, "definitions", None))
        for said in turns[:-1]:
            scratch.say(said)
        return watch(function, lambda: scratch.say(turns[-1]))
    # its own world: run in a context of its own, so what the scratch
    # session changes is its own and not the operator's that asked
    # (`executive.effect`: 'plan talk changed conversation')
    return contextvars.Context().run(apart)


def said(call: dict, function) -> str:
    """A call in words: what it was given, the lines it took, what it gave
    back and where."""
    lines, first = inspect.getsourcelines(function)
    text = {first + at: one.strip() for at, one in enumerate(lines)}
    out, told = [], set()
    for one in call["steps"]:
        if one["line"] in told:
            continue
        told.add(one["line"])
        changed = ", ".join(f"{key} = {value}" for key, value in
                            one["values"].items())
        out.append(f"line {one['line']} `{text.get(one['line'], '')}`"
                   + (f": {changed}" if changed else ""))
    if len(out) > SAID:
        out = out[:SAID // 2] + ["..."] + out[-(SAID - SAID // 2):]
    end = (f"raises {call['raises']}" if "raises" in call else
           f"returns {call.get('returns')} at line {call.get('line')} "
           f"`{text.get(call.get('line'), '')}`")
    return "; ".join(out + [end])


def function_at(file: str, name: str):
    """The function of this name in a file of the architecture's own code
    (`research/v689/reader.py`, `build`), or a method of a class there
    (`_ask_name` of Session); None where there is none."""
    import importlib
    module = file[:-3].replace("/", ".") if file.endswith(".py") else None
    if module is None:
        return None
    try:
        found = importlib.import_module(module)
    except Exception:                               # noqa: BLE001
        return None
    direct = getattr(found, name, None)
    if inspect.isfunction(direct):
        return direct
    for _, kind in inspect.getmembers(found, inspect.isclass):
        method = kind.__dict__.get(name)
        if inspect.isfunction(method):
            return method
    return None


#: how many calls deep a give-up is followed
DEEPEST = 3


def _called(node) -> str | None:
    return getattr(node.func, "id", None) or getattr(node.func, "attr", None)


def _callee(function, call: dict):
    """Where a call got the nothing it gave back: the function called on
    the line it returned at (`return act(...)`), else on the line that
    gave the last name None before it returned (`found = _one(...)`) --
    resolved through the names the call itself had bound, as a debugger
    would (a handler held in a variable is the function it holds). The
    function, else None."""
    import ast
    import textwrap
    lines, first = inspect.getsourcelines(function)
    tree = ast.parse(textwrap.dedent("".join(lines)))

    def at_line(line: int, kinds) -> str | None:
        for node in ast.walk(tree):
            if isinstance(node, kinds) and node.lineno + first - 1 == line \
                    and isinstance(node.value, ast.Call):
                return _called(node.value)
        return None
    name = at_line(call.get("line") or 0, ast.Return)
    steps = call["steps"]
    for at in range(len(steps) - 1, 0, -1):
        if name:
            break
        if "None" in steps[at]["values"].values():
            # the values of a step are as its line begins: the line before
            # gave them
            name = at_line(steps[at - 1]["line"], ast.Assign)
            break
    if not name:
        return None
    local, module = call.get("scope") or ({}, {})
    found = local.get(name) or module.get(name)
    return found if inspect.isfunction(found) else None


def look_inside(session, turns: list, file: str, name: str,
                wanted: str | None = None, depth: int = 0) -> dict | None:
    """A failing conversation said again with one of the architecture's own
    functions watched: the call that matters -- the one given the act read
    (`wanted`), else the last -- said, with the line it gave back at. Where
    it gave back nothing because a function it called did, that function
    is looked inside in turn (`DEEPEST` deep): `deeper`."""
    function = name if inspect.isfunction(name) else function_at(file, name)
    if function is None:
        return None
    name = function.__name__
    file = _relative(inspect.getsourcefile(function)) or file
    _, calls = replay(session, turns, function)
    if not calls:
        return {"said": f"{name} was never called", "calls": 0}
    call = next((one for one in calls if wanted and any(
        repr(wanted) == value for value in one["args"].values())),
        calls[-1])
    lines, first = inspect.getsourcelines(function)
    line = call.get("line")
    found = {"function": name, "file": file, "said": said(call, function),
             "calls": len(calls), "line": line,
             "code": lines[line - first].strip()
             if line and 0 <= line - first < len(lines) else "",
             "given": call["args"], "returns": call.get("returns")}
    if call.get("returns") == "None" and depth < DEEPEST:
        callee = _callee(function, call)
        if callee is not None and _relative(inspect.getsourcefile(callee)):
            found["deeper"] = look_inside(session, turns, file, callee,
                                          wanted, depth + 1)
    return found


def _relative(path: str | None) -> str | None:
    """A file of the architecture's own code, as the project names it;
    None for one outside it (a library's)."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except (TypeError, ValueError):
        return None


def deepest(found: dict | None) -> dict | None:
    """The last call followed: where the nothing came from."""
    while found and found.get("deeper"):
        found = found["deeper"]
    return found
