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


def _callee(function, call: dict) -> str | None:
    """Where a call got the nothing it gave back: the last name given None
    before it returned, and the function called on the line that gave it
    (`found = _one(...)` -> `_one`)."""
    import ast
    import textwrap
    steps = call["steps"]
    for at in range(len(steps) - 1, 0, -1):
        if "None" not in steps[at]["values"].values():
            continue
        # the values of a step are as its line begins: the line before
        # gave them
        given = steps[at - 1]["line"]
        lines, first = inspect.getsourcelines(function)
        tree = ast.parse(textwrap.dedent("".join(lines)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and \
                    node.lineno + first - 1 == given and \
                    isinstance(node.value, ast.Call):
                called = node.value.func
                return getattr(called, "id", None) or getattr(
                    called, "attr", None)
        return None
    return None


def look_inside(session, turns: list, file: str, name: str,
                wanted: str | None = None, depth: int = 0) -> dict | None:
    """A failing conversation said again with one of the architecture's own
    functions watched: the call that matters -- the one given the act read
    (`wanted`), else the last -- said, with the line it gave back at. Where
    it gave back nothing because a function it called did, that function
    is looked inside in turn (`DEEPEST` deep): `deeper`."""
    function = function_at(file, name)
    if function is None:
        return None
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
        if callee and function_at(file, callee) is not None:
            found["deeper"] = look_inside(session, turns, file, callee,
                                          wanted, depth + 1)
    return found


def deepest(found: dict | None) -> dict | None:
    """The last call followed: where the nothing came from."""
    while found and found.get("deeper"):
        found = found["deeper"]
    return found
