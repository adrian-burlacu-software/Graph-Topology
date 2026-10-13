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
            # one to be followed (`act`, a handler held in a variable, a
            # method of `self`)
            call["scope"] = (dict(frame.f_locals), frame.f_globals)
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
    if "." in name:
        # `StoreAsker.run`, as the project names a method
        kind, _, name = name.rpartition(".")
        held = getattr(found, kind, None)
        method = vars(held).get(name) if inspect.isclass(held) else None
        method = getattr(method, "__func__", method)
        return method if inspect.isfunction(method) else None
    direct = getattr(found, name, None)
    if inspect.isfunction(direct) and direct.__module__ == found.__name__:
        return direct
    # a method of a class the file defines, not one it imports (v689
    # server's `run` is StoreAsker's, not the Asker it imports)
    for _, kind in inspect.getmembers(found, inspect.isclass):
        if kind.__module__ != found.__name__:
            continue
        method = kind.__dict__.get(name)
        if inspect.isfunction(method):
            return method
    return None


#: how many calls deep a give-up is followed
DEEPEST = 3


def _dotted(node) -> str | None:
    """`turn.run`, `self._run`, `found`: a name or attribute chain as
    written; None for anything else."""
    import ast
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _bound(target) -> set:
    """The names a statement's target binds (`outcome, headline, trust`,
    `turn.answer`, `found["x"]` binds `found`)."""
    import ast
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_bound(one) for one in target.elts))
    if isinstance(target, ast.Subscript):
        return _bound(target.value)
    if isinstance(target, ast.Starred):
        return _bound(target.value)
    name = _dotted(target)
    return {name} if name else set()


def _reads(node) -> set:
    """The names and attribute chains a piece of code reads, each with its
    bases (`turn.run` reads `turn.run` and `turn`)."""
    import ast
    out = set()
    for one in ast.walk(node):
        if isinstance(one, (ast.Name, ast.Attribute)):
            name = _dotted(one)
            while name:
                out.add(name)
                name = name.rpartition(".")[0]
    return out


def _resolve(name: str, call: dict):
    """A called name resolved through what the call itself had bound, as a
    debugger would: a handler held in a variable is the function it holds,
    `self._run` the method of the object `self` was. The function, else
    None."""
    local, module = call.get("scope") or ({}, {})
    head, *rest = name.split(".")
    if head in local:
        found = local[head]
    elif head in module:
        found = module[head]
    else:
        return None
    for attr in rest:
        try:
            found = inspect.getattr_static(found, attr)
        except AttributeError:
            return None
        if isinstance(found, (staticmethod, classmethod)):
            found = found.__func__
    found = getattr(found, "__func__", found)
    return found if inspect.isfunction(found) else None


def _super(node, call: dict):
    """`super().run`: the method the class after the call's own class has,
    in the order its `self`'s type looks them up -- PinnedAsker.run's
    `return super().run(question)` is StoreAsker.run. None for anything
    else."""
    import ast
    if not (isinstance(node, ast.Attribute) and isinstance(
            node.value, ast.Call) and getattr(node.value.func, "id", None)
            == "super" and not node.value.args):
        return None
    local = (call.get("scope") or ({}, {}))[0]
    here, me = local.get("__class__"), local.get("self")
    if here is None or me is None:
        return None
    order = type(me).__mro__
    if here not in order:
        return None
    for kind in order[order.index(here) + 1:]:
        found = vars(kind).get(node.attr)
        if found is not None:
            found = getattr(found, "__func__", found)
            return found if inspect.isfunction(found) else None
    return None


def _callee(function, call: dict):
    """Where a call got what it gave up with, followed back through what it
    computed (a backward slice of the run): from the line it gave back at
    -- `return act(...)`, `return found`, or a handler's last line,
    `turn.answer = {"outcome": outcome, ...}` -- each name that line read
    back to the line that bound it, and on through what that line read.
    Of the calls of the architecture's own code on that path, the first
    that gave back nothing (`found = _one(...)` was None), else the first
    of all: the one the rest only passed on (`_generic`'s UNKNOWN is
    `summary_of(turn.run)`, and `turn.run` is what `self._run` gave). The
    function, else None."""
    import ast
    import textwrap
    lines, first = inspect.getsourcelines(function)
    tree = ast.parse(textwrap.dedent("".join(lines)))
    statements = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                       ast.If, ast.For, ast.While, ast.With, ast.Try)):
            statements.setdefault(node.lineno + first - 1, node)
    steps = call["steps"]
    taken = [one["line"] for one in steps]
    end = call.get("line") or (taken[-1] if taken else None)
    if end is None or end not in statements:
        return None
    # the lines taken, last first, from the one it gave back at
    order = [end] + [line for line in reversed(taken[:-1])
                     if line != end]
    start = statements[end]
    needed = _reads(start.value) if isinstance(
        start, (ast.Return, ast.Expr, ast.Assign, ast.AugAssign,
                ast.AnnAssign)) and getattr(start, "value", None) else set()
    if not needed:
        # a constant given back (`return None`): why is the condition it
        # was given back under (`if found is None:`)
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While)) and any(
                    one is start for one in ast.walk(node)
                    if isinstance(one, ast.stmt)):
                needed = _reads(node.test)
    path = [(end, start)] if needed or isinstance(
        getattr(start, "value", None), ast.Call) else []
    for line in order[1:]:
        node = statements.get(line)
        if not isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)) \
                or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) \
            else [node.target]
        bound = set().union(*(_bound(one) for one in targets))
        if not bound & needed:
            continue
        needed = (needed - bound) | _reads(node.value)
        path.append((line, node))
    # what each line on the path gave: the values of a step are as its
    # line begins, so the step after a line holds what that line bound
    gave = {}
    for at, one in enumerate(steps[:-1]):
        gave.setdefault(one["line"], steps[at + 1]["values"])
    calls = []
    for line, node in path:
        for one in ast.walk(node.value):
            if not isinstance(one, ast.Call):
                continue
            name = _dotted(one.func)
            found = _resolve(name, call) if name else _super(one.func, call)
            if found is None or not _relative(
                    inspect.getsourcefile(found)) or found is function:
                continue
            nothing = (call.get("returns") == "None" if line == end and
                       isinstance(node, ast.Return) else
                       "None" in (gave.get(line) or {}).values())
            calls.append((found, nothing))
    if not calls:
        return None
    gave_up = [found for found, nothing in calls if nothing]
    return gave_up[-1] if gave_up else calls[-1][0]


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
    # followed down whatever it gave back: a nothing, or a handler's
    # answer that gave up (`_generic` returns None whatever it says)
    if depth < DEEPEST:
        callee = _callee(function, call)
        if callee is not None and _relative(inspect.getsourcefile(callee)):
            found["deeper"] = look_inside(session, turns, file, callee,
                                          wanted, depth + 1)
    return found


#: the cases it answered right before, said again with a change in
REGRESSIONS = 4


def _owner(function):
    """The class a method is defined in, else the module: what holds the
    function by its name."""
    import importlib
    module = importlib.import_module(function.__module__)
    owner = module
    for part in function.__qualname__.split(".")[:-1]:
        if part == "<locals>":
            return None
        owner = getattr(owner, part, None)
        if owner is None:
            return None
    return owner


def changed_function(function, text: str):
    """The function as a changed file's text defines it, made as the
    running one was -- in its module's names, a method in its class (its
    `super()` the class's) -- not yet put in. None where the text does not
    define it."""
    import ast
    import textwrap
    owner = _owner(function)
    if owner is None:
        return None
    tree = ast.parse(text)
    holder = tree
    for part in function.__qualname__.split(".")[:-1]:
        holder = next((node for node in holder.body if isinstance(
            node, ast.ClassDef) and node.name == part), None)
        if holder is None:
            return None
    node = next((one for one in holder.body if isinstance(
        one, (ast.FunctionDef, ast.AsyncFunctionDef)) and
        one.name == function.__name__), None)
    if node is None:
        return None
    lines = text.splitlines(keepends=True)
    first = node.decorator_list[0].lineno if node.decorator_list \
        else node.lineno
    source = textwrap.dedent("".join(lines[first - 1:node.end_lineno]))
    names = dict(function.__globals__)
    # a method made in a class of its own, so that `super()` and its
    # private names compile as they do in its class
    wrapped = f"class {owner.__name__ if owner is not _module(function) else '_Here'}:\n" \
        + textwrap.indent(source, "    ")
    code = compile(wrapped, inspect.getsourcefile(function) or "<changed>",
                   "exec")
    exec(code, names)                               # noqa: S102
    made = vars(names[owner.__name__ if owner is not _module(function)
                      else "_Here"])[function.__name__]
    made = getattr(made, "__func__", made)
    for name, cell in zip(made.__code__.co_freevars,
                          made.__closure__ or ()):
        if name == "__class__":
            cell.cell_contents = owner
    return made


def _module(function):
    import importlib
    return importlib.import_module(function.__module__)


def answered(session, turns: list):
    """The turns said again to a scratch session like `session`: the last
    turn, as the page is given it."""
    import contextvars
    from research.v689.session import Session

    def apart():
        scratch = Session(session.asker, None, "v705-check", True,
                          getattr(session, "definitions", None))
        for said in turns[:-1]:
            scratch.say(said)
        return scratch.say(turns[-1])
    return contextvars.Context().run(apart)


def _as_page_turn(turn) -> dict:
    """A scratch session's turn as the conversation test reads a reply: its
    text, and its stance."""
    from research.v690.message import of_turn
    found = turn.as_dict()
    return {"reply": {"text": (turn.answer or {}).get("text") or "",
                      "message": {"stance": of_turn(found).stance}}}


def cases_of(case: dict, function) -> list:
    """The conversation test's own checks for a case: the case, and the
    conversations of its area it answered right that went the same way
    (the same operator fired) -- [(turns, check, should hold)]. Only those
    said without a project: those are what a scratch session says."""
    from research.v704 import selfmodel, suite
    model = selfmodel.held() or {}
    run = selfmodel.RUNS / f"{model.get('run')}.jsonl"
    turns = case.get("turns") or list(case.get("before") or []) + [
        case["said"]]
    found = [item for item in suite.SUITE if item.turns == turns
             and not item.project]
    if not found:
        return []
    item = found[0]
    out = [(item.turns, item.check, True)]
    if run.exists():
        import json
        operators = {one["operator"] for one in (case.get("fired") or
                                                 case.get("tried") or [])}
        for line in run.open(encoding="utf-8"):
            row = json.loads(line)
            if len(out) > REGRESSIONS or row["verdict"] != "right" or \
                    row["area"] != item.area or row["turns"] == item.turns \
                    or not operators & {one["operator"] for one in
                                        row.get("fired") or []}:
                continue
            same = [one for one in suite.SUITE if one.turns == row["turns"]
                    and not one.project]
            if same:
                out.append((same[0].turns, same[0].check, True))
    return out


def verify(session, cases: list, function, text: str) -> str | None:
    """A change of one of its own functions tried before it is written:
    the function as `text` has it put in for the while, the case said
    again and the cases it answered right -- None where the case now holds
    and none of the rest broke, else why not."""
    made = changed_function(function, text)
    if made is None:
        return f"the change does not define {function.__name__}"
    owner = _owner(function)
    before = vars(owner)[function.__name__]
    kept = before
    if isinstance(before, (staticmethod, classmethod)):
        made = type(before)(made)
    setattr(owner, function.__name__, made)
    try:
        for at, (turns, check, _) in enumerate(cases):
            try:
                turn = answered(session, turns)
            except Exception as bad:                # noqa: BLE001
                return (f"\"{turns[-1]}\" raises {type(bad).__name__}: "
                        f"{bad}")
            try:
                held = bool(check(_as_page_turn(turn), None))
            except Exception:                       # noqa: BLE001
                held = False
            if not held:
                said = " ".join(((turn.answer or {}).get("text") or "")
                                .split())[:80]
                return (f"\"{turns[-1]}\" still fails ({said})" if at == 0
                        else f"\"{turns[-1]}\", answered right before, now "
                             f"fails ({said})")
        return None
    finally:
        setattr(owner, function.__name__, kept)


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
