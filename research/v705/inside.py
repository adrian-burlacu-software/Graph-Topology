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


import contextvars as _contextvars
from contextlib import contextmanager

#: whether what runs is said to a scratch session -- an answer it is
#: checking, never a goal to act on: asking itself "What is needed to an
#: electric guitar?" for a goal's evidence was read as a goal, which asked
#: itself again, until the stack ran out
SCRATCH = _contextvars.ContextVar("v705_scratch", default=False)


@contextmanager
def _scratch(session, Session):
    """A scratch session like `session`, under a name of its own: what is
    kept per conversation is kept by its name (v691's scene is
    `SCENES[conversation]`), and replays under one name shared one scene
    -- `my dog is called Rex` said in one, `where is Mary?` in the next
    was the dog's scene's, and answered `which one?`. Dropped after."""
    import uuid
    name = f"v705-scratch-{uuid.uuid4().hex[:12]}"
    SCRATCH.set(True)
    # worked out afresh: a look inside sees the code run, not a cache's
    # answer (`StoreAsker.run`)
    from research.v689.server import FRESH
    FRESH.set(True)
    try:
        yield Session(session.asker, None, name, True,
                      getattr(session, "definitions", None))
    finally:
        try:
            from research.v691 import page as scenes
            scenes.SCENES.pop(name, None)
        except Exception:                           # noqa: BLE001
            pass


def replay(session, turns: list, function) -> tuple:
    """The turns said again to a scratch session like `session` -- the
    last with `function` watched: (the last turn, its calls)."""
    import contextvars
    from research.v689.session import Session

    def apart():
        with _scratch(session, Session) as scratch:
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


def wanted_of(function) -> str | None:
    """What a function says it gives, but nothing: `-> Mention | None` is
    `Mention`; None where it says nothing."""
    returned = inspect.signature(function).return_annotation
    if returned is inspect.Signature.empty:
        return None
    kinds = [one.strip() for one in str(returned).strip("'\"").split("|")
             if one.strip() not in ("None", "NoneType")]
    return " or ".join(kinds) if kinds else None


def imports_of(function) -> dict:
    """The imports of what is in reach (`in_reach`), by name: offered to a
    change that uses one (`fixing._with_imports`)."""
    wanted = wanted_of(function)
    return {written.split("(")[0]: line for written, line in
            (in_reach(function, wanted) if wanted else []) if line}


def statement_of(look: dict, function) -> str:
    """A call looked inside, said as the debugging editor was taught a bug
    (`teach_debugging.statement`): what it was given, what it gave, what
    it should have, and the run -- a give-up (None) should give what the
    function says it gives (`-> Mention | None`: a Mention)."""
    given = ", ".join(f"{key}={value}" for key, value in
                      (look.get("given") or {}).items()
                      if key not in ("self", "cls"))
    wanted = wanted_of(function)
    gave = look.get("returns")
    should = (f"should give a {wanted}" if gave == "None" and wanted else
              "should not give up")
    said = (f"{function.__name__}({given}) gives {gave} but {should}. "
            f"Fix it.\nThe run: {look.get('said')}")
    reach = in_reach(function, wanted) if gave == "None" and wanted else []
    if reach:
        # what in its project gives what it should: the editor is shown one
        # function, never the one that makes a Mention of `my dog`
        said += "\nWhat gives a " + wanted + ": " + "; ".join(
            one for one, _ in reach)
    return said


#: the functions a statement offers at most
REACH = 4


def _header(function) -> str:
    """`read_mention(tokens: list[str], at: int, ...) -> Mention | None`:
    a function's def line as written, on one line."""
    import ast
    import textwrap
    source = textwrap.dedent(inspect.getsource(function))
    node = ast.parse(source).body[0]
    lines = source.splitlines()[node.lineno - 1:node.body[0].lineno - 1]
    text = " ".join(" ".join(lines).split())
    text = text.split("def ", 1)[-1]
    return text[:-1] if text.endswith(":") else text


def in_reach(function, wanted: str) -> list:
    """The functions of its own code that give what a function should --
    by their annotations (`-> Mention | None`), those of its module first,
    then its package's: [(signature as written, import line)]."""
    import importlib
    import pkgutil
    import re as re_
    kinds = [one.strip() for one in wanted.split(" or ")]
    module = importlib.import_module(function.__module__)
    package = function.__module__.rpartition(".")[0]
    modules = [module]
    if package:
        held = importlib.import_module(package)
        for one in pkgutil.iter_modules(getattr(held, "__path__", [])):
            if one.name.startswith("test") or one.name == "__main__":
                continue
            name = f"{package}.{one.name}"
            if name == module.__name__:
                continue
            try:
                modules.append(importlib.import_module(name))
            except Exception:                       # noqa: BLE001
                continue
    out = []
    for held in modules:
        for name, found in vars(held).items():
            if not inspect.isfunction(found) or found is function or \
                    found.__module__ != held.__name__:
                continue
            returned = inspect.signature(found).return_annotation
            if returned is inspect.Signature.empty or not any(
                    re_.search(rf"\b{re_.escape(kind)}\b", str(returned))
                    for kind in kinds):
                continue
            written = _header(found)
            imported = None if held is module else \
                f"from {held.__name__} import {name}"
            out.append((written, imported))
            if len(out) >= REACH:
                return out
    return out


def looked_at(look: dict | None, name: str) -> dict | None:
    """The call of `name` among those a look followed, else None."""
    while look:
        if look.get("function") == name.rpartition(".")[2]:
            return look
        look = look.get("deeper")
    return None


#: the calls a composition tries at most
COMPOSED = 12


def _filled(callee, function) -> list:
    """Each way of calling `callee` from inside `function` with what it
    holds: a parameter filled by the name it has there, else by one of the
    same annotation, else an int by 0; one with a default filled only by
    its own name, a flag tried both ways. [argument text], fewest first."""
    import itertools
    mine = inspect.signature(function).parameters
    by_kind: dict = {}
    for name, one in mine.items():
        if name in ("self", "cls"):
            continue
        by_kind.setdefault(str(one.annotation), []).append(name)
    choices = []
    for name, one in inspect.signature(callee).parameters.items():
        if name in ("self", "cls") or one.kind in (one.VAR_POSITIONAL,
                                                  one.VAR_KEYWORD):
            continue
        given = None
        if name in mine:
            given = [name]
        elif one.annotation is not inspect.Parameter.empty and \
                by_kind.get(str(one.annotation)):
            given = by_kind[str(one.annotation)]
        has_default = one.default is not inspect.Parameter.empty
        if has_default:
            if isinstance(one.default, bool):
                choices.append([None, f"{name}={not one.default}"])
            elif given and name in mine:
                choices.append([f"{name}={name}"])
            continue
        if given:
            choices.append(given)
        elif "int" in str(one.annotation):
            choices.append(["0"])
        else:
            return []                   # a parameter nothing here fills
    out = []
    for picked in itertools.product(*choices):
        out.append(", ".join(one for one in picked if one))
    return out[:COMPOSED]


def compositions(function, look: dict, text: str) -> list:
    """The function's file with a call put in where it gave up: of each
    function in reach that gives what it should, filled from what the
    function holds (`_filled`), and given back where it gives something --
    [(the file's text, the call as written)]. Candidates by construction,
    for running the case to decide (`verify`)."""
    wanted = wanted_of(function)
    line = look.get("line")
    if not wanted or not line or look.get("returns") != "None":
        return []
    lines = text.splitlines(keepends=True)
    if not 0 < line <= len(lines):
        return []
    at = lines[line - 1]
    indent = at[:len(at) - len(at.lstrip())]
    ending = "\r\n" if at.endswith("\r\n") else "\n"
    made = "found"
    while made in inspect.getsource(function):
        made = "_" + made
    out = []
    module = _module(function)
    for written, imported in in_reach(function, wanted):
        name = written.split("(")[0]
        callee = getattr(module, name, None)
        if callee is None and imported:
            import importlib
            callee = getattr(importlib.import_module(
                imported.split()[1]), name, None)
        if callee is None:
            continue
        for args in _filled(callee, function):
            call = f"{name}({args})"
            put = [f"{indent}{made} = {call}{ending}",
                   f"{indent}if {made} is not None:{ending}",
                   f"{indent}    return {made}{ending}"]
            changed = "".join(lines[:line - 1] + put + lines[line - 1:])
            if imported:
                changed = _imported(changed, imported)
            out.append((changed, call))
            if len(out) >= COMPOSED:
                return out
    return out


def precedents(function, look: dict, text: str) -> list:
    """Where a function gave up because a call it made gave nothing
    (`act`: `whose = _whose(...)` was None), the code's own way past that
    elsewhere: another call of the same function falling back with `or`
    (`statement`: `_whose(words, ...) or mention(tokens, span, ...)`) --
    the fallback put after the call here, where every name it uses is one
    the function has. [(the file's text, the line as changed)]."""
    import ast
    deeper = (look or {}).get("deeper") or {}
    callee = deeper.get("function")
    if not callee or deeper.get("returns") != "None":
        return []
    lines, first = inspect.getsourcelines(function)
    tree = ast.parse(__import__("textwrap").dedent("".join(lines)))
    here = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                and _dotted(node.value.func) and \
                _dotted(node.value.func).rpartition(".")[2] == callee:
            here = node
            break
    if here is None:
        return []
    bound = set(inspect.signature(function).parameters) | {
        one.id for one in ast.walk(tree) if isinstance(one, ast.Name)
        and isinstance(one.ctx, ast.Store)} | set(function.__globals__)
    module = _module(function)
    sources = [module]
    package = function.__module__.rpartition(".")[0]
    if package:
        import importlib
        import pkgutil
        held = importlib.import_module(package)
        for one in pkgutil.iter_modules(getattr(held, "__path__", [])):
            if one.name.startswith("test") or one.name == "__main__":
                continue
            try:
                found = importlib.import_module(f"{package}.{one.name}")
            except Exception:                       # noqa: BLE001
                continue
            if found is not module:
                sources.append(found)
    fallbacks = []
    for held in sources:
        try:
            source = inspect.getsource(held)
        except (OSError, TypeError):
            continue
        for node in ast.walk(ast.parse(source)):
            if not (isinstance(node, ast.BoolOp) and isinstance(
                    node.op, ast.Or) and isinstance(node.values[0], ast.Call)
                    and (_dotted(node.values[0].func) or "").rpartition(
                        ".")[2] == callee):
                continue
            for other in node.values[1:]:
                used = {one.id for one in ast.walk(other)
                        if isinstance(one, ast.Name)}
                said = ast.get_source_segment(source, other)
                if said and used <= bound and said not in fallbacks:
                    fallbacks.append(said)
    file_lines = text.splitlines(keepends=True)
    line = here.lineno + first - 1
    end = here.end_lineno + first - 1
    old = "".join(file_lines[line - 1:end])
    call = ast.get_source_segment(textwrap_dedented(lines), here.value)
    out = []
    for fallback in fallbacks[:COMPOSED]:
        if call is None or call not in old:
            break
        changed_line = old.replace(call, f"{call} or {fallback}", 1)
        out.append(("".join(file_lines[:line - 1]) + changed_line +
                    "".join(file_lines[end:]), changed_line.strip()))
    return out


def textwrap_dedented(lines: list) -> str:
    import textwrap
    return textwrap.dedent("".join(lines))


def _imported(text: str, line: str) -> str:
    """The file with an import put after its last one at its top."""
    import ast
    tree = ast.parse(text)
    at = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            at = node.end_lineno
        elif at:
            break
    lines = text.splitlines(keepends=True)
    ending = "\r\n" if "\r\n" in text else "\n"
    return "".join(lines[:at] + [line + ending] + lines[at:])


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
    # what the changed file imports, as it would be imported (an import a
    # change was offered, `fixing._with_imports`)
    for one in tree.body:
        if isinstance(one, (ast.Import, ast.ImportFrom)):
            exec(compile(ast.Module([one], []), "<imports>", "exec"),  # noqa: S102
                 names)
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
        with _scratch(session, Session) as scratch:
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
    if case.get("check") is not None:
        # a case found by a way of looking (`evidence`) brings its check
        # -- a variant, the conversation it was made from as well
        from research.v704.suite import Conversation
        item = Conversation(case.get("area") or "", turns, case["check"])
        out = [(turns, case["check"], True)]
        if case.get("original"):
            out.append((*case["original"], True))
        # what it answers right of the same kind (`ideas.knowledge`)
        out += [(list(turns), check, True)
                for turns, check in case.get("guards") or ()]
    elif found:
        item = found[0]
        out = [(item.turns, item.check, True)]
    else:
        return []
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


#: whether a case holds said to a scratch session as the code is, by
#: session and turns
_NOW: dict = {}


def _holds_now(session, case: tuple) -> bool:
    turns, check, _ = case
    key = (id(session), tuple(turns))
    if key not in _NOW:
        try:
            _NOW[key] = bool(check(_as_page_turn(answered(session, turns)),
                                   None))
        except Exception:                           # noqa: BLE001
            _NOW[key] = False
    return _NOW[key]


def verify(session, cases: list, function, text: str) -> str | None:
    """A change of one of its own functions tried before it is written:
    the function as `text` has it put in for the while, the case said
    again and the cases it answered right -- None where the case now holds
    and none of the rest broke, else why not."""
    made = changed_function(function, text)
    if made is None:
        return f"the change does not define {function.__name__}"
    # what it answered right is what a scratch session answers right as it
    # is: the test's verdict came through the whole page ("where is Mary?"
    # answered by the scene, which a session alone answers "which one?")
    cases = [cases[0]] + [one for one in cases[1:]
                          if _holds_now(session, one)]
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
