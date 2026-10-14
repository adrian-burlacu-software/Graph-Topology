"""Tools found in its own code (v705): what thinking can call besides the
tools it was given -- any function of `research/` that says what it does
(a docstring) and asks only for what the pad holds, chosen by the encoder
for how near what it says is to what the thinking is about.

An LLM's tools are listed for it. Here none is: a function is a tool where
each parameter it needs is filled from the pad -- by its name (`session`,
`held`, `said`, `case`, `turns`, `file`, `function`, `goal`), else by its
annotation (a `str` is the case said, a `Session` the session) -- and
it is not the entry point of its module. What it gives back is observed.

**Guarded**: while a tool runs, an audit hook refuses whatever would change
the world -- a file opened to write, a file removed or renamed, a process
started, a connection made: a function found is only ever read with. A
tool that runs past `SLOW` seconds is left (still guarded) and not chosen
again.
"""
from __future__ import annotations

import ast
import contextvars
import importlib
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
#: how long a found tool may run before it is left
SLOW = 20.0
#: the found tools the encoder ranks for each choice (nearest first)
RANKED = 5
#: whether the running code is a found tool (`guarded`)
GUARDED: contextvars.ContextVar = contextvars.ContextVar("v705-guarded",
                                                         default=False)
#: what a guarded tool may not do: Python's audit events that change the
#: world (`open` is judged by its mode)
REFUSED = ("os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod",
           "os.truncate", "os.symlink", "os.link", "shutil.rmtree",
           "shutil.move", "shutil.copyfile", "subprocess.Popen", "os.system",
           "os.exec", "os.spawn", "os.posix_spawn", "os.startfile",
           "socket.connect", "socket.bind", "urllib.Request")
_WRITING = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_HOOKED: list = []
#: found tools, once read: [{module, name, params, doc}]
_FOUND: list = []
_VECTORS: dict = {}
#: tools left for running past `SLOW` (this process)
_SLOW: set = set()


def _audit(event: str, args) -> None:
    if not GUARDED.get():
        return
    if event == "open":
        mode, flags = (args[1] if len(args) > 1 else None), \
            (args[2] if len(args) > 2 else 0)
        writes = (isinstance(mode, str) and any(c in mode for c in "wax+")) \
            or (mode is None and isinstance(flags, int) and flags & _WRITING)
        if writes:
            raise PermissionError(f"a tool may not write {args[0]}")
        return
    if event in REFUSED:
        raise PermissionError(f"a tool may not {event}")


def _hook() -> None:
    if not _HOOKED:
        sys.addaudithook(_audit)
        _HOOKED.append(True)


# -- finding them -----------------------------------------------------------------

def found() -> list:
    """Every function of `research/` that is a tool by its form: at module
    level, public, with a docstring, not what `__main__` calls."""
    if _FOUND:
        return _FOUND
    for path in sorted((ROOT / "research").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        entry = set()
        for node in tree.body:
            if isinstance(node, ast.If) and "__main__" in ast.dump(node.test):
                entry |= {one.func.id for one in ast.walk(node)
                          if isinstance(one, ast.Call) and
                          isinstance(one.func, ast.Name)}
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or \
                    node.name.startswith("_") or node.name in entry:
                continue
            doc = ast.get_docstring(node)
            if not doc:
                continue
            arguments = node.args
            defaults = len(arguments.defaults)
            plain = arguments.posonlyargs + arguments.args
            params = [{"name": one.arg,
                       "kind": ast.unparse(one.annotation)
                       if one.annotation else "",
                       "needed": at < len(plain) - defaults}
                      for at, one in enumerate(plain)]
            params += [{"name": one.arg, "kind": ast.unparse(one.annotation)
                        if one.annotation else "", "needed": default is None}
                       for one, default in zip(arguments.kwonlyargs,
                                               arguments.kw_defaults)]
            _FOUND.append({"module": module, "name": node.name,
                           "params": params,
                           "doc": " ".join(doc.split("\n\n")[0].split())})
    return _FOUND


def holds(pad, mind) -> dict:
    """What the pad holds, by name: what a tool's parameters are filled
    from."""
    case = pad.case or {}
    said = case.get("said") or pad.goal
    out = {"goal": pad.goal, "said": said, "text": said, "question": said,
           "session": mind.session, "held": mind.held,
           "case": pad.case, "turns": list(case.get("turns") or [said])}
    if pad.place and pad.place.get("function"):
        out["file"] = pad.place["file"]
        out["function"] = pad.place["function"]
    return {name: value for name, value in out.items() if value is not None}


#: a parameter's annotation, by what of the pad fills it
_BY_KIND = {"str": "said", "Session": "session", "Project": "held",
            "dict": "case", "list[str]": "turns", "list": "turns"}


def filled(tool: dict, held: dict) -> dict | None:
    """The arguments a tool is called with from what the pad holds; None
    where a parameter it needs is not held."""
    out = {}
    for one in tool["params"]:
        if one["name"] in ("self", "cls"):
            return None
        if one["name"] in held:
            out[one["name"]] = held[one["name"]]
            continue
        kind = one["kind"].split("|")[0].strip().strip("'\"")
        by = _BY_KIND.get(kind)
        if one["needed"]:
            if by is None or by not in held:
                return None
            out[one["name"]] = held[by]
    return out


def _key(tool: dict) -> str:
    return f"{tool['module']}.{tool['name']}"


def ranked(about: str, held: dict, tried: set, most: int = RANKED) -> list:
    """The tools that can be called now and were not, nearest what the
    thinking is about first (MiniLM, as pretrained): [(likeness, tool)]."""
    import torch
    from research.v701 import likeness
    tools = [one for one in found() if _key(one) not in tried and
             _key(one) not in _SLOW and filled(one, held) is not None]
    if not tools:
        return []
    missing = [one for one in tools if _key(one) not in _VECTORS]
    for at in range(0, len(missing), 256):
        chunk = missing[at:at + 256]
        vectors = likeness._vectors(
            [f"{one['name'].replace('_', ' ')}: {one['doc']}"
             for one in chunk])
        for one, vector in zip(chunk, vectors):
            _VECTORS[_key(one)] = vector
    near = (likeness._vectors([about]) @
            torch.stack([_VECTORS[_key(one)] for one in tools]).T)[0]
    order = near.argsort(descending=True)[:most].tolist()
    return [(float(near[at]), tools[at]) for at in order]


def called(tool: dict, arguments: dict) -> dict:
    """The tool run, guarded and for `SLOW` seconds at most: {gave} or
    {failed} or {slow}."""
    _hook()
    out: dict = {}

    def run() -> None:
        GUARDED.set(True)
        try:
            function = getattr(importlib.import_module(tool["module"]),
                               tool["name"])
            out["gave"] = function(**arguments)
        except BaseException as bad:                # noqa: BLE001
            if isinstance(bad, KeyboardInterrupt):
                raise
            out["failed"] = f"{type(bad).__name__}: {bad}"
    thread = threading.Thread(target=contextvars.copy_context().run,
                              args=(run,), daemon=True)
    thread.start()
    thread.join(SLOW)
    if thread.is_alive():
        _SLOW.add(_key(tool))
        return {"slow": True}
    return out


def shown(value, longest: int = 160) -> str:
    """What a tool gave back, short enough to be a thought."""
    if isinstance(value, (list, tuple, set)) and len(value) > 3:
        head = ", ".join(repr(one)[:40] for one in list(value)[:3])
        return f"{len(value)} things: {head}, ..."[:longest]
    return " ".join(repr(value).split())[:longest]
