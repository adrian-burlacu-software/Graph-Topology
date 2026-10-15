"""What each of its functions means (v705): what it needs, what it gives,
what it changes -- as the maths side knows what each operation does, so
that code can be designed by reasoning from what is wanted back to what is
in hand, not guessed.

    python -m research.v705.semantics read      state/v705-semantics.json
    python -m research.v705.semantics watch     the kinds seen in runs
    python -m research.v705.semantics say NAME  one function, in words

Every way tried of inventing a helper failed at the same place: a teacher,
a designer, a search over expressions -- none knew what its own functions
*do*. `_subgoal`, the helper `carry` needed, takes the next subgoal, runs
`_step` on it, records the result in `state["results"]`, and when they run
out calls `_again`: a procedure made of what other functions give and
change. Knowing that is the first step of planning one.

**Read from the code** (`read`): of every function of `research/` (tests
aside) -- its parameters and their annotations; what it reads of each
(`state["weak"]`, `state.get("plan")`, `self.files`); what it changes of
each, or of `self`, or of a module's names (`state["at"] = ...`,
`.append`, `.setdefault`, `global`); what it gives back (its annotation,
and the kinds of what its returns say); the functions of its own code it
calls; the first line of its docstring.

**Watched in runs** (`watch`): the quick tests run once with every call of
its own code watched (`sys.setprofile`): the kinds of value each parameter
was given and each return gave.

**Indexed**, for planning backwards: who gives a kind of value, who
changes a field, who reads it.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEMANTICS = ROOT / "state" / "v705-semantics.json"
#: the methods that change what they are called on
CHANGING = {"append", "extend", "insert", "remove", "pop", "clear",
            "update", "setdefault", "add", "discard", "sort", "reverse",
            "popitem", "appendleft", "write"}
#: the tests watched at most (`watch`)
WATCHED = 400


def _root(node) -> str | None:
    """The name a target or a read is rooted in (`state` of
    `state["at"]`, `self` of `self.files[x]`)."""
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _field(node) -> str:
    """A target or a read as said (`state["at"]`, `self.files`): its root
    and its first field."""
    path = []
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        if isinstance(node, ast.Attribute):
            path.append(f".{node.attr}")
        else:
            key = node.slice
            path.append(f"[{ast.unparse(key)}]" if isinstance(
                key, ast.Constant) else "[...]")
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    return node.id + (path[-1] if path else "")


def _kind(node) -> str:
    """What kind of value an expression says it is, where it says."""
    if isinstance(node, ast.Constant):
        return "None" if node.value is None else type(node.value).__name__
    for kind, types in (("list", (ast.List, ast.ListComp)),
                        ("dict", (ast.Dict, ast.DictComp)),
                        ("set", (ast.Set, ast.SetComp)),
                        ("tuple", (ast.Tuple,)), ("str", (ast.JoinedStr,)),
                        ("bool", (ast.Compare, ast.BoolOp))):
        if isinstance(node, types):
            return kind
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return "bool"
    if isinstance(node, ast.Call):
        name = ast.unparse(node.func)
        return f"what {name} gives"
    if isinstance(node, ast.Name):
        return f"its {node.id}"
    return ast.unparse(node)[:30]


def meaning(node, module: str) -> dict:
    """What one function's code says it does."""
    arguments = node.args
    params = [one.arg for one in arguments.posonlyargs + arguments.args +
              arguments.kwonlyargs]
    kinds = {one.arg: ast.unparse(one.annotation) if one.annotation else ""
             for one in arguments.posonlyargs + arguments.args +
             arguments.kwonlyargs}
    held = set(params) | {"self", "cls"}
    reads: set = set()
    changes: set = set()
    gives: set = set()
    calls: set = set()
    # a method called is not a field read (`state.get`)
    called = {id(one.func) for one in ast.walk(node)
              if isinstance(one, ast.Call)}

    def keyed(call) -> str | None:
        """`state["results"]` of `state.setdefault("results", [])` or
        `state.get("results")`: the field a dict's method names."""
        if isinstance(call, ast.Call) and isinstance(
                call.func, ast.Attribute) and call.func.attr in (
                "setdefault", "get", "pop") and call.args and \
                isinstance(call.args[0], ast.Constant) and \
                _root(call.func.value) in held:
            return f"{_field(call.func.value)}[{call.args[0].value!r}]"
        return None
    for one in ast.walk(node):
        if isinstance(one, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = one.targets if isinstance(one, ast.Assign) \
                else [one.target]
            for target in targets:
                for part in ast.walk(target):
                    if isinstance(part, (ast.Attribute, ast.Subscript)) \
                            and _root(part) in held:
                        changes.add(_field(part))
                        break
        elif isinstance(one, ast.Delete):
            for target in one.targets:
                if _root(target) in held:
                    changes.add(_field(target))
        elif isinstance(one, ast.Global):
            changes |= {f"{module}.{name}" for name in one.names}
        elif isinstance(one, ast.Call):
            func = one.func
            if isinstance(func, ast.Attribute) and \
                    func.attr in CHANGING and _root(func.value) in held:
                # `state.setdefault("results", []).append(...)` changes
                # state["results"]; `state.setdefault("x", 1)` state["x"]
                changes.add(keyed(func.value) or (
                    keyed(one) if func.attr in ("setdefault", "pop")
                    else None) or _field(func.value) or
                    _root(func.value))
            elif keyed(one):
                reads.add(keyed(one))
            name = ast.unparse(func)
            if not name.startswith(("self.", "super(")) and "(" not in name:
                calls.add(name)
            elif name.startswith("self.") and name.count(".") == 1:
                calls.add(name)
        elif isinstance(one, (ast.Attribute, ast.Subscript)) and \
                isinstance(getattr(one, "ctx", None), ast.Load) and \
                id(one) not in called and _root(one) in held:
            said = _field(one)
            if said and said not in params:
                reads.add(said)
        elif isinstance(one, ast.Return) and one.value is not None:
            gives.add(_kind(one.value))
    doc = (ast.get_docstring(node) or "").split("\n\n")[0]
    return {"module": module, "params": params, "kinds": kinds,
            "returns": ast.unparse(node.returns) if node.returns else "",
            "reads": sorted(reads - changes), "changes": sorted(changes),
            "gives": sorted(gives), "calls": sorted(calls),
            "doc": " ".join(doc.split())}


def read() -> dict:
    """What the code says of every function of `research/` (tests
    aside): {file#qualname: meaning}, and the indexes."""
    out: dict = {}
    for path in sorted((ROOT / "research").rglob("*.py")):
        if "__pycache__" in path.parts or path.name.startswith("test"):
            continue
        file = path.relative_to(ROOT).as_posix()
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[f"{file}#{node.name}"] = meaning(node, module)
            elif isinstance(node, ast.ClassDef):
                for one in node.body:
                    if isinstance(one, (ast.FunctionDef,
                                        ast.AsyncFunctionDef)):
                        out[f"{file}#{node.name}.{one.name}"] = meaning(
                            one, module)
    found = {"functions": out}
    if SEMANTICS.exists():
        # what runs were seen to give: kept across a reading
        before = json.loads(SEMANTICS.read_text(encoding="utf-8"))
        for key, seen in (before.get("seen") or {}).items():
            found.setdefault("seen", {})[key] = seen
    found["index"] = index(out)
    SEMANTICS.parent.mkdir(parents=True, exist_ok=True)
    SEMANTICS.write_text(json.dumps(found), encoding="utf-8")
    counts = {"functions": len(out),
              "that change something": sum(bool(one["changes"])
                                           for one in out.values()),
              "fields changed": len(found["index"]["changes"]),
              "fields read": len(found["index"]["reads"])}
    print(json.dumps(counts))
    return counts


def _bare(field: str) -> str:
    """A field without what it is rooted in: `["at"]` of `state["at"]`,
    `.files` of `self.files` -- what two functions share when each names
    its own parameter differently."""
    for at, char in enumerate(field):
        if char in ".[":
            return field[at:]
    return field


def index(functions: dict) -> dict:
    """Who changes, reads and gives what: {changes: {field: [function]},
    reads: {...}, gives: {kind: [...]}} -- fields bare of their root."""
    out = {"changes": {}, "reads": {}, "gives": {}}
    for key, one in functions.items():
        for field in one["changes"]:
            out["changes"].setdefault(_bare(field), []).append(key)
        for field in one["reads"]:
            out["reads"].setdefault(_bare(field), []).append(key)
        for kind in [one["returns"]] + one["gives"]:
            if kind:
                out["gives"].setdefault(kind, []).append(key)
    return out


def _type(value) -> str:
    name = type(value).__name__
    if isinstance(value, (list, tuple, set)) and value:
        first = next(iter(value))
        return f"{name}[{type(first).__name__}]"
    if isinstance(value, dict) and value:
        key, one = next(iter(value.items()))
        return f"dict[{type(key).__name__}, {type(one).__name__}]"
    return name


def watch(most: int = WATCHED) -> dict:
    """The quick tests run with every call of its own code watched: the
    kinds of value each parameter was given and each return gave --
    {file#qualname: {param: {kind: count}, "->": {kind: count}}}."""
    from research.v705 import commits, inside, planted
    covered = json.loads(planted.COVERED.read_text(encoding="utf-8"))
    tests = [key for key, one in covered["tests"].items()
             if one["ok"] and one["seconds"] <= planted.QUICK][:most]
    seen: dict = {}
    places: dict = {}

    def where(code) -> str | None:
        name = code.co_filename
        path = places.get(name, False)
        if path is False:
            path = inside._relative(name) if "research" in name else None
            if path and (not path.startswith("research/") or
                         "/test" in path):
                path = None
            places[name] = path
        if not path:
            return None
        return f"{path}#{getattr(code, 'co_qualname', code.co_name)}"

    def profiler(frame, event, arg):
        if event not in ("call", "return"):
            return
        key = where(frame.f_code)
        if key is None or "<" in key:
            return
        one = seen.setdefault(key, {})
        if event == "call":
            code = frame.f_code
            for name in code.co_varnames[:code.co_argcount +
                                         code.co_kwonlyargcount]:
                if name in ("self", "cls") or name not in frame.f_locals:
                    continue
                kinds = one.setdefault(name, Counter())
                if sum(kinds.values()) < 50:
                    kinds[_type(frame.f_locals[name])] += 1
        else:
            kinds = one.setdefault("->", Counter())
            if sum(kinds.values()) < 50:
                kinds[_type(arg)] += 1

    for at, key in enumerate(tests):
        file, test = key.split("::")
        previous = sys.getprofile()
        sys.setprofile(profiler)
        timer = commits._watched(commits.LONGEST)
        try:
            commits.passes({"file": file, "test": test})
        except Exception:                           # noqa: BLE001
            pass
        finally:
            sys.setprofile(previous)
            timer.cancel()
        if at % 50 == 0:
            print(f"{at} of {len(tests)}", flush=True)
    found = json.loads(SEMANTICS.read_text(encoding="utf-8")) \
        if SEMANTICS.exists() else {}
    found["seen"] = {key: {name: dict(kinds) for name, kinds in one.items()}
                     for key, one in seen.items()}
    SEMANTICS.write_text(json.dumps(found), encoding="utf-8")
    counts = {"tests": len(tests), "functions seen": len(seen)}
    print(json.dumps(counts))
    return counts


def said(key: str, held: dict | None = None) -> str:
    """One function's meaning in words: what it needs, gives and
    changes."""
    held = held or json.loads(SEMANTICS.read_text(encoding="utf-8"))
    one = held["functions"].get(key)
    if one is None:
        return f"{key} is not a function of its own code."
    seen = (held.get("seen") or {}).get(key, {})
    name = key.split("#")[1]

    def kinds_of(param: str) -> str:
        kinds = seen.get(param) or {}
        written = one["kinds"].get(param)
        top = ", ".join(kind for kind, _ in sorted(
            kinds.items(), key=lambda pair: -pair[1])[:2])
        return written or top
    needs = ", ".join(
        param + (f" ({kinds_of(param)})" if kinds_of(param) else "")
        for param in one["params"] if param not in ("self", "cls"))
    gives = one["returns"] or ", ".join(one["gives"][:3])
    given = ", ".join(kind for kind, _ in sorted(
        (seen.get("->") or {}).items(), key=lambda pair: -pair[1])[:2])
    lines = [f"{name}: {one['doc']}" if one["doc"] else name,
             f"  needs: {needs or 'nothing'}"]
    if one["reads"]:
        lines.append(f"  reads: {', '.join(one['reads'][:8])}")
    lines.append(f"  gives: {gives or 'nothing'}"
                 + (f" (seen: {given})" if given else ""))
    if one["changes"]:
        lines.append(f"  changes: {', '.join(one['changes'][:8])}")
    if one["calls"]:
        lines.append(f"  calls: {', '.join(one['calls'][:10])}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("read", "watch", "say"))
    parser.add_argument("name", nargs="?", default="")
    options = parser.parse_args(argv)
    if options.job == "read":
        read()
    elif options.job == "watch":
        watch()
    else:
        held = json.loads(SEMANTICS.read_text(encoding="utf-8"))
        for key in held["functions"]:
            if key.split("#")[1] == options.name or key == options.name:
                print(said(key, held))
                print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
