"""Designing by means-ends over what its functions mean (v705): what a
failing test wants, as a goal state (`goal`), and procedures planned back
from it over what each function needs, gives and changes (`semantics`) --
each run again with the tests (`commits.verifier`).

    python -m research.v705.means goals     what each held-out bug's test wants
    python -m research.v705.means run       planned and judged

**The goal state** (`goal`): the failing test run, with the bug in, every
check it makes recorded (`assertEqual`, `assertIn`, `assertTrue`, ...):
the value looked at and what it should be, where among the test's names
the value lives (`state['sub']`, `lines`, `found.text`), and the call the
test made it with (`lines = doing.carry(state, ...)`). The first that
fails is the goal; those that pass are kept.

**Planning** (`plans`): back from the goal, as the maths side plans from a
spec -- the functions whose meaning changes the field the goal is of, or
gives the kind of value it wants; then, for what each needs and nothing
in scope holds, the functions that change or give that (one step more).
Each plan a procedure put in the function at the place: before each
return, or first -- its result kept where the goal looks (the value
returned, added to; the field named, set). A field no function changes is
changed by the plainest acts (`+= 1`, `.append`).
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import json
import sys
import textwrap
from pathlib import Path

from research.v705 import commits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "state" / "v705-means.json"
#: the plans tried for each bug at most, and the steps a plan takes
PLANS, STEPS = 80, 2
#: what a check records of a value at most
SHOWN = 160
#: the checks recorded, by unittest's name: (the value, what it should be)
CHECKS = {"assertEqual": (0, 1), "assertNotEqual": (0, 1),
          "assertIn": (1, 0), "assertNotIn": (1, 0), "assertTrue": (0, None),
          "assertFalse": (0, None), "assertIsNone": (0, None),
          "assertIsNotNone": (0, None), "assertGreater": (0, 1),
          "assertGreaterEqual": (0, 1), "assertLess": (0, 1),
          "assertCountEqual": (0, 1), "assertIs": (0, 1)}


# -- step 2: what the test wants -------------------------------------------------

def _path(value, names: dict) -> str | None:
    """Where among the test's names a value lives: `lines`, `state['sub']`,
    `found.text` (two steps down at most)."""
    for name, held in names.items():
        if name.startswith("_") or name == "self":
            continue
        if held is value:
            return name
    for name, held in names.items():
        if name.startswith("_") or name == "self":
            continue
        fields = held.items() if isinstance(held, dict) else (
            vars(held).items() if hasattr(held, "__dict__") and
            not isinstance(held, type) else [])
        for key, one in list(fields)[:60]:
            if one is value:
                return (f"{name}[{key!r}]" if isinstance(held, dict)
                        else f"{name}.{key}")
    return None


def _made_by(test: dict) -> dict:
    """{name: the call the test bound it with} -- `lines`:
    `doing.carry(state, change, held)`."""
    source = commits.functions((ROOT / test["file"]).read_text(
        encoding="utf-8")).get(test["test"], "")
    out = {}
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            for target in node.targets:
                for one in ast.walk(target):
                    if isinstance(one, ast.Name):
                        out[one.id] = ast.unparse(node.value)[:120]
    return out


def goal(case: dict) -> dict:
    """The failing test's checks, with the bug in: {failing, kept} -- each
    {check, path, got, wanted, made by}."""
    import unittest
    test = case["tests"][0]
    seen: list = []
    kept = {}
    made_by = _made_by(test)

    def watched(name, method):
        value_at, wanted_at = CHECKS[name]

        def check(self, *args, **kwargs):
            frame = sys._getframe(1)
            names = dict(frame.f_locals)
            value = args[value_at] if len(args) > value_at else None
            wanted = args[wanted_at] if wanted_at is not None and \
                len(args) > wanted_at else None
            path = _path(value, names)
            row = {"check": name, "path": path,
                   "got": repr(value)[:SHOWN],
                   "wanted": repr(wanted)[:SHOWN] if wanted_at is not None
                   else None,
                   "made by": made_by.get((path or "").split("[")[0]
                                          .split(".")[0])}
            try:
                out = method(self, *args, **kwargs)
            except AssertionError:
                row["failed"] = True
                seen.append(row)
                raise
            row["failed"] = False
            seen.append(row)
            return out
        return check

    for name in CHECKS:
        kept[name] = getattr(unittest.TestCase, name)
        setattr(unittest.TestCase, name, watched(name, kept[name]))
    try:
        commits.passes(test)
    finally:
        for name, method in kept.items():
            setattr(unittest.TestCase, name, method)
    failing = next((one for one in seen if one["failed"]), None)
    return {"failing": failing, "kept": [one for one in seen
                                         if not one["failed"]],
            "raised": None if failing else commits.LAST.get(
                "failure", "")[-300:]}


# -- step 3: planning back from it ---------------------------------------------

def _scope(source: str, upto: int | None = None) -> dict:
    """The names the function holds where it runs: its parameters, and
    what it binds -- before line `upto` of its source where given (a
    value offered at the first line, `args`, was not bound there yet) --
    {name: kind as said, or ""}."""
    tree = ast.parse(textwrap.dedent(source))
    node = tree.body[0]
    out = {one.arg: ast.unparse(one.annotation) if one.annotation else ""
           for one in node.args.posonlyargs + node.args.args +
           node.args.kwonlyargs}
    for one in ast.walk(node):
        if upto is not None and getattr(one, "lineno", 0) >= upto:
            continue
        if isinstance(one, ast.Assign):
            for target in one.targets:
                if isinstance(target, ast.Name):
                    out.setdefault(target.id, "")
        elif isinstance(one, ast.For) and isinstance(one.target, ast.Name):
            out.setdefault(one.target.id, "")
    return out


def _constants(case: dict, key: str) -> list:
    """The values its own code gives a key, in the file and beside it:
    `"change"` of `{"kind": "change", ...}` and of `x["kind"] = "change"`
    -- what kinds a step has, as the code says."""
    out = []
    # its own file first, then any of its code that names the key
    files = [ROOT / case["file"]] + [
        path for path in sorted((ROOT / "research").rglob("*.py"))
        if "__pycache__" not in path.parts]
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if f'"{key}"' not in text and f"'{key}'" not in text and \
                f"{key}=" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            pairs = []
            if isinstance(node, ast.Dict):
                pairs = list(zip(node.keys, node.values))
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Subscript):
                pairs = [(node.targets[0].slice, node.value)]
            elif isinstance(node, ast.keyword) and node.arg == key:
                pairs = [(ast.Constant(key), node.value)]
            for name, value in pairs:
                if isinstance(name, ast.Constant) and name.value == key \
                        and isinstance(value, ast.Constant) and \
                        isinstance(value.value, (str, int, bool)):
                    said = repr(value.value)
                    if said not in out:
                        out.append(said)
    return out[:8]


def _precedents(callee: str, name: str) -> list:
    """How the rest of its code calls a function: the expression each
    call gives the parameter `name` (by keyword) -- `kind=piece["kind"]`."""
    out = []
    for path in sorted((ROOT / "research").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if f"{callee}(" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        # where the parameter stands, of each function of that name: a call
        # gives it by place as often as by name (`_framing(piece["kind"])`)
        at = {one.args.args.index(arg) - (1 if one.args.args[0].arg in
                                          ("self", "cls") else 0)
              for one in ast.walk(tree)
              if isinstance(one, ast.FunctionDef) and one.name == callee
              for arg in one.args.args if arg.arg == name}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and \
                    ast.unparse(node.func).split(".")[-1] == callee:
                said = [ast.unparse(one.value) for one in node.keywords
                        if one.arg == name]
                said += [ast.unparse(node.args[one]) for one in at
                         if one < len(node.args)]
                for one in said:
                    if one not in out:
                        out.append(one)
    return out[:8]


def _filled(meaning: dict, scope: dict, seen: dict) -> str | None:
    """A call's arguments from what the function holds: each needed
    parameter by its name, else by its kind (written, or seen in runs);
    None where one is not held."""
    args = []
    for param in meaning["params"]:
        if param in ("self", "cls"):
            continue
        if param in scope:
            args.append(param)
            continue
        kind = meaning["kinds"].get(param) or next(iter(sorted(
            (seen.get(param) or {}).items(), key=lambda one: -one[1])),
            ("", 0))[0]
        same = [name for name, held in scope.items()
                if kind and held and held.split("[")[0] == kind.split("[")[0]]
        if same:
            args.append(same[0])
            continue
        return None
    return ", ".join(args)


def _bare(field: str) -> str:
    from research.v705 import semantics
    return semantics._bare(field)


def plans(case: dict, wanted: dict, source: str, held: dict) -> list:
    """Procedures planned back from the goal, as statements to put in:
    [(statements, why)] -- most direct first."""
    path = wanted.get("path") or ""
    scope = _scope(source)
    functions = held["functions"]
    seen_all = held.get("seen") or {}
    index = held["index"]
    own = case["function"].split(".")[-1]
    returns = [one for one in ast.walk(ast.parse(textwrap.dedent(source)))
               if isinstance(one, ast.Return) and one.value is not None]
    returned = ast.unparse(returns[-1].value) if returns else None
    field = _bare(path) if ("[" in path or "." in path) else ""
    root = path.split("[")[0].split(".")[0]
    candidates: list = []

    def callable_key(key: str) -> str | None:
        """How the function at `key` is called from the caller's file."""
        file, name = key.split("#")
        if "." in name:
            return None                             # a method: not planned
        return name if file == case["file"] else None

    # who changes the field the goal is of; who gives what it looks at
    givers = []
    if field:
        givers += [(key, "changes " + field)
                   for key in index["changes"].get(field, [])]
    wanted_kind = (wanted.get("wanted") or "")[:1]
    kind = {"[": "list", "{": "dict", "'": "str", '"': "str"}.get(
        wanted_kind)
    if kind:
        givers += [(key, "gives " + kind)
                   for key in index["gives"].get(kind, [])[:40]]
    for key, why in givers:
        name = callable_key(key)
        if not name or name == own:
            continue
        meaning = functions[key]
        args = _filled(meaning, scope, seen_all.get(key) or {})
        if args is None:
            # what it needs and nothing here holds: one step back, a
            # function of the file that gives it from what is held
            continue
        call = f"{name}({args})"
        if field and root in scope:
            candidates.append(([call], f"{name} {why}"))
        if returned and kind and why.startswith("gives"):
            candidates.append(([f"{returned} += {call}"] if returned.isidentifier()
                               else [f"return {call}"],
                               f"{name} {why}, added to what is returned"))
    # the plainest acts on the field the goal names, where no function does
    if field and root in scope:
        said = f"{root}{field}"
        try:
            got = ast.literal_eval(wanted.get("got") or "None")
            should = ast.literal_eval(wanted.get("wanted") or "None")
        except (ValueError, SyntaxError):
            got = should = None
        if isinstance(got, int) and isinstance(should, int) and \
                not isinstance(got, bool):
            step = "+= 1" if should > got else "-= 1"
            candidates.append(([f"{said} {step}"],
                               f"{said} moved on, as the goal has it"))
        if isinstance(should, list) and isinstance(got, list) and \
                len(should) == len(got) + 1:
            for name in [one for one in scope if one not in (root,)][:6]:
                candidates.append(([f"{root}.setdefault({field[1:-1]}, "
                                    f"[]).append({name})"],
                                   f"{name} kept in {said}"))
    # two steps: a function whose result another needs
    firsts = list(candidates)
    for statements, why in firsts[:10]:
        for more, also in firsts[:10]:
            if more is not statements:
                candidates.append((statements + more, f"{why}; {also}"))
    return candidates[:PLANS]


def _put(text: str, before: str, statements: list) -> list:
    """The file's text with the statements put in the function: before
    each return, and first -- [(text, where)]."""
    tree = ast.parse(textwrap.dedent(before))
    node = tree.body[0]
    lines = before.splitlines()
    out = []
    at = text.index(before)
    indent = text[text.rfind("\n", 0, at) + 1:at]
    spots = [(one.lineno, "before a return") for one in ast.walk(node)
             if isinstance(one, ast.Return)]
    first = node.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value,
                                                  ast.Constant):
        first = node.body[1] if len(node.body) > 1 else first
    spots.append((first.lineno, "first"))
    for line, where in spots:
        target = lines[line - 1]
        pad = target[:len(target) - len(target.lstrip())]
        if line > 1 and not pad:
            continue
        put = [pad + one for one in statements]
        if any(one.strip().startswith("return") for one in statements) and \
                where == "before a return":
            changed = lines[:line - 1] + put + lines[line:]
        else:
            changed = lines[:line - 1] + put + lines[line - 1:]
        made = "\n".join(changed)
        try:
            ast.parse(textwrap.dedent(made))
        except SyntaxError:
            continue
        out.append((text[:at] + made + text[at + len(before):], where))
    return out


# -- the goal said of the function itself ---------------------------------------

def _callee(call: str) -> str:
    """`carry` of `Q.carry(self.asked(...), ...)`."""
    import re
    found = re.match(r"\s*([\w\.]+)\s*\(", call or "")
    return found.group(1).split(".")[-1] if found else ""


def of_return(case: dict, row: dict, held: dict) -> str | None:
    """Where the goal's value is in what the function gives back: the
    rest of its path (`.value` of `found.value`, `` of `lines`) where the
    test's name was made by a call of the function, or of a function
    that calls it and gives back what it gives. None where it is not."""
    path = row.get("path") or ""
    own = case["function"].split(".")[-1]
    callee = _callee(row.get("made by") or "")
    if not path or not callee:
        return None
    root = path.split("[")[0].split(".")[0]
    if callee != own:
        callers = [key for key, one in held["functions"].items()
                   if key.split("#")[1].split(".")[-1] == callee and
                   any(name.split(".")[-1] == own for name in one["calls"])]
        if not callers:
            return None
    return path[len(root):]


def crashed(raised: str | None) -> dict | None:
    """What a test that never got to its checks needs, from how it fell:
    a parameter (`run() got an unexpected keyword argument 'timeout'`), an
    argument (`_framing() missing 1 required positional argument:
    'kind'`), a key (`KeyError: 'language'`), an attribute."""
    import re
    lines = [one.strip() for one in (raised or "").splitlines()
             if one.strip()]
    if not lines:
        return None
    last = lines[-1]
    found = re.search(r"(\w+)\(\) got an unexpected keyword argument "
                      r"'(\w+)'", last)
    if found:
        return {"needs": "parameter", "function": found.group(1),
                "name": found.group(2)}
    found = re.search(r"(\w+)\(\) missing \d+ required positional "
                      r"arguments?: '(\w+)'", last)
    if found:
        return {"needs": "argument", "function": found.group(1),
                "name": found.group(2)}
    found = re.match(r"KeyError: '([\w ]+)'", last)
    if found:
        return {"needs": "key", "name": found.group(1)}
    found = re.search(r"AttributeError: '\w+' object has no attribute "
                      r"'(\w+)'", last)
    if found:
        return {"needs": "attribute", "name": found.group(1)}
    return None


def _free(expression: str) -> set:
    """The names an expression reads."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return {"?"}
    import builtins
    return {one.id for one in ast.walk(tree) if isinstance(one, ast.Name)
            } - set(dir(builtins))


def _kind_of(said: str | None) -> str | None:
    """The kind of value a check wanted, as it said it (`'c'`: str)."""
    try:
        value = ast.literal_eval(said or "")
    except (ValueError, SyntaxError):
        return None
    return type(value).__name__


# -- designs: whole functions, changed --------------------------------------------

def _values(kind: str | None, scope: dict, case: dict, held: dict,
            near: str = "") -> list:
    """What could stand for a value of a kind, from what the function
    holds: the names of its scope (of that kind where known), and the
    calls of its file's functions that give that kind, their arguments
    filled from the scope -- names near `near` first."""
    out = []
    names = sorted(scope, key=lambda one: (near and near not in one, one))
    for name in names:
        written = scope[name]
        if not kind or not written or written.split("[")[0] == kind:
            out.append(name)
    for key, one in held["functions"].items():
        file, name = key.split("#")
        if file != case["file"] or "." in name or \
                name == case["function"].split(".")[-1]:
            continue
        gives = [one["returns"]] + one["gives"]
        if kind and not any(kind == (given or "").split("[")[0].split(" ")[0]
                            for given in gives):
            continue
        args = _filled(one, scope, (held.get("seen") or {}).get(key) or {})
        if args is not None:
            out.append(f"{name}({args})")
    near_first = sorted(out, key=lambda one: (near and near not in one,))
    return near_first[:12]


def _tree(source: str):
    return ast.parse(textwrap.dedent(source))


def _unparsed(tree) -> str:
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _calls_taking(source: str, name: str, case: dict, held: dict) -> list:
    """The calls in the function of its own code's functions that take a
    parameter named `name`."""
    takes = {key.split("#")[1].split(".")[-1] for key, one in
             held["functions"].items() if name in one["params"]}
    return [one for one in ast.walk(_tree(source))
            if isinstance(one, ast.Call) and
            (ast.unparse(one.func).split(".")[-1] in takes)]


def designs(case: dict, wanted: dict, source: str, held: dict) -> list:
    """Changed versions of the function, planned from the goal: [(source,
    why)] -- an interface a crash asks for, a value its return should
    give, a callback it should call, and the plans of `plans`."""
    out: list = []
    own = case["function"].split(".")[-1]
    scope = _scope(source)
    crash = wanted.get("crash")
    failing = wanted.get("failing") or {}

    def add(tree, why):
        made = _unparsed(tree)
        if made not in {one for one, _ in out}:
            out.append((made, why))

    if crash and crash["needs"] == "parameter" and crash["function"] == own:
        # a parameter it is asked to take: taken, and passed on to each
        # call inside that takes one of that name
        for passed in (False, True):
            tree = _tree(source)
            node = tree.body[0]
            node.args.kwonlyargs.append(ast.arg(crash["name"]))
            node.args.kw_defaults.append(ast.Constant(None))
            if passed:
                calls = _calls_taking(ast.unparse(node), crash["name"],
                                      case, held)
                if not calls:
                    continue
                for call in calls:
                    call.keywords.append(ast.keyword(
                        crash["name"], ast.Name(crash["name"], ast.Load())))
            add(tree, f"takes {crash['name']}"
                + (", passed on to what takes it" if passed else ""))
    if crash and crash["needs"] == "argument":
        # a call missing an argument: filled from what is held there, as
        # the rest of its code fills it, or a value its code gives that key
        at = next((one.lineno for one in ast.walk(_tree(source))
                   if isinstance(one, ast.Call) and
                   ast.unparse(one.func).split(".")[-1] ==
                   crash["function"]), None)
        here = _scope(source, at)
        offered = [one for one in _precedents(crash["function"],
                                              crash["name"])
                   if _free(one) <= set(here)] +             _constants(case, crash["name"]) +             _values(None, here, case, held, crash["name"])
        for value in list(dict.fromkeys(offered))[:10]:
            tree = _tree(source)
            calls = [one for one in ast.walk(tree)
                     if isinstance(one, ast.Call) and
                     ast.unparse(one.func).split(".")[-1] ==
                     crash["function"]]
            if not calls:
                break
            for call in calls:
                call.keywords.append(ast.keyword(
                    crash["name"], ast.parse(value, mode="eval").body))
            add(tree, f"{crash['function']} given {crash['name']}={value}")
    if crash and crash["needs"] in ("key", "attribute"):
        # a key of what it gives back: given, from what is held where the
        # dict is made, or a value its code gives that key
        dicts = [one for one in ast.walk(_tree(source))
                 if isinstance(one, ast.Dict)]
        here = _scope(source, dicts[0].lineno) if dicts else scope
        offered = _constants(case, crash["name"]) +             _values(None, here, case, held, crash["name"])
        for value in list(dict.fromkeys(offered))[:10]:
            tree = _tree(source)
            changed = False
            for one in ast.walk(tree):
                if isinstance(one, ast.Dict) and crash["needs"] == "key":
                    one.keys.append(ast.Constant(crash["name"]))
                    one.values.append(ast.parse(value, mode="eval").body)
                    changed = True
                    break
            if changed:
                add(tree, f"gives {crash['name']}: {value}")
    if failing:
        rest = of_return(case, failing, held)
        kind = _kind_of(failing.get("wanted"))
        if rest is not None:
            field = rest.lstrip(".") if rest.startswith(".") and \
                "[" not in rest else None
            last = max((one.lineno for one in ast.walk(_tree(source))
                        if isinstance(one, ast.Return)), default=None)
            here = _scope(source, last)
            offered = (_constants(case, field) if field else []) +                 _values(kind, here, case, held, field or "")
            for value in list(dict.fromkeys(offered))[:12]:
                tree = _tree(source)
                returns = [one for one in ast.walk(tree)
                           if isinstance(one, ast.Return) and
                           one.value is not None]
                if not returns:
                    break
                if field:
                    # the field of what it gives back, given the value
                    for ret in returns:
                        if isinstance(ret.value, ast.Call):
                            kept = [one for one in ret.value.keywords
                                    if one.arg != field]
                            ret.value.keywords = kept + [ast.keyword(
                                field, ast.parse(value, mode="eval").body)]
                    add(tree, f"what it gives back has {field} = {value}")
                elif rest == "":
                    for ret in returns:
                        ret.value = ast.parse(value, mode="eval").body
                    add(tree, f"gives back {value}")
        # a callback: the parameter the goal's value is filled through
        callbacks = [name for name in scope if name in
                     (held["functions"].get(f"{case['file']}#{case['function']}")
                      or {}).get("params", []) and name not in ("self",)]
        for name in callbacks:
            takers = [key for key, one in held["functions"].items()
                      if key.split("#")[0] == case["file"] and
                      name in one["params"] and name in one["calls"] and
                      "." not in key.split("#")[1] and
                      key.split("#")[1] != own]
            for key in takers:
                args = _filled(held["functions"][key], scope,
                               (held.get("seen") or {}).get(key) or {})
                if args is None:
                    continue
                call = f"{key.split('#')[1]}({args})"
                for statements in ([call],):
                    for spot in ("first", "before a return"):
                        made = _inserted(source, statements, spot)
                        if made:
                            out.append((made, f"{call}, which calls {name}"
                                        f" ({spot})"))
    for statements, why in plans(case, failing, source, held) \
            if failing else []:
        for spot in ("before a return", "first"):
            made = _inserted(source, statements, spot)
            if made:
                out.append((made, f"{why} ({spot})"))
    return out[:PLANS]


def _inserted(source: str, statements: list, spot: str) -> str | None:
    """The function with the statements put in: first, or before each
    return."""
    tree = _tree(source)
    node = tree.body[0]
    new = [ast.parse(one).body[0] for one in statements]

    class Before(ast.NodeTransformer):
        def generic_visit(self, one):
            super().generic_visit(one)
            for field in ("body", "orelse", "finalbody"):
                block = getattr(one, field, None)
                if not isinstance(block, list):
                    continue
                changed = []
                for statement in block:
                    if isinstance(statement, ast.Return):
                        changed += [ast.parse(ast.unparse(one_)).body[0]
                                    for one_ in new]
                    changed.append(statement)
                setattr(one, field, changed)
            return one
    if spot == "first":
        at = 1 if node.body and isinstance(node.body[0], ast.Expr) and \
            isinstance(node.body[0].value, ast.Constant) else 0
        node.body[at:at] = new
    else:
        Before().visit(node)
    try:
        return _unparsed(tree)
    except Exception:                               # noqa: BLE001
        return None


#: moves planned one after another at most, and the moves kept at each
DEPTH, BEAM = 3, 4


def _said(wanted: dict) -> str:
    """A goal as one line, to tell whether a move changed it."""
    if wanted.get("failing"):
        one = wanted["failing"]
        return f"{one['check']} {one.get('path')} {one['got']} "                f"{one.get('wanted')}"
    return json.dumps(wanted.get("crash"))


def deepen(case: dict, source: str, wanted: dict, held: dict, verify,
           old: str, name: str, row: dict, depth: int = 1,
           moves: tuple = ()) -> dict | None:
    """Means-ends, move by move: each design of the goal tried; one whose
    test passes is the answer; one that moves the goal -- the test gets
    past where it fell, to a check or a fall of its own -- is planned on
    from (`DEPTH` moves, `BEAM` kept at each). {moves, source} or None."""
    from research.v705 import inside
    moved = []
    for made_source, why in designs(case, wanted, source, held):
        text = commits._placed(made_source, case["before"], old)
        if text is None:
            continue
        said = verify(text)
        row.setdefault("tries", []).append(
            [depth, " / ".join(moves + (why,))[:160],
             (said or "holds")[:100]])
        if said is None:
            return {"moves": list(moves + (why,)), "source": made_source}
        if depth >= DEPTH or len(moved) >= BEAM:
            continue
        try:
            made = inside.changed_function(commits._current(case), text,
                                           filename=name)
            with commits.Put(commits._current(case), made):
                after = goal(case)
        except Exception:                           # noqa: BLE001
            continue
        after["crash"] = crashed(after["raised"])             if not after["failing"] else None
        if (after["failing"] or after["crash"]) and                 _said(after) != _said(wanted):
            moved.append((made_source, why, after))
    for made_source, why, after in moved:
        found = deepen(case, made_source, after, held, verify, old, name,
                       row, depth + 1, moves + (why,))
        if found:
            return found
    return None


def run(most: int = 0, which: str = "design") -> dict:
    """Each held-out bug (`design.cases`, or the history's measured ones):
    the goal from its test, the plans, each put in and judged by the
    tests and those beside them."""
    import importlib
    import linecache
    from research.v705 import design, inside
    held = json.loads((ROOT / "state" / "v705-semantics.json").read_text(
        encoding="utf-8"))
    if which == "design":
        found = design.cases()[:most or None]
    else:
        mined = {(one["sha"], one["function"]): one for one in
                 map(json.loads, commits.MINED.open(encoding="utf-8"))}
        found = [{**mined[(one["sha"], one["function"])],
                  "tests": one["fails"]}
                 for one in map(json.loads, commits.CHECKED.open(
                     encoding="utf-8")) if one["status"] == "bug"][
            :most or commits.MEASURED]
    counts = {"bugs": len(found), "goal": 0, "planned": 0, "fixed": 0}
    out = []
    for case in found:
        module = importlib.import_module(".".join(
            Path(case["file"]).with_suffix("").parts))
        helper = case.get("helper")
        kept_helper = getattr(module, helper) if helper else None
        disk = (ROOT / case["file"]).read_text(encoding="utf-8")
        old = disk.replace(commits.functions(disk)[case["function"]],
                           case["before"])
        if helper:
            old = design._without(old, case["helper now"])
        function = commits._current(case)
        name = str(ROOT / case["file"]) + commits.PLANTED
        linecache.cache[name] = (len(old), None, old.splitlines(True), name)
        row = {"function": case["function"], "helper": helper}
        try:
            if helper:
                delattr(module, helper)
            made = inside.changed_function(function, old, filename=name)
            with commits.Put(function, made):
                if "guards" not in case:
                    case["guards"] = commits._beside(case)
                wanted = goal(case)
                wanted["crash"] = crashed(wanted["raised"])                     if not wanted["failing"] else None
                row["goal"] = wanted["failing"]
                row["crash"] = wanted["crash"]
                if wanted["failing"]:
                    row["of return"] = of_return(case, wanted["failing"],
                                                 held)
                if wanted["failing"] or wanted["crash"]:
                    counts["goal"] += 1
                    verify = commits.verifier(case, commits._current(case))
                    found_ = deepen(case, case["before"], wanted, held,
                                    verify, old, name, row)
                    counts["planned"] += bool(row.get("tries"))
                    if found_:
                        row["fixed"] = found_
                    counts["fixed"] += bool(row.get("fixed"))
        except Exception as bad:                    # noqa: BLE001
            row["failed"] = f"{type(bad).__name__}: {bad}"[:200]
        finally:
            if helper:
                setattr(module, helper, kept_helper)
        out.append(row)
        print(json.dumps(row, default=repr)[:500], flush=True)
    OUT.write_text(json.dumps({**counts, "each": out}, indent=1,
                              default=repr), encoding="utf-8")
    print(json.dumps(counts))
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("run", "history"))
    parser.add_argument("--most", type=int, default=0)
    options = parser.parse_args(argv)
    run(options.most, "design" if options.job == "run" else "history")
    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(main())
