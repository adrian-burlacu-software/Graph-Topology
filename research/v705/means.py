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
import os
import sys
import textwrap
import time
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
        # a call, or a part of what one gives (`read(...)["examples"]`)
        if isinstance(node, ast.Assign) and isinstance(
                _steps(node.value)[0], ast.Call):
            for target in node.targets:
                for one in ast.walk(target):
                    if isinstance(one, ast.Name):
                        out[one.id] = ast.unparse(node.value)[:120]
    return out


def _steps(node) -> tuple:
    """An expression as what it starts from and the steps into it:
    `x[0]['args']` -- (Name x, [("index", 0), ("key", "args")]); a step
    not by a constant ends the steps there."""
    steps = []
    while True:
        if isinstance(node, ast.Subscript) and isinstance(
                node.slice, ast.Constant):
            kind = "index" if isinstance(node.slice.value, int) else "key"
            steps.insert(0, (kind, node.slice.value))
            node = node.value
        elif isinstance(node, ast.Attribute):
            steps.insert(0, ("attr", node.attr))
            node = node.value
        else:
            return node, steps


def fell_at(test: dict, raised: str) -> dict | None:
    """Where a test fell in its own lines, before a check could look --
    `examples[0]["args"]` an IndexError, the list empty -- read as a goal
    in part: the check that line is (`assertEqual(examples[0]["args"],
    [[1, 2], None])`), what it wants (a literal), and the steps into what
    the function gave back that it looks along (`read(...)["examples"]`,
    then `[0]["args"]`). None where it fell elsewhere."""
    import re
    path = str(ROOT / test["file"])
    frames = re.findall(r'File "([^"]+)", line (\d+)', raised or "")
    if not frames or Path(frames[-1][0]).resolve() != Path(path).resolve():
        return None
    line = int(frames[-1][1])
    holder, method = test["test"].split(".")
    tree = ast.parse((ROOT / test["file"]).read_text(encoding="utf-8"))
    cls = next((one for one in tree.body if isinstance(one, ast.ClassDef)
                and one.name == holder), None)
    function = next((one for one in (cls.body if cls else [])
                     if isinstance(one, ast.FunctionDef) and
                     one.name == method), None)
    if function is None:
        return None
    check = next((one for one in ast.walk(function)
                  if isinstance(one, ast.Call) and isinstance(
                      one.func, ast.Attribute) and one.func.attr in CHECKS
                  and one.lineno <= line <= one.end_lineno), None)
    if check is None:
        return None
    value_at, wanted_at = CHECKS[check.func.attr]
    if wanted_at is None or len(check.args) <= max(value_at, wanted_at):
        return None
    try:
        wanted = ast.literal_eval(check.args[wanted_at])
    except (ValueError, SyntaxError, TypeError):
        return None
    root, steps = _steps(check.args[value_at])
    if not isinstance(root, ast.Name):
        return None
    # the call the name was bound to, and its own steps
    made = None
    for one in ast.walk(function):
        if isinstance(one, ast.Assign) and one.lineno <= line and any(
                isinstance(target, ast.Name) and target.id == root.id
                for target in one.targets):
            made = one.value
    if made is None:
        return None
    call, before = _steps(made)
    if not isinstance(call, ast.Call):
        return None
    return {"check": check.func.attr, "path": ast.unparse(
        check.args[value_at]), "_fell": {"steps": before + steps,
                                         "wanted": wanted},
            "got": "(" + (raised or "").strip().splitlines()[-1][:80] + ")",
            "wanted": repr(wanted)[:SHOWN], "made by": ast.unparse(call),
            "failed": True}


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
            row = {"check": name, "path": path, "_raw": (value, wanted),
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
    raised = commits.LAST.get("failure", "")
    if failing is None and raised:
        # fallen in its own lines on the way to a check: that check, in part
        failing = fell_at(test, raised)
    return {"failing": failing, "kept": [one for one in seen
                                         if not one["failed"]],
            "raised": None if failing else raised[-300:]}


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
    found = re.search(r"(\w+)\(\) takes (\d+) positional arguments? but "
                      r"(\d+) (?:were|was) given", last)
    if found:
        return {"needs": "positional", "function": found.group(1),
                "count": int(found.group(3)) - int(found.group(2))}
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


def _passed(raised: str, function: str) -> list:
    """The values a failing call gave the function, read off the call as
    its traceback shows it (`read("pairs ...", "python")`): its literal
    arguments, in order; [] where it does not show one."""
    # the call as the traceback quotes it, not the error's own words
    # (`TypeError: read() takes ...`)
    lines = raised.rstrip().splitlines()
    raised = "\n".join(lines[:-1])
    at = raised.rfind(f"{function}(")
    if at < 0:
        return []
    depth, end = 0, None
    for index in range(at + len(function), len(raised)):
        char = raised[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        return []
    said = " ".join(line.strip() for line in raised[at:end].splitlines()
                    if not set(line.strip()) <= set("~^"))
    try:
        call = ast.parse(said, mode="eval").body
    except SyntaxError:
        return []
    out = []
    for one in getattr(call, "args", []):
        try:
            out.append(ast.literal_eval(one))
        except ValueError:
            out.append(None)
    return out


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
    if crash and crash["needs"] == "positional" and \
            crash["function"] == own:
        # more given than it takes: a parameter more, named as a key it
        # gives (`language`, once given) -- and, given, that key is it
        tree = _tree(source)
        node = tree.body[0]
        taken = {one.arg for one in node.args.args + node.args.kwonlyargs}
        keys = [one.value for one in ast.walk(tree)
                if isinstance(one, ast.Constant) and isinstance(
                    one.value, str) and one.value.isidentifier() and
                one.value not in taken]
        returned = _returned_name(source)
        # the key whose value in its code is what the failing call passes
        # first (`"python"` passed, `'language': 'python'` given)
        passed = _passed(wanted.get("raised") or "", own)[len(taken - {
            "self", "cls"}):]
        given = {}
        for one in ast.walk(tree):
            if isinstance(one, ast.Dict):
                for key_, value in zip(one.keys, one.values):
                    if isinstance(key_, ast.Constant) and \
                            isinstance(value, ast.Constant):
                        given[key_.value] = value.value
        keys = sorted(dict.fromkeys(keys), key=lambda one: (
            not any(_equal(given.get(one), value) for value in passed),))
        for key in keys[:6]:
            for sets in (False, True):
                tree = _tree(source)
                node = tree.body[0]
                node.args.args.append(ast.arg(key))
                node.args.defaults.append(ast.Constant(None))
                made = _unparsed(tree)
                if sets:
                    if not returned:
                        continue
                    made = _inserted(made, [
                        f"if {key} is not None:\n"
                        f"    {returned}[{key!r}] = {key}"],
                        "before a return")
                    if not made:
                        continue
                out.append((made, f"takes {key}" +
                            (f", which gives its {key}" if sets else "")))
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


# -- a leaf, by deduction -------------------------------------------------------------

#: the calls of the passing tests beside it taken as examples at most
KEPT_CALLS = 16
#: the calls beside it tried, one by one, as more of what one program
#: explains
SUBSET_TRIED = 8
#: the kinds of value an example's inputs keep (a store, an engine: not)
PLAIN_KINDS = (str, int, float, bool, list, tuple, dict, type(None))


def _calls_of(function, run) -> list:
    """Each call of the function while `run()` runs: (what it held when it
    gave back, what it gave back)."""
    code = function.__code__
    seen = []

    def profiler(frame, event, arg):
        if event == "return" and frame.f_code is code:
            seen.append((dict(frame.f_locals), arg))
    previous = sys.getprofile()
    sys.setprofile(profiler)
    try:
        run()
    except Exception:                               # noqa: BLE001
        pass
    finally:
        sys.setprofile(previous)
    return seen


def _at(value, path: list):
    for kind, step in path:
        value = value[step] if kind in ("key", "index") else \
            getattr(value, step)
    return value


def _located(value, root, depth: int = 2) -> list | None:
    """Where in what a function gave back a value is: [(kind, step)] --
    the same object first, else an equal one; None where it is not."""
    def walk(held, path, level, same):
        if (held is value) if same else _equal(held, value):
            return path
        if level >= depth:
            return None
        steps = []
        if isinstance(held, dict):
            steps = [("key", key, one) for key, one in held.items()]
        elif isinstance(held, (list, tuple)):
            steps = [("index", at, one) for at, one in enumerate(held)]
        elif hasattr(held, "__dict__") and not isinstance(held, type):
            steps = [("attr", key, one) for key, one in vars(held).items()
                     if not key.startswith("_")]
        for kind, step, one in steps[:60]:
            found = walk(one, path + [(kind, step)], level + 1, same)
            if found is not None:
                return found
        return None
    return walk(root, [], 0, True) or walk(root, [], 0, False)


def _equal(one, other) -> bool:
    try:
        return type(one) is type(other) and bool(one == other)
    except Exception:                               # noqa: BLE001
        return False


def _inputs(held: dict, returned_name: str | None) -> dict:
    """What a call held, as an example's inputs: plain values, not what
    it gives back."""
    return {name: value for name, value in held.items()
            if not name.startswith("_") and name != returned_name and
            isinstance(value, PLAIN_KINDS)}


def _returned_name(source: str) -> str | None:
    """The one name every return of the function gives back, if one."""
    names = {ast.unparse(one.value) for one in ast.walk(_tree(source))
             if isinstance(one, ast.Return) and one.value is not None}
    one = next(iter(names)) if len(names) == 1 else None
    return one if one and one.isidentifier() else None


def _library(function):
    """What a found program may call: its module's tables (a dict of
    strings) and functions of one argument; and the tables of strings
    anywhere in its code already loaded (`pytypes.BACK`)."""
    import inspect
    from research.v705.deduce import Library
    held = {}
    module = sys.modules.get(function.__module__)

    def table(value) -> bool:
        return isinstance(value, dict) and len(value) >= 3 and all(
            isinstance(key, str) and isinstance(one, str)
            for key, one in list(value.items())[:50])
    back = _calling(module, function) if module else set()
    for name, value in vars(module).items() if module else []:
        if name.startswith("__"):
            continue
        if table(value):
            held[name] = (value, None)
        elif inspect.isfunction(value) and value.__module__ == \
                module.__name__ and value is not function and \
                name not in back:
            try:
                needed = [one for one in inspect.signature(
                    value).parameters.values()
                    if one.default is inspect.Parameter.empty]
            except (TypeError, ValueError):
                continue
            if len(needed) == 1:
                held[name] = (value, None)
    for modname, other in list(sys.modules.items()):
        if not modname.startswith("research.") or other is module or \
                other is None:
            continue
        for name, value in list(vars(other).items()):
            if name.isupper() and table(value):
                alias = f"{modname.split('.')[-1]}_{name}"
                held.setdefault(alias, (value, f"from {modname} import "
                                               f"{name} as {alias}"))
    try:
        said = inspect.getsource(module) if module else ""
    except (OSError, TypeError):
        said = ""
    return Library(held, said)


def _calling(module, function) -> set:
    """The module's functions that call back into `function`, at once or
    through others of it: a program for it may not call them (`read`
    given `answered(text)`, which reads the request again)."""
    import inspect
    import types

    def names(code) -> set:
        out = set(code.co_names)
        for one in code.co_consts:
            if isinstance(one, types.CodeType):
                out |= names(one)
        return out
    calls = {}
    for name, value in vars(module).items():
        if inspect.isfunction(value) and value.__module__ == module.__name__:
            calls[name] = names(value.__code__)
        elif inspect.isclass(value) and value.__module__ == module.__name__:
            for one in vars(value).values():
                if inspect.isfunction(one):
                    calls.setdefault(value.__name__, set()).update(
                        names(one.__code__))
    # the name it is called by: its own, and its class's for a method
    target = {function.__name__, function.__qualname__.split(".")[0]}
    back, grown = set(), True
    while grown:
        grown = False
        for name, called in calls.items():
            if name not in back and called & (target | back):
                back.add(name)
                grown = True
    return back


#: what a deduced statement falling with says: it does not apply here (no
#: `def` line to search, a piece that is no literal), and what was there is
#: kept -- not a recursion or a stop
NOT_HERE = (LookupError, TypeError, ValueError, AttributeError, SyntaxError)


def _changes(program, inputs: dict, want, space: dict) -> bool:
    """Whether a program set where a call gives `want` now would change
    it: it gives something else, not falling as one that does not apply."""
    from research.v705.deduce import ANY, _same
    try:
        got = program(inputs, space)
    except NOT_HERE:
        return False
    except Exception:                               # noqa: BLE001
        return True
    return want is ANY or not _same(got, want)


def _in_part(steps: list, wanted, known: list):
    """What a check looking along `steps` and wanting `wanted` says of the
    value it starts from: an element at an index, a list that begins so
    (`deduce.Prefix`); a key, a dict with that key so and the others the
    values like it have (`known`), open (`deduce.ANY`)."""
    from research.v705.deduce import ANY, Prefix
    if not steps:
        return wanted
    (kind, step), rest = steps[0], steps[1:]
    if kind == "index" and isinstance(step, int) and step >= 0:
        inner = [one for value in known if isinstance(value, list)
                 for one in value]
        return Prefix([ANY] * step + [_in_part(rest, wanted, inner)])
    if kind == "key":
        dicts = [value for value in known if isinstance(value, dict)]
        keys = set().union(*(set(one) for one in dicts)) | {step}
        out = {key: ANY for key in keys}
        out[step] = _in_part(rest, wanted, [one[step] for one in dicts
                                            if step in one])
        return out
    return ANY


def _target(name: str, path: list) -> str:
    said = name
    for kind, step in path:
        said += f".{step}" if kind == "attr" else f"[{step!r}]"
    return said


def deduced(case: dict, source: str, wanted: dict, held: dict, verify,
            old: str, row: dict) -> list:
    """The leaf by deduction: which part of what the function gives back
    the check finds wrong; examples of it -- the failing call with what
    the check wants, the calls of the tests beside it with what they get
    now; a program found for them (`deduce.solve`), else one for the
    failing call and a condition telling the two apart; set before each
    return, judged by the tests. {moves, source} or None."""
    failing = wanted.get("failing") or {}
    raw = failing.get("_raw")
    fell = failing.get("_fell")
    if not raw and not fell:
        return []
    got, want = raw or (None, None)
    # the function as the moves so far made it, not as it was planted:
    # watched as it was, `read(text, 'python')` fell for want of the
    # language the first move gave it, and the call taken for the failing
    # one was the one before
    from research.v705 import inside
    function = commits._current(case)
    text = commits._placed(source, case["before"], old)
    try:
        made = inside.changed_function(function, text) if text else None
    except Exception:                               # noqa: BLE001
        made = None
    with commits.Put(function, made) if made is not None else \
            contextlib.nullcontext():
        return _deduced(case, source, wanted, made or function, raw, fell,
                        got, want, row)


def _deduced(case, source, wanted, function, raw, fell, got, want, row):
    from research.v705 import deduce
    calls = _calls_of(function, lambda: commits.passes(case["tests"][0]))
    name = _returned_name(source)
    # the parts the check finds wrong, and where each is in what it gave
    parts = []
    # the test's own calls before the one it fell after: what they gave
    # passed its checks, examples as good as those beside it
    earlier = []
    if fell and calls:
        steps = fell["steps"]
        cut = next((at for at, (kind, _) in enumerate(steps)
                    if kind == "index"), len(steps))
        path = steps[:cut]
        earlier = calls[:-1]
        known = []
        for _, returned in earlier:
            try:
                known.append(_at(returned, path))
            except Exception:                       # noqa: BLE001
                continue
        parts = [(path, _in_part(steps[cut:], fell["wanted"], known),
                  calls[-1][0])]
    for held_, returned in reversed(calls if raw else []):
        if isinstance(got, (tuple, list)) and isinstance(want, (tuple,
                                                                list)) \
                and len(got) == len(want):
            pairs = [(one, other) for one, other in zip(got, want)
                     if not _equal(one, other)]
        else:
            pairs = [(got, want)]
        found = [(_located(one, returned), other) for one, other in pairs]
        if found and all(path is not None for path, _ in found):
            parts = [(path, other, held_) for path, other in found]
            break
    if not parts:
        row.setdefault("deduction", []).append("no part of what it gives "
                                               "back is what the check "
                                               "looked at")
        return []
    beside = list(earlier)
    for test in case.get("guards") or []:
        beside += _calls_of(function, lambda test=test: commits.passes(test))
        if len(beside) >= KEPT_CALLS:
            break
    library = _library(function)
    statements = []
    for path, other, held_ in parts:
        failing_ex = [(_inputs(held_, name), other)]
        kept_ex, absent = [], []
        for one_held, returned in beside[:KEPT_CALLS]:
            try:
                kept_ex.append((_inputs(one_held, name),
                                _at(returned, path)))
            except (LookupError, AttributeError, TypeError):
                # a call that gives no such part is one it must stay unset
                # for (`yours` set on every request broke one that has none)
                absent.append(_inputs(one_held, name))
            except Exception:                       # noqa: BLE001
                continue
        found = deduce.solve(failing_ex + kept_ex, library)
        condition = None
        if found and absent:
            condition = deduce.condition(
                [inputs for inputs, _ in failing_ex + kept_ex], absent,
                library)
            if condition is None:
                found = []
        if not found:
            # what one program explains: the failing call and as many of
            # those beside it as it goes on explaining (the Python requests
            # of the test, not the TypeScript beside them) -- set apart from
            # the rest by a condition, the rest left as they were
            together = []
            # grown only where the failing call is known in part, and can
            # teach no rule alone: grown from a whole one, it took in a
            # TypeScript call whose signature is its second line
            for one in kept_ex[:SUBSET_TRIED] if deduce._partial(other) \
                    else []:
                if deduce.solve(failing_ex + together + [one], library):
                    together.append(one)
            apart = [one for one in kept_ex if one not in together] + [
                (inputs, deduce.ANY) for inputs in absent]
            found = deduce.solve(failing_ex + together, library)
            if found:
                # set apart only what it would change: where it gives what
                # is there now, or falls as one that does not apply (kept as
                # it was, `NOT_HERE`), nothing tells them apart
                space = library.space()
                apart = [(inputs, want) for inputs, want in apart
                         if _changes(found[0], inputs, want, space)]
            if found and apart:
                condition = deduce.condition(
                    [inputs for inputs, _ in failing_ex + together],
                    [inputs for inputs, _ in apart], library)
                if condition is None:
                    found = []
        if not found:
            row.setdefault("deduction", []).append(
                f"nothing found for {path} = {repr(other)[:60]}")
            if os.environ.get("V705_MISSED"):
                # the examples a miss was given, to look at
                with open(os.environ["V705_MISSED"], "a",
                          encoding="utf-8") as stream:
                    stream.write(repr(failing_ex + kept_ex) + "\n")
            return []
        program = found[0]
        target = _target(name or "_made", path) if path else None
        lines = [f"    {one}" for one in program.imports]
        if target:
            line = f"{target} = {program.expr}"
        else:
            line = f"return {program.expr}"
        # found of these examples, it holds where they do: where it does not
        # apply (no `def` line in a request: `re.search` finds none), what
        # was there is kept
        body = lines + [f"    {line}"]
        if condition:
            body = [f"    if {condition.expr}:"] + [
                "    " + one for one in body]
        # only what says "not here" (nothing found to index, a None to read
        # a group of): not a recursion or a stop, which kept swallowed ran
        # read -> answered -> read on without end
        statements.append("\n".join(["try:"] + body + [
            f"except ({', '.join(one.__name__ for one in NOT_HERE)}):",
            "    pass"]))
        row.setdefault("deduction", []).append(
            f"{_target('it', path)} = {program.expr}"
            + (f" if {condition.expr}" if condition else ""))
    if name is None and any(path for path, _, _ in parts):
        return []
    made = _inserted(source, statements, "before a return")
    if not made:
        return []
    # a move like any other: tried, and planned on from where the test
    # gets further (`deepen`)
    return [(made, "deduced " + "; ".join(row["deduction"][-len(parts):]))]


#: moves planned one after another at most, and the moves kept at each
DEPTH, BEAM = 5, 3


def _said(wanted: dict) -> str:
    """A goal as one line, to tell whether a move changed it."""
    if wanted.get("failing"):
        one = wanted["failing"]
        return (f"{one['check']} {one.get('path')} {one['got']} "
                f"{one.get('wanted')}")
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
    proposed = designs(case, wanted, source, held)
    if DEDUCED:
        proposed += deduced(case, source, wanted, held, verify, old, row)
    for made_source, why in proposed:
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
    # a goal no move of its own reaches: the code model told it exactly
    # -- the moves so far, what the check gets and should -- writes the
    # function as it stands now
    if LEAVES:
        for made_source, why in _leaf(case, source, wanted, moves):
            text = commits._placed(made_source, case["before"], old)
            if text is None:
                continue
            said = verify(text)
            row.setdefault("tries", []).append(
                [depth, " / ".join(moves + (why,))[:160],
                 (said or "holds")[:100]])
            if said is None:
                return {"moves": list(moves + (why,)), "source": made_source}
    return None


#: whether a goal no move reaches is given to the code model, and first
#: broken down by deduction
LEAVES = True
DEDUCED = True
#: whether one edit on the suspected lines is tried first (`repair`)
REPAIR = True
#: whether a change that holds is made as general as its tests let it be
GENERAL = True


def generalized(case: dict, found: dict, verify, old: str, row: dict
                ) -> dict:
    """A change that holds, made as general as the tests let it be: a
    constant it put in that a name of the failing call held is that name
    (`'language': 'python'` where `language` was 'python'); a statement it
    added that the tests hold without is taken out; a parameter it added
    that nothing reads is said (`interface only`). The tests judge each
    step; what they refuse stays as it was."""
    from research.v705 import inside
    source = found["source"]
    steps = []

    def holds(text_source: str) -> bool:
        text = commits._placed(text_source, case["before"], old)
        return text is not None and verify(text) is None

    # what the failing call held, by name, with the change in
    function = commits._current(case)
    text = commits._placed(source, case["before"], old)
    named = {}
    try:
        made = inside.changed_function(function, text)
        with commits.Put(function, made):
            calls = _calls_of(made, lambda: commits.passes(case["tests"][0]))
        for name, value in (calls[-1][0] if calls else {}).items():
            if isinstance(value, (str, int)) and not isinstance(value, bool) \
                    and value not in (0, 1, "") and not name.startswith("_"):
                named.setdefault(repr(value), []).append(name)
    except Exception:                               # noqa: BLE001
        pass
    before = _constants_of(case["before"])
    tree = _tree(source)
    for node in [one for one in ast.walk(tree)
                 if isinstance(one, ast.Constant)]:
        if repr(node.value) not in named or \
                before.get(repr(node.value), 0) >= _constants_of(
                    source).get(repr(node.value), 0):
            continue
        done = False
        for name in named[repr(node.value)]:
            kept = node.value
            # the name; else the name where it is given, what was there
            # where not (`language or 'python'`: read(text) gives none)
            for new in (ast.Name(id=name, ctx=ast.Load()),
                        ast.BoolOp(op=ast.Or(), values=[
                            ast.Name(id=name, ctx=ast.Load()),
                            ast.Constant(value=kept)])):
                parents = {child: one for one in ast.walk(tree)
                           for child in ast.iter_child_nodes(one)}
                if not _swapped(parents, node, new):
                    continue
                trying = _unparsed(tree)
                if holds(trying):
                    source = trying
                    steps.append(f"{kept!r} is {ast.unparse(new)}")
                    done = True
                    break
                _swapped(parents, new, node)
            if done:
                break
    # statements it put in (inserted lines all, not a line it changed: that
    # taken out is the code's own gone too), each taken out where the tests
    # hold without
    import difflib
    was = _unparsed(_tree(case["before"])).splitlines()
    for _ in range(12):
        source = _unparsed(_tree(source))
        tree = _tree(source)
        now = source.splitlines()
        inserted = set()
        for kind, _, _, start, end in difflib.SequenceMatcher(
                None, was, now, autojunk=False).get_opcodes():
            if kind == "insert":
                inserted |= set(range(start + 1, end + 1))
        added = [one for one in ast.walk(tree.body[0])
                 if isinstance(one, ast.stmt) and one is not tree.body[0]
                 and set(range(one.lineno, one.end_lineno + 1)) <= inserted]
        dropped = False
        for node in added:
            parents = {child: one for one in ast.walk(tree)
                       for child in ast.iter_child_nodes(one)}
            holder = parents.get(node)
            body = next((value for _, value in ast.iter_fields(holder)
                         if isinstance(value, list) and any(
                             one is node for one in value)), None)
            if body is None or len(body) < 2:
                continue
            at = [id(one) for one in body].index(id(node))
            body.pop(at)
            trying = _unparsed(tree)
            if holds(trying):
                source = trying
                steps.append("taken out, not needed: "
                             + ast.unparse(node).splitlines()[0][:60])
                dropped = True
                break
            body.insert(at, node)
        if not dropped:
            break
    # a parameter it added that nothing reads
    function_now = _tree(source).body[0]
    params_before = {one.arg for one in ast.walk(_tree(case["before"]))
                     if isinstance(one, ast.arg)}
    read = {one.id for one in ast.walk(function_now)
            if isinstance(one, ast.Name) and isinstance(one.ctx, ast.Load)}
    unread = [one.arg for one in function_now.args.args +
              function_now.args.kwonlyargs
              if one.arg not in params_before and one.arg not in read]
    if unread:
        row["interface only"] = unread
    row["generalized"] = steps
    return {"moves": list(found["moves"]) + steps, "source": source}


def _constants_of(source: str) -> dict:
    out: dict = {}
    for one in ast.walk(_tree(source)):
        if isinstance(one, ast.Constant):
            out[repr(one.value)] = out.get(repr(one.value), 0) + 1
    return out


def _swapped(parents: dict, node, new) -> bool:
    parent = parents.get(node)
    for name, value in ast.iter_fields(parent) if parent else []:
        if value is node:
            setattr(parent, name, new)
            parents[new] = parent
            return True
        if isinstance(value, list) and any(one is node for one in value):
            value[[id(one) for one in value].index(id(node))] = new
            parents[new] = parent
            return True
    return False

LEAF = ("A function of a Python project is being changed step by step "
        "toward a goal: {goal}\nThe steps taken so far: {moves}\nThe "
        "function as it is now:\n```python\n{source}\n```\nWhat its test "
        "checks now: {check}\nIn the same file, beside it:\n{reach}\nWrite "
        "the whole function changed so that the check holds -- the same "
        "name and parameters -- in one ```python block, nothing else.")


def _leaf(case: dict, source: str, wanted: dict, moves: tuple) -> list:
    """The code model's answers for a goal said exactly: [(source,
    why)]."""
    from research.v705 import coder
    one = wanted.get("failing")
    crash = wanted.get("crash")
    if one and one.get("wanted"):
        # what the check should get, told the model: not to be written in
        # as it is (`commits.copies_test`)
        case.setdefault("wanted values", []).append(one["wanted"])
    if one:
        check = (f"{one['check']} of {one.get('path') or 'a value'}"
                 f"{' (made by ' + one['made by'] + ')' if one.get('made by') else ''}"
                 f": it gets {one['got']}, it should be {one.get('wanted')}")
    elif crash:
        check = f"the test falls: {json.dumps(crash)} -- " \
                f"{(wanted.get('raised') or '').strip().splitlines()[-1]}"
    else:
        return []
    asked = LEAF.format(goal=case["message"],
                        moves="; ".join(moves) or "none",
                        source=textwrap.dedent(source), check=check,
                        reach="\n".join(commits.reach(case)))
    replies = coder.get().write([asked], longest=900, samples=coder.TRIES,
                                temperature=0.8)[0]
    out, seen = [], set()
    name = case["function"].split(".")[-1]
    for reply in replies:
        code = commits._code_of(reply)
        if not code:
            continue
        try:
            tree = ast.parse(textwrap.dedent(code))
        except SyntaxError:
            continue
        found = next((node for node in ast.walk(tree) if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
            node.name == name), None)
        if found is None:
            continue
        made = ast.unparse(found)
        if made not in seen:
            seen.add(made)
            out.append((made, f"the code model wrote it ({check[:60]})"))
    return out


def _case_key(case: dict) -> str:
    """What a case is, to know it again on a run taken up."""
    import hashlib
    return hashlib.sha1(f"{case['file']}#{case['function']}#"
                        f"{case['before']}".encode()).hexdigest()[:16]


def cases(which: str = "design", most: int = 0) -> list:
    """The bugs of a set: `design` (helpers taken out), `planted`
    (its own code made wrong by one edit, giving a wrong answer), or
    the history's measured ones -- each {file, function, before,
    tests, ...}."""
    from research.v705 import design
    if which == "design":
        found = design.cases()[:most or None]
    elif which == "planted":
        # its own code made wrong by one edit, held out, giving a wrong
        # answer (`planted`): the function as wrong is the one put back
        # judged as every other set is (`commits._beside`: the class's
        # tests and those anywhere that run it), not four of its covering
        from research.v705 import planted
        found = []
        for row in map(json.loads, planted.PLANTED.open(encoding="utf-8")):
            for bug in row["bugs"]:
                if row["held"] and planted._taught(row, bug) and \
                        planted.failing(bug) == "wrong":
                    tests = [{"file": one["file"], "test": one["test"]}
                             for one in bug["fails"]]
                    found.append({
                        "file": row["file"], "function": row["function"],
                        "before": bug["wrong"], "tests": tests,
                        "message": f"make {tests[0]['test']} pass"})
        found = found[:most or 30]
    else:
        mined = {(one["sha"], one["function"]): one for one in
                 map(json.loads, commits.MINED.open(encoding="utf-8"))}
        found = [{**mined[(one["sha"], one["function"])],
                  "tests": one["fails"]}
                 for one in map(json.loads, commits.CHECKED.open(
                     encoding="utf-8")) if one["status"] == "bug"][
            :most or commits.MEASURED]
    return found


def run(most: int = 0, which: str = "design", only: str = "") -> dict:
    """Each held-out bug (`design.cases`, or the history's measured ones):
    the goal from its test, the plans, each put in and judged by the
    tests and those beside them."""
    import importlib
    import linecache
    from research.v705 import design, inside, repair
    held = json.loads((ROOT / "state" / "v705-semantics.json").read_text(
        encoding="utf-8"))
    found = cases(which, most)
    if only:
        found = [one for one in found if one["function"] == only]
    counts = {"bugs": len(found), "goal": 0, "planned": 0, "fixed": 0}
    out = []
    # each bug's row kept as it is done: stopped and run again, it takes
    # up after the last one
    partial = OUT.with_suffix(".partial.jsonl")
    done = {}
    if partial.exists():
        for line in partial.open(encoding="utf-8"):
            one = json.loads(line)
            done[one["key"]] = one["row"]
    for case in found:
        key = _case_key(case)
        if key in done:
            row = done[key]
            counts["goal"] += bool(row.get("goal") or row.get("crash"))
            counts["planned"] += bool(row.get("tries"))
            counts["fixed"] += bool(row.get("fixed"))
            out.append(row)
            continue
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
        row = {"function": case["function"], "helper": helper, "key": key}
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
                    # a slip first: one edit on the lines the failing test
                    # runs most apart from the rest (`repair`); a design only
                    # where none holds
                    # how long each way took, to set beside what it gave
                    clock = time.time()
                    found_ = repair.search(case, commits._current(case), old,
                                           verify, row) if REPAIR else None
                    row["seconds"] = {"repair": round(time.time() - clock)}
                    if found_ is None:
                        clock = time.time()
                        found_ = deepen(case, case["before"], wanted, held,
                                        verify, old, name, row)
                        row["seconds"]["plan"] = round(time.time() - clock)
                    if found_ and GENERAL:
                        clock = time.time()
                        found_ = generalized(case, found_, verify, old, row)
                        row["seconds"]["general"] = round(time.time() - clock)
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
        with partial.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"key": key, "row": row},
                                    default=repr) + "\n")
        print(json.dumps(row, default=repr)[:500], flush=True)
    OUT.write_text(json.dumps({**counts, "each": out}, indent=1,
                              default=repr), encoding="utf-8")
    print(json.dumps(counts))
    return counts


def judge(which: str) -> dict:
    """Each change a run held (state/<out>) against the true function
    (`truth.real`): every call of it the tests make, and every quick test
    that runs it -- the held ones counted again as real or not, and kept
    in each row as `truth`."""
    from research.v705 import truth
    found = {_case_key(case): case for case in cases(which)}
    data = json.loads(OUT.read_text(encoding="utf-8"))
    counts = {"held": 0, "real": 0}
    for at, row in enumerate(data["each"]):
        # by its key; rows written before they kept one, by their place
        case = found.get(row.get("key")) or (
            list(found.values())[at] if len(data["each"]) == len(found)
            else None)
        if not row.get("fixed") or case is None:
            continue
        counts["held"] += 1
        true = commits._current(case)
        disk = (ROOT / case["file"]).read_text(encoding="utf-8")
        old = disk.replace(commits.functions(disk)[case["function"]],
                           case["before"])
        text = commits._placed(row["fixed"]["source"], case["before"], old)
        row["truth"] = truth.real(case, true, text) if text else {
            "real": False, "why": "not placed"}
        counts["real"] += bool(row["truth"]["real"])
        print(case["function"], json.dumps(row["truth"])[:300], flush=True)
    data["real"] = counts["real"]
    OUT.write_text(json.dumps(data, indent=1, default=repr),
                   encoding="utf-8")
    print(json.dumps(counts))
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("run", "history", "planted",
                                        "judge-run", "judge-history",
                                        "judge-planted"))
    parser.add_argument("--most", type=int, default=0)
    parser.add_argument("--out", default="")
    parser.add_argument("--only", default="",
                        help="the one function whose case is run")
    options = parser.parse_args(argv)
    global OUT
    if options.out:
        OUT = ROOT / "state" / options.out
    if options.job.startswith("judge-"):
        which = options.job.split("-", 1)[1]
        judge({"run": "design"}.get(which, which))
        return 0
    run(options.most, {"run": "design"}.get(options.job, options.job),
        options.only)
    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(main())
