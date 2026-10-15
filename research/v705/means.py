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

def _scope(source: str) -> dict:
    """The names the function holds where it runs: its parameters, and
    what it binds -- {name: kind as said, or ""}."""
    tree = ast.parse(textwrap.dedent(source))
    node = tree.body[0]
    out = {one.arg: ast.unparse(one.annotation) if one.annotation else ""
           for one in node.args.posonlyargs + node.args.args +
           node.args.kwonlyargs}
    for one in ast.walk(node):
        if isinstance(one, ast.Assign):
            for target in one.targets:
                if isinstance(target, ast.Name):
                    out.setdefault(target.id, "")
        elif isinstance(one, (ast.For, ast.comprehension)) and \
                isinstance(one.target, ast.Name):
            out.setdefault(one.target.id, "")
    return out


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
                row["goal"] = wanted["failing"]
                row["raised"] = wanted["raised"]
                if wanted["failing"]:
                    counts["goal"] += 1
                    planned = plans(case, wanted["failing"], case["before"],
                                    held)
                    row["plans"] = len(planned)
                    counts["planned"] += bool(planned)
                    verify = commits.verifier(case, commits._current(case))
                    tried = 0
                    for statements, why in planned:
                        for text, where in _put(old, case["before"],
                                                statements):
                            tried += 1
                            if verify(text) is None:
                                row["fixed"] = {"why": why, "where": where,
                                                "statements": statements}
                                break
                        if row.get("fixed"):
                            break
                    row["tried"] = tried
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
