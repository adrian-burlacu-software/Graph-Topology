"""Design by examples (v705): what a missing helper must give, found from
the failing test, and a helper that gives it found by search -- not
guessed by a model. The designer, told a goal and a caller, wrote helpers
of the house's style that were none of what was needed (0 of 8); what is
needed is said by the test, as values.

1. **Where, and what** (`angels`): each place in the function a value is
   made -- what it returns, what it binds, what an `if` asks -- watched
   while the failing test runs, and at each, other values tried in turn,
   one occurrence at a time: the values of its own kind in scope there,
   and the plain ones (`True`, `None`, `0`, `[]`, ...). Where the failing
   test then passes, the place and the value it needed: an **angelic
   value** (SemFix, Angelix).
2. **Examples**: each time that place ran -- in the failing test, and in
   the tests that pass beside it -- what was in scope, and the value it
   must give: the angelic one where the test needed it, else the one it
   gave (the passing tests are kept passing).
3. **Search** (`search`): expressions grown from what was in scope and the
   functions in reach of the file, each evaluated on every example; one
   whose values are those of another is not grown again (observational
   equivalence: v696's forward trie, 2b); the first giving every value
   needed is the answer.
4. **The helper**: that expression made a function of what it reads,
   named for what it gives, put in beside the caller and called at the
   place -- judged by the tests and those beside them (`commits.verifier`).

    python -m research.v705.angelic run     the held-out design bugs
"""
from __future__ import annotations

import argparse
import ast
import itertools
import json
import sys
import time
from pathlib import Path

from research.v705 import commits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "state" / "v705-angelic.json"
#: places watched, occurrences of each tried, values tried at each
PLACES, OCCURRENCES, VALUES = 24, 4, 16
#: expressions the search grows at most, and how long it may take
GROWN, SECONDS = 40000, 90.0
#: the plain values every place is offered
PLAIN = (True, False, None, 0, 1, -1, "", (), [], {})


# -- watching the places -----------------------------------------------------

class Angels:
    """The watch over one function's places: each made value goes through
    `__angel__(place, value, scope)` -- recorded, and given back as it was
    unless an override says otherwise for that occurrence."""

    def __init__(self) -> None:
        self.seen: dict = {}            # place -> [(scope, value)]
        self.count: dict = {}
        self.override: dict = {}        # (place, occurrence) -> value
        self.recording = True

    def reset(self) -> None:
        self.count = {}

    def __call__(self, place: int, value, scope: dict):
        at = self.count.get(place, 0)
        self.count[place] = at + 1
        if self.recording:
            self.seen.setdefault(place, []).append(
                ({name: one for name, one in scope.items()
                  if not name.startswith("__")}, value))
        return self.override.get((place, at), value)


def instrumented(source: str) -> tuple:
    """The function's source with each place a value is made put through
    `__angel__`: (source, {place: (line, kind, what)})."""
    import textwrap
    tree = ast.parse(textwrap.dedent(source))
    places: dict = {}

    class Wrap(ast.NodeTransformer):
        def _wrap(self, node, kind, what):
            if len(places) >= PLACES or node is None:
                return node
            place = len(places)
            places[place] = (node.lineno, kind, what)
            return ast.Call(
                func=ast.Name("__angel__", ast.Load()),
                args=[ast.Constant(place), node,
                      ast.Call(ast.Name("locals", ast.Load()), [], [])],
                keywords=[])

        def visit_Return(self, node):
            self.generic_visit(node)
            if node.value is not None:
                node.value = self._wrap(node.value, "return",
                                        ast.unparse(node.value)[:60])
            return node

        def visit_Assign(self, node):
            self.generic_visit(node)
            node.value = self._wrap(node.value, "binds " + ", ".join(
                ast.unparse(one) for one in node.targets)[:40],
                ast.unparse(node.value)[:60])
            return node

        def visit_If(self, node):
            self.generic_visit(node)
            node.test = self._wrap(node.test, "if",
                                   ast.unparse(node.test)[:60])
            return node

        # a comprehension's names are its own: locals() there is not the
        # function's -- left unwatched
        def visit_Lambda(self, node):
            return node

    function = tree.body[0]
    function.body = [Wrap().visit(one) for one in function.body]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), places


def _pool(value, scope: dict) -> list:
    """The values tried at a place for one occurrence: the plain ones of
    its kind, those of its kind in scope, and for a flag its opposite."""
    out = []
    if isinstance(value, bool):
        return [not value]
    for one in list(scope.values()) + list(PLAIN):
        if one is value:
            continue
        if value is None or one is None or type(one) is type(value) or (
                isinstance(one, (int, float)) and
                isinstance(value, (int, float)) and
                not isinstance(one, bool)):
            try:
                if one == value:
                    continue
            except Exception:                       # noqa: BLE001
                continue
            if not any(one is seen for seen in out):
                out.append(one)
        if len(out) >= VALUES:
            break
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        out += [value + 1, value - 1]
    return out[:VALUES]


def _run(test: dict) -> bool:
    timer = commits._watched(commits.LONGEST)
    try:
        return commits.passes(test)
    except BaseException:                           # noqa: BLE001
        return False
    finally:
        timer.cancel()


def angels(case: dict, function, made_from: str, module) -> dict:
    """The places of the function, watched in the failing test and those
    beside it; of each, an occurrence in the failing test whose value
    changed makes it pass -- {place, occurrence, value, places, seen}."""
    source, places = instrumented(made_from)
    watch = Angels()
    space = dict(vars(module))
    space["__angel__"] = watch
    from research.v705 import inside
    owner = inside._owner(function)
    wrapped = (f"class _Here:\n" + "\n".join(
        "    " + line for line in source.splitlines())) \
        if owner is not module else source
    exec(compile(wrapped, "<angelic>", "exec"), space)  # noqa: S102
    made = vars(space["_Here"])[function.__name__] \
        if owner is not module else space[function.__name__]
    made = getattr(made, "__func__", made)
    if owner is not module:
        for name, cell in zip(made.__code__.co_freevars,
                              made.__closure__ or ()):
            if name == "__class__":
                cell.cell_contents = owner
    failing = case["tests"][0]
    found: dict = {"places": places}
    with commits.Put(function, made):
        # what each place gives, in the tests beside it: kept as it is
        watch.recording = True
        for test in case.get("guards") or []:
            watch.reset()
            _run(test)
        beside = {place: list(seen) for place, seen in watch.seen.items()}
        watch.seen = {}
        watch.reset()
        _run(failing)
        failing_seen = {place: list(seen)
                        for place, seen in watch.seen.items()}
        watch.recording = False
        for place in sorted(failing_seen, key=lambda one: -one):
            for at, (scope, value) in enumerate(
                    failing_seen[place][:OCCURRENCES]):
                for angel in _pool(value, scope):
                    watch.override = {(place, at): angel}
                    watch.reset()
                    if _run(failing):
                        found.update(place=place, occurrence=at,
                                     value=angel, was=value,
                                     failing=failing_seen[place],
                                     beside=beside.get(place, []))
                        watch.override = {}
                        return found
        watch.override = {}
    return found


# -- searching for what gives it ---------------------------------------------

def _same(one, other) -> bool:
    try:
        if type(one) is not type(other) and not (
                isinstance(one, (int, float)) and
                isinstance(other, (int, float))):
            return False
        return bool(one == other)
    except Exception:                               # noqa: BLE001
        return False


def _key(values) -> str:
    try:
        return repr(values)[:2000]
    except Exception:                               # noqa: BLE001
        return str(id(values))


UNARY = {"len({})": len, "not {}": lambda x: not x,
         "{}[0]": lambda x: x[0], "{}[-1]": lambda x: x[-1],
         "sorted({})": sorted, "set({})": set, "list({})": list,
         "sum({})": sum, "bool({})": bool, "str({})": str,
         "{} is None": lambda x: x is None,
         "{} is not None": lambda x: x is not None}
BINARY = {"{} + {}": lambda x, y: x + y, "{} - {}": lambda x, y: x - y,
          "{} * {}": lambda x, y: x * y, "{} == {}": lambda x, y: x == y,
          "{} != {}": lambda x, y: x != y, "{} < {}": lambda x, y: x < y,
          "{} <= {}": lambda x, y: x <= y, "{} in {}": lambda x, y: x in y,
          "{} and {}": lambda x, y: x and y, "{} or {}": lambda x, y: x or y,
          "max({}, {})": max, "min({}, {})": min,
          "{}[{}]": lambda x, y: x[y],
          "{}.get({})": lambda x, y: x.get(y)}


def search(examples: list, functions: dict) -> str | None:
    """An expression over the names in scope giving every example's value:
    [(scope, wanted)] -- grown bottom-up, kept apart by what it gives
    (observational equivalence), functions in reach called on what is in
    hand. None where nothing within `GROWN` or `SECONDS` does."""
    from research.v705 import tools
    tools._hook()
    tools.GUARDED.set(True)
    try:
        return _search(examples, functions)
    finally:
        tools.GUARDED.set(False)


def _search(examples: list, functions: dict) -> str | None:
    started = time.time()
    wanted = [value for _, value in examples]
    names = sorted(set.intersection(*(set(scope) for scope, _ in examples))) \
        if examples else []
    level: dict = {}                    # key of values -> (expression, values)
    grown = 0

    def offer(said: str, values: list) -> str | None:
        nonlocal grown
        grown += 1
        if any(isinstance(one, _Failed) for one in values):
            return None
        if all(_same(one, want) for one, want in zip(values, wanted)):
            return said
        key = _key(values)
        if key not in level:
            level[key] = (said, values)
        return None

    def each(function) -> list:
        out = []
        for scope, _ in examples:
            try:
                out.append(function(scope))
            except Exception as bad:                # noqa: BLE001
                out.append(_Failed(bad))
        return out

    for name in names:
        found = offer(name, each(lambda scope, name=name: scope[name]))
        if found:
            return found
    for constant in PLAIN:
        found = offer(repr(constant), [constant] * len(examples))
        if found:
            return found
    # attributes of what is in hand, one level
    for said, values in list(level.values()):
        first = values[0]
        for attribute in [one for one in dir(first)
                          if not one.startswith("_")][:30]:
            try:
                if callable(getattr(first, attribute)):
                    continue
            except Exception:                       # noqa: BLE001
                continue
            found = offer(f"{said}.{attribute}", [
                _attr(one, attribute) for one in values])
            if found:
                return found
    for _ in range(2):
        hand = list(level.values())
        for said, values in hand:
            for form, function in UNARY.items():
                found = offer(form.format(said), _apply(function, values))
                if found:
                    return found
                if grown > GROWN or time.time() - started > SECONDS:
                    return None
            for name, function in functions.items():
                found = offer(f"{name}({said})", _apply(function, values))
                if found:
                    return found
        for (one, first), (other, second) in itertools.product(hand,
                                                               repeat=2):
            for form, function in BINARY.items():
                found = offer(form.format(one, other),
                              _apply(function, first, second))
                if found:
                    return found
            for name, function in functions.items():
                found = offer(f"{name}({one}, {other})",
                              _apply(function, first, second))
                if found:
                    return found
            if grown > GROWN or time.time() - started > SECONDS:
                return None
    return None


class _Failed:
    def __init__(self, bad) -> None:
        self.bad = bad

    def __repr__(self) -> str:
        return f"<failed {type(self.bad).__name__}>"


def _attr(value, attribute):
    if isinstance(value, _Failed):
        return value
    try:
        return getattr(value, attribute)
    except Exception as bad:                        # noqa: BLE001
        return _Failed(bad)


def _apply(function, *columns) -> list:
    out = []
    for row in zip(*columns):
        if any(isinstance(one, _Failed) for one in row):
            out.append(row[0] if isinstance(row[0], _Failed) else row[1])
            continue
        try:
            out.append(function(*row))
        except Exception as bad:                    # noqa: BLE001
            out.append(_Failed(bad))
    return out


def in_reach(module, own: str) -> dict:
    """The functions of the caller's file a helper may call: those of its
    module, defined there, not the caller, taking one or two arguments."""
    import inspect
    out = {}
    for name, one in vars(module).items():
        if not inspect.isfunction(one) or one.__module__ != \
                module.__name__ or name == own or name.startswith("test"):
            continue
        try:
            needed = [p for p in inspect.signature(one).parameters.values()
                      if p.default is inspect.Parameter.empty and
                      p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        except (TypeError, ValueError):
            continue
        if 1 <= len(needed) <= 2:
            out[name] = one
    return out


# -- the helper, put in and called ---------------------------------------------

def helper_of(expression: str, place: tuple, own: str) -> tuple:
    """The expression as a function of the names it reads: (its source,
    its name, the call that stands for it)."""
    tree = ast.parse(expression, mode="eval")
    import builtins
    reads = sorted({one.id for one in ast.walk(tree)
                    if isinstance(one, ast.Name) and
                    isinstance(one.ctx, ast.Load)} - set(dir(builtins)))
    _, kind, what = place
    named = kind.split(" ", 1)[1].split(",")[0].strip() \
        if kind.startswith("binds ") else own.split(".")[-1].strip("_")
    name = f"_{named.replace('.', '_').strip('_')}_found"
    source = (f"def {name}({', '.join(reads)}):\n"
              f"    \"\"\"What `{what}` must be ({kind}) for its tests: "
              f"found from what they need.\"\"\"\n"
              f"    return {expression}\n")
    return source, name, f"{name}({', '.join(reads)})"


def _placed(text: str, caller_now: str, place: tuple, call: str) -> str | None:
    """The file's text with the place's expression, in the caller, put as
    the call (the first line holding it)."""
    line, kind, what = place
    lines = caller_now.splitlines()
    at = line - 1
    if at >= len(lines):
        return None
    tree = ast.parse(__import__("textwrap").dedent(caller_now))
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.Return) and node.lineno == line and \
                node.value is not None and kind == "return":
            target = node.value
        elif isinstance(node, ast.Assign) and node.lineno == line and \
                kind.startswith("binds"):
            target = node.value
        elif isinstance(node, ast.If) and node.lineno == line and \
                kind == "if":
            target = node.test
        if target is not None:
            said = ast.get_source_segment(
                __import__("textwrap").dedent(caller_now), target)
            if said and said in caller_now:
                changed = caller_now.replace(said, call, 1)
                return text.replace(caller_now, changed)
    return None


def run(most: int = 0) -> dict:
    """Each held-out design bug (`design.cases`), its helper taken out as
    well as the caller put back: the angelic value found, the examples,
    the search, and the helper put in and called -- judged by the tests
    and those beside them."""
    import importlib
    import linecache
    from research.v705 import design, inside
    found = design.cases()[:most or None]
    counts = {"bugs": len(found), "angel found": 0, "searched": 0,
              "fixed": 0}
    out = []
    for case in found:
        module = importlib.import_module(".".join(
            Path(case["file"]).with_suffix("").parts))
        kept_helper = getattr(module, case["helper"])
        disk = (ROOT / case["file"]).read_text(encoding="utf-8")
        old = design._without(disk.replace(commits.functions(disk)[
            case["function"]], case["before"]), case["helper now"])
        function = commits._current(case)
        name = str(ROOT / case["file"]) + commits.PLANTED
        linecache.cache[name] = (len(old), None, old.splitlines(True), name)
        row = {"function": case["function"], "helper": case["helper"]}
        try:
            delattr(module, case["helper"])
            made = inside.changed_function(function, old, filename=name)
            if "guards" not in case:
                with commits.Put(function, made):
                    case["guards"] = commits._beside(case)
            angel = angels(case, function, case["before"], module)
            row["places"] = {str(key): list(value) for key, value in
                             angel["places"].items()}
            if "place" in angel:
                counts["angel found"] += 1
                place = angel["places"][angel["place"]]
                row["angel"] = {"place": list(place),
                                "occurrence": angel["occurrence"],
                                "value": repr(angel["value"])[:200],
                                "was": repr(angel["was"])[:200]}
                examples = [(scope, angel["value"] if at ==
                             angel["occurrence"] else value)
                            for at, (scope, value) in
                            enumerate(angel["failing"])] + \
                    list(angel["beside"])
                expression = search(examples, in_reach(
                    module, case["function"].split(".")[-1]))
                counts["searched"] += expression is not None
                row["expression"] = expression
                if expression:
                    helper, helper_name, call = helper_of(
                        expression, place, case["function"])
                    text = _placed(old, case["before"], place, call)
                    if text is not None:
                        text = design._with(text, helper, case["function"])
                        space = dict(vars(module))
                        exec(compile(helper, name, "exec"),     # noqa: S102
                             space)
                        setattr(module, helper_name, space[helper_name])
                        try:
                            with commits.Put(function, made):
                                why = commits.verifier(
                                    case, commits._current(case))(text)
                        finally:
                            delattr(module, helper_name)
                        row["verified"] = why or "holds"
                        row["helper source"] = helper
                        counts["fixed"] += why is None
        except Exception as bad:                    # noqa: BLE001
            row["failed"] = f"{type(bad).__name__}: {bad}"[:200]
        finally:
            setattr(module, case["helper"], kept_helper)
        out.append(row)
        print(json.dumps(row)[:600], flush=True)
    OUT.write_text(json.dumps({**counts, "each": out}, indent=1,
                              default=repr), encoding="utf-8")
    print(json.dumps(counts))
    return counts


def run_planted(most: int = 20) -> dict:
    """The same, on held-out planted bugs that give a wrong answer (no
    helper taken out): where a value is all that is wrong, whether the
    angel and the search find it, and the change holds."""
    import json as json_
    import linecache
    from research.v705 import inside, planted
    covered = json_.loads(planted.COVERED.read_text(
        encoding="utf-8"))["functions"]
    bugs = [(row, bug) for row in map(json_.loads, planted.PLANTED.open(
        encoding="utf-8")) if row["held"] for bug in row["bugs"]
        if planted._taught(row, bug) and planted.failing(bug) == "wrong"]
    counts = {"bugs": 0, "angel found": 0, "searched": 0, "fixed": 0}
    out = []
    for row, bug in bugs[:most]:
        counts["bugs"] += 1
        case = {"file": row["file"], "function": row["function"],
                "tests": [{"file": one["file"], "test": one["test"]}
                          for one in bug["fails"]],
                "guards": [one for one in covered[row["place"]][:4]
                           if one["test"] not in {f["test"] for f in
                                                  bug["fails"]}]}
        text = (ROOT / row["file"]).read_text(encoding="utf-8")
        old = text.replace(row["right"], bug["wrong"])
        function = commits._current(case)
        name = str(ROOT / row["file"]) + commits.PLANTED
        linecache.cache[name] = (len(old), None, old.splitlines(True), name)
        found = {"function": row["function"], "kind": bug["kind"]}
        try:
            module = __import__("importlib").import_module(".".join(
                Path(row["file"]).with_suffix("").parts))
            made = inside.changed_function(function, old, filename=name)
            angel = angels(case, function, bug["wrong"], module)
            if "place" in angel:
                counts["angel found"] += 1
                place = angel["places"][angel["place"]]
                found["angel"] = [list(place), repr(angel["value"])[:100],
                                  repr(angel["was"])[:100]]
                examples = [(scope, angel["value"] if at ==
                             angel["occurrence"] else value)
                            for at, (scope, value) in
                            enumerate(angel["failing"])] + \
                    list(angel["beside"])
                expression = search(examples, in_reach(
                    module, row["function"].split(".")[-1]))
                found["expression"] = expression
                if expression:
                    counts["searched"] += 1
                    changed = _placed(old, bug["wrong"], place, expression)
                    if changed is not None:
                        with commits.Put(function, made):
                            why = commits.verifier(
                                case, commits._current(case))(changed)
                        found["verified"] = why or "holds"
                        counts["fixed"] += why is None
        except Exception as bad:                    # noqa: BLE001
            found["failed"] = f"{type(bad).__name__}: {bad}"[:160]
        out.append(found)
        print(json.dumps(found, default=repr)[:400], flush=True)
    print(json.dumps(counts))
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("run", "planted"))
    parser.add_argument("--most", type=int, default=0)
    options = parser.parse_args(argv)
    if options.job == "planted":
        run_planted(options.most or 20)
    else:
        run(options.most)
    return 0


if __name__ == "__main__":
    sys.exit(main())
