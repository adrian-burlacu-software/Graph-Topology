"""Repair by one edit (v705): the lines a failing test runs that the tests
beside it run less, most suspected first (Ochiai over lines), and on each
the edits a slip is undone by -- an operator for its like, a constant one
off or another the code uses, a name for another bound there, an
attribute for another of its receiver, a call taken off or put round, two
arguments swapped, a condition turned round, a statement taken out --
each tried by the tests (`commits.verifier`).

The templates are the usual ones of program repair (mutation operators,
PraPR, TBar), not the edits bugs were planted with (`pyediting`): what is
planted is meant to be the slips programmers make, and these are the edits
that undo such slips, so the two meet -- measured by kind of bug, and by
history bugs no edit planted.

    search(case, function, old, verify, row)   {moves, source} or None
"""
from __future__ import annotations

import ast
import copy
import math
import sys
import textwrap

from research.v705 import commits

#: the lines tried, most suspected first, and the edits tried in all
LINES, MOST = 8, 400
#: how long a test run for its lines may take (they took under three
#: seconds when the coverage was gathered)
SPECTRUM_SECONDS = 20
#: the calls put round a value where one is missing
WRAPS = ("len", "str", "int", "list", "sorted", "set", "tuple", "bool",
         "abs", "min", "max", "sum", "any", "all", "reversed", "repr")
OPERATORS = {
    ast.Compare: [ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE],
    "in": [ast.In, ast.NotIn], "is": [ast.Is, ast.IsNot],
    ast.BinOp: [ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Div, ast.Mod],
}


# -- where: lines by spectrum ------------------------------------------------------

def _lines_run(function, run) -> set:
    """The lines of a function's code (and the code inside it) that run."""
    monitoring = sys.monitoring
    tool = next(one for one in range(6) if monitoring.get_tool(one) is None)
    monitoring.use_tool_id(tool, "v705-repair")
    seen: set = set()
    codes = [function.__code__]
    for code in codes:
        codes += [one for one in code.co_consts if hasattr(one, "co_code")]

    def line(code, number):
        seen.add(number)
        return monitoring.DISABLE
    try:
        monitoring.register_callback(tool, monitoring.events.LINE, line)
        for code in codes:
            monitoring.set_local_events(tool, code, monitoring.events.LINE)
        monitoring.restart_events()
        # watched as every test is: the bug itself may never end on one
        # (_split_top planted wrong grew to 20 GB here, unwatched)
        timer = commits._watched(SPECTRUM_SECONDS)
        try:
            run()
        except Exception:                           # noqa: BLE001
            pass
        finally:
            timer.cancel()
    finally:
        for code in codes:
            monitoring.set_local_events(tool, code, 0)
        monitoring.register_callback(tool, monitoring.events.LINE, None)
        monitoring.free_tool_id(tool)
    return seen


def suspected(case: dict, function) -> list:
    """[(line of the function's source, from 1, score)] most suspected
    first: run by the failing tests, less by those beside them."""
    first = function.__code__.co_firstlineno
    failing = [_lines_run(function, lambda test=test: commits.passes(test))
               for test in case["tests"]]
    passing = [_lines_run(function, lambda test=test: commits.passes(test))
               for test in case.get("guards") or []]
    scores = []
    for number in set().union(*failing) if failing else set():
        ef = sum(number in one for one in failing)
        ep = sum(number in one for one in passing)
        scores.append((number - first + 1,
                       ef / math.sqrt(len(failing) * (ef + ep))))
    scores.sort(key=lambda one: (-one[1], one[0]))
    return scores


# -- what: the edits on a line -------------------------------------------------------

def _parents(tree) -> dict:
    out = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            out[child] = node
    return out


def _replace(parents: dict, node, new) -> bool:
    parent = parents.get(node)
    if parent is None:
        return False
    for name, value in ast.iter_fields(parent):
        if value is node:
            setattr(parent, name, new)
            return True
        if isinstance(value, list) and any(one is node for one in value):
            value[[id(one) for one in value].index(id(node))] = new
            return True
    return False


def _bound(tree) -> list:
    """The names a function binds: its parameters and what it assigns."""
    function = tree.body[0]
    names = [one.arg for one in function.args.args + function.args.kwonlyargs
             if one.arg != "self"]
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.append(node.id)
    return list(dict.fromkeys(names))


def _edits(tree, node, parents, context: dict) -> list:
    """[(what was done, function(tree copy, node copy, parents) -> bool)]
    of one node."""
    out = []
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        op = type(node.ops[0])
        family = OPERATORS["in"] if op in OPERATORS["in"] else \
            OPERATORS["is"] if op in OPERATORS["is"] else \
            OPERATORS[ast.Compare]
        for other in family:
            if other is not op:
                out.append((f"{op.__name__} -> {other.__name__}",
                            lambda t, n, p, o=other: _set_op(n, o)))
        out.append(("sides swapped", lambda t, n, p: _swap_sides(n)))
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS[ast.BinOp]:
        for other in OPERATORS[ast.BinOp]:
            if other is not type(node.op):
                out.append((f"{type(node.op).__name__} -> {other.__name__}",
                            lambda t, n, p, o=other: _set_binop(n, o)))
    if isinstance(node, ast.BoolOp):
        other = ast.Or if isinstance(node.op, ast.And) else ast.And
        out.append((f"-> {other.__name__}",
                    lambda t, n, p, o=other: _set_boolop(n, o)))
        for at in range(len(node.values)):
            out.append((f"part {at} of it taken out",
                        lambda t, n, p, a=at: _drop_value(n, p, a)))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        out.append(("not taken off", lambda t, n, p: _replace(p, n,
                                                             n.operand)))
    if context.get("condition"):
        out.append(("turned round", lambda t, n, p: _replace(
            p, n, ast.UnaryOp(op=ast.Not(), operand=n))))
    if isinstance(node, ast.Constant) and not isinstance(parents.get(node),
                                                         ast.Expr):
        # (a docstring is no value)
        for value in _constants_for(node.value, context):
            out.append((f"{node.value!r} -> {value!r}",
                        lambda t, n, p, v=value: _set_constant(n, v)))
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and \
            node.id in context["bound"]:
        for other in context["bound"]:
            if other != node.id:
                out.append((f"{node.id} -> {other}",
                            lambda t, n, p, o=other: _set_name(n, o)))
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
        for other in context["attributes"]:
            if other != node.attr:
                out.append((f".{node.attr} -> .{other}",
                            lambda t, n, p, o=other: _set_attr(n, o)))
    if isinstance(node, ast.Call):
        if len(node.args) == 1 and not node.keywords:
            out.append((f"{ast.unparse(node.func)}(...) taken off",
                        lambda t, n, p: _replace(p, n, n.args[0])))
        for at in range(len(node.args) - 1):
            out.append((f"arguments {at} and {at + 1} swapped",
                        lambda t, n, p, a=at: _swap_args(n, a)))
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript, ast.Call)) \
            and isinstance(getattr(node, "ctx", ast.Load()), ast.Load) \
            and not isinstance(parents.get(node), ast.Call):
        for name in context["wraps"]:
            out.append((f"{name}(...) put round",
                        lambda t, n, p, f=name: _replace(p, n, ast.Call(
                            func=ast.Name(id=f, ctx=ast.Load()), args=[n],
                            keywords=[]))))
    return out


def _set_op(node, op) -> bool:
    node.ops = [op()]
    return True


def _set_binop(node, op) -> bool:
    node.op = op()
    return True


def _set_boolop(node, op) -> bool:
    node.op = op()
    return True


def _drop_value(node, parents, at) -> bool:
    values = [one for index, one in enumerate(node.values) if index != at]
    return _replace(parents, node, values[0] if len(values) == 1 else
                    ast.BoolOp(op=node.op, values=values))


def _swap_sides(node) -> bool:
    node.left, node.comparators = node.comparators[0], [node.left]
    return True


def _set_constant(node, value) -> bool:
    node.value = value
    return True


def _set_name(node, name) -> bool:
    node.id = name
    return True


def _set_attr(node, name) -> bool:
    node.attr = name
    return True


def _swap_args(node, at) -> bool:
    node.args[at], node.args[at + 1] = node.args[at + 1], node.args[at]
    return True


def _constants_for(value, context: dict) -> list:
    """What a constant may have been meant as: one off, the plain ones, a
    flag turned, another of its kind the code says."""
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, int):
        return list(dict.fromkeys(one for one in (
            value + 1, value - 1, 0, 1, -1) + tuple(
            one for one in context["numbers"] if one != value)
            if one != value))[:8]
    if isinstance(value, str):
        return [one for one in context["strings"] if one != value][:12]
    if value is None:
        return []
    return []


def _context(tree, module_source: str) -> dict:
    function = tree.body[0]
    strings = [one.value for one in ast.walk(function)
               if isinstance(one, ast.Constant) and isinstance(one.value, str)
               and len(one.value) < 40]
    numbers = [one.value for one in ast.walk(function)
               if isinstance(one, ast.Constant) and isinstance(one.value, int)
               and not isinstance(one.value, bool)]
    receivers: dict = {}
    try:
        module = ast.parse(module_source)
    except SyntaxError:
        module = function
    for node in ast.walk(module):
        if isinstance(node, ast.Attribute):
            receivers.setdefault(ast.unparse(node.value), set()).add(
                node.attr)
    attributes = sorted({one.attr for one in ast.walk(function)
                         if isinstance(one, ast.Attribute)})
    called = [one.func.id for one in ast.walk(function)
              if isinstance(one, ast.Call) and isinstance(one.func, ast.Name)
              and len(one.args) == 1]
    return {"bound": _bound(tree), "strings": list(dict.fromkeys(strings)),
            "numbers": list(dict.fromkeys(numbers)),
            "receivers": receivers, "attributes": attributes,
            "wraps": list(dict.fromkeys(called + list(WRAPS)))}


def candidates(source: str, lines: list, module_source: str = "") -> list:
    """[(line, what was done, the function's source with it done)] of the
    edits on these lines, in their order, at most `MOST`."""
    tree = ast.parse(textwrap.dedent(source))
    context = _context(tree, module_source)
    function = tree.body[0]
    conditions = {id(node.test) for node in ast.walk(function)
                  if isinstance(node, (ast.If, ast.While, ast.IfExp))} | {
        id(one) for node in ast.walk(function)
        if isinstance(node, ast.comprehension) for one in node.ifs}
    nodes = list(ast.walk(function))
    out = []
    for line in lines:
        for index, node in enumerate(nodes):
            if getattr(node, "lineno", None) != line or node is function:
                continue
            here = dict(context)
            here["condition"] = id(node) in conditions
            if isinstance(node, ast.Attribute):
                here["attributes"] = sorted(context["receivers"].get(
                    ast.unparse(node.value), set()))[:12]
            for what, edit in _edits(tree, node, _parents(tree), here):
                made = copy.deepcopy(tree)
                target = list(ast.walk(made.body[0]))[index]
                try:
                    if not edit(made, target, _parents(made)):
                        continue
                    said = ast.unparse(ast.fix_missing_locations(made))
                    compile(said, "<repair>", "exec")
                except Exception:                   # noqa: BLE001
                    continue
                out.append((line, what, said))
        for index, node in enumerate(nodes):
            if getattr(node, "lineno", None) != line:
                continue
            if isinstance(node, ast.stmt) and not isinstance(
                    node, (ast.FunctionDef, ast.Return)):
                made = copy.deepcopy(tree)
                target = list(ast.walk(made.body[0]))[index]
                parents = _parents(made)
                holder = parents.get(target)
                for name, value in ast.iter_fields(holder):
                    if isinstance(value, list) and any(
                            one is target for one in value) and len(value) > 1:
                        value.remove(target)
                        out.append((line, "statement taken out",
                                    ast.unparse(made)))
                        break
        if len(out) >= MOST * 10:
            break
    # the few edits of every line before the many of one (a value put
    # inside each of sixteen calls): operators, constants, a call off, two
    # swapped, a condition turned; then names, attributes, a statement out;
    # then calls put round
    order = {line: at for at, line in enumerate(lines)}
    out.sort(key=lambda one: (_group(one[1]), order[one[0]]))
    seen, kept = set(), []
    for one in out:
        if one[2] not in seen:
            seen.add(one[2])
            kept.append(one)
    return kept[:MOST]


def _group(what: str) -> int:
    if "put round" in what:
        return 2
    if what.startswith(".") or what == "statement taken out" or (
            " -> " in what and what.split(" -> ")[0].isidentifier() and
            what.split(" -> ")[0] not in _OPERATOR_NAMES):
        return 1
    return 0


_OPERATOR_NAMES = {one.__name__ for family in OPERATORS.values()
                   for one in family} | {"And", "Or"}


def search(case: dict, function, old: str, verify, row: dict) -> dict | None:
    """The first one-edit change on the most suspected lines the tests pass:
    {moves, source} or None; each try kept in the row."""
    import inspect
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError):
        source = textwrap.dedent(case["before"])
    if "guards" not in case:
        case["guards"] = commits._beside(case)
    lines = [line for line, score in suspected(case, function)
             if score > 0][:LINES]
    row["suspected"] = lines
    module = sys.modules.get(function.__module__)
    try:
        module_source = inspect.getsource(module) if module else ""
    except (OSError, TypeError):
        module_source = ""
    tried, held = 0, []
    for line, what, made in candidates(source, lines, module_source):
        text = commits._placed(made, case["before"], old)
        if text is None:
            continue
        tried += 1
        said = verify(text)
        if said is None:
            held.append((line, what, made, text))
            if len(held) >= KEPT:
                break
    if not held:
        row["repair"] = {"tried": tried}
        return None
    # of those the tests pass, the one that changes least of what it does
    # where the tests beside it call it, and on calls like those (fuzzed):
    # a slip undone changes the calls it went wrong on, `and` made `or`
    # changes many it did not
    calls = truth_recorded(function, case.get("guards") or [])
    changed = _changed(function, [one[3] for one in held], calls)
    fuzzed = _changed(function, [one[3] for one in held], _fuzzed(calls))
    best = min(range(len(held)), key=lambda at: (
        (fuzzed[at] + changed[at]) if RANK == "both" else
        changed[at] if RANK == "calls" else fuzzed[at] if RANK == "fuzz"
        else 0, at))
    line, what, made, _ = held[best]
    row["repair"] = {"tried": tried, "line": line, "edit": what,
                     "held": [{"edit": one[1], "line": one[0],
                               "calls": count, "fuzzed": fuzz,
                               "source": one[2]}
                              for one, count, fuzz in zip(held, changed,
                                                          fuzzed)]}
    return {"moves": [f"line {line}: {what}"], "source": made}


#: how the changes that hold are chosen among: by what they change of the
#: tests' calls and of fuzzed ones ("both"), of one of them, or the first
RANK = "both"
#: fuzzed calls made at most
FUZZED = 300


def truth_recorded(function, tests: list) -> list:
    from research.v705 import truth
    return truth._recorded(function, tests)


def _variants(value) -> list:
    """Values like one, a step from it."""
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, int):
        return [value + 1, value - 1, 0]
    if isinstance(value, float):
        return [value * 2, -value, 0.0]
    if isinstance(value, str):
        out = ["", value.upper(), value + value[-1:]]
        if len(value) > 1:
            out += [value[1:], value[:-1]]
        return out
    if isinstance(value, list):
        out = [[], list(reversed(value))]
        if value:
            out += [value[1:], value + value[:1]]
        return out
    if isinstance(value, tuple):
        return [tuple(one) for one in _variants(list(value))]
    if isinstance(value, dict) and value:
        return [{key: one for key, one in value.items() if key != drop}
                for drop in list(value)[:3]]
    return []


def _fuzzed(calls: list) -> list:
    """Calls like the tests' own, one argument a step from what it was."""
    import copy
    import random
    rng = random.Random(7057)
    out = []
    for held, gave in calls:
        for name, value in held.items():
            if name == "self":
                continue
            for other in _variants(value):
                one = copy.copy(held)
                one[name] = other
                out.append((one, None))
    rng.shuffle(out)
    return out[:FUZZED]


#: the changes the tests pass kept to choose among
KEPT = 6


def _changed(function, texts: list, calls: list) -> list:
    """Of each changed file's text, how many of these calls of the function
    it answers otherwise than the function does now."""
    import copy
    from research.v705 import inside, truth
    out = []
    for text in texts:
        try:
            made = inside.changed_function(function, text)
        except Exception:                           # noqa: BLE001
            out.append(len(calls) + 1)
            continue
        count = 0
        for held, _ in calls:
            now, again, mine = (copy.deepcopy(held) for _ in range(3))
            try:
                gave = truth._sealed(lambda: truth._called(function, now))
                twice = truth._sealed(lambda: truth._called(function, again))
            except Exception:                       # noqa: BLE001
                continue
            if not truth._alike(gave, twice):
                continue
            try:
                got = truth._sealed(lambda: truth._called(made, mine))
            except Exception as bad:                # noqa: BLE001
                got = bad
            count += not (truth._alike(got, gave) and truth._alike(mine,
                                                                   now))
        out.append(count)
    return out
