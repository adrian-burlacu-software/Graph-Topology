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
import itertools
import math
import sys
import textwrap

from research.v705 import commits

#: the lines tried, most suspected first, the edits ordered (by where they
#: stand and by the code model), and those of them tried by the tests
LINES, POOL, MOST = 40, 600, 400
#: the parts a condition may be given at a place, at most
CONJUNCTS = 30
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
    # `self` too: `pinned._cache` for `self._cache` was out of reach
    names = [one.arg for one in function.args.args + function.args.kwonlyargs]
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
        # a condition that is only its subject: what it was compared with
        # gone (`if 'json':` for `if kind == 'json':`; `if key:` for `if key
        # in self._cache:`)
        if isinstance(node, ast.Constant):
            for name in context["bound"][:12]:
                out.append((f"{name} == {node.value!r}",
                            lambda t, n, p, m=name: _replace(p, n, ast.Compare(
                                left=ast.Name(id=m, ctx=ast.Load()),
                                ops=[ast.Eq()], comparators=[n]))))
        if isinstance(node, (ast.Name, ast.Attribute)):
            for holder in context["containers"][:10]:
                if holder != ast.unparse(node):
                    out.append((f"in {holder}",
                                lambda t, n, p, h=holder: _replace(
                                    p, n, ast.Compare(
                                        left=n, ops=[ast.In()],
                                        comparators=[ast.parse(
                                            h, mode="eval").body]))))
    if context.get("condition") or isinstance(node, ast.BoolOp) and \
            isinstance(node.op, ast.And):
        # a part of the condition missing (`parts[0] == 'at'` for `len(parts)
        # == 3 and parts[0] == 'at' and parts[1] == thing`): one more part
        # of the names there, each made one edit
        for said in context["conjuncts"][:CONJUNCTS]:
            out.append((f"and {said}",
                        lambda t, n, p, s=said: _replace(p, n, ast.BoolOp(
                            op=ast.And(), values=(
                                list(n.values) if isinstance(n, ast.BoolOp)
                                and isinstance(n.op, ast.And) else [n])
                            + [ast.parse(s, mode="eval").body]))))
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) \
            and not (isinstance(parents.get(node), ast.Call) and
                     parents[node].func is node):
        # a method not called (`K.category` for `K.category(one.name)`)
        for name in context["names"][:10]:
            out.append((f"called with {name}",
                        lambda t, n, p, m=name: _replace(p, n, ast.Call(
                            func=n, args=[ast.parse(m, mode="eval").body],
                            keywords=[]))))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
            node.func.id in ("self", "cls"):
        # the object called for one of its methods (`self(arrow, scope)`
        # for `self._lambda(arrow, scope)`)
        for method in context["methods"][:16]:
            out.append((f"self.{method}(...)",
                        lambda t, n, p, m=method: _set_callee(n, m)))
    if context.get("joined") and (isinstance(node, ast.IfExp) or isinstance(
            node, ast.Constant) and isinstance(node.value, str)):
        # a piece of a string left off its end (`'I-'` for `'I-' + name`)
        for name in context["names"][:8]:
            out.append((f"+ {name}",
                        lambda t, n, p, m=name: _replace(p, n, ast.BinOp(
                            left=n, op=ast.Add(),
                            right=ast.parse(m, mode="eval").body))))
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
        # a parameter the callee takes that the call leaves to its default,
        # given what is in hand by its name (`checker()` for
        # `checker(spec.language)`, `P.library()` for
        # `P.library(language=spec.language)`): the plumbing a feature
        # wants threaded through code already there
        for name, said in _unpassed(node, context):
            out.append((f"{name}={said} passed",
                        lambda t, n, p, k=name, s=said: _pass(n, k, s)))
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


def _callee(node, context: dict):
    """What a call calls, found in the module as it runs (or the class of
    the method being changed, for `self.m(...)`); None where not."""
    namespace = context.get("namespace") or {}
    func = node.func
    if isinstance(func, ast.Name):
        return namespace.get(func.id)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id == "self" and context.get("owner") is not None:
            return getattr(context["owner"], func.attr, None)
        base = namespace.get(func.value.id)
        return getattr(base, func.attr, None) if base is not None else None
    return None


def _unpassed(node, context: dict) -> list:
    """[(parameter, what in hand to give it)]: the parameters the callee
    takes that this call does not give, each with what is in hand by its
    name -- the name itself, or a field so named of a name (as the module
    reads that field off a receiver so named)."""
    import inspect
    target = _callee(node, context)
    if target is None or not callable(target):
        return []
    try:
        parameters = list(inspect.signature(target).parameters.values())
    except (TypeError, ValueError):
        return []
    given = {one.arg for one in node.keywords if one.arg}
    positional = [one for one in parameters if one.kind in (
        one.POSITIONAL_ONLY, one.POSITIONAL_OR_KEYWORD)]
    given |= {one.name for one in positional[:len(node.args)]}
    out = []
    for one in parameters:
        if one.name in given or one.name in ("self", "cls") or one.kind in (
                one.VAR_POSITIONAL, one.VAR_KEYWORD, one.POSITIONAL_ONLY):
            continue
        if one.name in context["bound"]:
            out.append((one.name, one.name))
        for holder in context["bound"]:
            if one.name in context["receivers"].get(holder, set()):
                out.append((one.name, f"{holder}.{one.name}"))
    return out[:8]


def _pass(node, name: str, said: str) -> bool:
    node.keywords.append(ast.keyword(arg=name, value=ast.parse(
        said, mode="eval").body))
    return True


def _set_callee(node, method) -> bool:
    node.func = ast.Attribute(value=node.func, attr=method, ctx=ast.Load())
    return True


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
            if one != value))[:12]
    if isinstance(value, float):
        return [one for one in dict.fromkeys(context["numbers"])
                if one != value][:8]
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
    # the numbers of the whole module: `-1` meant `0.5`, said elsewhere
    numbers += [one.value for one in ast.walk(module)
                if isinstance(one, ast.Constant) and isinstance(
                    one.value, (int, float)) and not isinstance(
                    one.value, bool)][:40]
    # what holds things, as the function says them (`self._cache`)
    containers = [ast.unparse(one) for one in ast.walk(function)
                  if isinstance(one, (ast.Attribute, ast.Name)) and
                  isinstance(one.ctx, ast.Load) and
                  isinstance(_parents(tree).get(one), ast.Subscript)]
    # names a call or a piece may take: bound, and a comprehension's own
    names = _bound(tree) + [one.id for node in ast.walk(function)
                            if isinstance(node, ast.comprehension)
                            for one in ast.walk(node.target)
                            if isinstance(one, ast.Name)]
    # and their fields, as the module reads them off a receiver so named
    names += [f"{one}.{field}" for one in list(names)
              for field in sorted(receivers.get(one, set()))[:4]]
    methods = sorted(receivers.get("self", set()))
    # a field compared with a value somewhere in the module
    compared = list(dict.fromkeys(
        (one.left.attr, one.comparators[0].value) for one in ast.walk(module)
        if isinstance(one, ast.Compare) and len(one.ops) == 1 and isinstance(
            one.ops[0], (ast.Eq, ast.NotEq)) and isinstance(
            one.left, ast.Attribute) and isinstance(one.comparators[0],
                                                    ast.Constant)
        and isinstance(one.comparators[0].value, (str, int))))
    # parts a condition may lack: what it indexes counted, a part of it
    # compared with a name, two names compared
    subscripts = list(dict.fromkeys(
        ast.unparse(one) for one in ast.walk(function)
        if isinstance(one, ast.Subscript) and isinstance(one.ctx, ast.Load)
        and len(ast.unparse(one)) < 30))
    indexed = list(dict.fromkeys(
        ast.unparse(one.value) for one in ast.walk(function)
        if isinstance(one, ast.Subscript) and isinstance(one.value, ast.Name)))
    plain = [one for one in _bound(tree) if one != "self"][:8]
    conjuncts = [f"len({one}) == {number}" for one in indexed[:4]
                 for number in sorted({abs(n) for n in numbers if
                                       isinstance(n, int) and n < 8}
                                      | {1, 2, 3})[:4]]
    conjuncts += [f"{one} == {name}" for one in subscripts[:6]
                  for name in plain if name not in one]
    conjuncts += [f"{one} == {other}" for one, other in itertools.combinations(
        plain, 2)][:12]
    return {"bound": _bound(tree), "strings": list(dict.fromkeys(strings)),
            "numbers": list(dict.fromkeys(numbers)),
            "receivers": receivers, "attributes": attributes,
            "containers": list(dict.fromkeys(containers)),
            "names": list(dict.fromkeys(names)), "methods": methods,
            "conjuncts": list(dict.fromkeys(conjuncts)),
            "compared": compared,
            # the keys the module subscripts with, as strings
            "keys": list(dict.fromkeys(
                one.slice.value for one in ast.walk(module)
                if isinstance(one, ast.Subscript) and isinstance(
                    one.slice, ast.Constant) and isinstance(
                    one.slice.value, str)))[:60],
            "wraps": list(dict.fromkeys(called + list(WRAPS)))}


#: helpers offered at a place at most
HELPERS = 8


def _helpers(function, context: dict) -> list:
    """[(name, its arguments as this function would give them)]: the
    module's own functions this one does not call yet whose parameters
    without a default it holds by name."""
    import inspect
    namespace = context.get("namespace") or {}
    module = namespace.get("__name__")
    called = {ast.unparse(one.func) for one in ast.walk(function)
              if isinstance(one, ast.Call)}
    out = []
    for name, value in namespace.items():
        if not inspect.isfunction(value) or value.__module__ != module or                 name == function.name or name in called:
            continue
        try:
            needed = [one.name for one in inspect.signature(
                value).parameters.values()
                if one.default is inspect.Parameter.empty and one.kind in (
                    one.POSITIONAL_ONLY, one.POSITIONAL_OR_KEYWORD)]
        except (TypeError, ValueError):
            continue
        if needed and all(one in context["bound"] for one in needed):
            out.append((name, needed))
    return out


def declared(annotation, namespace: dict) -> list:
    """The fields a parameter's class declares (`bug: Bug`: name, source,
    ..., language), looked up in the module's namespace; none where it is
    not found."""
    if annotation is None:
        return []
    said = annotation.value if isinstance(annotation, ast.Constant) and         isinstance(annotation.value, str) else ast.unparse(annotation)
    found = namespace.get(said.split(".")[0]) if said else None
    for part in said.split(".")[1:]:
        found = getattr(found, part, None)
    if not isinstance(found, type):
        return []
    out = []
    for one in reversed(found.__mro__):
        out += list(getattr(one, "__annotations__", {}))
    return list(dict.fromkeys(out))


def _copy(tree):
    """A copy of a function's tree to edit: parsed again from the text it
    was parsed from, where it keeps it (`candidates`) -- the same tree, made
    by the parser many times faster than `deepcopy` copies one."""
    text = getattr(tree, "text", None)
    return ast.parse(text) if text is not None else copy.deepcopy(tree)


def _dispatched(tree, helpers: list, context: dict) -> list:
    """`if <a case>: return <helper>(...)` first in the function: the case
    a field compared with a value as the module compares it elsewhere
    (`x.language == 'python'` said somewhere is what makes `bug.language
    == 'python'` a case)."""
    function = tree.body[0]
    at = 1 if function.body and isinstance(function.body[0], ast.Expr) and         isinstance(getattr(function.body[0], "value", None), ast.Constant)         else 0
    line = function.body[at].lineno if len(function.body) > at else         function.lineno
    # of what the function is given: all there is where the case is put,
    # first in it (`node.kind` of a loop's `node` cannot be asked there) --
    # and what each has, as the module reads it off one so named or as its
    # class declares it (`bug: Bug`, `Bug.language`)
    given = function.args.posonlyargs + function.args.args +         function.args.kwonlyargs
    cases = [f"{holder.arg}.{field} == {value!r}"
             for holder in given
             for field, value in context.get("compared", [])
             if field in context["receivers"].get(holder.arg, set()) | set(
                 declared(holder.annotation, context.get("namespace", {})))
             ][:12]
    out = []
    for name, args in helpers[:HELPERS]:
        for case in cases:
            made = _copy(tree)
            made.body[0].body.insert(at, ast.parse(
                f"if {case}:" + chr(10) + f"    return {name}({', '.join(args)})"
            ).body[0])
            out.append((line, f"if {case}: return {name}(...)",
                        ast.unparse(made)))
    return out


def _copied(tree, context: dict) -> list:
    """`if 'k' in a: b['k'] = a['k']` after the last statement that stores
    into `b`, in the block it stands in: a field the function reads off
    one dict carried into the one it fills (`if 'file' in node:
    where['file'] = node['file']` beside its own `if 'said' in node:
    where['said'] = node['said']`) -- `k` a key the module subscripts
    somewhere that this function does not store into `b` yet."""
    function = tree.body[0]

    def key(node):
        return node.slice.value if isinstance(node.slice, ast.Constant) and             isinstance(node.slice.value, str) else None
    read, stored = set(), {}
    for node in ast.walk(function):
        if isinstance(node, ast.Subscript) and isinstance(
                node.value, ast.Name) and key(node):
            if isinstance(node.ctx, ast.Store):
                stored.setdefault(node.value.id, set()).add(key(node))
            else:
                read.add(node.value.id)
    # the last statement of any block storing into each dict, by its place
    # in the walk: (the block's index path, its position)
    last = {}
    for holder in ast.walk(function):
        for field, body in ast.iter_fields(holder):
            if not isinstance(body, list):
                continue
            for at, statement in enumerate(body):
                if not isinstance(statement, ast.stmt):
                    continue
                for node in ast.walk(statement):
                    if isinstance(node, ast.Subscript) and isinstance(
                            node.ctx, ast.Store) and isinstance(
                            node.value, ast.Name):
                        last.setdefault(node.value.id, {})[id(holder)] = (
                            statement.lineno, at, field, id(holder))
    keys = context.get("keys", [])
    nodes = list(ast.walk(function))
    out = []
    for into, already in stored.items():
        # in the block of the last store, and beside the statement holding
        # it in the block around (`if 'said' in node:` stands beside its own)
        for line, at, field, holder_id in list(last.get(into, {}).values())[
                -2:]:
            index = next(number for number, one in enumerate(nodes)
                         if id(one) == holder_id)
            out += _copies(tree, index, field, at, line, into, read, keys,
                           already)[:COPIED - len(out)]
            if len(out) >= COPIED:
                return out
    return out


#: the copies of a field made at most (made all and cut after, a function
#: reading many dicts made hundreds)
COPIED = 120


def _copies(tree, index, field, at, line, into, read, keys, already) -> list:
    """Each key copied from each dict read into `into`, put in at `at` of
    the `field` block of the walk's `index`th node."""
    out = []
    for source in sorted(read - {into}):
        for name in [one for one in keys if one not in already][:30]:
            if len(out) >= COPIED:
                return out
            made = _copy(tree)
            holder = list(ast.walk(made.body[0]))[index]
            getattr(holder, field).insert(at + 1, ast.parse(
                f"if {name!r} in {source}:" + chr(10) +
                f"    {into}[{name!r}] = {source}[{name!r}]").body[0])
            out.append((line, f"{into}[{name!r}] = {source}[{name!r}]",
                        ast.unparse(made)))
    return out


def candidates(source: str, lines: list, module_source: str = "",
               where: dict | None = None) -> list:
    """[(line, what was done, the function's source with it done)] of the
    edits on these lines, in their order, at most `MOST`."""
    text = textwrap.dedent(source)
    tree = ast.parse(text)
    tree.text = text
    context = _context(tree, module_source)
    # the module as it runs and the method's class: what calls call
    context.update(where or {})
    function = tree.body[0]
    conditions = {id(node.test) for node in ast.walk(function)
                  if isinstance(node, (ast.If, ast.While, ast.IfExp))} | {
        id(one) for node in ast.walk(function)
        if isinstance(node, ast.comprehension) for one in node.ifs}
    nodes = list(ast.walk(function))
    # what each node is in, walked once: the tree is never changed here (each
    # edit is made on a copy), and walked again for each node of a long
    # function, it was most of a change step (`page.replies`: 400 lines)
    parents = _parents(tree)

    everything = list(ast.walk(tree))
    at_of = {id(node): at for at, node in enumerate(everything)}
    above = [at_of[id(parents[node])] if node in parents else None
             for node in everything]

    def fresh():
        # a copy to edit (`_copy`), its nodes and what each is in: the same
        # parse, so the same walk -- what each is in found by place, not by
        # walking it again (3,000 walks of `page.replies`: a minute)
        made = ast.parse(text)
        nodes_ = list(ast.walk(made))
        return made, nodes_, {node: nodes_[up] for node, up in
                              zip(nodes_, above) if up is not None}
    out = []
    # a feature's own shape: a case handed to a helper of the module that
    # takes what this function holds -- `if bug.language == 'python':
    # return _repair_python(bug, budget, rewrite)`; and a default given by
    # one (`... if name else _super(...)` for `... else None`)
    helpers = _helpers(function, context)
    out += _dispatched(tree, helpers, context)
    out += _copied(tree, context)
    for index, node in enumerate(nodes):
        if isinstance(node, ast.Constant) and node.value is None and                 getattr(node, "lineno", None) in lines:
            for name, args in helpers[:HELPERS]:
                made, made_nodes, made_parents = fresh()
                target = made_nodes[index + 1]
                call = ast.parse(f"{name}({', '.join(args)})",
                                 mode="eval").body
                if _replace(made_parents, target, call):
                    out.append((node.lineno, f"None -> {name}(...)",
                                ast.unparse(made)))
    for line in lines:
        for index, node in enumerate(nodes):
            if getattr(node, "lineno", None) != line or node is function:
                continue
            here = dict(context)
            here["condition"] = id(node) in conditions
            parent = parents.get(node)
            here["joined"] = isinstance(parent, ast.Call) and any(
                one is node for one in parent.args)
            if isinstance(node, ast.Attribute):
                here["attributes"] = sorted(context["receivers"].get(
                    ast.unparse(node.value), set()))[:12]
            for what, edit in _edits(tree, node, parents, here):
                made, made_nodes, made_parents = fresh()
                target = made_nodes[index + 1]
                try:
                    if not edit(made, target, made_parents):
                        continue
                    said = ast.unparse(made)
                    compile(said, "<repair>", "exec")
                except Exception:                   # noqa: BLE001
                    continue
                out.append((line, what, said))
        for index, node in enumerate(nodes):
            if getattr(node, "lineno", None) != line:
                continue
            if isinstance(node, ast.stmt) and not isinstance(
                    node, (ast.FunctionDef, ast.Return)):
                made, made_nodes, made_parents = fresh()
                target = made_nodes[index + 1]
                holder = made_parents.get(target)
                for name, value in ast.iter_fields(holder):
                    if isinstance(value, list) and any(
                            one is target for one in value) and len(value) > 1:
                        value.remove(target)
                        out.append((line, "statement taken out",
                                    ast.unparse(made)))
                        break
        if len(out) >= POOL * 4:
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
    return kept[:POOL]


def _group(what: str) -> int:
    if "put round" in what:
        return 2
    if what.startswith((".", "and ")) or what == "statement taken out" or (
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
    # every line the failing tests run, most suspected first: the code
    # model's order sorts their edits with where they stand (five true
    # lines stood past the first eight)
    lines = [line for line, score in suspected(case, function)
             if score > 0][:LINES]
    row["suspected"] = lines
    module = sys.modules.get(function.__module__)
    try:
        module_source = inspect.getsource(module) if module else ""
    except (OSError, TypeError):
        module_source = ""
    owner = module
    for part in function.__qualname__.split(".")[:-1]:
        owner = getattr(owner, part, None)
    where = {"namespace": vars(module) if module else {},
             "owner": owner if owner is not module else None}
    found = _ordered(source, candidates(source, lines, module_source, where),
                     row)
    held, tried = _tried(found[:MOST], case, old, verify)
    if not held:
        # no one edit: two -- each of the first edits by how far it gets
        # the failing tests (more pass, or they fail otherwise), the best
        # of them each given a second edit, tried in the code model's order
        held, more = _paired(case, function, old, verify, row, found, lines,
                             module_source, where)
        tried += more
    if not held:
        row["repair"] = {"tried": tried}
        return None
    # of those the tests pass, the one that changes least of what it does
    # where the tests beside it call it: a slip undone changes the calls it
    # went wrong on, `and` made `or` changes many it did not (calls fuzzed
    # from these were measured too, and chose no better)
    calls = truth_recorded(function, case.get("guards") or [])
    changed = _changed(function, [one[3] for one in held], calls)
    best = min(range(len(held)), key=lambda at: (changed[at], at))
    line, what, made, _ = held[best]
    row["repair"] = {"tried": tried, "line": line, "edit": what,
                     "held": [{"edit": one[1], "line": one[0],
                               "calls": count, "source": one[2]}
                              for one, count in zip(held, changed)]}
    return {"moves": [f"line {line}: {what}"], "source": made}


def _ordered(source: str, found: list, row: dict, pool: int = 0) -> list:
    """Edits in the order of where they stand (suspected lines, few edits
    first) and of how natural the code model finds each, the two ranks
    summed -- the true edit in the first six for 11 of 14 bugs, against 3
    by where they stand alone; one batch on the GPU."""
    found = found[:pool] if pool else found
    if not found:
        return found
    from research.v705 import natural
    try:
        gain = natural.gained(ast.unparse(ast.parse(source)),
                              [one[2] for one in found])
    except Exception as bad:                        # noqa: BLE001
        row["natural"] = f"{type(bad).__name__}: {bad}"[:120]
        return found
    by = {at: rank for rank, at in enumerate(sorted(
        range(len(found)), key=lambda at: -gain[at]))}
    return [found[at] for at in sorted(range(len(found)),
                                       key=lambda at: (at + by[at], at))]


def _tried(found: list, case: dict, old: str, verify) -> tuple:
    """Each edit judged by the tests, until `KEPT` hold: (held, tried)."""
    tried, held = 0, []
    for line, what, made in found:
        text = commits._placed(made, case["before"], old)
        if text is None:
            continue
        tried += 1
        if verify(text) is None:
            held.append((line, what, made, text))
            if len(held) >= KEPT:
                break
    return held, tried


#: of how many first edits progress is measured, how many changes are kept
#: going at each depth, and how many edits each is given (of how many
#: ordered)
FIRSTS, BEAM, SECONDS, SECOND_POOL = 120, 8, 40, 200
#: the changes the tests pass kept to choose among (each kept runs every
#: guard: six made a run four times as long, for no more real)
KEPT = 3


def _progress(case: dict, function, text: str, before: dict) -> tuple:
    """How far an edit gets the failing tests: (how many pass, how many
    fail otherwise than they did)."""
    from research.v705 import inside
    try:
        made = inside.changed_function(function, text)
    except Exception:                               # noqa: BLE001
        return (-1, -1)
    passed = moved = 0
    with commits.Put(function, made):
        for test in case["tests"]:
            said = commits.failure(test)
            if not commits.LAST.get("failure"):
                passed += 1
            elif said != before.get(test["test"]):
                moved += 1
    return (passed, moved)


def _relined(case: dict, function, lines: list) -> list:
    """The suspected lines of a function's source as they stand once it is
    written out again (`ast.unparse`): by the nodes on them."""
    import inspect
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError):
        source = textwrap.dedent(case["before"])
    tree = ast.parse(source)
    again = ast.parse(ast.unparse(tree))
    pairs = list(zip(ast.walk(tree), ast.walk(again)))
    out = []
    for line in lines:
        for one, other in pairs:
            if isinstance(one, ast.stmt) and one.lineno == line and                     other.lineno not in out:
                out.append(other.lineno)
                break
    return out


def closeness(got, wanted) -> float:
    """How near a value a check got is to the one it wanted, from 0 to 1:
    equal 1; strings by how much of them match, numbers by how far apart
    for their size, lists by what they share, dicts by the keys they share
    and how near those keys' values are -- a gradient where the tests give
    only pass or fail."""
    import difflib
    try:
        if type(got) is type(wanted) and got == wanted:
            return 1.0
    except Exception:                               # noqa: BLE001
        pass
    if isinstance(got, str) and isinstance(wanted, str):
        return difflib.SequenceMatcher(None, got, wanted).ratio()
    if isinstance(got, (int, float)) and isinstance(wanted, (int, float))             and not isinstance(got, bool) and not isinstance(wanted, bool):
        return 1.0 / (1.0 + abs(got - wanted) / (abs(wanted) + 1.0))
    if isinstance(got, (list, tuple)) and isinstance(wanted, (list, tuple)):
        if not got and not wanted:
            return 1.0
        mine = [repr(one) for one in got]
        shared = sum(min(mine.count(one), [repr(x) for x in wanted].count(
            one)) for one in set(mine))
        return shared / max(len(got), len(wanted))
    if isinstance(got, dict) and isinstance(wanted, dict):
        keys = set(got) | set(wanted)
        if not keys:
            return 1.0
        return sum(closeness(got[key], wanted[key]) for key in
                   set(got) & set(wanted)) / len(keys)
    return 0.0


def _fitness(case: dict, function, text: str) -> float:
    """How far a change gets the failing tests, from 0 to 1: each passing
    1, each failing as near as its first failed check got to what it
    wanted (`closeness`, at most 0.9), a fall before any check 0."""
    from research.v705 import inside, means
    try:
        made = inside.changed_function(function, text)
    except Exception:                               # noqa: BLE001
        return -1.0
    total = 0.0
    with commits.Put(function, made):
        for test in case["tests"]:
            found = means.goal({"tests": [test]})
            failing = found.get("failing")
            if failing is None and not found.get("raised"):
                total += 1.0
            elif failing and failing.get("_raw"):
                got, wanted = failing["_raw"]
                total += 0.9 * closeness(got, wanted)
    return total / max(1, len(case["tests"]))


#: how many edits a change may make at most: two -- three and four found
#: nothing more on the planted bugs and took 24 min on one (the measure
#: between pass and fail ties many edits, and a deeper beam only widens)
DEPTH = 2


def _paired(case, function, old, verify, row, found, lines,
            module_source, where=None) -> tuple:
    """Changes of more than one edit, depth by depth (`DEPTH`): at each,
    the changes kept going each given edits in the code model's order;
    an edit kept where it gets the failing tests further than the change
    it was made on (`_fitness`), the `BEAM` furthest going on; one whose
    failing tests all pass judged by every test. The fewest edits that
    hold win (the search stops at the first depth any holds) --
    (held, tried)."""
    import inspect
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError):
        source = textwrap.dedent(case["before"])
    root = commits._placed(ast.unparse(ast.parse(source)), case["before"],
                           old)
    frontier = [(None, [], _fitness(case, function, root) if root else 0.0)]
    # an edited function is written out again without its comments: the
    # suspected lines as they stand there
    relined = _relined(case, function, lines)
    held, tried, kept = [], 0, []
    for depth in range(1, DEPTH + 1):
        children = []
        for made_so_far, moves, fit in frontier:
            if made_so_far is None:
                edits = found[:FIRSTS]
            else:
                edits = _ordered(made_so_far, candidates(
                    made_so_far, relined, module_source, where), row,
                    pool=SECOND_POOL)[:SECONDS]
            for line, what, made in edits:
                text = commits._placed(made, case["before"], old)
                if text is None:
                    continue
                tried += 1
                reached = _fitness(case, function, text)
                if reached >= 1.0 - 1e-9 and depth > 1:
                    if verify(text) is None:
                        held.append((line, " / ".join(moves + [what]),
                                     made, text))
                        if len(held) >= KEPT:
                            break
                elif reached > fit:
                    children.append((made, moves + [what], reached))
            if len(held) >= KEPT:
                break
        kept.append([[" / ".join(one[1]), round(one[2], 3)]
                     for one in sorted(children, key=lambda one: -one[2])[
                         :BEAM]])
        if held or not children:
            break
        frontier = sorted(children, key=lambda one: -one[2])[:BEAM]
    row["beam"] = kept
    return held[:KEPT], tried


def truth_recorded(function, tests: list) -> list:
    from research.v705 import truth
    return truth._recorded(function, tests)


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
