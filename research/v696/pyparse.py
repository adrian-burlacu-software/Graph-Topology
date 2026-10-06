"""Python read back into the search's own trees.

What a writer, a person or a project writes in Python is trusted once it is
a tree the search could have built (`parse.py` says the same of
TypeScript): every node one of Python's library operators (`pylibrary.py`)
or one of the engine's own (a conditional, an element, a range, a form with
its callback), with its type.

An expression is read by the very templates its operators are written
with: `s.upper()` is the operator whose template `{0}.upper()` it matches,
so what is read and what is printed cannot part. What is not an
expression is read as one:

    a name bound once (`y = ...`)        its value, where it is read
    `if c: return a` ... `return b`      a if c else b
    out = []; for x in xs: out.append(e) [e for x in xs] (if ...: a filter)
    acc = v; for x in xs: acc = f(acc,x) functools.reduce over xs from v
    for x in xs: if c: return True       any(c for x in xs) (all, next)
    for i in range(a, b): ...            the same over list(range(a, b))

Anything else -- a `while`, an exception, a mutation the tree cannot
carry -- is refused, with why (`why`).

    parse(source, entry, params) -> Expr | None
"""
from __future__ import annotations

import ast
import functools

from research.v696 import program as P
from research.v696 import pylibrary
from research.v696 import pytypes

#: how big an inlined name may make a tree before it is refused
LARGEST = 400


class Unread(Exception):
    pass


# -- the library, by the templates it is written with ---------------------------

def _template_tree(template: str, count: int):
    text = template.format(*[f"_a{at}" for at in range(count)])
    return ast.parse(text, mode="eval").body


@functools.lru_cache(maxsize=None)
def _index(types: tuple):
    """Each operator with its template parsed: (op, template tree)."""
    lib = pylibrary.library(types)
    out = []
    for op in lib.ops:
        if op.py:
            try:
                out.append((op, _template_tree(op.py, len(op.needs))))
            except SyntaxError:
                continue
    return out


def _match(pattern, node, caught: dict) -> bool:
    """Whether `node` is `pattern`, its `_aN` names catching the
    subtrees there."""
    if isinstance(pattern, ast.Name) and pattern.id.startswith("_a"):
        at = int(pattern.id[2:])
        if at in caught:
            return ast.dump(caught[at]) == ast.dump(node)
        caught[at] = node
        return True
    if type(pattern) is not type(node):
        return False
    for name, value in ast.iter_fields(pattern):
        if name in ("ctx", "type_comment", "kind"):
            continue
        other = getattr(node, name, None)
        if isinstance(value, list):
            if not isinstance(other, list) or len(value) != len(other):
                return False
            if not all(_match(a, b, caught) if isinstance(a, ast.AST)
                       else a == b for a, b in zip(value, other)):
                return False
        elif isinstance(value, ast.AST):
            if not isinstance(other, ast.AST) or \
                    not _match(value, other, caught):
                return False
        elif value != other:
            return False
    return True


# -- reading --------------------------------------------------------------------

TYPES = P.TYPES


class Reader:
    def __init__(self, tree: ast.Module, types=TYPES) -> None:
        self.tree = tree
        self.types = tuple(sorted(set(types)))
        self.functions = {node.name: node for node in tree.body
                          if isinstance(node, ast.FunctionDef)}
        self.helpers: dict = {}
        #: the functions being read, outermost first: a call to one of
        #: them is the function calling itself
        self.reading: dict = {}
        #: names made for state carried through loops, never twice (an inner
        #: loop's would hide an outer one's)
        self.made = 0

    # -- an expression ------------------------------------------------------

    def expr(self, node, env: dict) -> P.Expr:
        if isinstance(node, ast.Constant):
            return self._constant(node.value)
        if isinstance(node, ast.Name):
            if node.id in env:
                found = env[node.id]
                if found is None:
                    raise Unread(f"{node.id} is read before it is made")
                return found
            if node.id in ("True", "False", "None"):
                return self._constant({"True": True, "False": False,
                                       "None": None}[node.id])
            raise Unread(f"a name nothing here makes: {node.id}")
        if isinstance(node, (ast.List, ast.Tuple)) and all(
                isinstance(one, ast.Constant) for one in node.elts):
            values = [one.value for one in node.elts]
            kind = self._kind_of(values[0]) if values else "number"
            if not all(self._kind_of(one) == kind for one in values):
                raise Unread("a list of mixed types")
            return P.const(values, f"{kind}[]")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self._apply_named("!", [self._truth(self.expr(
                node.operand, env))])
        if isinstance(node, ast.Attribute) and isinstance(
                node.value, ast.Name) and node.value.id == "math" and \
                node.attr in ("pi", "e", "inf", "tau"):
            import math
            return P.const(getattr(math, node.attr), "number")
        if isinstance(node, ast.Tuple) and node.elts:
            parts = [self.expr(one, env) for one in node.elts]
            return P.apply(P.tuple_op([one.type for one in parts]), parts)
        if isinstance(node, ast.IfExp):
            test = self._truth(self.expr(node.test, env))
            yes, no = self.expr(node.body, env), self.expr(node.orelse, env)
            return self._ternary(test, yes, no)
        if isinstance(node, ast.BoolOp):
            parts = [self._truth(self.expr(one, env)) for one in node.values]
            symbol = "&&" if isinstance(node.op, ast.And) else "||"
            out = parts[0]
            for one in parts[1:]:
                out = self._apply_named(symbol, [out, one])
            return out
        if isinstance(node, ast.Compare) and len(node.ops) > 1:
            # a < b < c: each pair, all of them
            parts, left = [], node.left
            for op, right in zip(node.ops, node.comparators):
                parts.append(ast.Compare(left=left, ops=[op],
                                         comparators=[right]))
                left = right
            return self.expr(ast.BoolOp(op=ast.And(), values=parts), env)
        if isinstance(node, ast.Compare) and isinstance(node.ops[0],
                                                        ast.NotIn):
            inner = ast.Compare(left=node.left, ops=[ast.In()],
                                comparators=node.comparators)
            return self._apply_named("!", [self.expr(inner, env)])
        if isinstance(node, ast.Subscript) and not isinstance(node.slice,
                                                              ast.Slice):
            base, at = self.expr(node.value, env), self.expr(node.slice, env)
            for op in P.INDEX:
                if op.needs == (base.type, at.type):
                    return P.apply(op, [base, at])
            raise Unread(f"an element of a {base.type}")
        if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
            return self._comprehension(node, env)
        if isinstance(node, ast.Call):
            found = self._form_call(node, env)
            if found is not None:
                return found
            helper = self._helper_call(node, env)
            if helper is not None:
                return helper
        return self._templated(node, env)

    def _constant(self, value) -> P.Expr:
        if value is None:
            raise Unread("None as a value")
        return P.const(value, self._kind_of(value))

    @staticmethod
    def _kind_of(value) -> str:
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, (int, float)):
            return "number"
        if isinstance(value, str):
            return "string"
        raise Unread(f"a constant of {type(value).__name__}")

    def _truth(self, expr: P.Expr) -> P.Expr:
        if expr.type == "boolean":
            return expr
        if expr.type == "number":
            return self._apply_named("!==", [expr, P.const(0, "number")])
        if expr.type == "string":
            return self._apply_named("!==", [expr, P.const("", "string")])
        if expr.type.endswith("[]"):
            return self._apply_named("!==", [self._apply_named(
                "length", [expr]), P.const(0, "number")])
        raise Unread(f"a {expr.type} taken as true or false")

    def _ternary(self, test, yes, no) -> P.Expr:
        if yes.type != no.type:
            raise Unread(f"two ways giving {yes.type} and {no.type}")
        return P.apply(P.Op("?:", "ternary", ("boolean", yes.type,
                                              yes.type), yes.type),
                       [test, yes, no])

    def _apply_named(self, name: str, args) -> P.Expr:
        for op, _ in _index(self.types):
            if op.name == name and op.needs == tuple(one.type
                                                     for one in args):
                return P.apply(op, args)
        raise Unread(f"no {name} of {[one.type for one in args]}")

    def _templated(self, node, env: dict) -> P.Expr:
        """The operator whose template this is, its arguments read -- each
        argument once, whichever operators share its shape (`a + b` is a
        number's, a string's and a list's): read again for each, a deep
        expression took time without end."""
        tried = False
        read: dict = {}

        def once(part):
            key = id(part)
            if key not in read:
                try:
                    read[key] = self.expr(part, env)
                except Unread as bad:
                    read[key] = bad
            return read[key]

        for op, pattern in _index(self.types):
            caught: dict = {}
            if not _match(pattern, node, caught):
                continue
            if len(caught) != len(op.needs):
                continue
            tried = True
            args = [once(caught[at]) for at in range(len(op.needs))]
            if any(isinstance(one, Unread) for one in args):
                continue
            if tuple(one.type for one in args) == op.needs:
                return P.apply(op, args)
        # what the library writes another way: list(range(a, b)) and
        # range(a, b) are the counting loop's numbers; [*xs, x] is append
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "list" and len(node.args) == 1 and isinstance(
                    node.args[0], ast.Call) and isinstance(
                    node.args[0].func, ast.Name) and \
                    node.args[0].func.id == "range":
                return self._range(node.args[0], env)
            if name == "range":
                return self._range(node, env)
            if name == "len" and len(node.args) == 1:
                inner = self.expr(node.args[0], env)
                return self._apply_named("length", [inner])
        what = ast.unparse(node)
        raise Unread(("no operator of these types: " if tried else
                      "nothing in the library is written ") + what[:80])

    def _range(self, call, env) -> P.Expr:
        args = [self.expr(one, env) for one in call.args]
        if len(args) == 1:
            args = [P.const(0, "number"), args[0]]
        if len(args) != 2 or any(one.type != "number" for one in args):
            raise Unread("a range with a step")
        return P.apply(P.RANGE, args)

    # -- forms ----------------------------------------------------------------

    def _forms(self, name: str, receiver: str, extra: int = 0):
        lib = pylibrary.library(self.types)
        return [form for form in lib.forms if form.name == name
                and form.receiver == receiver and len(form.extra) == extra]

    def _over(self, node, env):
        """What a loop or a comprehension goes over: a list, the characters
        of a string, the numbers of a range."""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "range":
            return self._range(node, env)
        found = self.expr(node, env)
        if found.type == "string":
            return self._apply_named("chars", [found])
        if not found.type.endswith("[]"):
            raise Unread(f"a loop over a {found.type}")
        return found

    def _target(self, target, over: P.Expr):
        """(element name, index name or None, iterable) of `for t in it`:
        `for i, x in enumerate(xs)` reads its index too."""
        if isinstance(target, ast.Name):
            return target.id, None
        raise Unread("a loop that unpacks")

    def _loop_of(self, target, iter_node, env):
        if isinstance(iter_node, ast.Call) and isinstance(
                iter_node.func, ast.Name) and \
                iter_node.func.id == "enumerate" and \
                isinstance(target, ast.Tuple) and len(target.elts) == 2 \
                and all(isinstance(one, ast.Name) for one in target.elts):
            over = self._over(iter_node.args[0], env)
            return target.elts[1].id, target.elts[0].id, over
        over = self._over(iter_node, env)
        name, index = self._target(target, over)
        return name, index, over

    def _with_form(self, name: str, over: P.Expr, element: str,
                   index: str | None, body_node, env, extra=()):
        """`name` over `over`, the callback's body read with the element
        (and its index) in scope."""
        item = over.type[:-2]
        inner = dict(env)
        inner[element] = P.param(element, item)
        if index:
            inner[index] = P.param(index, "number")
        body = self.expr(body_node, inner)
        if name in ("filter", "some", "every", "find", "findIndex"):
            body = self._truth(body)
        for form in self._forms(name, over.type, len(extra)):
            want = form.body if form.body != "U" else body.type
            if want != body.type:
                continue
            op = form.op(body.type)
            spare = "i" if element != "i" else "j"
            scope = [(element, item), (index or spare, "number")][
                :len(form.scope)]
            return P.apply(op, [over, P.lambda_(scope, body), *extra])
        raise Unread(f"no {name} over {over.type} giving {body.type}")

    def _comprehension(self, node, env) -> P.Expr:
        if len(node.generators) != 1:
            raise Unread("a comprehension of several loops")
        loop = node.generators[0]
        element, index, over = self._loop_of(loop.target, loop.iter, env)
        if loop.ifs:
            test = loop.ifs[0]
            for more in loop.ifs[1:]:
                test = ast.BoolOp(op=ast.And(), values=[test, more])
            filtered = self._with_form("filter", over, element, index, test,
                                       env)
            if isinstance(node.elt, ast.Name) and node.elt.id == element:
                return filtered
            if index:
                raise Unread("an index after a filter")
            return self._with_form("map", filtered, element, None, node.elt,
                                   env)
        if isinstance(node.elt, ast.Name) and node.elt.id == element \
                and not index:
            return over
        return self._with_form("map", over, element, index, node.elt, env)

    def _form_call(self, node: ast.Call, env):
        func = node.func
        name = func.id if isinstance(func, ast.Name) else (
            func.attr if isinstance(func, ast.Attribute) else None)
        args = node.args
        keys = {one.arg: one.value for one in node.keywords}
        if name in ("any", "all", "sum") and len(args) == 1 and \
                isinstance(args[0], (ast.GeneratorExp, ast.ListComp)) and \
                len(args[0].generators) == 1 and not args[0].generators[0].ifs:
            loop = args[0].generators[0]
            element, index, over = self._loop_of(loop.target, loop.iter, env)
            form = {"any": "some", "all": "every", "sum": "sumOf"}[name]
            return self._with_form(form, over, element, index, args[0].elt,
                                   env)
        if name in ("sorted", "max", "min") and len(args) == 1 and \
                set(keys) == {"key"} and isinstance(keys["key"], ast.Lambda):
            lam = keys["key"]
            over = self._over(args[0], env)
            form = {"sorted": "sortedBy", "max": "maxBy", "min": "minBy"}[name]
            return self._with_form(form, over, lam.args.args[0].arg, None,
                                   lam.body, env)
        if name in ("map", "filter") and len(args) == 2 and \
                not keys and isinstance(func, ast.Name):
            over = self._over(args[1], env)
            if isinstance(args[0], ast.Lambda) and \
                    len(args[0].args.args) == 1:
                element = args[0].args.args[0].arg
                body = args[0].body
            elif isinstance(args[0], ast.Name):
                element = "x"
                body = ast.Call(func=ast.Name(id=args[0].id, ctx=ast.Load()),
                                args=[ast.Name(id="x", ctx=ast.Load())],
                                keywords=[])
            else:
                return None
            return self._with_form(name, over, element, None, body, env)
        if name in ("list", "sorted") and len(args) == 1 and not keys \
                and isinstance(func, ast.Name) and isinstance(
                    args[0], ast.Call) and isinstance(
                    args[0].func, ast.Name) and \
                args[0].func.id in ("map", "filter"):
            inner = self.expr(args[0], env)
            if name == "list":
                return inner
            return self._apply_named("sorted", [inner])
        if name == "reduce" and len(args) in (2, 3) and isinstance(
                args[0], ast.Lambda) and len(args[0].args.args) == 2:
            lam = args[0]
            over = self._over(args[1], env)
            item = over.type[:-2]
            acc, element = (one.arg for one in lam.args.args)
            start = self.expr(args[2], env) if len(args) == 3 else None
            kind = start.type if start is not None else item
            inner = dict(env)
            inner[acc] = P.param(acc, kind)
            inner[element] = P.param(element, item)
            body = self.expr(lam.body, inner)
            extra = (start,) if start is not None else ()
            for form in self._forms("reduce", over.type, len(extra)):
                if form.body not in (body.type, "U"):
                    continue
                op = form.op(body.type)
                return P.apply(op, [over, P.lambda_(
                    [(acc, kind), (element, item), ("i", "number")], body),
                    *extra])
            raise Unread("a reduce of these types")
        return None

    # -- helpers: other functions of the file -----------------------------------

    def _helper_call(self, node: ast.Call, env):
        if not (isinstance(node.func, ast.Name)
                and node.func.id in self.functions):
            return None
        name = node.func.id
        if name in self.reading:
            # the function calling itself: an operator naming it, its body
            # the tree being read (`pyprint.prelude` defines it from that)
            names, needs, gives = self.reading[name]
            args = [self.expr(one, env) for one in node.args]
            if tuple(one.type for one in args) != needs:
                raise Unread(f"{name} calls itself with other types")
            return P.apply(P.Op(name, "recurse", needs, gives,
                                params=names), args)
        op = self.helpers.get(name)
        if op is None:
            fn = self.functions[name]
            params = [(one.arg, pytypes.engine(ast.unparse(one.annotation)
                                               if one.annotation else None))
                      for one in fn.args.args]
            body = self.function(name, params)
            op = P.Op(name, "helper", tuple(kind for _, kind in params),
                      body.type, params=tuple(n for n, _ in params),
                      body=body)
            self.helpers[name] = op
        args = [self.expr(one, env) for one in node.args]
        if tuple(one.type for one in args) != op.needs:
            raise Unread(f"{name} called with other types")
        return P.apply(op, args)

    # -- statements -------------------------------------------------------------

    def function(self, name: str, params=None) -> P.Expr:
        node = self.functions.get(name)
        if node is None:
            raise Unread(f"no function {name}")
        if params is None:
            params = [(one.arg, pytypes.engine(ast.unparse(one.annotation)
                                               if one.annotation else None))
                      for one in node.args.args]
        if len(params) != len(node.args.args):
            raise Unread("its parameters are not the request's")
        env = {}
        for (wanted, kind), one in zip(params, node.args.args):
            env[one.arg] = P.param(wanted, kind)
        gives = pytypes.engine(ast.unparse(node.returns)) \
            if node.returns is not None else "any"
        if name in self.reading:
            raise Unread(f"{name} read inside itself")
        self.reading[name] = (tuple(wanted for wanted, _ in params),
                              tuple(kind for _, kind in params), gives)
        try:
            return self._body(node, env)
        finally:
            self.reading.pop(name, None)

    def _body(self, node, env) -> P.Expr:
        body = [one for one in node.body if not (
            isinstance(one, ast.Expr) and isinstance(one.value, ast.Constant)
            and isinstance(one.value.value, str))]
        found = self.block(body, env)
        if found is None:
            raise Unread("a way through it returns nothing")
        if found.size > LARGEST:
            raise Unread("too large read as one expression")
        return found

    def block(self, statements: list, env: dict):
        """What these statements return, read in order: a tree, or None
        where they fall through (env then holds the names they made)."""
        at = 0
        while at < len(statements):
            one = statements[at]
            if isinstance(one, ast.Return):
                if one.value is None:
                    raise Unread("a return of nothing")
                return self.expr(one.value, env)
            if isinstance(one, ast.Pass):
                at += 1
                continue
            if isinstance(one, ast.Assign) and len(one.targets) == 1 and \
                    isinstance(one.targets[0], ast.Name):
                env[one.targets[0].id] = self.expr(one.value, env)
                at += 1
                continue
            if isinstance(one, ast.AnnAssign) and isinstance(one.target,
                                                             ast.Name) \
                    and one.value is not None:
                env[one.target.id] = self.expr(one.value, env)
                at += 1
                continue
            if isinstance(one, ast.AugAssign) and isinstance(one.target,
                                                             ast.Name):
                symbol = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*",
                          ast.Div: "/", ast.FloorDiv: "//",
                          ast.Mod: "%"}.get(type(one.op))
                if symbol is None:
                    raise Unread("an assignment by an operator")
                value = ast.BinOp(left=ast.Name(id=one.target.id,
                                                ctx=ast.Load()),
                                  op=one.op, right=one.value)
                env[one.target.id] = self.expr(value, env)
                at += 1
                continue
            if isinstance(one, ast.If):
                yes_env, no_env = dict(env), dict(env)
                yes = self.block(list(one.body), yes_env)
                no = self.block(list(one.orelse), no_env) if one.orelse \
                    else None
                test = self._truth(self.expr(one.test, env))
                if yes is not None and no is not None:
                    return self._ternary(test, yes, no)
                if yes is not None or no is not None:
                    # a guard: the rest is the other way through
                    rest_env = no_env if yes is not None else yes_env
                    rest = self.block(statements[at + 1:], rest_env)
                    if rest is None:
                        raise Unread("a guard and nothing returned after")
                    return (self._ternary(test, yes, rest) if yes is not None
                            else self._ternary(test, rest, no))
                for name in set(yes_env) | set(no_env):
                    a, b = yes_env.get(name), no_env.get(name)
                    if a is None or b is None:
                        env.pop(name, None)
                    elif a.source() == b.source():
                        env[name] = a
                    else:
                        env[name] = self._ternary(test, a, b)
                at += 1
                continue
            if isinstance(one, ast.While):
                self._while(one, env)
                at += 1
                continue
            if isinstance(one, ast.Assign) and len(one.targets) == 1 and \
                    isinstance(one.targets[0], ast.Tuple) and all(
                        isinstance(name, ast.Name)
                        for name in one.targets[0].elts):
                # a, b = b, a + b: every value read before any is bound
                if isinstance(one.value, ast.Tuple) and \
                        len(one.value.elts) == len(one.targets[0].elts):
                    values = [self.expr(part, env)
                              for part in one.value.elts]
                    for name, value in zip(one.targets[0].elts, values):
                        env[name.id] = value
                else:
                    whole = self.expr(one.value, env)
                    for at_, name in enumerate(one.targets[0].elts):
                        env[name.id] = P.apply(P.element_op(whole.type, at_),
                                               [whole, P.const(at_, "number")])
                at += 1
                continue
            if isinstance(one, ast.For):
                found = self._for(one, statements[at + 1:], env)
                if found is not None:
                    kind, value = found
                    if kind == "return":
                        return value
                at += 1
                continue
            raise Unread(f"a {type(one).__name__} statement")
        return None

    def _for(self, loop: ast.For, after: list, env: dict):
        """A loop as a form: its accumulator folded, its early return
        searched. ("return", tree) where it decides what is returned."""
        if loop.orelse:
            raise Unread("a for with an else")
        element, index, over = self._loop_of(loop.target, loop.iter, env)
        body = list(loop.body)
        # for x in xs: if c: return A  ...  return B
        if len(body) == 1 and isinstance(body[0], ast.If) and \
                not body[0].orelse and len(body[0].body) == 1 and \
                isinstance(body[0].body[0], ast.Return) and after and \
                isinstance(after[0], ast.Return) and after[0].value:
            found = body[0].body[0].value
            rest = after[0].value
            if isinstance(found, ast.Constant) and isinstance(
                    rest, ast.Constant) and isinstance(found.value, bool) \
                    and isinstance(rest.value, bool) and \
                    found.value != rest.value:
                some = self._with_form("some", over, element, index,
                                       body[0].test, env)
                return "return", (some if found.value else
                                  self._apply_named("!", [some]))
            if isinstance(found, ast.Name) and found.id == element:
                found_one = self._with_form("find", over, element, index,
                                            body[0].test, env)
                default = self.expr(rest, env)
                if default.type != found_one.type:
                    raise Unread("a search with another default")
                return "return", found_one
        # the names the body changes: one accumulator, folded
        changed = self._changes(loop)
        changed.discard(element)
        if index:
            changed.discard(index)
        outer = sorted(name for name in changed if name in env)
        if not outer:
            # nothing it changes outlives it
            return None
        if len(outer) > 1:
            self._carried(over, element, index, body, outer, env)
            return None
        acc = outer[0]
        start = env[acc]
        item = over.type[:-2]
        inner = dict(env)
        inner[acc] = P.param(acc, start.type)
        inner[element] = P.param(element, item)
        if index:
            inner[index] = P.param(index, "number")
        appended = self._appends(body, acc, inner)
        if appended == "complex":
            # appends among other steps: each written `acc = [*acc, v]`,
            # and the loop folded as any other
            body = self._as_assignment(body, acc)
        elif appended is not None:
            # out = []; for x in xs: out.append(e) / if c: out.append(e)
            test, value = appended
            if start.kind == "const" and start.value == []:
                source = over
                if test is not None:
                    source = self._with_form("filter", over, element, index,
                                             test, env)
                    index = None
                if isinstance(value, ast.Name) and value.id == element \
                        and not index:
                    env[acc] = source
                else:
                    env[acc] = self._with_form("map", source, element, index,
                                               value, env)
                return None
            body = self._as_assignment(body, acc)
        step = self.block(list(body) + [ast.Return(value=ast.Name(
            id=acc, ctx=ast.Load()))], inner)
        if step is None or step.type != start.type:
            raise Unread("a loop's step of another type")
        for form in self._forms("reduce", over.type, 1):
            if form.body not in (step.type, "U"):
                continue
            op = form.op(step.type)
            spare = index or next(one for one in ("i", "j", "k", "_i")
                                  if one not in (element, acc))
            env[acc] = P.apply(op, [over, P.lambda_(
                [(acc, start.type), (element, item), (spare, "number")],
                step), start])
            return None
        raise Unread("no fold of these types")

    # -- state carried through a loop: several names as one tuple ------------

    def _state(self, names: list, env: dict):
        """(type, start, inner env) of names carried together: one name as
        itself, several as a tuple `s` whose elements they are."""
        if len(names) == 1:
            name = names[0]
            kind = env[name].type
            return kind, env[name], {name: P.param(name, kind)}, name
        types = [env[name].type for name in names]
        kind = P.tuple_of(types)
        start = P.apply(P.tuple_op(types), [env[name] for name in names])
        made = self._fresh()
        held = P.param(made, kind)
        inner = {name: P.apply(P.element_op(kind, at),
                               [held, P.const(at, "number")])
                 for at, name in enumerate(names)}
        return kind, start, inner, made

    def _returned(self, names: list):
        if len(names) == 1:
            return ast.Name(id=names[0], ctx=ast.Load())
        return ast.Tuple(elts=[ast.Name(id=one, ctx=ast.Load())
                               for one in names], ctx=ast.Load())

    def _unpacked(self, names: list, value: P.Expr, env: dict) -> None:
        if len(names) == 1:
            env[names[0]] = value
            return
        for at, name in enumerate(names):
            env[name] = P.apply(P.element_op(value.type, at),
                                [value, P.const(at, "number")])

    def _carried(self, over, element, index, body, names, env) -> None:
        """A loop changing several names: folded, the names a tuple."""
        for name in names:
            if self._appends(body, name, env) is not None:
                body = self._as_assignment(body, name)
        kind, start, carried, held = self._state(names, env)
        item = over.type[:-2]
        inner = dict(env)
        inner.update(carried)
        inner[element] = P.param(element, item)
        if index:
            inner[index] = P.param(index, "number")
        step = self.block(list(body) + [ast.Return(
            value=self._returned(names))], inner)
        if step is None or step.type != kind:
            raise Unread("a loop's step of another type")
        for form in self._forms("reduce", over.type, 1):
            if form.body != "U":
                continue
            spare = index or next(one for one in ("i", "j", "k", "_i")
                                  if one not in (element, held))
            folded = P.apply(form.op(kind), [over, P.lambda_(
                [(held, kind), (element, item), (spare, "number")], step),
                start])
            self._unpacked(names, folded, env)
            return
        raise Unread("no fold of these types")

    def _while(self, loop: ast.While, env: dict) -> None:
        """`while c: ...` as the engine's loop over the names it changes."""
        if loop.orelse:
            raise Unread("a while with an else")
        changed = self._changes(loop)
        names = sorted(one for one in changed if one in env)
        if not names:
            raise Unread("a while changing nothing it began with")
        body = list(loop.body)
        for name in names:
            if self._appends(body, name, env) is not None:
                body = self._as_assignment(body, name)
        kind, start, carried, held = self._state(names, env)
        inner = dict(env)
        inner.update(carried)
        test = self._truth(self.expr(loop.test, inner))
        update = self.block(body + [ast.Return(value=self._returned(names))],
                            dict(inner))
        if update is None or update.type != kind:
            raise Unread("a while's step of another type")
        scope = [(held, kind)]
        looped = P.apply(P.while_op(kind), [start, P.lambda_(scope, test),
                                            P.lambda_(scope, update)])
        self._unpacked(names, looped, env)

    @staticmethod
    def _changes(loop) -> set:
        """The names a loop's body binds -- in tuples too, and the lists it
        appends to. A change a tree cannot carry -- an element or an
        attribute set, a call made for what it does, a return or a break
        left to the loop -- is refused here, never passed over."""
        changed = set()
        for node in ast.walk(loop):
            if node is loop:
                continue
            if isinstance(node, (ast.Return, ast.Break, ast.Continue,
                                 ast.Yield, ast.Raise, ast.Try,
                                 ast.Global, ast.Nonlocal, ast.Delete)):
                raise Unread(f"a {type(node).__name__} in a loop")
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) \
                    else [node.target]
                for target in targets:
                    for one in ast.walk(target):
                        if isinstance(one, (ast.Subscript, ast.Attribute)):
                            raise Unread("a loop setting an element")
                        if isinstance(one, ast.Name):
                            changed.add(one.id)
            if isinstance(node, ast.Expr) and isinstance(node.value,
                                                         ast.Call):
                call = node.value
                if not (isinstance(call.func, ast.Attribute)
                        and call.func.attr == "append"
                        and isinstance(call.func.value, ast.Name)):
                    raise Unread("a call in a loop made for what it does")
                changed.add(call.func.value.id)
        return changed

    def _fresh(self) -> str:
        self.made += 1
        return f"_s{self.made}"

    def _appends(self, body: list, acc: str, env: dict):
        """(test or None, value) where the body is `acc.append(v)`, or
        `if c: acc.append(v)` -- else None."""
        def append_of(stmt):
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value,
                                                         ast.Call):
                call = stmt.value
                if isinstance(call.func, ast.Attribute) and \
                        call.func.attr == "append" and isinstance(
                        call.func.value, ast.Name) and \
                        call.func.value.id == acc and len(call.args) == 1:
                    return call.args[0]
            return None
        if len(body) == 1:
            value = append_of(body[0])
            if value is not None:
                return None, value
            if isinstance(body[0], ast.If) and not body[0].orelse and \
                    len(body[0].body) == 1:
                value = append_of(body[0].body[0])
                if value is not None:
                    return body[0].test, value
        if any(append_of(one) is not None for one in ast.walk(
                ast.Module(body=body, type_ignores=[]))
               if isinstance(one, ast.stmt)):
            return "complex"
        return None

    def _as_assignment(self, body: list, acc: str) -> list:
        """`acc.append(v)` written as `acc = [*acc, v]`, wherever it is."""
        class Swap(ast.NodeTransformer):
            def visit_Expr(self, node):
                call = node.value
                if isinstance(call, ast.Call) and isinstance(
                        call.func, ast.Attribute) and \
                        call.func.attr == "append" and isinstance(
                        call.func.value, ast.Name) and \
                        call.func.value.id == acc:
                    return ast.Assign(
                        targets=[ast.Name(id=acc, ctx=ast.Store())],
                        value=ast.List(elts=[
                            ast.Starred(value=ast.Name(id=acc,
                                                       ctx=ast.Load()),
                                        ctx=ast.Load()), call.args[0]],
                            ctx=ast.Load()), lineno=node.lineno)
                return node
        tree = ast.fix_missing_locations(Swap().visit(
            ast.Module(body=[ast.parse(ast.unparse(one)).body[0]
                             for one in body], type_ignores=[])))
        return tree.body


def _types_of(params) -> tuple:
    types = set(P.TYPES)
    for _, kind in params or ():
        types.add(kind)
        if kind.endswith("[]"):
            types.add(kind[:-2])
    return tuple(types)


def parse(source: str, entry: str, params) -> P.Expr | None:
    """`entry`'s return as the search's tree, or None where it is more
    than one."""
    try:
        return Reader(ast.parse(source), _types_of(params)).function(
            entry, list(params) if params is not None else None)
    except (Unread, SyntaxError, KeyError, IndexError, TypeError,
            ValueError, RecursionError):
        return None


def why(source: str, entry: str, params) -> str | None:
    """Why `entry` is not read, or None where it is."""
    try:
        Reader(ast.parse(source), _types_of(params)).function(
            entry, list(params) if params is not None else None)
        return None
    except SyntaxError as bad:
        return f"a syntax error: {bad.msg}"
    except (Unread, KeyError, IndexError, TypeError, ValueError,
            RecursionError) as bad:
        return str(bad) or type(bad).__name__
