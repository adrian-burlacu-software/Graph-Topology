"""Changes a request asks for, made by their shape (v705 into the change
path, v700 `fixing`): the editors write a change's text, and cannot write
some -- `gained should take a message, empty by default`: twenty answers,
none touched the `def` line. The architecture has those changes as edits
of a function's tree (`repair.candidates`: an operator for its like, a
value passed on to a call that takes it, a case handed to a helper, a
field carried over) and one more here: a parameter the request names,
with a plain default. Each is ranked by a language model reading the
request before it (`natural.gained`, `said`), the first few offered to the
change path as answers are -- checked, judged -- and put into the text as
written (`splice`), its comments kept.

    offered(statement, part, path, text, held)   [(what, made part, gain)]
"""
from __future__ import annotations

import ast
import functools
import keyword
import re
import sys
import textwrap

#: the changes offered to the change path, and of how many ranked
OFFERED, POOL = 6, 400
#: the defaults a parameter a request names is given
DEFAULTS = (None, "", 0, False)


def _taking(tree, statement: str) -> list:
    """[(line, what, source)]: a parameter named by a word of the request
    that the function does not hold, with each plain default -- after the
    parameters it has (keyword-only where it takes `*args`)."""
    function = tree.body[0]
    held = {one.arg for one in ast.walk(function) if isinstance(one, ast.arg)}
    held |= {one.id for one in ast.walk(function) if isinstance(one, ast.Name)}
    words = [one for one in dict.fromkeys(re.findall(
        r"\b[A-Za-z_][A-Za-z0-9_]*\b", statement))
        if len(one) >= 3 and not keyword.iskeyword(one) and one not in held
        and one.isidentifier()]
    out = []
    for word in words[:24]:
        for default in DEFAULTS:
            made = ast.parse(ast.unparse(tree))
            args = made.body[0].args
            if args.vararg is not None:
                args.kwonlyargs.append(ast.arg(arg=word))
                args.kw_defaults.append(ast.Constant(default))
            else:
                args.args.append(ast.arg(arg=word))
                args.defaults.append(ast.Constant(default))
            out.append((function.lineno, f"takes {word}={default!r}",
                        ast.unparse(ast.fix_missing_locations(made))))
    return out


def _where(path: str) -> dict:
    """The module of a project file as it runs, where it is one this
    process has loaded (its own code): what its calls call."""
    name = path.removesuffix(".py").replace("/", ".").replace("\\", ".")
    module = sys.modules.get(name)
    if module is None and name.startswith("research."):
        # its own code not loaded yet: loaded, as the tests load it
        import importlib
        try:
            module = importlib.import_module(name)
        except Exception:
            module = None
    return {"namespace": vars(module)} if module else {}


def _takes(func, name: str, namespace: dict) -> bool:
    """Whether the function a call calls (`P.library`) has a parameter
    so named, looked up in the module it is called in."""
    import inspect
    chain = []
    while isinstance(func, ast.Attribute):
        chain.append(func.attr)
        func = func.value
    if not isinstance(func, ast.Name) or func.id not in namespace:
        return False
    found = namespace[func.id]
    try:
        for attr in reversed(chain):
            found = getattr(found, attr)
        return name in inspect.signature(found).parameters
    except (AttributeError, TypeError, ValueError):
        return False


@functools.lru_cache(maxsize=4096)
def _compared(text: str) -> tuple:
    """The fields a file compares with a value (`x.language == "python"`):
    ((field, value), ...)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ()
    return tuple(dict.fromkeys(
        (one.left.attr, one.comparators[0].value) for one in ast.walk(tree)
        if isinstance(one, ast.Compare) and len(one.ops) == 1 and isinstance(
            one.ops[0], (ast.Eq, ast.NotEq)) and isinstance(
            one.left, ast.Attribute) and isinstance(one.comparators[0],
                                                    ast.Constant)
        and isinstance(one.comparators[0].value, (str, int))))


def _given(function, where: dict) -> str:
    """The classes a function's parameters are annotated with, as written
    (`class Bug: ... language: str = "typescript"`), where they are found."""
    import inspect
    namespace = where.get("namespace", {})
    out = []
    for one in function.args.args + function.args.kwonlyargs:
        if one.annotation is None:
            continue
        found = namespace.get(ast.unparse(one.annotation).strip("'\""))
        if isinstance(found, type):
            try:
                out.append(inspect.getsource(found))
            except (OSError, TypeError):
                continue
    return "\n".join(dict.fromkeys(out))


def _identifier(said: str) -> str:
    """Words a request marks, as the name they would be in code
    (`commit message` -> `commit_message`)."""
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", said)
    return "_".join(words) if words and not said.strip().isidentifier() \
        else said.strip()


def _literal(said: str):
    import ast as _ast
    try:
        return True, _ast.literal_eval(said.strip())
    except (ValueError, SyntaxError):
        return False, None


def asked(spans: dict, part: str, path: str, text: str,
          said: str = "", project: dict | None = None) -> list:
    """The changes of the kind a request's roles name (the reader's
    `spans`, `teach_change_talk`): [(what, the part with it in as
    written)] -- a function taking what it marks `NEW` (its `DEFAULT`, or
    each plain one); the call it marks `CALLEE` given the `VALUE`; the
    `CASE` handed to the `HELPER`. None of a kind not asked for."""
    from research.v705 import repair, splice
    source = textwrap.dedent(part)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef,
                                                      ast.AsyncFunctionDef)):
        return []
    function = tree.body[0]
    made: list = []
    news = spans.get("NEW", [])
    # a name said in words read as one span each (`the commit message`:
    # `commit`, `message`) -- joined as well
    for new in news + ([" ".join(news)] if len(news) > 1 else []):
        name = _identifier(new)
        if not name.isidentifier() or keyword.iskeyword(name):
            continue
        given = [_literal(one) for one in spans.get("DEFAULT", [])]
        defaults = [value for ok, value in given if ok] or list(DEFAULTS)
        for default in defaults:
            changed = ast.parse(ast.unparse(tree))
            args = changed.body[0].args
            if args.vararg is not None:
                args.kwonlyargs.append(ast.arg(arg=name))
                args.kw_defaults.append(ast.Constant(default))
            else:
                args.args.append(ast.arg(arg=name))
                args.defaults.append(ast.Constant(default))
            made.append((f"takes {name}={default!r}", ast.unparse(changed)))
    where = _where(path)
    # a value handed on to a call and a case handed to a helper are said
    # alike (`repair should hand python bugs to _repair_python` was read
    # as the one, asked as the other), and which named thing is the call is
    # the code's to say (`give library the spec.language`: both read as
    # the call): each thing such a request names is tried as either, the
    # call one the function makes, the value one that parses, a case a
    # plain word of what is named -- the checks and the judge deciding
    named = list(dict.fromkeys(one.strip() for role in (
        "VALUE", "CALLEE", "CASE", "HELPER") for one in spans.get(role, [])))
    values = calls = named
    cases = list(dict.fromkeys(word for one in named for word in re.findall(
        r"[A-Za-z]+", one) if word.isalpha() and len(one.split()) <= 3
        and "." not in one and not one.startswith("_")))
    for value in values:
        for callee in calls:
            if value == callee:
                continue
            # the keyword the callee takes it by: its own parameter so
            # named where it can be found, else the value's own last name
            last = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", value)
            keyword_ = last[-1] if last else None
            for at, node in enumerate(ast.walk(function)):
                if not isinstance(node, ast.Call) or not ast.unparse(
                        node.func).endswith(callee.strip()) or keyword_ is None:
                    continue
                changed = ast.parse(ast.unparse(tree))
                target = list(ast.walk(changed.body[0]))[at]
                try:
                    target.keywords.append(ast.keyword(
                        arg=keyword_, value=ast.parse(value.strip(),
                                                      mode="eval").body))
                except SyntaxError:
                    continue
                made.append((f"{callee}(..., {keyword_}={value})",
                             ast.unparse(changed)))
    # a value said with no call it is passed on to (`In Solver._ops, give
    # library the spec.language` read with no call): passed by its name to
    # each call there whose function takes a parameter of that name
    if values and not any(what.startswith(tuple(f"{one}(" for one in calls))
                          for what, _ in made):
        namespace = where.get("namespace", {})
        for value in values:
            last = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", value)
            if not last:
                continue
            for at, node in enumerate(ast.walk(function)):
                if not isinstance(node, ast.Call) or any(
                        one.arg == last[-1] for one in node.keywords) or                         not _takes(node.func, last[-1], namespace):
                    continue
                changed = ast.parse(ast.unparse(tree))
                target = list(ast.walk(changed.body[0]))[at]
                try:
                    target.keywords.append(ast.keyword(
                        arg=last[-1], value=ast.parse(value.strip(),
                                                      mode="eval").body))
                except SyntaxError:
                    continue
                made.append((f"{ast.unparse(node.func)}(..., "
                             f"{last[-1]}={value})", ast.unparse(changed)))
    dispatches = []
    for case in cases:
        for helper in calls:
            if case == helper or not helper.isidentifier():
                continue
            context = repair._context(tree, text)
            context.update(where)
            helpers = [one for one in repair._helpers(function, context)
                       if one[0] == helper.strip()] or [
                (helper.strip(), [one.arg for one in function.args.args
                                  if one.arg not in ("self", "cls")])]
            wanted = case.strip().strip("'\"")
            # the fields compared with it in the module, or else each a
            # field compared with something there or one what the function
            # is given declares (`Bug.language`, no `== "python"` left in
            # editing.py once repair's own was taken out)
            fields = [field for field, _ in context.get("compared", [])] + [
                field for one in function.args.args + function.args.kwonlyargs
                for field in repair.declared(one.annotation,
                                             where.get("namespace", {}))]
            # compared with it here, or anywhere in the project: in
            # editing.py none was left once repair's own was taken out,
            # and `.language == "python"` is said in fourteen files
            everywhere = context.get("compared", []) + [
                one for other, said_ in (project or {}).items()
                if other != path and other.endswith(".py")
                for one in _compared(said_)]
            context["compared"] = [one for one in dict.fromkeys(everywhere)
                                   if str(one[1]) == wanted] or [
                (field, wanted) for field in dict.fromkeys(fields)]
            for line, what, source_ in repair._dispatched(tree, helpers,
                                                          context):
                dispatches.append((what, source_))
    # the likeliest of the cases so made by the code model, alone: the judge
    # reads `bug.name == 'python'` as it reads `bug.language == 'python'`
    dispatches = list(dict((source_, what) for what, source_ in
                           dispatches).items())
    if len(dispatches) > 1:
        from research.v705 import natural
        # the request said ahead of each: `hand python bugs` makes
        # `bug.language` likelier than `bug.source` (said nothing, the
        # model chose `bug.source == 'python'`)
        gains = natural.gained(source, [one for one, _ in dispatches],
                               said=said, given=_given(function, where))
        dispatches = [max(zip(gains, dispatches))[1]]
    made += [(what, source_) for source_, what in dispatches]
    # each change once: made from two pairs named, it is one change
    unique: dict = {}
    for what, source_ in made:
        unique.setdefault(source_, what)
    made = [(what, source_) for source_, what in unique.items()]
    out = []
    for what, source_ in made:
        put = splice.spliced(part, source_)
        if put is not None and put != part:
            out.append((what, put))
    return out


def offered(statement: str, part: str, path: str, text: str,
            held) -> list:
    """The changes of `part` (one function, as written) the request may
    ask for, made by their shape, most likely first given the request:
    [(what, the part with it in as written, gain)]."""
    from research.v705 import natural, repair, splice
    source = textwrap.dedent(part)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef,
                                                      ast.AsyncFunctionDef)):
        return []
    lines = list(range(1, len(source.splitlines()) + 1))
    found = _taking(tree, statement) + repair.candidates(
        source, lines, text, _where(path))
    found = found[:POOL]
    if not found:
        return []
    plain = ast.unparse(tree)
    sources = [one[2] for one in found]
    # how much more likely the request makes each edit than it is alone:
    # alone, an edit is likelier for being shorter (a statement taken out
    # led every request); against itself, length is the same both sides
    gains = [said - alone for said, alone in zip(
        natural.gained(plain, sources, said=statement, held=held),
        natural.gained(plain, sources, held=held))]
    out = []
    for at in sorted(range(len(found)), key=lambda at: -gains[at]):
        made = splice.spliced(part, found[at][2])
        if made is None or made == part:
            continue
        out.append((found[at][1], made, round(gains[at], 3)))
        if len(out) >= OFFERED:
            break
    return out
