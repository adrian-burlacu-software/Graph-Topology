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


#: the uses of a parameter put in that are offered, of how many made
USES, USES_POOL = 3, 200


def _used(taken: str, name: str, text: str, said: str, path: str) -> list:
    """[(what, source)]: the function with `name` taken (`taken`) and read
    by one edit more, the likeliest `USES` by the code model after the
    request."""
    from research.v705 import repair
    try:
        tree = ast.parse(taken)
    except SyntaxError:
        return []
    lines = list(range(1, len(taken.splitlines()) + 1))
    read_before = _read(tree) & {one.arg for one in ast.walk(tree)
                                 if isinstance(one, ast.arg)}
    found = []
    for _, what, source_ in repair.candidates(taken, lines, text,
                                              _where(path)):
        try:
            now = _read(ast.parse(source_))
        except SyntaxError:
            continue
        # it read, and every parameter read before still read: `said ->
        # commit` left `said` unread, what every caller gives it lost
        if name in now and read_before <= now:
            found.append((f"takes {name}, {what}", source_))
    # beside a parameter where that is read (`said or commit_message`),
    # its default as that one's: what stands in for it where it is not
    # given (taken as `commit=0`, it was of no kind beside `said: str = ""`)
    function = tree.body[0]
    defaults = dict(zip([one.arg for one in function.args.args][::-1],
                        function.args.defaults[::-1]))
    defaults.update({one.arg: value for one, value in zip(
        function.args.kwonlyargs, function.args.kw_defaults) if value})
    # (what a parameter is, as its annotation or a default of a value says
    # -- `None` says only that it may be left out: `held or commit`, of
    # `held=None`, was the code model's likeliest)
    kin = {one for one, value in defaults.items() if one != name and
           isinstance(value, ast.Constant) and value.value is not None}
    for at, node in enumerate(ast.walk(function)):
        if not (isinstance(node, ast.Name) and node.id in kin and
                isinstance(node.ctx, ast.Load)):
            continue
        for first, second in ((node.id, name), (name, node.id)):
            changed = ast.parse(taken)
            _default(changed.body[0], name, defaults[node.id].value)
            target = list(ast.walk(changed.body[0]))[at]
            for parent in ast.walk(changed.body[0]):
                for field, value in ast.iter_fields(parent):
                    if value is target:
                        setattr(parent, field, ast.parse(
                            f"({first} or {second})", mode="eval").body)
                    elif isinstance(value, list) and any(
                            one is target for one in value):
                        value[[one is target for one in value].index(
                            True)] = ast.parse(f"({first} or {second})",
                                               mode="eval").body
            found.insert(0, (f"takes {name}, {node.id} -> {first} or "
                                f"{second}", ast.unparse(changed)))
    found = list({one[1]: one for one in found}.values())[:USES_POOL]
    if len(found) <= USES:
        return found
    from research.v705 import natural
    gains = natural.gained(ast.unparse(tree), [one[1] for one in found],
                           said=said, given=_given(tree.body[0],
                                                   _where(path)))
    ranked = sorted(range(len(found)), key=lambda at: -gains[at])
    return [found[at] for at in ranked[:USES]]


def _read(tree) -> set:
    """The names a tree reads."""
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)}


def _default(function, name: str, value) -> None:
    """A parameter's default made `value`, in place."""
    args = function.args
    if name in [one.arg for one in args.kwonlyargs]:
        at = [one.arg for one in args.kwonlyargs].index(name)
        args.kw_defaults[at] = ast.Constant(value)
        return
    names = [one.arg for one in args.args]
    if name in names:
        at = names.index(name) - (len(names) - len(args.defaults))
        if at >= 0:
            args.defaults[at] = ast.Constant(value)


def _likeliest(tree, made: list, said: str, where: dict) -> tuple:
    """Of changes alike but for one choice ([(what, source)]), the one the
    code model finds likeliest after the request and what the function is
    given -- each scored against the function as unparsed, as they are
    (as written, every window is the whole function)."""
    from research.v705 import natural
    gains = natural.gained(ast.unparse(tree), [one for _, one in made],
                           said=said, given=_given(tree.body[0], where))
    best = max(range(len(made)), key=lambda at: gains[at])
    return made[best]


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
    # `commit`, `message`) -- joined as well, where said one after the
    # other (not `commit message window`, of `... and score each window`)
    lowered = " ".join(said.lower().split())
    joined = [" ".join(news[first:last]) for first in range(len(news))
              for last in range(first + 2, len(news) + 1)
              if " ".join(news[first:last]).lower() in lowered]
    for new in news + joined:
        name = _identifier(new)
        if not name.isidentifier() or keyword.iskeyword(name):
            continue
        given = [_literal(one) for one in spans.get("DEFAULT", [])]
        defaults = [value for ok, value in given if ok] or list(DEFAULTS)
        taking = []
        for default in defaults:
            changed = ast.parse(ast.unparse(tree))
            args = changed.body[0].args
            if args.vararg is not None:
                args.kwonlyargs.append(ast.arg(arg=name))
                args.kw_defaults.append(ast.Constant(default))
            else:
                args.args.append(ast.arg(arg=name))
                args.defaults.append(ast.Constant(default))
            taking.append((f"takes {name}={default!r}", ast.unparse(changed)))
        # no default said: the likeliest by the code model, the request
        # ahead -- the judge read `commit=None`, `commit=''`, `commit=0`
        # alike (0.28 to 0.36), so none was clearly the change
        if len(taking) > 1:
            taking = [_likeliest(tree, taking, said, _where(path))]
        made += taking
        # and taken to be used: the parameter put in, and one edit of the
        # function's own kinds reading it (a name it stands for, `x or
        # it`, put on a string, given to a call taking it) -- `take the
        # commit message and score each window after it`: taking it is
        # half of what was asked
        if taking:
            made += _used(taking[0][1], name, text, said, path)
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
    # (the request said ahead of each: `hand python bugs` makes
    # `bug.language` likelier than `bug.source`)
    if len(dispatches) > 1:
        dispatches = [_likeliest(tree, [(what, source_) for source_, what
                                        in dispatches], said, where)[::-1]]
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
