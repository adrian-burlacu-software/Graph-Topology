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
    return {"namespace": vars(module)} if module else {}


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


def asked(spans: dict, part: str, path: str, text: str) -> list:
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
    for new in spans.get("NEW", []):
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
    for value in spans.get("VALUE", []):
        for callee in spans.get("CALLEE", []):
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
    for case in spans.get("CASE", []):
        for helper in spans.get("HELPER", []):
            context = repair._context(tree, text)
            context.update(where)
            helpers = [one for one in repair._helpers(function, context)
                       if one[0] == helper.strip()] or [
                (helper.strip(), [one.arg for one in function.args.args
                                  if one.arg not in ("self", "cls")])]
            wanted = case.strip().strip("'\"")
            context["compared"] = [one for one in context.get(
                "compared", []) if str(one[1]) == wanted] or [
                (field, wanted) for field, _ in context.get("compared", [])]
            for line, what, source_ in repair._dispatched(tree, helpers,
                                                          context):
                made.append((what, source_))
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
