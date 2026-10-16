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
