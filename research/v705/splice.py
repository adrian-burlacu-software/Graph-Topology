"""A change made on a function's tree, put into its text as written
(v705): only the statements that differ are written again, each at its own
indentation; the `def` line where the signature changed; every other line
-- its comments, its layout -- as it was. An edit made by its shape
(`repair.candidates`) is made on the tree and written back whole by
`ast.unparse`, which drops every comment: written so into a file, one
changed operator rewrote the function.

    spliced(original, made)   the original text with the change in, or None
"""
from __future__ import annotations

import ast
import difflib
import textwrap


def _function(tree):
    found = tree.body[0] if tree.body else None
    return found if isinstance(found, (ast.FunctionDef,
                                       ast.AsyncFunctionDef)) else None


def _said(node, indent: str) -> list:
    """A statement written out at an indentation, as lines."""
    return [indent + line if line else line
            for line in ast.unparse(node).splitlines()]


def _header(function) -> str:
    """The `def` line of a function as its tree says it."""
    bare = ast.FunctionDef(name=function.name, args=function.args,
                           body=[ast.Pass()], decorator_list=[],
                           returns=function.returns, type_comment=None,
                           type_params=getattr(function, "type_params", []))
    if isinstance(function, ast.AsyncFunctionDef):
        bare = ast.AsyncFunctionDef(**{key: getattr(bare, key)
                                       for key in bare._fields})
    ast.fix_missing_locations(bare)
    return ast.unparse(bare).splitlines()[0]


def _appended(old, new) -> list | None:
    """The parameters `new` has after all of `old`'s, as written (`message:
    str = ''`), where that is all that changed; None otherwise."""
    if old.vararg or old.kwonlyargs or old.kwarg or old.posonlyargs or \
            new.vararg or new.kwonlyargs or new.kwarg or new.posonlyargs:
        return None
    if len(new.args) <= len(old.args) or [ast.dump(one) for one in new.args[
            :len(old.args)]] != [ast.dump(one) for one in old.args]:
        return None
    defaults = [None] * (len(new.args) - len(new.defaults)) + list(
        new.defaults)
    kept = [None] * (len(old.args) - len(old.defaults)) + list(old.defaults)
    if [ast.dump(one) if one else None for one in defaults[:len(old.args)]] \
            != [ast.dump(one) if one else None for one in kept]:
        return None
    out = []
    for arg, default in zip(new.args[len(old.args):],
                            defaults[len(old.args):]):
        said = ast.unparse(arg)
        if default is not None:
            said += (" = " if arg.annotation else "=") + ast.unparse(default)
        out.append(said)
    return out


def _bodies(node) -> list:
    """The statement lists a statement holds, by field."""
    return [(name, value) for name, value in ast.iter_fields(node)
            if isinstance(value, list) and value and all(
                isinstance(one, ast.stmt) for one in value)]


def _same_shell(one, other) -> bool:
    """Two compound statements alike but for what their bodies hold."""
    if type(one) is not type(other):
        return False

    def said(value):
        if isinstance(value, ast.AST):
            return ast.dump(value)
        if isinstance(value, list):
            return [said(item) for item in value]
        return value
    for name, value in ast.iter_fields(one):
        if isinstance(value, list) and value and all(
                isinstance(item, ast.stmt) for item in value):
            continue
        if said(value) != said(getattr(other, name)):
            return False
    return True


def _replace_in(lines: list, was: list, now: list, indent: str, edits: list,
                after_line: int) -> bool:
    """The edits that turn the statement list `was` into `now`: [(first
    line, last line, new lines)] in `lines`' numbering (from 1) -- by the
    statements' dumps lined up, a statement that differs written again,
    or gone into, where only its bodies differ."""
    dumps_was = [ast.dump(one) for one in was]
    dumps_now = [ast.dump(one) for one in now]
    matcher = difflib.SequenceMatcher(None, dumps_was, dumps_now,
                                      autojunk=False)
    for kind, a0, a1, b0, b1 in matcher.get_opcodes():
        if kind == "equal":
            continue
        if kind == "replace" and a1 - a0 == b1 - b0:
            for old, new in zip(was[a0:a1], now[b0:b1]):
                if _same_shell(old, new):
                    for (name, inner), (_, inner_new) in zip(
                            _bodies(old), _bodies(new)):
                        first = inner[0]
                        inside = lines[first.lineno - 1][:first.col_offset]
                        if not _replace_in(lines, inner, inner_new, inside,
                                           edits, first.lineno - 1):
                            return False
                else:
                    edits.append((old.lineno, old.end_lineno,
                                  _said(new, indent)))
            continue
        new_lines = [line for one in now[b0:b1] for line in _said(one, indent)]
        if a1 > a0:
            edits.append((was[a0].lineno, was[a1 - 1].end_lineno, new_lines))
        else:
            # put in after the statement before, or first in the block
            at = was[a0 - 1].end_lineno if a0 else after_line
            edits.append((at + 1, at, new_lines))
    return True


def spliced(original: str, made: str) -> str | None:
    """The original function's text with the change `made` makes in it,
    every line it does not change as written; None where the two are not
    one function or the change cannot be put in so."""
    indent = original[:len(original) - len(original.lstrip())]
    indent = indent.split("\n")[-1]
    source = textwrap.dedent(original)
    try:
        before, after = ast.parse(source), ast.parse(textwrap.dedent(made))
    except SyntaxError:
        return None
    old, new = _function(before), _function(after)
    if old is None or new is None:
        return None
    lines = source.splitlines()
    edits: list = []
    if ast.dump(old.args) != ast.dump(new.args) or (
            ast.dump(old.returns) if old.returns else None) != (
            ast.dump(new.returns) if new.returns else None):
        # the `def` line -- over several lines, all of them to the one
        # ending `:` -- written again
        head = old.lineno
        end = head
        while end <= len(lines) and not lines[end - 1].rstrip().endswith(":"):
            end += 1
        added = _appended(old.args, new.args)
        if added is not None and (ast.dump(old.returns) if old.returns
                                  else None) == (ast.dump(new.returns)
                                                 if new.returns else None):
            # only parameters put on the end: those written in before the
            # `)` closing the ones there, the rest of the line as it was
            # (`said: str = ""` was written `said: str=''` again)
            said = "\n".join(lines[head - 1:end])
            close = said.rfind(")", 0, said.rfind("->") if "->" in said
                               else said.rfind(":"))
            if close > 0:
                before_ = said[:close].rstrip()
                joined = ", " if not before_.endswith("(") else ""
                said = before_ + joined + ", ".join(added) + said[close:]
                edits.append((head, end, said.split("\n")))
            else:
                edits.append((head, end, [lines[head - 1][:old.col_offset] +
                                          _header(new)]))
        else:
            edits.append((head, end, [lines[head - 1][:old.col_offset] +
                                      _header(new)]))
    body_indent = lines[old.body[0].lineno - 1][:old.body[0].col_offset]
    if not _replace_in(lines, old.body, new.body, body_indent, edits,
                       old.body[0].lineno - 1):
        return None
    for first, last, new_lines in sorted(edits, key=lambda one: -one[0]):
        lines[first - 1:last] = new_lines
    out = "\n".join(lines) + ("\n" if source.endswith("\n") else "")
    try:
        if ast.dump(ast.parse(out)) != ast.dump(after):
            return None
    except SyntaxError:
        return None
    return textwrap.indent(out, indent) if indent else out
