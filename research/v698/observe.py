"""Pasted Python answered by running it (2026-10-20, the core made
reliable).

Pasted code was "explained" by the operators it is made of, named as the
TypeScript engine names them (`f is made of %, ===, Array.filter`), and
never run: what it printed, what it returned for 5, how many times its loop
went round, where it breaks -- each a question of what it does, answered
by what it was seen to do:

- **a function** called on what the question gives (`for 5`), else on
  inputs of the kinds its parameters are used as -- typical ones, to show
  what it does, and edge ones (empty, zero, one), to find where it breaks;
- **a script** run, what it prints kept, and how many times each loop's
  body ran (traced, not guessed);
- what a result is of its input, where it is one of its parts (`xs[1]`:
  the second element).

Run apart (`python -I`, a time limit), on a copy -- never in the server.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys

#: how long a run may take
SECONDS = 10

#: inputs by the kind a parameter is used as: (typical, edge)
INPUTS = {
    "list": ([[1, 2, 3, 4], [5, 3, 8]], [[], [7]]),
    "dict": ([{"a": 1, "b": 2}], [{}]),
    "str": (["  Hello World  ", "abc"], [""]),
    "int": ([5, 12], [0, -3]),
}

#: what runs the code: given {source, calls, trace}, says {printed, error,
#: loops: {line: count}, calls: [{call, value | raised}]}
HARNESS = r'''
import io, json, sys, contextlib, traceback
given = json.loads(sys.stdin.read())
counts = {}
watched = set(given["trace"])
def tracer(frame, event, arg):
    if frame.f_code.co_filename == "<pasted>":
        if event == "line" and frame.f_lineno in watched:
            counts[frame.f_lineno] = counts.get(frame.f_lineno, 0) + 1
        return tracer
    return None
out = {"printed": "", "error": None, "loops": {}, "calls": []}
space = {"__name__": "__pasted__"}
printed = io.StringIO()
sys.settrace(tracer)
try:
    with contextlib.redirect_stdout(printed):
        exec(compile(given["source"], "<pasted>", "exec"), space)
except BaseException as bad:
    out["error"] = f"{type(bad).__name__}: {bad}"
sys.settrace(None)
out["printed"] = printed.getvalue()[:2000]
out["loops"] = {str(k): v for k, v in counts.items()}
for call in given["calls"]:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            value = eval(call, space)
        out["calls"].append({"call": call, "value": repr(value)[:300],
                             "type": type(value).__name__})
    except BaseException as bad:
        out["calls"].append({"call": call,
                             "raised": f"{type(bad).__name__}: {bad}"[:200]})
print(json.dumps(out))
'''


def run(source: str, calls: list = (), trace: list = ()) -> dict:
    """The source run apart, then each call evaluated after it."""
    try:
        done = subprocess.run(
            [sys.executable, "-I", "-c", HARNESS], input=json.dumps(
                {"source": source, "calls": list(calls),
                 "trace": list(trace)}), capture_output=True, text=True,
            timeout=SECONDS)
    except subprocess.TimeoutExpired:
        return {"printed": "", "error": f"it ran past {SECONDS} s",
                "loops": {}, "calls": []}
    try:
        return json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"printed": "", "error": (done.stderr or "it did not run")
                [-300:], "loops": {}, "calls": []}


# -- what a parameter is used as -------------------------------------------------------

def _kinds(function: ast.FunctionDef) -> list:
    """Each parameter's kind, by how the body uses it: iterated, measured
    or indexed -- a list; `.items()` -- a dict; a string's methods -- a
    string; arithmetic -- a number."""
    out = []
    strings = set(dir(str)) - set(dir(list)) - set(dir(dict))
    for arg in function.args.args:
        name = arg.arg
        kind = None
        if arg.annotation is not None:
            said = ast.unparse(arg.annotation).lower()
            kind = next((one for one in ("list", "dict", "str", "int")
                         if one in said), None)
        for node in ast.walk(function):
            if kind:
                break
            if isinstance(node, ast.Attribute) and isinstance(
                    node.value, ast.Name) and node.value.id == name:
                kind = "dict" if node.attr in ("items", "keys", "values") \
                    else "str" if node.attr in strings else "list"
            elif isinstance(node, (ast.For, ast.comprehension)) and \
                    isinstance(node.iter, ast.Name) and node.iter.id == name:
                kind = "list"
            elif isinstance(node, ast.Call) and isinstance(
                    node.func, ast.Name) and node.func.id in (
                    "len", "sum", "sorted", "max", "min", "list", "set") \
                    and any(isinstance(one, ast.Name) and one.id == name
                            for one in node.args):
                kind = "list"
            elif isinstance(node, ast.Subscript) and isinstance(
                    node.value, ast.Name) and node.value.id == name:
                kind = "list"
            elif isinstance(node, ast.BinOp) and any(
                    isinstance(one, ast.Name) and one.id == name
                    for one in (node.left, node.right)):
                kind = "int"
            elif isinstance(node, ast.Compare) and isinstance(
                    node.left, ast.Name) and node.left.id == name:
                kind = "int"
        out.append(kind or "int")
    return out


def _given(said: str, count: int) -> list | None:
    """The arguments a question gives, outside its code (`what does this
    return for 5?`): its literals, as many as the function takes."""
    outside = re.sub(r"```.*?```", " ", said, flags=re.S)
    found = []
    for token in re.findall(r"\[[^\]]*\]|\{[^}]*\}|'[^']*'|\"[^\"]*\"|"
                            r"-?\d+(?:\.\d+)?", outside):
        try:
            found.append(ast.literal_eval(token))
        except (ValueError, SyntaxError):
            continue
    if count and len(found) == count:
        return found
    return None


def _ordinal(at: int) -> str:
    words = ("first", "second", "third", "fourth", "fifth")
    return words[at] if at < len(words) else f"{at + 1}th"


def _part_of(args: list, value: str) -> str:
    """What a result is of its input, where it is one of its parts: `the
    second element (index 1)`."""
    try:
        got = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return ""
    for one in args:
        if isinstance(one, list) and len(one) >= 2 and got in one and \
                one.count(got) == 1 and \
                type(got) is type(one[one.index(got)]):
            at = one.index(got)
            return f"the {_ordinal(at)} element (index {at})"
    return ""


# -- what it was seen to do -------------------------------------------------------------

def observed(said: str, source: str) -> dict:
    """What running pasted Python shows: {entry, asked, examples, breaks,
    printed, loops, error, facts: [what to say, in order]}."""
    try:
        tree = ast.parse(source)
    except SyntaxError as bad:
        return {"facts": [f"It is not Python I can read: {bad.msg} "
                          f"(line {bad.lineno})."], "error": bad.msg}
    functions = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef,
                                      ast.AsyncFunctionDef))]
    loops = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)) and node.body:
            loops[node.body[0].lineno] = node.lineno
    out = {"facts": [], "examples": [], "breaks": [], "entry": None}
    calls, kinds, own = [], [], []
    asked = None
    if functions:
        function = functions[-1]
        entry = function.name
        out["entry"] = entry
        kinds = _kinds(function)
        asked = _given(said, len(kinds))
        if asked is not None:
            calls.append(f"{entry}({', '.join(map(repr, asked))})")
        own = list(calls)
        if kinds:
            typical = [INPUTS[kind][0] for kind in kinds]
            edge = [INPUTS[kind][1] for kind in kinds]
            for at in range(max(len(one) for one in typical)):
                calls.append(f"{entry}(" + ", ".join(
                    repr(one[at % len(one)]) for one in typical) + ")")
            for at in range(max(len(one) for one in edge)):
                calls.append(f"{entry}(" + ", ".join(
                    repr(one[at % len(one)]) for one in edge) + ")")
        else:
            calls.append(f"{entry}()")
        # each call once: the one asked not again among the examples
        calls = own + [one for one in dict.fromkeys(calls[len(own):])
                       if one not in own]
    seen = run(source, calls, list(loops))
    out.update(printed=seen["printed"], error=seen["error"])
    if seen["error"]:
        out["facts"].append(f"Run as it is, it stops: {seen['error']}.")
    if functions and asked is not None and seen["calls"]:
        first = seen["calls"][0]
        out["asked"] = first
        out["facts"].append(
            f"{first['call']} gives {first['value']}." if "value" in first
            else f"{first['call']} raises {first['raised']}.")
    if seen["printed"].strip():
        lines = seen["printed"].strip().splitlines()
        out["facts"].append("It prints " + ", ".join(lines[:12])
                            + (" …" if len(lines) > 12 else "")
                            + (" (each on a line of its own)."
                               if len(lines) > 1 else "."))
    for body, line in sorted(loops.items(), key=lambda one: one[1]):
        count = seen["loops"].get(str(body), 0)
        if not functions or count:
            out.setdefault("loops", {})[line] = count
            out["facts"].append(f"The loop on line {line} runs {count} "
                                f"time{'s' if count != 1 else ''}.")
    rest = seen["calls"][1:] if asked is not None else seen["calls"]
    for one in rest:
        if "raised" in one:
            out["breaks"].append(one)
        else:
            out["examples"].append(one)
    if out["examples"]:
        shown = []
        for one in out["examples"][:3]:
            part = ""
            try:
                args = list(ast.literal_eval(
                    "(" + one["call"][one["call"].index("(") + 1:-1] + ",)"))
            except (ValueError, SyntaxError):
                args = []
            part = _part_of(args, one["value"])
            shown.append(f"{one['call']} == {one['value']}"
                         + (f" ({part})" if part else ""))
        out["facts"].append("Run on examples: " + "; ".join(shown) + ".")
    for one in out["breaks"][:2]:
        empty = any(token in one["call"] for token in ("([]", "({}", "(''",
                                                        "(\"\""))
        out["facts"].append(f"It breaks on {one['call']}"
                            + (" (an empty input)" if empty else "")
                            + f": {one['raised']}.")
    return out
