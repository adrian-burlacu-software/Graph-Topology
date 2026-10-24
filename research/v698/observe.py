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
import difflib
import json
import random
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


def _written_call(said: str, entry: str) -> str | None:
    """The call of `entry` a question writes out, outside its code, as it
    writes it -- read by Python's own parser, so its brackets and strings
    are whole (`f((1, ), (1, ), (1, 2))`, `f(dict(e=1), 'u')`, `f(range(100,
    120))`): its literals picked out one by one lost tuples, nested lists
    and every argument that is no literal (CRUXEval: 25 of 800 asked calls
    run wrong or not at all)."""
    outside = re.sub(r"```.*?```", " ", said, flags=re.S)
    for found in re.finditer(rf"(?<![\w.]){re.escape(entry)}\s*\(", outside):
        start = found.start()
        for end in range(found.end(), len(outside)):
            if outside[end] != ")":
                continue
            text = outside[start:end + 1]
            try:
                node = ast.parse(text, mode="eval").body
            except SyntaxError:
                continue
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == entry:
                return text
            break
    return None


def literals(said: str) -> list:
    """The Python literals a question says outside its code, each whole --
    the longest piece from each place one may begin that Python reads as a
    literal (`[[1, 2], []]`, `'a, b'`, `False` -- a truth asked was no
    value, 47 of CRUXEval's 49 then answered by explaining)."""
    outside = re.sub(r"```.*?```", " ", said, flags=re.S)
    out, at = [], 0
    while at < len(outside):
        if (outside[at] not in "[{('\"-0123456789" and not re.match(
                r"(True|False|None)\b", outside[at:])) or (
                at and (outside[at - 1].isalnum() or outside[at - 1] == "_")):
            at += 1
            continue
        for end in range(len(outside), at, -1):
            piece = outside[at:end].rstrip(" ?.!,;:")
            try:
                value = ast.literal_eval(piece)
            except (ValueError, SyntaxError, TypeError, MemoryError,
                    RecursionError):
                continue
            out.append(value)
            at += len(piece)
            break
        else:
            at += 1
    return out


#: what the code model is asked of an input that gives a result
INPUT = ("Here is a Python function:\n```python\n{code}\n```\nGive an input "
         "for which {entry} returns {wanted}. Answer with the call alone, as "
         "{entry}(...).")
#: inputs the code model proposes, each run
TRIED = 8


def found_input(source: str, entry: str, wanted, model=None) -> dict:
    """An input that gives `wanted`, found by running: first the search
    alone (`searched`, from inputs of the kinds its parameters are used as
    and the pieces of what is wanted), then from the calls the code model
    proposes -- what is said is what was run, never a guess (CRUXEval input
    prediction: of the 134 its eight proposed calls missed, 86 found by the
    search alone). {call, tried} or {tried} where none gives it."""
    found = searched(source, entry, wanted, _seeds(source, entry, wanted))
    if found.get("call"):
        return found
    if model is None:
        from research.v705 import coder
        model = coder.big()
    asked = INPUT.format(code=source.rstrip(), entry=entry,
                         wanted=repr(wanted))
    replies = model.write([asked], longest=120, samples=TRIED,
                          temperature=0.8)[0]
    calls = []
    for reply in replies if isinstance(replies, list) else [replies]:
        call = _written_call(reply.replace("```python", " ").replace(
            "```", " "), entry)
        if call and call not in calls:
            calls.append(call)
    again = searched(source, entry, wanted, calls)
    return {**again, "tried": found["tried"] + again["tried"]}


# -- an input searched for --------------------------------------------------------------

#: calls run at once; rounds of growing the nearest; how many nearest are
#: grown each round, and how many calls each argument of one grows
BATCH, ROUNDS, NEAREST, EACH = 120, 4, 4, 40


def _likeness(given, wanted) -> float:
    """How near a result is to the one wanted: 2 when it is it, else how
    alike they are written (a tenth more for the same type)."""
    if type(given) is type(wanted) and given == wanted:
        return 2.0
    return difflib.SequenceMatcher(None, repr(given), repr(wanted)).ratio() \
        + (0.1 if type(given) is type(wanted) else 0)


def _arguments(call: str) -> list | None:
    """A call's arguments as values, where each is one written out."""
    try:
        node = ast.parse(call, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Call) or node.keywords:
        return None
    try:
        return [ast.literal_eval(one) for one in node.args]
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def _call(entry: str, args: list) -> str:
    return f"{entry}({', '.join(repr(one) for one in args)})"


def _pieces(wanted) -> list:
    """Values a wanted result may be made from: itself, turned round, an end
    off, its parts, its length; a number's neighbours."""
    out = [wanted]
    if isinstance(wanted, (str, list, tuple)):
        out += [wanted[::-1], wanted[1:], wanted[:-1], len(wanted)]
        out += list(wanted)[:12]
        if isinstance(wanted, (list, tuple)):
            out.append(list(wanted))
            out += [str(one) for one in wanted[:6]]
        else:
            out += [list(wanted), wanted.strip(), wanted.lower(),
                    wanted.upper(), wanted.swapcase(), wanted.split()]
    elif isinstance(wanted, dict):
        out += [list(wanted), list(wanted.values()), list(wanted.items())]
        out += list(wanted)[:6] + list(wanted.values())[:6]
    elif isinstance(wanted, bool):
        out += [not wanted, 0, 1, ""]
    elif isinstance(wanted, (int, float)):
        out += [wanted + 1, wanted - 1, -wanted, wanted * 2, 0, 1]
        if wanted:
            out.append(wanted // 2)
        out.append(str(wanted))
    return out


def _constants(source: str) -> list:
    """The values the code itself writes: its strings and numbers (`find(
    ",")` wants a comma in what it is given)."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(
                node.value, (str, int, float)) and not isinstance(
                node.value, bool) and node.value not in out:
            out.append(node.value)
    return out[:20]


def _as(value, seq):
    if isinstance(value, str):
        return "".join(str(one) for one in seq)
    return tuple(seq) if isinstance(value, tuple) else seq


def _near(value, wanted, rng, own=()) -> list:
    """Values one small edit from a value: an element or letter dropped,
    put in or changed (from the result wanted or the code's own values),
    the whole turned round, doubled, halved or made as long as what is
    wanted; a number's neighbours; a key of a dict left out."""
    out = []
    parts = _pieces(wanted) + list(own)
    if isinstance(value, (str, list, tuple)):
        seq = list(value)
        out += [value[::-1], value + value, value[:len(value) // 2]]
        if isinstance(wanted, (str, list, tuple)) and seq and \
                len(wanted) != len(seq):
            out.append(_as(value, (seq * len(wanted))[:len(wanted)]))
        for at in range(len(seq) + 1):
            if at < len(seq):
                out.append(_as(value, seq[:at] + seq[at + 1:]))
            fill = rng.choice((list(wanted) if isinstance(
                wanted, (str, list, tuple)) and wanted else parts) +
                list(own))
            if isinstance(value, str):
                fill = fill if isinstance(fill, str) else str(fill)[:1]
            out.append(_as(value, seq[:at] + [fill] + seq[at:]))
            if at < len(seq):
                out.append(_as(value, seq[:at] + [fill] + seq[at + 1:]))
    elif isinstance(value, bool):
        out.append(not value)
    elif isinstance(value, (int, float)):
        out += [value + 1, value - 1, -value, value * 2,
                value // 2 if isinstance(value, int) else value / 2]
    elif isinstance(value, dict):
        for key in list(value)[:6]:
            out.append({k: v for k, v in value.items() if k != key})
    out += [one for one in parts if type(one) is type(value)]
    return out


def _seeds(source: str, entry: str, wanted) -> list:
    """Calls to begin from: inputs of the kinds its parameters are used as,
    what is wanted given as the first argument or as every one."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    function = next((node for node in tree.body if isinstance(
        node, ast.FunctionDef) and node.name == entry), None)
    if function is None or not function.args.args:
        return []
    kinds = [kind or "str" for kind in _kinds(function)]
    out = [_call(entry, [INPUTS[kind][at][0] for kind in kinds])
           for at in (0, 1)]
    rest = [INPUTS[kind][0][0] for kind in kinds[1:]]
    out += [_call(entry, [one] + rest) for one in _pieces(wanted)[:8]]
    out += [_call(entry, [one] * len(kinds)) for one in _pieces(wanted)[:4]]
    return out


def searched(source: str, entry: str, wanted, calls: list,
             seed: int = 0) -> dict:
    """An input that gives `wanted`, by running: the calls run, the nearest
    misses grown by one small edit of an argument, the nearest of those
    grown again (four rounds). {call, tried} or {tried}."""
    rng = random.Random(seed)
    try:
        own = _constants(source)
    except SyntaxError:
        return {"tried": 0}
    seen, scored, tried, todo = set(), [], 0, list(calls)
    for _ in range(ROUNDS + 1):
        todo = [one for one in dict.fromkeys(todo) if one not in seen][
            :BATCH * 3]
        if not todo:
            break
        for at in range(0, len(todo), BATCH):
            batch = todo[at:at + BATCH]
            seen.update(batch)
            tried += len(batch)
            for one in run(source, batch).get("calls") or ():
                try:
                    given = ast.literal_eval(one.get("value", ""))
                except (ValueError, SyntaxError, TypeError, MemoryError,
                        RecursionError):
                    continue
                like = _likeness(given, wanted)
                if like == 2.0:
                    return {"call": one["call"], "tried": tried}
                scored.append((like, one["call"]))
        scored.sort(key=lambda one: -one[0])
        todo = []
        for _, call in scored[:NEAREST]:
            args = _arguments(call) or []
            made = []
            for place, value in enumerate(args):
                for other in _near(value, wanted, rng, own):
                    made.append(_call(entry, args[:place] + [other] +
                                      args[place + 1:]))
            rng.shuffle(made)
            todo += made[:EACH * max(1, len(args))]
    return {"tried": tried}


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
        # the function the question names, else the last defined (CRUXEval
        # defines helpers after `f`: the question read with the wrong name)
        outside = re.sub(r"```.*?```", " ", said, flags=re.S)
        named = [one for one in functions
                 if re.search(rf"(?<![\w.]){re.escape(one.name)}(?!\w)",
                              outside)]
        function = (named or functions)[-1]
        entry = function.name
        out["entry"] = entry
        kinds = _kinds(function)
        written = _written_call(said, entry)
        asked = None if written else _given(said, len(kinds))
        if written:
            # the call as the question writes it; its arguments, where each
            # is a literal, for what the result is of them
            calls.append(written)
            try:
                asked = [ast.literal_eval(one) for one in
                         ast.parse(written, mode="eval").body.args]
            except (ValueError, SyntaxError, TypeError):
                asked = []
        elif asked is not None:
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
