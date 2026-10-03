"""Code in the conversation: a request for a function, read and solved by
v696, with everything it did kept on the answer (`PLAN.md`, 2).

A request is English, and as much else as was said of it -- a TypeScript
signature, examples of calls and what they give:

    reverse the words of a sentence
    function reverseWords(s: string): string
    reverseWords("a b c") == "c b a"

Without a signature the examples give one: the name they call, the types
of what they pass and get back. Without either there is nothing to call or
to check by, and the answer asks for an example.

What is solved and how is v696's, whole (`solve`): the reader of meaning
reads the request, the six risk estimators score it and the matrix says
how it is searched (`risk.py`), the three 360M writers write programs for it
(`sketcher.proposals`, their writing kept in `state/`), and the judged
search finds what meets the examples and chooses among it (`search.py`).
The answer carries all of it -- what was read, every program written and
what became of it, the search's events, how the answer was chosen and
whether a second program confirmed it -- and the program, whole.
"""
from __future__ import annotations

import dataclasses
import json
import re
import threading
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "state"
#: where what the writers wrote is kept: a request asked again is answered
#: from it (`sketcher.CACHE`)
WRITTEN = STATE / "v697-written.jsonl"
#: the search's budget, as in v696's measurements
BUDGET = 5000
ROUNDS = 2

SIGNATURE = re.compile(
    r"function\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*:\s*([\w\[\]<>|, ]+?)\s*"
    r"(?=\{|;|$|\n)", re.M)
#: what stands between a call and what it gives, in an example
ARROW = re.compile(r"\s*(===|==|=>|->|→|should return|returns|gives)\s*")
CALL = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\(")
#: a request for code in plain words
ASKED = re.compile(
    r"\b(write|make|create|implement|code|give me|show me|need)\b[^.?!]*"
    r"\b(function|method|program|typescript)\b", re.I)


def _balanced(text: str, start: int, close: str) -> int:
    """Where the bracket opened before `start` closes: quotes and nested
    brackets skipped. -1 if it does not."""
    depth, quote, at = 1, "", start
    pairs = {"(": ")", "[": "]", "{": "}"}
    while at < len(text):
        char = text[at]
        if quote:
            if char == "\\":
                at += 1
            elif char == quote:
                quote = ""
        elif char in "\"'`":
            quote = char
        elif char in pairs:
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return at if char == close else -1
        at += 1
    return -1


def _value_end(text: str, start: int, name: str) -> int:
    """Where an example's value ends: at a line's end or a `;`, or at a
    comma or `and` before the next call of the same function -- outside
    brackets and quotes."""
    depth, quote, at = 0, "", start
    while at < len(text):
        char = text[at]
        if quote:
            if char == "\\":
                at += 1
            elif char == quote:
                quote = ""
        elif char in "\"'`":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0:
            if char in "\n;":
                return at
            rest = text[at:]
            if re.match(rf"(,|\s+and)\s+{re.escape(name)}\(", rest):
                return at
        at += 1
    return len(text)


def _literal(text: str):
    """A TypeScript literal as the value Node makes of it."""
    from research.v696.checker import checker
    row = checker().values([], [[]], [text])[0][0]
    if "error" in row:
        raise ValueError(text)
    return row["value"]


def _type(value) -> str | None:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        kinds = {_type(one) for one in value}
        if len(kinds) == 1 and None not in kinds:
            return f"{kinds.pop()}[]"
    return None


#: what a parameter is called when only its type is known
NAMES = {"string": "s", "number": "n", "boolean": "flag"}


def read(text: str) -> dict:
    """A request taken apart: {english, entry, params, returns, examples,
    signature (as given or made), made (which of them was inferred),
    missing (what it would take to answer), asked (whether it asks for
    code at all)}."""
    out = {"said": text, "entry": None, "params": [], "returns": None,
           "examples": [], "signature": None, "made": [], "missing": None}
    spans = []
    found = SIGNATURE.search(text)
    if found:
        from research.v696.tasks import _params
        out["entry"] = found.group(1)
        out["params"] = [list(one) for one in _params(found.group(2))]
        out["returns"] = found.group(3).strip()
        spans.append(found.span())
    for call in CALL.finditer(text):
        name = call.group(1)
        if text[:call.start()].rstrip().endswith("function") or \
                (out["entry"] and name != out["entry"]):
            continue
        close = _balanced(text, call.end(), ")")
        if close < 0:
            continue
        arrow = ARROW.match(text, close + 1)
        if not arrow:
            continue
        end = _value_end(text, arrow.end(), name)
        said = text[arrow.end():end].strip().rstrip(".,")
        try:
            args = _literal(f"[{text[call.end():close]}]")
            value = _literal(said)
        except ValueError:
            continue
        out["entry"] = out["entry"] or name
        out["examples"].append({"args": args, "value": value,
                                "said": text[call.start():end].strip()})
        spans.append((call.start(), end))
    english = text
    for start, end in sorted(spans, reverse=True):
        english = english[:start] + " " + english[end:]
    english = re.sub(r"\b(e\.g\.|for example|for instance|like)\s*[:,]?\s*$",
                     "", " ".join(english.split()), flags=re.I)
    out["english"] = english.strip(" ,;:.") + ("." if english.strip(
        " ,;:.") else "")
    out["asked"] = bool(found or len(out["examples"]) >= 2
                        or ASKED.search(text)
                        or (out["examples"] and ASKED.search(english)))
    if not out["params"] and out["examples"]:
        _inferred(out)
    if out["entry"] and out["returns"] and out["signature"] is None:
        said = ", ".join(f"{name}: {kind}" for name, kind in out["params"])
        out["signature"] = (f"function {out['entry']}({said}): "
                            f"{out['returns']}")
    if out["signature"] is None and out["missing"] is None:
        out["missing"] = ("how it is called: an example such as "
                          "reverse(\"ab\") == \"ba\", or its TypeScript "
                          "signature")
    return out


def _inferred(out: dict) -> None:
    """A signature from the examples: their types, position by position."""
    width = {len(one["args"]) for one in out["examples"]}
    if len(width) != 1:
        out["missing"] = "examples that call it with the same arguments"
        return
    kinds = []
    for at in range(width.pop()):
        seen = {_type(one["args"][at]) for one in out["examples"]} - {None}
        if len(seen) != 1:
            out["missing"] = (f"argument {at + 1}'s type: the examples do "
                              f"not show one (give the signature)")
            return
        kinds.append(seen.pop())
    returns = {_type(one["value"]) for one in out["examples"]} - {None}
    if len(returns) != 1:
        out["missing"] = "what it returns: the examples do not show one type"
        return
    names, params = set(), []
    for kind in kinds:
        base = "xs" if kind.endswith("[]") else NAMES.get(kind, "x")
        name, n = base, 2
        while name in names:
            name, n = f"{base}{n}", n + 1
        names.add(name)
        params.append([name, kind])
    out["params"], out["returns"] = params, returns.pop()
    out["made"] = ["signature"]


# -- the models, loaded once ---------------------------------------------------

class Tools:
    """The reader of meaning, the risk estimators sharing its encoder, and
    the writers -- each loaded the first time it is needed, and kept."""

    _lock = threading.Lock()
    _one = None

    def __init__(self) -> None:
        from research.v696 import reader as R
        from research.v696 import risk
        self.reader = R.Reader.load(R.LLM / risk.READER)
        self.estimators = risk.Estimators(
            features=risk.Features(reader=self.reader))
        self.writers: dict = {}

    @classmethod
    def get(cls) -> "Tools":
        with cls._lock:
            if cls._one is None:
                cls._one = Tools()
            return cls._one

    def writer(self, name: str):
        from research.v696 import sketcher
        if name not in self.writers:
            self.writers[name] = sketcher.Sketcher(sketcher.LLM / name)
        return self.writers[name]


# -- solving, everything kept --------------------------------------------------

def _plain(value):
    """What JSON keeps of a value: tuples as lists, the rest as text."""
    if isinstance(value, dict):
        return {str(key): _plain(one) for key, one in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(one) for one in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _top(probs: dict, most: int = 10, floor: float = 0.05) -> list:
    return [[name, round(p, 3)] for name, p in sorted(
        probs.items(), key=lambda one: -one[1])[:most] if p >= floor]


def solve(text: str) -> dict:
    """Everything done for a request, and its answer (`code`)."""
    from research.v696 import risk, sketcher
    from research.v696.experiment import CONFIGS
    from research.v696.parse import parse
    from research.v696.reader import request as said_as
    from research.v696.search import Solver
    from research.v696.spec import Spec

    started = time.time()
    timings = {}
    asked = read(text)
    out = {"request": asked, "timings": timings}
    if asked["signature"] is None:
        out["answer"] = {"status": "missing", "code": None}
        return out
    name = f"chat-{asked['entry']}-{zlib.crc32(text.encode()):08x}"
    spec = Spec(name, [tuple(one) for one in asked["params"]],
                asked["returns"],
                [(list(one["args"]), one["value"])
                 for one in asked["examples"]],
                entry=asked["entry"], english=asked["english"])
    tools = Tools.get()

    # the request read: what the reader of meaning expects of its program
    probs = tools.reader.read([said_as(spec)])[0]
    spec.expected = {"uses": probs["uses"], "behaviour": probs["behaviour"],
                     "probs": probs}
    out["read"] = {"returns": _top(probs["returns"], 3),
                   "uses": _top(probs["uses"]),
                   "behaviour": _top(probs["behaviour"]),
                   "root": _top(probs["root"], 3)}
    timings["read"] = round(time.time() - started, 2)

    # its risks, and what the matrix says of its search
    mark = time.time()
    from research.v696.risk import _surface
    spec.risk = tools.estimators(
        [said_as(spec)], [_surface(spec.english, spec.params, spec.returns,
                                   spec.examples)])[0]
    spec.moves = risk.moves(spec.risk, risk.MOVES - {"budget"})
    out["risk"] = {"scores": spec.risk,
                   "names": risk.NAMES, "high": risk.HIGH,
                   "cells": list(spec.moves.cells),
                   "moves": sorted(spec.moves.on),
                   "matrix": {cell: sorted(
                       set().union(*(risk.DIAGONAL[f] for f in
                                     cell.split("×")))
                       | risk.PAIRED.get(tuple(cell.split("×")), set()))
                       for cell in spec.moves.cells}}
    timings["risk"] = round(time.time() - mark, 2)

    # the writers: every one asked once, whatever met before it -- the
    # matrix's `readings`, for every request here: what is answered on the
    # page says whether a second writer arrived at the same (confirmed) --
    # and asked again where nothing yet meets the examples
    mark = time.time()
    STATE.mkdir(exist_ok=True)
    sketcher.CACHE = WRITTEN
    names = sketcher.PROPOSERS.split(",")
    searched = spec.moves
    spec.moves = dataclasses.replace(searched,
                                     on=searched.on | {"readings"})
    try:
        for at, one in enumerate(names):
            sketcher.proposals(tools.writer(one), [spec], rounds=ROUNDS,
                               keep=at > 0)
    finally:
        spec.moves = searched
    out["writers"] = _writers(spec, names, parse)
    timings["write"] = round(time.time() - mark, 2)

    # the search: judged, the writers' programs admitted first
    mark = time.time()
    answer = {"status": "unverified", "code": None}
    if spec.examples:
        solver = Solver(CONFIGS["meet+repair+forms+proposals"],
                        budget=BUDGET)
        got = solver.solve(spec)
        out["search"] = {"route": got.route, "evaluated": got.evaluated,
                         "seconds": round(got.seconds, 2),
                         "refused": got.rejected,
                         "confirmed": got.confirmed,
                         "events": _plain(solver.events)}
        if got.program is not None:
            answer = _answer(spec, got.program, out["writers"])
            answer.update(status="confirmed" if got.confirmed else "met",
                          route=got.route)
        else:
            answer = {"status": "unsolved", "code": None}
    elif spec.proposals:
        # nothing to check by: what the first writer wrote that runs
        answer = _answer(spec, spec.proposals[0], out["writers"])
        answer["status"] = "unverified"
    timings["search"] = round(time.time() - mark, 2)
    timings["total"] = round(time.time() - started, 2)
    out["answer"] = answer
    return _plain(out)


def _writers(spec, names: list, parse) -> list:
    """Every program each writer wrote for the request, round by round, as
    written -- and whether it read into a tree, ran, met the examples."""
    from research.v696 import sketcher
    from research.v696.spec import Spec
    out = []
    kept = sketcher._cached()
    for name in names:
        rounds = []
        for key, written in kept.items():
            writer, request, round_, _ = key.split("|")
            if writer != name or request != spec.name:
                continue
            programs = []
            for text in written:
                alone = Spec(spec.name, spec.params, spec.returns,
                             spec.examples, entry=spec.entry)
                sketcher._read(alone, [text], parse)
                tree = alone.proposals[0] if alone.proposals else None
                programs.append({
                    "text": alone.sources[0] if alone.sources else text,
                    "read": tree is not None,
                    "program": tree.source() if tree is not None else None,
                    "meets": bool(tree is not None and spec.examples
                                  and sketcher._meets(spec, tree))})
            rounds.append({"round": int(round_), "programs": programs})
        if rounds:
            out.append({"writer": name, "rounds": sorted(
                rounds, key=lambda one: one["round"])})
    return out


def _answer(spec, program, writers: list) -> dict:
    """The answer, whole: as a writer wrote it where it is one of theirs
    (people's code reads better than the search's tree), and as the search
    holds it -- printed exactly from its tree, helpers first."""
    printed = spec.function(program)
    written = next((one["text"] for writer in writers
                    for round_ in writer["rounds"]
                    for one in round_["programs"]
                    if one["program"] == program.source()), None)
    return {"code": written or printed, "printed": printed,
            "as": "written" if written else "printed", "entry": spec.entry,
            "program": program.source(),
            "examples": len(spec.examples)}


def spoken(found: dict) -> str:
    """What is said of it, in a sentence: the code is on the answer."""
    asked, answer = found["request"], found["answer"]
    entry = asked.get("entry") or "it"
    count = len(asked["examples"])
    examples = f"{count} example{'s' if count != 1 else ''}"
    status = answer["status"]
    if status == "missing":
        return f"Show me {asked['missing']}."
    if status == "confirmed":
        return (f"Here is {entry}: it meets your {examples}, and a second "
                f"program, written apart, does the same beyond them.")
    if status == "met":
        return (f"Here is {entry}: it meets your {examples}, but nothing "
                f"written apart confirms it beyond them -- check it.")
    if status == "unverified":
        if answer.get("code"):
            return (f"Here is {entry}, as written and unchecked: give me an "
                    f"example call and what it should return, and I will "
                    f"check it.")
        return f"Nothing I wrote for {entry} reads as a program."
    written = sum(len(r["programs"]) for w in found.get("writers", ())
                  for r in w["rounds"])
    return (f"I could not write {entry} so that it meets your {examples}: "
            f"{written} programs were written and none met them, and the "
            f"search found nothing in "
            f"{found.get('search', {}).get('evaluated', 0)} candidates.")


def answered(text: str) -> dict:
    """The conversation's answer to a request for code: said in a
    sentence, with everything done for it (`code`)."""
    try:
        found = solve(text)
    except Exception as bad:                       # noqa: BLE001
        import traceback
        found = {"request": read(text), "answer": {
            "status": "failed", "code": None,
            "error": f"{type(bad).__name__}: {bad}",
            "trace": traceback.format_exc()[-2000:]}}
        return {"outcome": "acted", "source": "programming",
                "text": f"Writing it failed: {type(bad).__name__}: {bad}",
                "spoken": f"Writing it failed: {type(bad).__name__}.",
                "code": found}
    said = spoken(found)
    return {"outcome": "acted", "source": "programming", "text": said,
            "spoken": said, "code": found}


if __name__ == "__main__":
    import sys
    print(json.dumps(answered(" ".join(sys.argv[1:]) or
                              'reverse a string\nrev("ab") == "ba"\n'
                              'rev("abc") == "cba"'), indent=1)[:6000])
