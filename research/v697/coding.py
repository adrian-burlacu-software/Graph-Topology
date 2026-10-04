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
#: What later layers know of a request, for writing its code: callables
#: from the request as read (`read`) to {"english": said to the writers,
#: "library": operators offered to the search, "used": what was used}.
#: v698's taught concepts register here (`knowledge.context`).
CONTEXT: list = []


def known_of(asked: dict) -> dict:
    """Everything later layers know of a request, together."""
    out = {"english": "", "library": [], "used": []}
    for one in CONTEXT:
        found = one(asked) or {}
        if found.get("english"):
            out["english"] = " ".join(filter(None, (out["english"],
                                                    found["english"])))
        out["library"] += list(found.get("library", ()))
        out["used"] += list(found.get("used", ()))
    return out


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
#: a request to make code: a word for making it, then what is made. *Can
#: you find any bugs in the code* is a question about code, not a request
#: for some: `can you` and `code` alone do not make one.
ASKED = re.compile(
    r"\b(write|make|create|implement|code up|give me|i (?:want|need|'d "
    r"like)|we need|build|generate)\b[^.?!]*"
    r"\b(function|method|program|script|snippet|typescript code|"
    r"code (?:that|to|which|for))\b", re.I)
#: a request to print, not to return: what it prints is what it does
PRINTS = re.compile(r"\b(print(s|ed|ing)?|outputs?|displays?|console)\b",
                    re.I)
#: what the request's own words say comes out: `prints out "Hello World"`
QUOTED = re.compile(
    r"\b(print(?:s|ed|ing)?|outputs?|displays?|logs?|says?|returns?|"
    r"gives? back)\b(?:\s+out)?(?:\s+(?:the\s+)?(?:text|string|words?|"
    r"message|line))?\s*:?\s*[\"\u201c']([^\"\u201d']+)[\"\u201d']", re.I)


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
    yours = None
    if found:
        from research.v696.tasks import _params
        out["entry"] = found.group(1)
        out["params"] = [list(one) for one in _params(found.group(2))]
        out["returns"] = found.group(3).strip()
        spans.append(found.span())
        # a whole function, body and all: the person's own code (`what
        # about function day(n) { ... }`) -- a program to try, not words
        body = re.match(r"\s*\{", text[found.end():])
        if body:
            close = _balanced(text, found.end() + body.end(), "}")
            if close > 0:
                yours = (found.start(), close + 1)
                out["yours"] = text[found.start():close + 1]
                spans[-1] = yours
    for call in CALL.finditer(text):
        name = call.group(1)
        if text[:call.start()].rstrip().endswith("function") or \
                (out["entry"] and name != out["entry"]) or \
                (yours and yours[0] <= call.start() < yours[1]):
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
    out["mode"] = "prints" if PRINTS.search(out["english"]) else "returns"
    words = QUOTED.search(out["english"])
    out["from_words"] = ({"value": words.group(2), "said": words.group(0)}
                         if words else None)
    out["asked"] = bool(found or len(out["examples"]) >= 2
                        or ASKED.search(text)
                        or (out["examples"] and ASKED.search(english)))
    if not out["params"] and out["examples"]:
        _inferred(out)
    if out["entry"] and out["returns"] and out["signature"] is None:
        said = ", ".join(f"{name}: {kind}" for name, kind in out["params"])
        out["signature"] = (f"function {out['entry']}({said}): "
                            f"{out['returns']}")
    if out["signature"] is None and out["missing"] is None \
            and len(out["english"].split()) < 3:
        out["missing"] = ("what it should do, in words, or how it is "
                          "called: an example such as reverse(\"ab\") == "
                          "\"ba\", or its TypeScript signature")
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


# -- the ways of writing asked for (v698) ---------------------------------------

class _Held:
    """The ways of writing a request is held to (v698 `ways.py`: *use a
    switch*, *make it recursive*), checked on each program as written --
    its shape read once."""

    def __init__(self, ways, before=None) -> None:
        from research.v698 import ways as W
        self.W, self.ways, self.shapes = W, list(ways or ()), {}
        code, entry = before or (None, None)
        self.before = self.shape(code, entry) if code and entry else None

    def shape(self, text: str, entry: str) -> dict | None:
        from research.v696.checker import CheckerError, checker
        key = (text, entry)
        if key not in self.shapes:
            try:
                self.shapes[key] = checker().shape(text, entry)
            except CheckerError:
                self.shapes[key] = None
        return self.shapes[key]

    def fits(self, text: str, entry: str) -> bool:
        return self.W.fits(self.shape(text, entry), self.ways, self.before)

    def missing(self, text: str, entry: str) -> list:
        return self.W.missing(self.shape(text, entry), self.ways,
                              self.before)

    def said(self, ways=None) -> str:
        return self.W.said(self.ways if ways is None else ways)

    def told(self) -> str:
        """What the writers are told of them."""
        return f"(write it {self.said()})"

    def rewritten(self, answer: str, entry: str, inputs: list, code: str,
                  out: dict, also=()) -> str | None:
        """The answer, rewritten the ways asked by the writers -- kept only
        where it is so written and does what the answer does, on every
        input (`inputs`: the examples and inputs varied from them). What
        each writer wrote is kept in `out["rewritten"]`."""
        from research.v696 import sketcher
        from research.v696.checker import CheckerError, checker
        from research.v696.teach_sketch import prompt

        def does(source):
            try:
                rows = checker().run(source, entry, inputs)
            except CheckerError:
                return None
            return [[row.get("value"), "error" in row, row.get("printed")]
                    for row in rows]

        want = does(answer)
        tried = out.setdefault("rewritten", {"inputs": inputs[:12],
                                             "programs": []})["programs"]
        if want is None:
            return None
        best = [len(self.missing(answer, entry)), None]

        def consider(source: str, by: str) -> bool:
            """Kept: written every way asked and doing the same (True) --
            or, written more of them and doing the same, the best yet."""
            record = {"writer": by, "text": source,
                      "ways missing": self.missing(source, entry)}
            tried.append(record)
            if len(record["ways missing"]) >= best[0]:
                return False
            record["same"] = does(source) == want
            if not record["same"]:
                return False
            if not record["ways missing"]:
                return True
            best[:] = [len(record["ways missing"]), source]
            return False

        # first what needs no writer: the search's own printing of it
        # (one line, ?:, no loops), and either, restyled the ways that are
        # syntax alone (an arrow, a switch, ifs)
        for base, how in [(answer, "restyled")] + [
                (one, "printed") for one in also if one]:
            text = base
            for way in self.ways:
                try:
                    text = checker().restyle(text, entry, way) or text
                except CheckerError:
                    pass
            for source, by in ((base, how), (text, "restyled")):
                if source != answer and consider(source, by):
                    return source
        said = prompt(f"Rewrite this function {self.said()}, so that it "
                      f"does exactly what it does:\n{answer}", code, None)
        tools = Tools.get()
        for name in sketcher.PROPOSERS.split(","):
            key = f"{name}|rewrite|0|{zlib.crc32(said.encode())}"
            texts = sketcher._cached().get(key)
            if texts is None:
                writer = tools.writer(name)
                writer.torch.manual_seed(sketcher.SEED
                                         + zlib.crc32(said.encode()))
                texts = writer.write([said], greedy=True)[0]
                sketcher._keep(key, texts)
            for one in texts:
                source, signature = _function(one)
                if signature is None or signature.group(1) != entry:
                    # an arrow, as written in its code block
                    source = (one.split("```")[1].partition("\n")[2]
                              if "```" in one else one).strip()
                    shape = self.shape(source, entry)
                    if not shape or shape["entry"] is None:
                        continue
                if consider(source, name):
                    return source
        # none written every way asked: the one written the most of them
        # that does what the answer does
        return best[1]

    def json(self, text: str | None, entry: str | None) -> dict:
        missing = self.missing(text, entry) if text and entry else \
            list(self.ways)
        return {"asked": self.ways, "said": self.said(),
                "missing": missing, "missing said": self.said(missing)}


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


def solve(text: str, ways=(), before=None, yours=None) -> dict:
    """Everything done for a request, and its answer (`code`). `ways`: the
    ways of writing it is held to (v698), `before`: (code, entry) of the
    answer it changes, for the ways relative to it (`shorter`); `yours`: a
    function the person wrote for it -- tried as one more writer's."""
    from research.v696 import risk, sketcher
    from research.v696.experiment import CONFIGS
    from research.v696.parse import parse
    from research.v696.reader import request as said_as
    from research.v696.search import Solver
    from research.v696.spec import Spec

    started = time.time()
    timings = {}
    asked = read(text)
    yours = yours or asked.get("yours")
    if yours:
        asked["yours"] = yours
    out = {"request": asked, "timings": timings}
    held = _Held(ways, before) if ways else None
    if asked["missing"]:
        out["answer"] = {"status": "missing", "code": None}
        return out
    known = known_of(asked)
    if known.get("used"):
        out["knowledge"] = known["used"]
    words = asked["from_words"]
    if asked["mode"] == "returns" and asked["signature"] \
            and not asked["params"] and words and not asked["examples"]:
        # `returns "Hello World!"` of a function that takes nothing: what
        # the words say it returns is its example
        asked["examples"].append({"args": [], "value": words["value"],
                                  "said": words["said"]})
        asked["made"].append("an example from your words")
    if asked["mode"] == "prints" or not asked["signature"] \
            or not asked["examples"]:
        return _open(asked, text, out, started, held)
    name = f"chat-{asked['entry']}-{zlib.crc32(text.encode()):08x}"
    spec = Spec(name, [tuple(one) for one in asked["params"]],
                asked["returns"],
                [(list(one["args"]), one["value"])
                 for one in asked["examples"]],
                entry=asked["entry"],
                english=" ".join(filter(None, (
                    asked["english"], known.get("english"),
                    held.told() if held else None))),
                library=list(known.get("library", ())))
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
    # in a conversation, a program that gives nothing (or throws) beyond the
    # examples is not chosen over one that answers there: `day` without a
    # default over `day(n % 7)`, whatever the risks say -- tried on values
    # of its types far from the examples too (`day(7)`)
    spec.moves = dataclasses.replace(spec.moves,
                                     on=spec.moves.on | {"total", "wide"})
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
    if yours:
        # the person's own: one more writer, `you` -- checked as theirs are
        mine = Spec(spec.name, spec.params, spec.returns, spec.examples,
                    entry=spec.entry)
        sketcher._read(mine, [yours], parse)
        tree = mine.proposals[0] if mine.proposals else None
        sketcher._read(spec, [yours], parse, writer="you")
        out["writers"].append({"writer": "you", "rounds": [{
            "round": 0, "programs": [{
                "text": yours, "read": tree is not None,
                "program": tree.source() if tree is not None else None,
                "meets": bool(tree is not None and spec.examples
                              and sketcher._meets(spec, tree))}]}]})
    fitting = {}
    if held:
        # each program as its writers wrote it: written the ways asked, or
        # not -- the same tree may be written with a switch and with ifs
        for writer in out["writers"]:
            for round_ in writer["rounds"]:
                for one in round_["programs"]:
                    if not one["read"]:
                        continue
                    one["ways missing"] = held.missing(one["text"],
                                                       spec.entry)
                    if not one["ways missing"]:
                        fitting.setdefault(one["program"], one["text"])
        spec.shaped = (lambda program: program.source() in fitting
                       or held.fits(spec.function(program), spec.entry))
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
            answer = _answer(spec, got.program, out["writers"], fitting)
            answer.update(status="confirmed" if got.confirmed else "met",
                          route=got.route)
        else:
            answer = {"status": "unsolved", "code": None}
    elif spec.proposals:
        # nothing to check by: what the first writer wrote that runs
        first = next((one for one in spec.proposals
                      if one.source() in fitting), spec.proposals[0])
        answer = _answer(spec, first, out["writers"], fitting)
        answer["status"] = "unverified"
    timings["search"] = round(time.time() - mark, 2)
    timings["total"] = round(time.time() - started, 2)
    if held and answer.get("code") and held.missing(answer["code"],
                                                    spec.entry):
        # nothing that meets the examples was written the ways asked: the
        # answer itself, rewritten so, where it then does what it did
        from research.v696 import meaning as M
        pairs = [(list(args), value) for args, value in spec.examples]
        inputs = [args for args, _ in pairs] + (M.probes(pairs) if pairs
                                                else [])
        code = "\n".join([spec.signature()] + [
            one["said"] for one in asked["examples"]])
        better = held.rewritten(answer["code"], spec.entry, inputs or [[]],
                                code, out, also=[answer.get("printed")])
        if better:
            answer.update(code=better, rewritten=answer["code"],
                          **{"as": "rewritten"})
    if held:
        answer["ways"] = held.json(answer.get("code"), spec.entry)
    out["answer"] = answer
    return _plain(out)


# -- the open path: no examples to search by, or something printed -----------

#: inputs a program is run on where no example gives any, by type
DEFAULTS = {"number": [0, 1, 5, -3], "string": ["", "a", "hello world", "Abc"],
            "boolean": [True, False], "number[]": [[], [1, 2, 3], [5, -1, 0]],
            "string[]": [[], ["a", "b"], ["hello", "world"]],
            "boolean[]": [[], [True, False]]}


#: a function as people write it: types where they are given, a result type
#: or not (then it gives nothing: `void`)
WRITTEN_FUNCTION = re.compile(
    r"function\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*(?::\s*([^{]+?))?\s*\{")


class _Signature:
    """A written function's name, parameters and result, as SIGNATURE's
    groups are read."""

    def __init__(self, found) -> None:
        self._groups = (found.group(1), found.group(2),
                        (found.group(3) or "void").strip())

    def group(self, at: int) -> str:
        return self._groups[at - 1]


def _function(text: str):
    """The function a writer wrote, and its signature, or (text, None).
    Statements written with no function around them -- a program that
    prints, as people write one -- are taken as a function of nothing,
    `main`."""
    if "```" in text:
        inside = text.split("```")[1]
        text = inside.partition("\n")[2] or inside
    text = text.strip()
    found = WRITTEN_FUNCTION.search(text)
    if found is None and text and not text.startswith(("//", "/*")):
        text = "function main(): void {\n  " + text.replace(
            "\n", "\n  ") + "\n}"
        found = WRITTEN_FUNCTION.search(text)
    return text, (_Signature(found) if found else None)


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(text)).lower().split())


def _open(asked: dict, text: str, out: dict, started: float,
          held: _Held | None = None) -> dict:
    """Where there are no examples to search by, or what is asked is
    printed: every writer writes for the request as said; each program is
    run -- on the examples, or on inputs of its types -- and what it does is
    what is compared. The answer is what the most writers arrived at apart
    (confirmed by two), and what meets the examples where there are any,
    the request's own words among them (`prints out "Hello World"`)."""
    from research.v696 import meaning as M
    from research.v696 import sketcher
    from research.v696.checker import CheckerError, checker
    from research.v696.teach_sketch import prompt, said_meaning
    timings = out["timings"]
    tools = Tools.get()
    STATE.mkdir(exist_ok=True)
    sketcher.CACHE = WRITTEN
    english = " ".join(filter(None, (asked["english"],
                                     known_of(asked).get("english"),
                                     held.told() if held else None)))
    code = "\n".join([asked["signature"] or ""] + [
        one["said"] for one in asked["examples"]]).strip()
    request = f"open-{zlib.crc32(text.encode()):08x}"
    probs = tools.reader.read([(english, code)])[0]
    out["read"] = {"returns": _top(probs["returns"], 3),
                   "uses": _top(probs["uses"]),
                   "behaviour": _top(probs["behaviour"]),
                   "root": _top(probs["root"], 3)}
    timings["read"] = round(time.time() - started, 2)

    words = asked["from_words"]
    wanted = [(list(one["args"]), one["value"]) for one in asked["examples"]]

    def meets(entry, params, rows) -> bool:
        if asked["mode"] == "prints" and words and not wanted:
            # what the words say is printed, by a function of nothing
            return not params and _norm("\n".join(
                rows[0].get("printed", []))) == _norm(words["value"])
        if asked["mode"] == "prints":
            return all(_norm("\n".join(row.get("printed", []))) == _norm(value)
                       for row, (_, value) in zip(rows, wanted))
        if words and not wanted:
            return not params and "value" in rows[0] and \
                _norm(rows[0]["value"]) == _norm(words["value"])
        return all("value" in row and json.dumps(row["value"]) ==
                   json.dumps(value) for row, (_, value) in zip(rows, wanted))

    mark = time.time()
    written, programs = [], []
    names = sketcher.PROPOSERS.split(",")
    for round_ in range(ROUNDS):
        for name in names:
            writer = tools.writer(name)
            meaning = (said_meaning(probs) if writer.settings["meaning"]
                       else None)
            said = prompt(english, code, meaning)
            key = (f"{name}|{request}|{round_}|"
                   f"{zlib.crc32(said.encode())}")
            texts = sketcher._cached().get(key)
            if texts is None:
                writer.torch.manual_seed(
                    sketcher.SEED + zlib.crc32(request.encode()) + round_)
                texts = writer.write([said], greedy=round_ == 0)[0]
                sketcher._keep(key, texts)
            row = {"writer": name, "round": round_, "programs": []}
            for one in texts:
                source, signature = _function(one)
                entry = signature.group(1) if signature else None
                record = {"text": source, "read": False, "program": None,
                          "meets": False}
                row["programs"].append(record)
                if entry is None or (asked["entry"] and
                                     entry != asked["entry"]):
                    continue
                from research.v696.tasks import _params
                params = [kind for _, kind in _params(signature.group(2))]
                programs.append({"writer": name, "text": source,
                                 "entry": entry, "params": params,
                                 "returns": signature.group(3).strip(),
                                 "record": record})
            written.append(row)
        if round_ == 0 and not wanted and not words:
            break
    if asked.get("yours"):
        # the person's own: one more writer, `you`
        source, signature = _function(asked["yours"])
        record = {"text": source, "read": False, "program": None,
                  "meets": False}
        written.append({"writer": "you", "round": 0, "programs": [record]})
        if signature is not None:
            from research.v696.tasks import _params
            programs.append({"writer": "you", "text": source,
                             "entry": signature.group(1),
                             "params": [kind for _, kind in
                                        _params(signature.group(2))],
                             "returns": signature.group(3).strip(),
                             "record": record})
    # one signature for the request: as given, or the most writers' types
    shapes = {}
    for one in programs:
        # what it prints is what is compared: what it returns is no part
        # of its shape then
        shapes.setdefault((tuple(one["params"]), one["returns"] if
                           asked["mode"] != "prints" else "void"),
                          []).append(one)
    if asked["params"] or asked["signature"]:
        shape = (tuple(kind for _, kind in asked["params"]),
                 asked["returns"] if asked["mode"] != "prints" else "void")
    else:
        shape = max(shapes, key=lambda k: (len({one["writer"] for one in
                                                shapes[k]}),
                                           len(shapes[k])), default=None)
    candidates = shapes.get(shape, []) if shape else []
    if wanted:
        inputs = [args for args, _ in wanted] + M.probes(wanted)
    elif shape and shape[0]:
        inputs = [[DEFAULTS.get(kind, [None])[at % len(DEFAULTS.get(
            kind, [None]))] for kind in shape[0]] for at in range(4)]
    else:
        inputs = [[]]
    groups: dict = {}
    for one in candidates:
        try:
            rows = checker().run(one["text"], one["entry"], inputs)
        except CheckerError:
            continue
        if all("error" in row for row in rows):
            continue
        one["record"]["read"] = True
        if (wanted or words) and not meets(one["entry"], one["params"],
                                           rows[:max(len(wanted), 1)]):
            continue
        one["record"]["meets"] = bool(wanted or words)
        if held:
            one["record"]["ways missing"] = held.missing(one["text"],
                                                         one["entry"])
        does = tuple(json.dumps([row.get("value", "!") if "error" not in row
                                 else "!", row.get("printed", [])])
                     for row in rows)
        group = groups.setdefault(does, {"authors": set(), "programs": [],
                                         "does": [json.loads(d) for d in
                                                  does]})
        group["authors"].add(one["writer"])
        group["programs"].append(one)
    timings["write"] = round(time.time() - mark, 2)
    ranked = sorted(groups.values(), key=lambda g: (-len(g["authors"]),
                                                    -len(g["programs"])))
    out["writers"] = [{"writer": name, "rounds": [
        {"round": row["round"], "programs": row["programs"]}
        for row in written if row["writer"] == name]}
        for name in names + (["you"] if asked.get("yours") else [])]
    out["open"] = {
        "inputs": inputs[:12], "mode": asked["mode"],
        "signature": None if shape is None else
        f"({', '.join(shape[0])}) => {shape[1]}",
        "groups": [{"authors": sorted(g["authors"]),
                    "programs": [one["text"] for one in g["programs"][:3]],
                    "count": len(g["programs"]), "does": g["does"][:12]}
                   for g in ranked]}
    if not ranked:
        out["answer"] = {"status": "unsolved", "code": None,
                         "mode": asked["mode"]}
    else:
        best = ranked[0]
        chosen = best["programs"][0]
        if held:
            # written the ways asked: the most agreed-on behaviour that
            # has such a program, and that program
            fit = [g for g in ranked if any(not one["record"].get(
                "ways missing") for one in g["programs"])]
            if fit:
                best = fit[0]
                chosen = next(one for one in best["programs"]
                              if not one["record"].get("ways missing"))
        chosen["record"]["program"] = "the answer"
        agreed = len(best["authors"]) >= 2
        status = (("confirmed" if agreed else "met") if wanted or words
                  else ("agreed" if agreed else "unverified"))
        out["answer"] = {"status": status, "code": chosen["text"],
                         "entry": chosen["entry"], "as": "written",
                         "route": "agreement", "program": "the answer",
                         "examples": len(wanted) + bool(words and not wanted),
                         "mode": asked["mode"],
                         "by": sorted(best["authors"])}
        if held and held.missing(chosen["text"], chosen["entry"]):
            better = held.rewritten(chosen["text"], chosen["entry"], inputs,
                                    code, out)
            if better:
                out["answer"].update(code=better, rewritten=chosen["text"],
                                     **{"as": "rewritten"})
        if held:
            out["answer"]["ways"] = held.json(out["answer"]["code"],
                                              chosen["entry"])
    timings["total"] = round(time.time() - started, 2)
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


def _answer(spec, program, writers: list, fitting=None) -> dict:
    """The answer, whole: as a writer wrote it where it is one of theirs
    (people's code reads better than the search's tree) -- as written the
    ways asked, where one was (`fitting`) -- and as the search holds it:
    printed exactly from its tree, helpers first."""
    printed = spec.function(program)
    written = (fitting or {}).get(program.source()) or next(
        (one["text"] for writer in writers
                    for round_ in writer["rounds"]
                    for one in round_["programs"]
                    if one["program"] == program.source()), None)
    return {"code": written or printed, "printed": printed,
            "as": "written" if written else "printed", "entry": spec.entry,
            "program": program.source(),
            "examples": len(spec.examples)}


def spoken(found: dict) -> str:
    """What is said of it -- and, where the person wrote a version of their
    own, what became of it."""
    said = _ways_said(found)
    mine = next((one for writer in found.get("writers", ())
                 if writer["writer"] == "you" for round_ in writer["rounds"]
                 for one in round_["programs"]), None)
    if mine is None or not (found.get("answer") or {}).get("code"):
        return said
    if found["answer"]["code"] == mine["text"]:
        return f"{said} It is the version you wrote."
    if not mine["read"]:
        return (f"{said} Yours I could not read as a program of its "
                f"signature.")
    if found["request"].get("examples") and not mine["meets"]:
        return f"{said} Yours does not meet the examples."
    return (f"{said} Yours was weighed with the writers' programs; this one "
            f"was chosen over it (ask what else was written to see both).")


def _ways_said(found: dict) -> str:
    """What is said of it, in a sentence or two: the code is on the
    answer -- and, held to ways of writing it, whether it is so written."""
    said = _spoken(found)
    ways = (found.get("answer") or {}).get("ways")
    if not ways or not (found["answer"].get("code")):
        return said
    from research.v698.ways import said as said_as
    done = said_as([one for one in ways["asked"]
                    if one not in ways["missing"]])
    rewritten = found["answer"].get("as") == "rewritten"
    if rewritten:
        inputs = len(found["rewritten"]["inputs"])
        how = (f"Rewritten {done}, as you asked: it does what it did on all "
               f"{inputs} inputs tried")
    else:
        how = f"It is written {done}, as you asked"
    if not ways["missing"]:
        return f"{said} {how}."
    tried = len((found.get("rewritten") or {}).get("programs", ()))
    why = ("no program written that meets the examples is"
           + (f", and none of {tried} rewrites of it that is does the same"
              if tried else ""))
    if done:
        return f"{said} {how} -- but not {ways['missing said']}: {why}."
    return f"{said} It is not written {ways['missing said']}: {why}."


def _spoken(found: dict) -> str:
    asked, answer = found["request"], found["answer"]
    entry = asked.get("entry") or "it"
    count = len(asked["examples"])
    examples = f"{count} example{'s' if count != 1 else ''}"
    status = answer["status"]
    if status == "missing":
        return f"Show me {asked['missing']}."
    entry = answer.get("entry") or entry
    if answer.get("route") == "agreement" and status != "unsolved":
        by = answer.get("by") or []
        shown = ("prints" if answer.get("mode") == "prints" else "gives")
        checked = ("it meets your words" if asked.get("from_words")
                   and not asked["examples"] else f"it meets your {examples}")
        if status == "confirmed":
            return (f"Here is {entry}: {checked}, and {len(by)} writers "
                    f"arrived at it apart.")
        if status == "met":
            return (f"Here is {entry}: {checked}; one writer wrote it, "
                    f"nothing apart confirms it -- check it.")
        if status == "agreed":
            return (f"Here is {entry}, as {len(by)} writers wrote it apart "
                    f"(they agree on what it {shown}); give me an example "
                    f"to check it by.")
        return (f"Here is {entry}, as one writer wrote it, unchecked: give "
                f"me an example call and what it should give, and I will "
                f"check it.")
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
    if answer.get("mode") == "prints" or "open" in found:
        return (f"Nothing the writers wrote ({written} programs) does what "
                f"you asked, run as it was written.")
    return (f"I could not write {entry} so that it meets your {examples}: "
            f"{written} programs were written and none met them, and the "
            f"search found nothing in "
            f"{found.get('search', {}).get('evaluated', 0)} candidates.")


def answered(text: str, ways=(), before=None, yours=None) -> dict:
    """The conversation's answer to a request for code: said in a
    sentence, with everything done for it (`code`). `ways`, `before`,
    `yours`: as `solve` takes them."""
    try:
        found = solve(text, ways, before, yours)
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
