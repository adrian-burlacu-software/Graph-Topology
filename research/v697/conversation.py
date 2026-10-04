"""Code in a conversation: what was asked for is kept, and what is said
after it is about it (`PLAN.md`, "Code in a conversation").

Each conversation has a workspace -- the last request for code, as read, and
everything solving it came to. What is said next is one of:

    a request       something new asked for (`coding.read`)
    more of it      an example of the same function, its signature, or a
                    clarification (*it should return "Hello World!"*): put
                    together with what was asked and solved again
    a call          `total([4, 5])`, *run it on [4, 5]*: the answer run
    a question      about the answer -- what it does, why it is the answer,
                    how sure, what else was written, its risks -- answered
                    from what was kept of writing it, and by running it

A question is taken as about the code only where the turn before was code,
or where it names the code (`the function`, its name): *what does it do*
after *there is a beagle* is about the beagle.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from research.v697 import coding


@dataclass
class Workspace:
    #: the request as last read (`coding.read`), and what solving it came to
    request: dict | None = None
    found: dict | None = None
    #: the conversation turn the code was last answered at
    turn: int = -10
    #: the ways of writing asked for so far (v698 `ways.py`): *use a
    #: switch*, then *make it recursive* -- the code is held to them all
    ways: list | None = None


WORKSPACES: dict = {}


def workspace(key) -> Workspace:
    return WORKSPACES.setdefault(key, Workspace())


#: a clarification of what was asked: *it should*, *make it*, *also*
MORE = re.compile(r"^\s*(it|that|this|the (function|program|code)|and|also|"
                  r"but)\b.{0,40}\b(should|must|needs? to|has to|ought to|"
                  r"is supposed to)\b|^\s*(make|change|have) (it|that)\b|"
                  r"^\s*(actually|instead|no,?)\b", re.I)
QUESTIONS = (
    ("explain", re.compile(r"\bwhat (does|did|do) (it|that|this|the "
                           r"(code|function|program)|\w+\(\)) (do|does)\b|"
                           r"\bexplain\b|\bdescribe\b|\bhow does (it|that|"
                           r"this) work\b|\bwhat is (it|that) doing\b", re.I)),
    ("why", re.compile(r"\bwhy\b|\bhow did you (get|write|find|choose|pick|"
                       r"come)\b|\bwhere did (it|that) come from\b|"
                       r"\bwho wrote\b", re.I)),
    ("sure", re.compile(r"\bhow sure\b|\bare you sure\b|\bconfirmed\b|"
                        r"\bis (it|that) (right|correct)\b|\bcan i trust\b|"
                        r"\bdoes it work\b", re.I)),
    ("others", re.compile(r"\b(what|which) else\b|\balternatives?\b|\bother "
                          r"(programs?|versions?|ways?|answers?|ones?)\b|"
                          r"\banything else\b|\bdifferent\b", re.I)),
    ("risk", re.compile(r"\brisks?\b|\bcould go wrong\b|\bdanger|"
                        r"\bweak(ness)?\b", re.I)),
)
NAMES_CODE = re.compile(r"\b(the|your|this|that) (code|function|program|"
                        r"answer)\b", re.I)
RUN = re.compile(r"\brun (it|that|this)\s+(on|with)\s+(.+)$", re.I)


def classify(text: str, space: Workspace, turn: int) -> tuple:
    """(kind, detail): what the utterance is to the conversation's code, or
    (None, None) where it is not about code at all."""
    asked = coding.read(text)
    known = space.request
    entry = (known or {}).get("entry") or ((space.found or {}).get(
        "answer") or {}).get("entry")
    same = bool(entry) and asked["entry"] == entry
    if known and same and (asked["examples"] or asked["signature"]):
        return "more", asked
    if asked["asked"] and (asked["signature"] or asked["examples"]
                           or not known or not MORE.search(text)):
        return "request", asked
    recent = turn - space.turn <= 1
    if not known or not space.found:
        return None, None
    if entry:
        call = re.search(rf"(?<![\w.]){re.escape(entry)}\(", text)
        if call and not asked["examples"]:
            close = coding._balanced(text, call.end(), ")")
            if close > 0:
                return "call", text[call.end():close]
    ran = RUN.search(text)
    if ran and (recent or NAMES_CODE.search(text)):
        return "call", ran.group(3).strip().rstrip(".?")
    if MORE.search(text) and (recent or NAMES_CODE.search(text)):
        return "more", asked
    about = recent or NAMES_CODE.search(text) or (
        entry and re.search(rf"\b{re.escape(entry)}\b", text))
    if about:
        for kind, pattern in QUESTIONS:
            if pattern.search(text):
                return "question", kind
    return None, None


def _merged(known: dict, said: dict, text: str) -> str:
    """What was asked, and what was said of it since, as one request."""
    lines = [known["english"]]
    english = said["english"].strip(" .")
    if english and english.lower() not in known["english"].lower():
        lines.append(said["english"])
    signature = said["signature"] if (said["signature"] and "signature"
                                      not in said["made"]) else (
        known["signature"] if "signature" not in known["made"] else None)
    if signature:
        lines.append(signature)
    seen = set()
    for one in known["examples"] + said["examples"]:
        # an example the words gave is in the English already
        if one["said"] not in seen and "(" in one["said"]:
            seen.add(one["said"])
            lines.append(one["said"])
    return "\n".join(line for line in lines if line)


def answered(text: str, space: Workspace, turn: int) -> dict | None:
    kind, detail = classify(text, space, turn)
    if kind is None:
        return None
    return respond(kind, detail, text, space, turn)


def respond(kind: str, detail, text: str, space: Workspace,
            turn: int, ways=()) -> dict:
    """What a turn of each kind does -- whoever read what kind it is (the
    hand rules above, or v698's encoder). `ways`: the ways of writing the
    turn asks for (v698's encoder), added to those asked before."""
    if kind == "more" and not space.request:
        kind, detail = "request", coding.read(text)
    if kind in ("request", "more"):
        request = text if kind == "request" else _merged(space.request,
                                                         detail, text)
        from research.v698 import ways as W
        held = [] if kind == "request" else list(space.ways or ())
        for one in ways or ():
            held = W.merged(held, one)
        before = None
        if kind == "more":
            before = (((space.found or {}).get("answer") or {}).get("code"),
                      ((space.found or {}).get("answer") or {}).get("entry"))
        # a function the person wrote for it, this turn: tried with the rest
        yours = detail.get("yours") if isinstance(detail, dict) else None
        answer = coding.answered(request, ways=held, before=before,
                                 yours=yours)
        space.ways = held
        found = answer["code"]
        if kind == "more":
            found["continued"] = {"said": text, "asked": request}
            answer["text"] = answer["spoken"] = (
                "Taking that with what you asked before: "
                + answer["spoken"][:1].lower() + answer["spoken"][1:])
        space.request, space.found, space.turn = (found["request"], found,
                                                  turn)
        answer["suggest"] = _settle(found)
        return answer
    space.turn = turn
    if kind == "call":
        return _call(space, detail, text)
    return _question(space, detail, text)


def _settle(found: dict) -> list:
    """What to ask next of an answer nothing confirms: the example that
    would settle it, at inputs of its types no example covers -- where it
    gives nothing or throws first, then far from the examples (`day(7) ==
    ?` where only `day(0)` was given) -- and whether it has bugs."""
    from research.v696 import meaning as M
    from research.v696.checker import CheckerError, checker
    answer = found.get("answer") or {}
    code, entry = answer.get("code"), answer.get("entry")
    out = [{"text": "are there any bugs in it", "send": True}]
    if not code or not entry or answer.get("status") == "confirmed":
        return out
    request = found.get("request") or {}
    given = [list(one["args"]) for one in request.get("examples") or ()]
    probes = [args for args in M.wide_probes(
        [kind for _, kind in request.get("params") or ()])
        if args not in given]
    try:
        rows = checker().run(code, entry, probes) if probes else []
    except CheckerError:
        rows = []
    ranked = sorted(zip(probes, rows), key=lambda one: (
        "error" not in one[1] and one[1].get("value") is not None,
        -sum(abs(x) for x in one[0] if isinstance(x, (int, float)))))
    asks = [{"text": f"{entry}({', '.join(map(json.dumps, args))}) == ?",
             "send": False} for args, _ in ranked[:2]]
    return asks + out


# -- what is said of the code it answered -------------------------------------

def _code(space: Workspace) -> tuple:
    answer = (space.found or {}).get("answer") or {}
    return answer.get("code"), answer.get("entry")


def _inputs(space: Workspace) -> list:
    """What the answer is run on to say what it does: the examples, and
    inputs varied from them -- or, with none, inputs of its types."""
    from research.v696 import meaning as M
    request = space.request or {}
    pairs = [(list(one["args"]), one["value"]) for one in
             request.get("examples") or ()]
    if pairs:
        return [args for args, _ in pairs] + M.probes(pairs)
    return (space.found.get("open") or {}).get("inputs") or [[]]


def _said(value) -> str:
    return json.dumps(value) if not isinstance(value, str) else \
        json.dumps(value, ensure_ascii=False)


def _reply(space: Workspace, text: str, kind: str, spoken: str,
           looked: dict, blocks=()) -> dict:
    code, entry = _code(space)
    found = {"request": space.request,
             "followup": {"asked": kind, "said": text, "looked": looked},
             "blocks": list(blocks),
             "answer": {"status": "about", "code": None, "entry": entry}}
    return {"outcome": "acted", "source": "programming", "text": spoken,
            "spoken": spoken, "code": coding._plain(found)}


def _call(space: Workspace, args: str, text: str) -> dict:
    from research.v696.checker import checker
    code, entry = _code(space)
    if not code:
        return _reply(space, text, "call", "There is no program yet to run.",
                      {})
    try:
        values = coding._literal(f"[{args}]")
    except ValueError:
        return _reply(space, text, "call",
                      f"I could not read {args} as arguments.", {})
    row = checker().run(code, entry, [values])[0]
    shown = f"{entry}({', '.join(map(_said, values))})"
    if "error" in row:
        said = f"{shown} fails: {row['error']}."
    else:
        said = f"{shown} gives {_said(row.get('value'))}."
    if row.get("printed"):
        said += " It prints: " + " / ".join(row["printed"]) + "."
    return _reply(space, text, "call", said, {"call": shown, "ran": row})


def _question(space: Workspace, kind: str, text: str) -> dict:
    code, _ = _code(space)
    if not code and kind in ("explain", "others"):
        return _reply(space, text, kind, "There is no program yet: nothing "
                      "written did what was asked. Give me an example of "
                      "what it should do, and I will try again.", {})
    return {"explain": _explain, "why": _why, "sure": _sure,
            "others": _others, "risk": _risk}[kind](space, text)


def _explain(space: Workspace, text: str) -> dict:
    """What it is made of, as the compiler reads it, and what running it
    shows -- read, not made up: the exact readings of v696's meaning."""
    from research.v696 import meaning as M
    from research.v696.checker import checker
    code, entry = _code(space)
    uses, root = M.structure(code, entry)
    made = [one for one in sorted(uses) if one not in
            ("a name", "a literal", "let", "const", "return")]
    inputs = _inputs(space)
    rows = checker().run(code, entry, inputs)
    pairs = [(args, row["value"]) for args, row in zip(inputs, rows)
             if "value" in row]
    printed = [line for row in rows for line in row.get("printed", [])]
    behaviour = sorted(one for one in M.behaviour(pairs)
                       if M.checkable(one)) if pairs else []
    said = (f"{entry} is made of {', '.join(made)}" if made else
            f"{entry} uses nothing the library declares")
    said += f", and its result comes from {root}." if root else "."
    if behaviour:
        shown = ("your examples and inputs varied from them"
                 if (space.request or {}).get("examples")
                 else "inputs of its types")
        said += (f" Run on {len(pairs)} inputs ({shown}), its result is "
                 f"always: {'; '.join(behaviour[:5])}.")
    if printed:
        said += f" It prints: {' / '.join(dict.fromkeys(printed))}."
    runs = [{"call": f"{entry}({', '.join(map(_said, args))})",
             "gives": row.get("value", row.get("error")),
             "printed": row.get("printed")} for args, row in
            zip(inputs, rows)]
    return _reply(space, text, "explain", said,
                  {"uses": made, "root": root, "behaviour": behaviour,
                   "runs": runs},
                  [{"title": entry, "kind": "the program explained",
                    "code": code}])


def _chosen(found: dict) -> dict | None:
    return next((one for one in (found.get("search") or {}).get("events", ())
                 if one["event"] == "chosen"), None)


def _why(space: Workspace, text: str) -> dict:
    found = space.found
    answer = found["answer"]
    entry = answer.get("entry")
    written = sum(len(r["programs"]) for w in found.get("writers", ())
                  for r in w["rounds"])
    meeting = sum(p["meets"] for w in found.get("writers", ())
                  for r in w["rounds"] for p in r["programs"])
    chosen = _chosen(found)
    if chosen:
        group = next((g for g in chosen["behaviours"] if any(
            p["program"] == chosen["program"] for p in g["programs"])), {})
        who = ", ".join(group.get("authors", [])) or "a writer"
        said = (f"{written} programs were written for {entry}, {meeting} met "
                f"your examples. This one came from {who} (route "
                f"{answer.get('route')}); it was chosen by {chosen['by']}"
                + (", and it is confirmed: more than one of them arrived "
                   "at what it does." if chosen["confirmed"] else
                   ", but no one else arrived at what it does beyond the "
                   "examples."))
        looked = {"chosen": chosen}
    elif found.get("open"):
        groups = found["open"]["groups"]
        top = groups[0] if groups else {}
        said = (f"{written} programs were written for {entry}; run, they "
                f"did {len(groups)} different things. The answer is what "
                f"the most writers arrived at apart: "
                f"{', '.join(top.get('authors', [])) or 'none'}.")
        looked = {"groups": groups}
    else:
        said, looked = "I have nothing kept of how it was found.", {}
    return _reply(space, text, "why", said, looked)


def _sure(space: Workspace, text: str) -> dict:
    found = space.found
    status = found["answer"].get("status")
    risk = found.get("risk") or {}
    high = [f"{risk['names'][f]} ({f}{score})" for f, score in
            (risk.get("scores") or {}).items() if score >= risk.get("high", 2)]
    said = {"confirmed": "As sure as checking can make me: it meets what "
                         "you gave, and another writer or the search "
                         "arrived at the same apart.",
            "met": "It meets what you gave, but nothing written apart "
                   "confirms it beyond that.",
            "agreed": "Writers agree on it, but nothing you gave checks "
                      "it: give me an example.",
            "unverified": "Not sure: one writer wrote it and nothing checks "
                          "it.",
            }.get(status, f"It is {status}.")
    if high:
        said += f" What makes this request risky: {', '.join(high)}."
    return _reply(space, text, "sure", said, {"status": status,
                                              "risk": risk})


def _others(space: Workspace, text: str) -> dict:
    """What else met the examples, and where it differs from the answer:
    an input to say what should come out (`entry(args) == ...`)."""
    found = space.found
    entry = found["answer"].get("entry")
    inputs = _inputs(space)
    chosen = _chosen(found)
    blocks, rows = [], []
    if chosen:
        mine = next((g for g in chosen["behaviours"] if any(
            p["program"] == chosen["program"] for p in g["programs"])), None)
        for group in chosen["behaviours"]:
            if group is mine:
                continue
            at = next((n for n, (a, b) in enumerate(zip(group["does"],
                                                       mine["does"]))
                       if a != b), None)
            # the probes the search judged by start with these inputs
            where = inputs[len((space.request or {}).get("examples", ()))
                           + at] if at is not None and \
                len((space.request or {}).get("examples", ())) + at < \
                len(inputs) else None
            rows.append({"authors": group["authors"],
                         "programs": [p["program"] for p in
                                      group["programs"]],
                         "differs at": where,
                         "it gives": group["does"][at] if at is not None
                         else None,
                         "the answer gives": mine["does"][at]
                         if at is not None else None})
            blocks.append({"title": entry, "kind": "also met the examples, "
                           f"by {', '.join(group['authors'])}",
                           "code": group["programs"][0]["program"]})
    elif found.get("open"):
        for group in found["open"]["groups"][1:]:
            rows.append({"authors": group["authors"],
                         "programs": group["programs"]})
            blocks.append({"title": entry, "kind": "written too, by "
                           f"{', '.join(group['authors'])}",
                           "code": group["programs"][0]})
    if not rows:
        said = "Nothing else that was written or found did anything else."
    else:
        said = f"{len(rows)} other behaviour(s) met what you gave."
        first = next((one for one in rows if one.get("differs at")), None)
        if first:
            call = f"{entry}({', '.join(map(_said, first['differs at']))})"
            said += (f" They part at {call}: the answer gives "
                     f"{first['the answer gives']}, another "
                     f"{first['it gives']}. Tell me which is right "
                     f"({call} == ...) and I will take it.")
    return _reply(space, text, "others", said, {"others": rows}, blocks)


def _risk(space: Workspace, text: str) -> dict:
    risk = (space.found or {}).get("risk")
    if not risk:
        return _reply(space, text, "risk", "Its risks were not read: it was "
                      "answered without a search.", {})
    parts = [f"{risk['names'][f]} {score}" for f, score in
             risk["scores"].items()]
    said = (f"Its six risks (0-3): {', '.join(parts)}. "
            + (f"High together: {', '.join(risk['cells'])}; so the search "
               f"was run with these moves: "
               f"{', '.join(risk['moves']) or 'none'}."
               if risk["cells"] else "Nothing is high: ordinary review."))
    return _reply(space, text, "risk", said, {"risk": risk})
