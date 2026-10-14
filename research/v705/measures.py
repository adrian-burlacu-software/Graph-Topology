"""How its replies read, as checks (v705): a goal about its replies --
"make your replies to goals more concise" -- made something a change can
be tried against, so the thinking has cases where the conversation test
has none.

- **What it measures**: the measure and the way a goal asks for, read by
  two heads taught over MiniLM (`teach_measures`) -- `concise` is less
  length; None where it asks for no measure of its replies.
- **Of which replies**: the act its words are nearest (`teach_goals`'
  choices: a goal, a question about code, ...), and the messages its
  reader was taught as that act are said to a scratch session.
- **The check**: a reply holds where the measure moved the way asked by
  `MARGIN` and the reply still says what it said -- MiniLM finds it
  `KEEP` near the reply as it was. The worst reply is the case; the rest
  are kept as they are (`guards`).
- **Where its words come from** (`made`): every call of its own code
  watched while the case is said again; each function that gave back a
  piece of the reply, the one that gave back the most first.
- **What a change can be** (`dropped`): each statement of the function
  taken out in turn -- the reply without what it added.
"""
from __future__ import annotations

import json
import random
import re
import sys

from research.encoder import LLM

HEAD = LLM / "measures-head"
#: how sure the head must be that the goal asks for a measure
SURE = 0.6
#: messages of each act said again
SAMPLES = 4
#: how far a measure must move for a reply to hold: replays are seeded
#: (`inside._scratch`), so only by more than a word or two
MARGIN = 0.1
#: how near (MiniLM) the reply must stay to what it said
KEEP = 0.75
#: acts within this much of the nearest are taken too
LEAD = 0.03
#: a piece of a reply: at least this many characters
PIECE = 12
#: the places a case is given at most
PLACES = 8
SEED = 7056
_HEADS: dict = {}
_CASES: dict = {}


# -- the measures (each a function of a reply) -------------------------------

def _sentences(reply: str) -> list:
    return [one for one in re.split(r"(?<=[.!?:])\s+|\n+", reply.strip())
            if one.strip()]


def length(reply: str) -> int:
    """The words of a reply: how long it is."""
    return len(reply.split())


def repetition(reply: str) -> float:
    """Repetition in a reply: the share of its word triples it said
    before."""
    words = reply.lower().split()
    triples = [tuple(words[at:at + 3]) for at in range(len(words) - 2)]
    if not triples:
        return 0.0
    return 1 - len(set(triples)) / len(triples)


def questions(reply: str) -> int:
    """The questions in a reply: how much it asks back."""
    return reply.count("?")


#: every measure, by its name (what the heads were taught) -- how many
#: sentences and how long they run were taught too, and taken apart from
#: length by neither the teacher nor the heads: a wordy reply is all three
MEASURES = {"length": length, "repetition": repetition,
            "questions": questions}


# -- reading a goal -------------------------------------------------------------

def read(goal: str) -> dict | None:
    """{measure, way ("min" | "max"), sure} a goal asks of its replies;
    None where it asks for no measure, or the heads are not taught."""
    import torch
    from research.v701 import likeness
    if "heads" not in _HEADS:
        if not (HEAD / "heads.pt").exists():
            return None
        saved = torch.load(HEAD / "heads.pt", weights_only=True)
        heads = {}
        for key, size in (("measure", len(saved["names"])), ("way", 2)):
            head = torch.nn.Linear(saved[key]["weight"].shape[1], size)
            head.load_state_dict(saved[key])
            heads[key] = head.eval()
        _HEADS.update(heads=heads, names=saved["names"])
    vector = likeness._vectors([goal])
    with torch.no_grad():
        which = _HEADS["heads"]["measure"](vector).softmax(-1)[0]
        way = _HEADS["heads"]["way"](vector).softmax(-1)[0]
    # about some measure, sure enough, then which: length, sentences and
    # their length share the words a person uses (`concise`), and none
    # of them alone was sure of "make your answers shorter"
    names = _HEADS["names"]
    some = 1 - float(which[names.index("none")])
    if some < SURE:
        return None
    at = max((one for one in range(len(names)) if names[one] != "none"),
             key=lambda one: float(which[one]))
    return {"measure": names[at], "way": "max" if way[1] > way[0] else "min",
            "sure": round(some, 3)}


def acts(goal: str) -> list:
    """The acts whose replies the goal is about: the nearest of the acts
    the reader tells apart, and those within `LEAD` of it."""
    from research.v701.likeness import nearest
    from research.v704.teach_goals import CHOICES
    said = {f"replies to {what}": act for act, what in CHOICES
            if act != "none"}
    ranked = nearest(goal, list(said))
    return [said[one] for score, one in ranked
            if ranked[0][0] - score <= LEAD]


def inputs(goal: str, most: int = SAMPLES) -> list:
    """Messages the reader was taught as those acts (held out from its
    teaching), `most` of each."""
    from research.v698 import teach_code_talk as K
    wanted = acts(goal)
    found: dict = {}
    for path in sorted(K.DATA.glob("valid-*.jsonl")):
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            if row.get("act") in wanted:
                found.setdefault(row["act"], []).append(" ".join(row["words"]))
    rng = random.Random(SEED)
    out = []
    for act in wanted:
        messages = sorted(set(found.get(act) or []))
        rng.shuffle(messages)
        out += messages[:most]
    return out


# -- the checks -----------------------------------------------------------------

def kept(before: str, after: str) -> bool:
    """Whether a reply still says what it said: MiniLM finds it `KEEP`
    near it."""
    from research.v701.likeness import nearest
    if not after.strip():
        return False
    return nearest(before, [after])[0][0] >= KEEP


def moved(value: float, before: float, way: str) -> bool:
    if way == "min":
        return value <= before * (1 - MARGIN) and before > 0
    return value >= before * (1 + MARGIN) + (1 if before == 0 else 0)


def holds(measure: str, way: str, before: str):
    """The check of a reply: moved the way asked, and still saying what it
    said."""
    function, was = MEASURES[measure], MEASURES[measure](before)

    def check(page, _=None) -> bool:
        text = page["reply"]["text"]
        return moved(function(text), was, way) and kept(before, text)
    check.said = f"{measure} {'down' if way == 'min' else 'up'} by " \
                 f"{int(MARGIN * 100)}%, saying what it said"
    return check


def stays(measure: str, way: str, before: str):
    """The check of a reply not the case: not moved the other way, and
    still saying what it said."""
    function, was = MEASURES[measure], MEASURES[measure](before)

    def check(page, _=None) -> bool:
        text = page["reply"]["text"]
        value = function(text)
        worse = value > was if way == "min" else value < was
        return not worse and kept(before, text)
    return check


def cases(goal: str, session) -> list:
    """The goal's cases: each message of the acts it is about said again,
    its reply measured -- the worst the case, carrying the rest as guards;
    [] where the goal asks for no measure of its replies."""
    asked = read(goal)
    if asked is None or session is None:
        return []
    key = (goal, id(session))
    if key in _CASES:
        return _CASES[key]
    from research.v705 import evidence, inside
    said = []
    for message in inputs(goal):
        turn = inside.answered(session, [message])
        text = inside._as_page_turn(turn)["reply"]["text"]
        if text.strip():
            said.append((MEASURES[asked["measure"]](text), message, text,
                         turn))
    said.sort(key=lambda one: one[0], reverse=asked["way"] == "min")
    out = []
    for value, message, text, turn in said[:1]:
        guards = [([other], stays(asked["measure"], asked["way"], reply))
                  for _, other, reply, _ in said[1:]]
        out.append(evidence.case_of(
            turn, [message], holds(asked["measure"], asked["way"], text),
            area="measures", answer=text, measure={**asked, "is": value},
            guards=guards))
    _CASES[key] = out
    return out


# -- where its words come from --------------------------------------------------

def made(session, turns: list, reply: str) -> list:
    """The functions of its own code that gave back a piece of the reply
    while the turns are said again, the one that gave back the most first
    -- [{function, file, gave}] (`PLACES` at most)."""
    import contextvars
    from research.v705 import inside
    pieces = [one for one in _sentences(reply) if len(one) >= PIECE]
    if not pieces:
        return []
    found: dict = {}
    places: dict = {}

    def gives(value) -> int:
        texts = [value] if isinstance(value, str) else [
            one for one in inside._fields(value) if isinstance(one, str)]
        return sum(len(piece) for piece in pieces
                   if any(piece in text for text in texts))

    def profiler(frame, event, arg):
        if event != "return":
            return
        name = frame.f_code.co_filename
        path = places.get(name, False)
        if path is False:
            path = inside._relative(name) if "research" in name else None
            if path and (not path.startswith("research/") or
                         "/test" in path or "v705/" in path):
                path = None
            places[name] = path
        if not path or arg is None:
            return
        size = gives(arg)
        if size:
            function = getattr(frame.f_code, "co_qualname",
                               frame.f_code.co_name)
            if size > found.get((function, path), 0):
                found[(function, path)] = size

    def apart():
        from research.v689.session import Session
        with inside._scratch(session, Session) as scratch:
            for said in turns[:-1]:
                scratch.say(said)
            previous = sys.getprofile()
            sys.setprofile(profiler)
            try:
                scratch.say(turns[-1])
            finally:
                sys.setprofile(previous)
    contextvars.Context().run(apart)
    ranked = sorted(found.items(), key=lambda one: -one[1])
    return [{"function": function, "file": path, "gave": size}
            for (function, path), size in ranked
            if "<" not in function][:PLACES]


def feeding(function) -> list:
    """The functions of its own code what `function` gives back is made
    from: each call on the backward slice of its returns -- `return found`,
    `found = _reply(..., said, ...)`, `said = joined(pieces, ...)`, `pieces
    = pieces_of(...)` -- resolved through its module and the imports in
    it. Templates filled later (`pieces_of`) give back no piece of the
    reply's text, and are where what it says is chosen."""
    import ast
    import inspect
    import textwrap
    from research.v705 import inside
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    except (OSError, TypeError, SyntaxError):
        return []
    body = tree.body[0]
    wanted: set = set()
    for node in ast.walk(body):
        if isinstance(node, ast.Return) and node.value is not None:
            wanted |= inside._reads(node.value)
    binding = [node for node in ast.walk(body)
               if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
               and node.value is not None]
    calls: list = []
    sliced: set = set()
    grown = True
    while grown:
        grown = False
        for node in binding:
            targets = node.targets if isinstance(node, ast.Assign)                 else [node.target]
            bound = set().union(*(inside._bound(one) for one in targets))
            if id(node) in sliced or not bound & wanted:
                continue
            sliced.add(id(node))
            wanted |= inside._reads(node.value)
            grown = True
            calls += [one for one in ast.walk(node.value)
                      if isinstance(one, ast.Call)]
    for node in ast.walk(body):
        if isinstance(node, ast.Return) and node.value is not None:
            calls += [one for one in ast.walk(node.value)
                      if isinstance(one, ast.Call)]
    # names the function imports where it runs, then its module's
    scope = dict(getattr(function, "__globals__", {}))
    for node in ast.walk(body):
        if isinstance(node, ast.ImportFrom) and node.module:
            for one in node.names:
                try:
                    module = __import__(node.module, fromlist=[one.name])
                    scope[one.asname or one.name] = getattr(module, one.name)
                except (ImportError, AttributeError):
                    pass
    out = []
    for call in calls:
        name = inside._dotted(call.func)
        if not name:
            continue
        head, *rest = name.split(".")
        found = scope.get(head)
        for part in rest:
            found = getattr(found, part, None)
        found = inspect.unwrap(found) if callable(found) else None
        if not inspect.isfunction(found) or found is function:
            continue
        path = inside._relative(inspect.getsourcefile(found))
        if path and path.startswith("research/") and "/test" not in path                 and found not in out:
            out.append(found)
    return out


# -- what a change can be -------------------------------------------------------

def dropped(function, text: str) -> list:
    """The file's text with each statement of the function taken out in
    turn (a `pass` where it stood) -- [(changed text, what was taken)];
    not its docstring, nor a return."""
    import ast
    import inspect
    lines, first = inspect.getsourcelines(function)
    source = "".join(lines)
    import textwrap
    tree = ast.parse(textwrap.dedent(source))
    body = tree.body[0]
    file_lines = text.splitlines(keepends=True)
    out = []
    for node in ast.walk(body):
        # statements only: an `x if y else z` has a body too, an expression
        held = [getattr(node, field, None) for field in ("body", "orelse",
                                                         "finalbody")]
        for one in [one for block in held if isinstance(block, list)
                    for one in block]:
            if not isinstance(one, ast.stmt) or isinstance(
                    one, (ast.Return, ast.FunctionDef, ast.Pass)) or (
                    isinstance(one, ast.Expr) and
                    isinstance(one.value, ast.Constant)):
                continue
            start = first + one.lineno - 1
            end = first + one.end_lineno - 1
            original = file_lines[start - 1:end]
            indent = re.match(r"\s*", original[0]).group(0)
            changed = file_lines[:start - 1] + [indent + "pass\n"] + \
                file_lines[end:]
            what = " ".join("".join(original).split())[:80]
            out.append(("".join(changed), f"without `{what}`"))
    return out


def _spliced(file_lines: list, first: int, indent: int, node,
             new: str) -> str:
    """The file's text with a node of a function's dedented source put as
    `new` (ast's columns are bytes of the dedented line)."""
    start, end = first + node.lineno - 1, first + node.end_lineno - 1
    head = file_lines[start - 1].encode()
    tail = file_lines[end - 1].encode()
    before = head[:indent + node.col_offset].decode()
    after = tail[indent + node.end_col_offset:].decode()
    return "".join(file_lines[:start - 1] + [before + new + after] +
                   file_lines[end:])


def smaller(function, text: str) -> list:
    """The file's text with one expression of the function made less, in
    turn: a `+` without one of its sides (`frame + PRECEDENT`), a number
    halved (`most: int = 90`), an `x if c else y` as one of its branches
    -- [(changed text, what was made less)]."""
    import ast
    import inspect
    import textwrap
    lines, first = inspect.getsourcelines(function)
    indent = len(lines[0]) - len(lines[0].lstrip())
    source = textwrap.dedent("".join(lines))
    tree = ast.parse(source)
    file_lines = text.splitlines(keepends=True)
    out = []

    def said(node) -> str:
        return " ".join((ast.get_source_segment(source, node) or "").split())

    for node in ast.walk(tree.body[0]):
        if isinstance(node, ast.Expr) and isinstance(node.value,
                                                      ast.Constant):
            continue
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            for kept, gone in ((node.left, node.right),
                               (node.right, node.left)):
                if said(kept):
                    out.append((_spliced(file_lines, first, indent, node,
                                         ast.get_source_segment(source,
                                                                kept)),
                                f"`{said(node)[:60]}` without "
                                f"`{said(gone)[:40]}`"))
        elif isinstance(node, ast.IfExp):
            for kept in (node.body, node.orelse):
                out.append((_spliced(file_lines, first, indent, node,
                                     ast.get_source_segment(source, kept)),
                            f"`{said(node)[:60]}` as `{said(kept)[:40]}`"))
        elif isinstance(node, ast.Constant) and type(node.value) in (
                int, float) and node.value not in (0, 1, -1) and \
                node.lineno == node.end_lineno:
            less = node.value // 2 if isinstance(node.value, int) \
                else node.value / 2
            out.append((_spliced(file_lines, first, indent, node,
                                 repr(less)),
                        f"{node.value} halved"))
    return out
