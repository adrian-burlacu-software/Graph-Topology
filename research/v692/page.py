"""Mathematics on the page: v692's acts in v689's conversation.

Importing this registers one act with `v689.session` (`contributes`, as
v691 does) and R33 with v687's reasoner (`semantics`). The act is proposed
when the encoder reads an utterance as asking something of mathematics
(`reading.read`), and it answers with what doing it came to (`doing.do`).

**A conversation keeps what it was told.** *Let x be 5*, then *is x
prime*, then *what is x squared plus 1*: each conversation has a
workspace of what its variables were said to be, put into everything asked
after (`Workspace`) -- and of its functions, *let f(x) = x^2 + 1*, then
*what is f(3)*. A measure that could not be worked out for want of a given
stays open there: *what is the area of a rectangle with length 4* is
answered by asking for the width, and *the width is 6* finishes it. It is the conversation's, like v691's scene -- v689's
event types are closed, and what `x` is here is not a fact about the world.

**Answers are shaped as the rest of the system speaks.** A value is
`retrieved` (`68 — worked out`), a yes or a no `verified` or `denied`,
something taken in `noted`: the outcomes v690's decoder was taught to say
(`v690/message.py`). Until the decoder has been taught mathematics too, what
is said is the text as written (`spoken`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from research.v687.executive import ANSWERED, Operator
from research.v689 import session as v689
from research.v692 import doing, reading, semantics  # noqa: F401 (R33)

#: Above v689's own acts and v691's orders and questions, below setting a
#: world up: reading an utterance as mathematics is the encoder's call, and
#: it was taught to leave everything else alone.
UTILITY = 205.0

@dataclass
class Workspace:
    """What a conversation's variables were said to be."""

    bound: dict = field(default_factory=dict)
    #: the last result, for `what is that doubled`
    last: object = None
    #: a measure asked for that was short of a given, until it is given
    pending: dict | None = None


WORKSPACES: dict = {}


def workspace(session) -> Workspace:
    key = getattr(session, "conversation", "") or id(session)
    return WORKSPACES.setdefault(key, Workspace())


def said(memory) -> str:
    turn = memory.get("turn")
    return getattr(turn, "said", "") or memory["reading"].said


#: conversation -> (utterance, what it was read as): `proposes` runs every
#: cycle, and reading is the encoder's work, done once.
_READ: dict = {}


def read_for(session, text: str):
    key = getattr(session, "conversation", "") or id(session)
    was = _READ.get(key)
    if was is None or was[0] != text:
        was = (text, reading.read(text))
        _READ[key] = was
    return was[1]


def answered(session, utterance: str) -> dict | None:
    """The answer to an utterance, or None where it asks no mathematics."""
    from research.v692.speaking import answer_of

    read = read_for(session, utterance)
    if read is None:
        return None
    space = workspace(session)
    if read.trouble:
        return {"outcome": "unknown", "source": "mathematics",
                "text": f"I could not read that as mathematics "
                        f"({read.trouble})"}
    parts = dict(read.parts)
    if read.kind is not None:
        parts["KIND"] = read.kind
    act = read.act
    if act == "given" and space.pending is not None:
        # What was asked for, given: the question it finishes, again.
        act, parts = "measure", doing.continued(space.pending, parts)
    result = doing.do(act, parts, space.bound)
    if act == "let" and result.stance == "noted":
        space.bound.update(result.value)
    elif result.value is not None:
        space.last = result.value
    if act == "measure":
        space.pending = parts if result.needs else None
    answer = answer_of(utterance, act, parts, result,
                       read.symbols)["answer"]
    answer["mathematics"].update(chance=read.chance,
                                 kind=read.kind.name if read.kind else None)
    said_plainly = (
        # What is said of a proposition written in symbols (`p | ~p is a
        # tautology`), and what a measure finished by a given is about
        # (`the area`, not the width that was given): in both the decoder
        # is handed a sentence v690 reads as ordinary English, and says it
        # back in the wrong words. The page says those as they are.
        any(one in result.text for one in "&|~")
        or (act == "measure" and read.act == "given"))
    if said_plainly and result.stance == "value" and result.about:
        result.text = f"{result.about} is {result.text}"
    if not _decoder_says_mathematics() or result.stance == "unknown"             or said_plainly:
        # Said as written until the decoder has been taught mathematics --
        # and always where it could not be worked out, because what is
        # said then is what is missing (`I need the width of the
        # rectangle`), which no outcome's shape carries.
        written = result.text
        if result.because and result.because not in written:
            written = f"{written}: {result.because}"
        answer["spoken"] = _sentence(written)
    return answer


def _sentence(text: str) -> str:
    """Said as a sentence: ended, and opened -- unless it opens with a
    variable, which keeps its case (`p | ~p is a tautology`)."""
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    first = text.split(" ", 1)[0] if text else ""
    if len(first) == 1 and first.isalpha() and first not in ("a", "i"):
        return text
    return text[:1].upper() + text[1:]


def replies(session) -> list:
    """v692's act, as an operator for v689's act executive."""
    def proposes(memory) -> bool:
        read = read_for(session, said(memory))
        if read is None:
            return False
        if read.act == "given" and workspace(session).pending is None:
            # `the width is 6` is mathematics only where something was
            # asked for; otherwise it is somebody saying something.
            return False
        if not read.parts and read.kind is None:
            # Nothing was read to do it to: `are all prime numbers odd` is
            # a claim about a kind, which the reasoner answers (R33), not
            # a sum to work out.
            return False
        return True

    def apply(memory):
        answer = answered(session, said(memory))
        if answer is None:
            return None
        memory["turn"].answer = answer
        return ANSWERED

    return [Operator(name="mathematics", apply=apply, proposes=proposes,
                     utility=UTILITY,
                     rule="what is asked of mathematics, worked out")]


def _decoder_says_mathematics() -> bool:
    """Whether the decoder in use was taught mathematics (`subjects` in its
    `decoder.json`): until it is, answers are said as written."""
    import json
    try:
        from research.v690 import decoder
        found = json.loads((decoder.MODEL / "decoder.json").read_text(
            encoding="utf-8"))
    except Exception:                                # noqa: BLE001
        return False
    return "math" in found.get("subjects", ())


v689.contributes(replies)


def _register_round_trip() -> None:
    try:
        from research.v690 import speaking as v690
        from research.v692 import speaking
    except Exception:                                # noqa: BLE001
        return
    v690.contributes(speaking.traced)


_register_round_trip()
