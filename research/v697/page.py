"""Code on the page: v696 as an act of v689's conversation.

Importing this registers one act with `v689.session` (`contributes`, as
v691 and v692 do): it is proposed when an utterance asks for a function
(`coding.read`), and it answers with what solving it came to
(`coding.answered`) -- a sentence to say, and everything done for it under
the answer's `code`.

What is read is the utterance as typed (`said_as_typed`): v689's reading
puts what was said into its own normal form, and a TypeScript signature or
a string in an example has to keep its case and its quotes.
"""
from __future__ import annotations

import threading

from research.v687.executive import ANSWERED, Operator
from research.v689 import session as v689
from research.v697 import coding

#: Above mathematics (205): a request for code may hold `f(3) == 10`,
#: which mathematics would otherwise try to work out.
UTILITY = 210.0

_TYPED = threading.local()


def said_as_typed(text: str | None) -> None:
    """What the page was sent, for the turn about to be said on this
    thread (v697's server sets it before each turn)."""
    _TYPED.text = text


def _typed(memory) -> str:
    turn = memory.get("turn")
    return (getattr(_TYPED, "text", None) or getattr(turn, "said", "")
            or memory["reading"].said)


#: utterance -> its reading, as `proposes` runs every cycle
_READ: dict = {}


def _asks(text: str) -> bool:
    if text not in _READ:
        _READ.clear()
        _READ[text] = coding.read(text)
    return bool(_READ[text]["asked"])


def replies(session) -> list:
    def proposes(memory) -> bool:
        return _asks(_typed(memory))

    def apply(memory):
        memory["turn"].answer = coding.answered(_typed(memory))
        return ANSWERED

    return [Operator(name="programming", apply=apply, proposes=proposes,
                     utility=UTILITY,
                     rule="a function asked for: written, searched, checked "
                          "by its examples (v696)")]


v689.contributes(replies)
