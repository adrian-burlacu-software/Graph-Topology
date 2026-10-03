"""Code on the page: v696 as an act of v689's conversation.

Importing this registers one act with `v689.session` (`contributes`, as
v691 and v692 do). It is proposed when an utterance asks for a function, or
says more of the one asked for, or asks about it (`conversation.classify`),
and it answers from the conversation's code workspace
(`conversation.answered`) -- a sentence to say, and everything done for it
under the answer's `code`.

What is read is the utterance as typed (`said_as_typed`): v689's reading
puts what was said into its own normal form, and a TypeScript signature or
a string in an example has to keep its case and its quotes.
"""
from __future__ import annotations

import threading

from research.v687.executive import ANSWERED, Operator
from research.v689 import session as v689
from research.v697 import conversation

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


def _turn(memory) -> int:
    return getattr(memory.get("turn"), "number", 0) or 0


#: (conversation, utterance, turn) -> what it is to the code: `proposes`
#: runs every cycle, and reading it is done once
_SEEN: dict = {}


def _kind(session, memory):
    key = (getattr(session, "conversation", "") or id(session),
           _typed(memory), _turn(memory))
    if key not in _SEEN:
        _SEEN.clear()
        space = conversation.workspace(key[0])
        _SEEN[key] = conversation.classify(key[1], space, key[2])[0]
    return _SEEN[key]


def replies(session) -> list:
    def proposes(memory) -> bool:
        return _kind(session, memory) is not None

    def apply(memory):
        space = conversation.workspace(
            getattr(session, "conversation", "") or id(session))
        answer = conversation.answered(_typed(memory), space, _turn(memory))
        if answer is None:
            return None
        memory["turn"].answer = answer
        return ANSWERED

    return [Operator(name="programming", apply=apply, proposes=proposes,
                     utility=UTILITY,
                     rule="code: a function asked for, more said of it, or "
                          "a question about it -- written, searched, "
                          "checked, kept (v696)")]


v689.contributes(replies)
