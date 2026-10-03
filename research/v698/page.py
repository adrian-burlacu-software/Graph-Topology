"""The project in the conversation: v698's acts in v689's executive.

Two acts, registered as v697's code is (`v689.session.contributes`):

    pasted code   code pasted with a question about it (*explain this*):
                  read, explained, and kept as the conversation's code
    the project   a question about the project the editor sent: answered
                  from its outline, its calls and the compiler

Both read the utterance as typed (`v697.page.said_as_typed`). Both stand
above v697's code (210): a function the project names is the project's
before it is the last answer's, and pasted code is read, not asked for.
"""
from __future__ import annotations

from research.v687.executive import ANSWERED, Operator
from research.v689 import session as v689
from research.v697 import conversation
from research.v697 import page as v697_page
from research.v698 import asking
from research.v698 import project as Pj

PASTED, PROJECT = 212.0, 211.0


def _key(session):
    return getattr(session, "conversation", "") or id(session)


def replies(session) -> list:
    def typed(memory) -> str:
        return v697_page._typed(memory)

    def pastes(memory) -> bool:
        return asking.pasted(typed(memory)) is not None

    def explain(memory):
        text = typed(memory)
        space = conversation.workspace(_key(session))
        memory["turn"].answer = asking.read_pasted(
            text, asking.pasted(text), space)
        space.turn = v697_page._turn(memory)
        return ANSWERED

    def about(memory) -> bool:
        key = _key(session)
        return asking.route(typed(memory), Pj.project(key),
                            conversation.workspace(key),
                            v697_page._turn(memory), key)[0] is not None

    def answer(memory):
        key = _key(session)
        found = asking.answered(typed(memory), Pj.project(key),
                                conversation.workspace(key),
                                v697_page._turn(memory), key)
        if found is None:
            return None
        memory["turn"].answer = found
        return ANSWERED

    return [Operator(name="pasted code", apply=explain, proposes=pastes,
                     utility=PASTED,
                     rule="code pasted with a question about it: read by "
                          "the compiler, run, kept as the conversation's"),
            Operator(name="about code", apply=answer, proposes=about,
                     utility=PROJECT,
                     rule="a question about code -- the project, a file, a "
                          "function, the conversation's code: a subject and "
                          "an aspect, read apart")]


v689.contributes(replies)
