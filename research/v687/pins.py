"""Which sense the reader chose, for the length of one question.

R6 says inference runs per sense and never per word string, and every rule
here obeys it -- but until now the *choice* of sense was the engine's alone.
It picks well (v684's evidence-weighted disambiguation exists because picking
badly sent 348 facts to the part of a gunlock), and it still picks wrongly
sometimes: `why does a dog bark` resolves `bark` to the covering of a tree,
because that is the synset ConceptNet's crawl hung its facts on.

A pin is the reader overruling that for one word. It is request-scoped state,
which is exactly what a context variable is for: the server is threaded, and
each thread gets its own context, so two readers pinning different senses of
`mouse` at the same moment do not see each other's choice.

Where a pin bites is worth being exact about, because a control that looks as
though it does something everywhere and only works in places is worse than no
control:

    v684's subject      yes -- `is a hammer a tool` already took a sense
    R23's event         yes -- this is the one that gets `bark` wrong
    class words         yes -- `class_concept` takes the primary sense of
                        `bird`, `whale`, `mouse` for R21, R24 and R25

    R16, R17            no. XCSLB *ships* a sense key for its concepts, so
                        the join is given rather than guessed, and overruling
                        it would mean disagreeing with the corpus about what
                        its own concept means.
    R22                 no. It matches the words of an object string; no
                        synset is chosen on the way in.

`applies_to` reports that, so the page can say "nothing here resolves this
word" instead of offering a choice that would be ignored.
"""
from __future__ import annotations

import contextvars

#: word -> synset id, for the request being served.
_PINNED: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "pinned_senses", default={})


def use(pinned: dict[str, str] | None) -> None:
    """Set the pins for this request. Call once, at the top of `ask`."""
    _PINNED.set({word.lower(): sense for word, sense in (pinned or {}).items()
                 if word and sense})


def of(word: str) -> str | None:
    """The sense the reader pinned for this word, if any."""
    return _PINNED.get().get((word or "").lower())


def all_pins() -> dict[str, str]:
    return dict(_PINNED.get())


def parse(values: list[str]) -> dict[str, str]:
    """Read `pin=bark:bark.v.01` parameters into a mapping."""
    pinned: dict[str, str] = {}
    for value in values or []:
        word, _, sense = value.partition(":")
        if word.strip() and sense.strip():
            pinned[word.strip().lower()] = sense.strip()
    return pinned


#: Which rules consult a sense for a word from the question, in the words the
#: page shows the reader.
APPLIES = {
    "subject": "as the subject of a fact question (R1-R9)",
    "event": "as the event in a why / what-happens question (R23)",
    "phrase": "as the thing a backwards question asks about (R22)",
    "class": "as a class to search or count within (R16, R21, R24, R25)",
}


def applies_to(word: str, engine) -> list[str]:
    """The ways a pin on this word would change the answer.

    Empty means nothing here resolves it to a sense, and the page says that
    instead of offering a choice that goes nowhere.
    """
    uses: list[str] = []
    lowered = (word or "").lower()
    if engine.reasoner.senses_of(lowered):
        uses.append(APPLIES["subject"])
        uses.append(APPLIES["event"])
        uses.append(APPLIES["phrase"])
    if engine.profiles.subtypes(lowered):
        uses.append(APPLIES["class"])
    return uses


#: The one place a pin is deliberately ignored, worth saying out loud.
NOT_APPLIED = ("R17 reads the sense key XCSLB ships with each of its concepts, "
               "so a profile is not guessing and has nothing to overrule.")
