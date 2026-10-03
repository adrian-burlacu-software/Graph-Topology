"""Code talk read by the encoder: what a message does, what it asks, of
what, and which of its words say so.

The shared encoder (`research/encoder.py`), taught code talk as a subject
of its own (`teach_code_talk.py`, `v689/teach_reader.py --subject code`),
has four heads for it:

    code_act      none | ask | make | change | run | teach
    code_aspect   what is asked: explain | where | size | callers | calls |
                  bugs | sure | why | others | risk | files | functions
    code_subject  none | project | file | named | last
    code_role     each word: the subject's phrase (SUBJ), or -- teaching --
                  the concept (CONCEPT) and its members (MEMBER)

Nothing here matches a word: what the encoder reads is taken as read, and
what a SUBJ phrase *names* is resolved exactly afterwards (`asking.py`:
the project's index, its files, the conversation's code). Below `FLOOR`
the message is not taken as code talk, and the turn is left to the rest of
the conversation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from research import encoder

HEADS = ("code_act", "code_aspect", "code_subject", "code_role")
#: how sure the encoder must be that a message is code talk
FLOOR = 0.6


@dataclass
class Reading:
    act: str
    chance: float
    aspect: str
    aspect_chance: float
    subject: str
    words: list
    roles: list
    #: the words of each span, as typed (case kept): SUBJ, CONCEPT, MEMBER
    spans: dict = field(default_factory=dict)
    #: the next readings of the act, for whoever wants to see them
    acts: list = field(default_factory=list)
    #: the words known to be code here, as typed (`_known`)
    known: list = field(default_factory=list)

    def json(self) -> dict:
        return {"act": self.act, "chance": round(self.chance, 3),
                "aspect": self.aspect,
                "aspect chance": round(self.aspect_chance, 3),
                "subject": self.subject, "words": self.words,
                "roles": self.roles, "spans": self.spans,
                "known": self.known,
                "acts": [[one, round(p, 3)] for one, p in self.acts[:4]]}


def taught() -> bool:
    """Whether the reader the encoder loads was taught code talk -- read
    off its labels, without loading it."""
    import json
    try:
        labels = json.loads((encoder.MODEL / "labels.json").read_text(
            encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return "code_act" in labels.get("heads", {})


def available() -> bool:
    """Whether the reader in use was taught code talk."""
    try:
        return "code_act" in encoder.LOADED.get().heads
    except Exception:                               # noqa: BLE001
        return False


def _typed(text: str, said: list) -> list:
    """Each word as it was typed: the encoder reads lower case, a name is
    resolved as written (`loudMain`)."""
    out, at, lower = [], 0, text.lower()
    for word in said:
        found = lower.find(word, at)
        if found < 0:
            out.append(word)
            continue
        out.append(text[found:found + len(word)])
        at = found + len(word)
    return out


def _spans(typed: list, roles: list) -> dict:
    spans: dict = {}
    current = None
    for word, role in zip(typed, roles):
        if role.startswith("B-"):
            current = role[2:]
            spans.setdefault(current, []).append([word])
        elif role.startswith("I-") and current == role[2:]:
            spans[current][-1].append(word)
        else:
            current = None
    return {name: [" ".join(one) for one in found]
            for name, found in spans.items()}


def _known(said: list, known) -> list:
    """`CODE` beside each word that names something known to be code -- a
    function or file of the project, the conversation's code -- looked up,
    as a word's spaCy tag is: `greeting(` and `greeting's` are `greeting`."""
    import re
    lower = {str(one).lower() for one in known or ()}
    return ["CODE" if re.split(r"[(\[.'\"?,!]", word)[0].strip() in lower
            or word in lower else "" for word in said]


def read(text: str, known=()) -> Reading | None:
    """What the encoder reads a message as, or None where the reader in use
    was not taught code talk. `known`: the names known to be code here."""
    from research.v692.corpus import words
    if not available():
        return None
    said = words(text)
    if not said:
        return None
    found = encoder.read(said, tags=_known(said, known), heads=HEADS)
    act, chance = found["code_act"][0]
    aspect, aspect_chance = found["code_aspect"][0]
    subject = found["code_subject"][0][0]
    roles = found["code_role"]
    typed = _typed(text, said)
    tags = _known(said, known)
    return Reading(act, chance, aspect, aspect_chance, subject, said, roles,
                   _spans(typed, roles), found["code_act"],
                   [word for word, tag in zip(typed, tags) if tag])


def code_talk(text: str, known=()) -> Reading | None:
    """The reading, where it is code talk sure enough to take the turn."""
    found = read(text, known)
    if found is None or found.act == "none" or found.chance < FLOOR:
        return None
    return found
