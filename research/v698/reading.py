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

and, of a request or a change, the set of ways of writing it asks for
(`ways.py`: switch, recursive, iterative, map, filter, no-loop, ...) -- any
number at once -- read by an encoder of its own (`asked_ways.py`: the
reader of meaning, as the risk estimators read a request).

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
    #: the ways of writing asked for (`ways.WAYS`), in the order said
    ways: list = field(default_factory=list)

    def json(self) -> dict:
        return {"act": self.act, "chance": round(self.chance, 3),
                "aspect": self.aspect,
                "aspect chance": round(self.aspect_chance, 3),
                "subject": self.subject, "words": self.words,
                "roles": self.roles, "spans": self.spans,
                "known": self.known, "ways": self.ways,
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
    # a request with its examples under it (`sum_evens([1, 2]) == 2`): the
    # words are read, as the reader and the ways were taught -- with the
    # examples, `write a python function ...` was read as a change, and no
    # language at all
    from research.v697.coding import read as taken
    asked = taken(text)
    if asked.get("examples") and asked.get("english"):
        text = asked["english"]
    said = words(text)
    if not said:
        return None
    if not encoder.whole(said):
        # longer than the encoder reads (an exercise's instructions, its
        # examples, then `modify the supplied files: accumulate.py`): read
        # by its parts, the one that asks something done taken as what is
        # asked -- the rest is what it is asked of
        part = _asking(text, known)
        if part is not None:
            text, said = part, words(part)
    found = encoder.read(said, tags=_known(said, known), heads=HEADS)
    # what can be meant depends on the conversation's state: what is said
    # back to a plan only where one is held -- read with its state word
    # (v704 `teach_plan_talk.MARKER`); elsewhere the likeliest of the rest
    # (`what parameters does lend take?` was read as a query of a plan)
    from research.v704.teach_plan_talk import MARKER, PLAN_ACTS
    if said[0] != MARKER:
        found = {**found, "code_act": [one for one in found["code_act"]
                                       if one[0] not in PLAN_ACTS]}
    act, chance = found["code_act"][0]
    aspect, aspect_chance = found["code_aspect"][0]
    subject = found["code_subject"][0][0]
    roles = found["code_role"]
    typed = _typed(text, said)
    tags = _known(said, known)
    ways = []
    if act in ("make", "change") and chance >= FLOOR:
        from research.v698 import asked_ways
        # as it was taught: the words, lower case, as the reader reads them
        ways = [one for one, _ in asked_ways.read(" ".join(said))]
    return Reading(act, chance, aspect, aspect_chance, subject, said, roles,
                   _spans(typed, roles), found["code_act"],
                   [word for word, tag in zip(typed, tags) if tag], ways)


#: what is asked done to code: one of them read surely, together, is code
#: talk however the reader shares between them
DOING = ("make", "change", "goal")
#: what a doing is of, said by name: the project, a file, something in it
NAMED = ("project", "file", "named")


def _parts(text: str) -> list:
    """A long message's parts, as it is laid out: its paragraphs, code
    blocks left out, and a paragraph too long to be read whole its
    lines and sentences."""
    import re
    from research.v692.corpus import words
    plain = re.sub(r"```.*?```", "\n\n", text, flags=re.S)
    out = []
    for paragraph in re.split(r"\n\s*\n", plain):
        paragraph = paragraph.strip()
        said = words(paragraph) if paragraph else []
        if not said:
            continue
        if encoder.whole(said):
            out.append(paragraph)
        else:
            out += [one.strip() for one in
                    re.split(r"(?<=[.!?:])\s+|\n", paragraph) if one.strip()]
    return out


def _asking(text: str, known=()) -> str | None:
    """The part of a long message that asks something done -- read as a
    doing surely, one saying what of by name before one that does not
    (`Implement the accumulate operation ...` and `modify the supplied
    files: accumulate.py` are both changes; the second says of what) --
    or None where no part does."""
    from research.v692.corpus import words
    best, found = 0.0, None
    for part in _parts(text):
        said = words(part)
        if not said or not encoder.whole(said):
            continue
        read = encoder.read(said, tags=_known(said, known),
                            heads=("code_act", "code_subject"))
        doing = sum(chance for act, chance in read["code_act"]
                    if act in DOING)
        if read["code_act"][0][0] not in DOING or doing < FLOOR:
            continue
        score = doing + (read["code_subject"][0][0] in NAMED)
        if score > best:
            best, found = score, part
    return found


def code_talk(text: str, known=()) -> Reading | None:
    """The reading, where it is code talk sure enough to take the turn --
    the act surely, or a doing where the doings together are sure (`clean
    this up`: change 0.53, goal 0.46 -- left to the social reading, it was
    said goodbye to)."""
    found = read(text, known)
    if found is None or found.act == "none":
        return None
    if found.chance < FLOOR and not (found.act in DOING and sum(
            chance for act, chance in found.acts if act in DOING) >= FLOOR):
        return None
    return found
