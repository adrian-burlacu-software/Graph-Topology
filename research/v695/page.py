"""Actions on the page: how a goal is reached, and who can.

Importing this adds two acts to v691's page (`page.adds`), asked before
the page reads the utterance itself:

    achieve   *how would I get over a fence*       the actions that reach it,
                                                   that the asker can do
    able      *could you jump over a fence*        whether I, or you, can --
              *can i fly*                          by what each of us is

`achieve` takes a goal only when something reaches it; a goal nothing
reaches is left to what was there before. `able` takes only the two taking
part: *can a dog swim* is a question about dogs, and v688's.

Who is who: the one asking is a person, and the program is a computer
program (`v689.discourse`), so *could you jump* is asked of a computer
program -- which VerbNet says jumps no more than a rock does.
"""
from __future__ import annotations

from research.v689.discourse import ADDRESSEE_KIND, SPEAKER_KIND
from research.v691 import hearing
from research.v691 import page as v691_page
from research.v695 import achieving, can

#: The two taking part, as said back to the other.
SWAPPED = {"me": "you", "you": "me", "my": "your", "your": "my",
           "myself": "yourself", "yourself": "myself"}
#: Who a pronoun is, as the kind `can` asks about.
WHO = {"i": SPEAKER_KIND, "we": SPEAKER_KIND, "you": ADDRESSEE_KIND}


def _achieves(scene, text: str) -> bool:
    goal = achieving.read(text)
    return goal is not None and bool(achieving.answer(goal))


def achieve(scene, heard) -> str:
    goal = achieving.read(heard.said)
    return achieving.answer(goal) if goal is not None else ""


def asked_able(text: str):
    """(who, kind, verb, rest) for *can you / could I* + a doing, or None."""
    words = hearing.parse(text)
    if len(words) < 3 or words[0].tag != "MD" or words[0].text not in (
            "can", "could"):
        return None
    root = next((one for one in words if one.dep == "ROOT"), None)
    if root is None or not root.tag.startswith("VB"):
        return None
    subject = hearing.children(words, root.index, {"nsubj"})
    if not subject or subject[0].text not in WHO:
        return None
    if hearing.children(words, root.index, {"ccomp", "xcomp", "dative"}):
        return None                  # `can you tell me ...`: a request
    # Said back, the two taking part change places: `see me` is `see you`.
    rest = " ".join(SWAPPED.get(one.text, one.text)
                    for one in words[root.index + 1:] if one.tag != ".")
    return subject[0].text, WHO[subject[0].text], root.lemma, rest


def _able(scene, text: str) -> bool:
    found = asked_able(text)
    # Only a verdict is said: `unattested` of the program is left to v689,
    # which has always answered it.
    return found is not None and can.can(found[1], found[2]).verdict != \
        "unattested"


def able(scene, heard) -> str:
    who, kind, verb, rest = asked_able(heard.said)
    found = can.can(kind, verb)
    doing = f"{verb} {rest}".strip()
    subject = "I" if who == "you" else "you"
    if found.verdict == "yes":
        return f"Yes, {subject} can {doing} -- {found.why}."
    if found.verdict == "no":
        return (f"No, {subject} can't {doing} -- {subject} "
                f"{'am' if subject == 'I' else 'are'} "
                f"{can.kind_word(kind)}, and {found.why}.")
    return ""


v691_page.adds("achieve", _achieves, achieve, utility=v691_page.ASK,
               rule="how a goal is reached: actions the asker can do")
v691_page.adds("able", _able, able, utility=v691_page.ASK,
               rule="whether you or I can do something, by what we are")
