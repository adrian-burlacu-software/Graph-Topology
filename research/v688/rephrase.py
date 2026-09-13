"""A request put as the question inside it, before anything is read.

    do you know if a dog can swim       can a dog swim
    tell me whether a cat has fur       does a cat have fur
    is it true that dogs bark           do dogs bark
    is it likely that a bird can fly    can a bird fly     (and says so)
    can't a dog swim                    can a dog swim     (and says so)
    why can't a penguin fly             can a penguin fly  (asked why)
    how come birds fly                  do birds fly       (asked why)
    describe a cat                      tell me about a cat
    define a hammer                     what is a hammer
    what does bark mean                 what is a bark
    name three birds                    what kinds of birds are there
    list animals that fly               which animals fly

Every one of these was answered about something else. `do you know if a dog
can swim` asked whether a computer program knows things, `doesn't a cat have
fur` became `does a doesn't a cat have fur`, and `describe a cat` was re-asked
as `what is a describe a cat`. The question was there all along, in words a
parser was never going to take apart, so they are taken apart first -- by
string operations, which is what this repository uses where it can: three
regular expressions here have had their escapes mangled in transit.

**Why.** A why-question about a kind asks the yes or no inside it and what
that rests on, so it is asked as the yes or no and marked `why`; `negative`
says whether the premise was a denial (`why can't a penguin fly`), so the
answer can say whether the premise holds. Only a kind is taken this way --
`a penguin`, `birds`. `why does the dog bark` may be about one dog, and `why
does it rain` about no kind at all, so they are left for the conversation and
for v687's scripts.

Two shapes are left exactly as said and marked as questions, because a rule
downstream reads them whole: an analogy (`fins are to fish as what are to
birds`, R24) and a conditional (`if a dog had wings could it fly`, which R18
refuses by name).
"""
from __future__ import annotations

from dataclasses import dataclass

#: `do you know if ...`: the question is what follows.
EMBEDDING = ("do you know if ", "do you know whether ",
             "can you tell me if ", "can you tell me whether ",
             "could you tell me if ", "could you tell me whether ",
             "tell me if ", "tell me whether ", "i wonder if ",
             "i wonder whether ", "i want to know if ",
             "i want to know whether ", "please tell me if ",
             "please tell me whether ")

#: `is it true that ...`: the frame adds nothing to what follows.
TRUTH = ("is it true that ", "is it the case that ", "is it so that ")

#: `is it likely that ...`: asked as whether it holds, and said so.
LIKELY = ("is it likely that ", "is it possible that ",
          "is it probable that ", "is it unlikely that ")

#: What a statement puts after its subject and a question puts first.
AUX = frozenset({"can", "could", "is", "are", "was", "were", "has", "have",
                 "does", "do", "did", "will", "would", "should", "must",
                 "may", "might"})

#: The question words, in front of which nothing is moved.
WH = frozenset({"what", "who", "whom", "whose", "which", "why", "how",
                "where", "when"})

#: A negative question's opener, and the positive one it asks with.
NEGATIVE = {"can't": "can", "cannot": "can", "couldn't": "could",
            "isn't": "is", "aren't": "are", "wasn't": "was",
            "weren't": "were", "hasn't": "has", "haven't": "have",
            "doesn't": "does", "don't": "do", "didn't": "did",
            "won't": "will", "wouldn't": "would", "shouldn't": "should",
            "mustn't": "must"}

#: What `name three birds` may put before the kind.
COUNTING = frozenset({"me", "a", "an", "the", "some", "all", "any", "few",
                      "several", "many", "one", "two", "three", "four",
                      "five", "six", "seven", "eight", "nine", "ten"})

#: Words that end in `s` and are not a plural kind: `why does it`, `why is
#: this`.
NOT_KINDS = frozenset({"this", "these", "those", "us", "its", "his", "yes",
                       "is", "was", "as"})

NEGATIVE_NOTE = ("a negative question asks the same thing as the positive "
                 "one, so it is answered as that")
LIKELY_NOTE = ("asked as whether it holds: nothing here measures how likely "
               "it is")


@dataclass
class Rephrased:
    text: str
    note: str = ""
    #: A question already, whatever it opens with.
    asking: bool = False
    #: Asked why: the text is the yes or no whose grounds are wanted.
    why: bool = False
    #: The why's premise was a denial: `why can't a penguin fly`.
    negative: bool = False


def with_article(phrase: str) -> str:
    """`hammer` as `a hammer`; a phrase with a determiner as it is."""
    first = phrase.split(" ", 1)[0].lower()
    if first in ("a", "an", "the", "some", "any") or not phrase:
        return phrase
    return ("an " if first[:1] in "aeiou" else "a ") + phrase


def singular_verb(verb: str) -> str:
    """`barks` as `bark`, `flies` as `fly`, `watches` as `watch`."""
    if verb.endswith("ies") and len(verb) > 4:
        return verb[:-3] + "y"
    if verb.endswith(("ches", "shes", "sses", "xes", "zes")):
        return verb[:-2]
    if verb.endswith("s") and not verb.endswith("ss"):
        return verb[:-1]
    return verb


def polar(clause: str) -> str:
    """`a dog can swim` as `can a dog swim`; a question as it already is.

    A denial inside the statement is dropped with the rest of the frame:
    `a penguin can't fly` asks `can a penguin fly`."""
    words = clause.split()
    lower = [word.lower() for word in words]
    if not words or lower[0] in AUX or lower[0] in WH:
        return clause
    for index in range(1, min(len(words), 5)):
        aux = NEGATIVE.get(lower[index], lower[index])
        if aux not in AUX:
            continue
        subject = " ".join(words[:index])
        after = words[index + 1:]
        if lower[index] == aux and after[:1] and after[0].lower() == "not":
            after = after[1:]
        rest = " ".join(after)
        if aux in ("has", "have"):
            opener = "does" if aux == "has" else "do"
            return f"{opener} {subject} have {rest}".strip()
        return f"{aux} {subject} {rest}".strip()
    if len(words) >= 3 and lower[0] in ("a", "an", "the"):
        return (f"does {' '.join(words[:2])} {singular_verb(lower[2])} "
                f"{' '.join(words[3:])}").strip()
    return f"do {clause}"


def denies(clause: str) -> bool:
    """Is this yes-or-no, or statement, a denial?"""
    words = [word.lower() for word in clause.split()]
    return any(word in NEGATIVE or word in ("not", "never", "no")
               for word in words[:5])


def about_a_kind(words: list[str]) -> bool:
    """`a penguin ...`, `birds ...`: a kind, not one thing or none."""
    if not words:
        return False
    first = words[0]
    return first in ("a", "an") or (
        first.endswith("s") and not first.endswith("ss")
        and first not in NOT_KINDS)


def _why(stripped: str, lower: str) -> Rephrased | None:
    """`why can't a penguin fly`, `how come birds fly`."""
    for opener in ("why ", "how come "):
        if not lower.startswith(opener):
            continue
        body = stripped[len(opener):].strip()
        words = body.lower().split()
        if not words:
            return None
        if words[0] in AUX or words[0] in NEGATIVE:
            after = words[1:]
            if after[:1] == ["not"]:
                after = after[1:]
            if not about_a_kind(after):
                return None
            asked = rephrase(body).text if words[0] in NEGATIVE or (
                len(words) > 1 and words[1] == "not") else body
            return Rephrased(asked, why=True, negative=denies(body))
        if opener == "how come " and about_a_kind(words):
            return Rephrased(polar(body), why=True, negative=denies(body))
        return None
    return None


def rephrase(text: str) -> Rephrased:
    said = " ".join((text or "").replace(chr(8217), "'").split())
    stripped = said.rstrip("?.! ")
    lower = stripped.lower()
    if lower.startswith("please "):
        stripped, lower = stripped[7:], lower[7:]
    if lower.endswith(" please"):
        stripped, lower = stripped[:-7].rstrip(", "), lower[:-7].rstrip(", ")
    words = lower.split()
    if not words:
        return Rephrased(stripped)

    asked_why = _why(stripped, lower)
    if asked_why is not None:
        return asked_why

    for frame in EMBEDDING + TRUTH:
        if lower.startswith(frame):
            return Rephrased(polar(stripped[len(frame):]))
    for frame in LIKELY:
        if lower.startswith(frame):
            return Rephrased(polar(stripped[len(frame):]), LIKELY_NOTE)

    if words[0] in NEGATIVE and len(words) > 1:
        return Rephrased(f"{NEGATIVE[words[0]]} {stripped.split(' ', 1)[1]}",
                         NEGATIVE_NOTE)
    if len(words) > 2 and words[0] in AUX and words[1] == "not":
        kept = stripped.split()
        return Rephrased(" ".join([kept[0].lower()] + kept[2:]),
                         NEGATIVE_NOTE)

    if words[0] == "describe" and len(words) > 1:
        return Rephrased("tell me about " + stripped.split(" ", 1)[1])
    if words[0] == "define" and len(words) > 1:
        return Rephrased("what is " + with_article(stripped.split(" ", 1)[1]))
    if (lower.startswith("what does ") and lower.endswith(" mean")
            and len(words) > 3):
        return Rephrased("what is " + with_article(stripped[10:-5].strip()))
    if lower.startswith("what is the meaning of ") and len(words) > 5:
        return Rephrased("what is " + with_article(stripped[23:].strip()))

    if words[0] in ("name", "list") and len(words) > 1:
        body = words[1:]
        while body and body[0] in COUNTING:
            body = body[1:]
        for joiner in ("that", "which", "who"):
            if joiner in body[1:]:
                at = body.index(joiner, 1)
                return Rephrased(f"which {' '.join(body[:at])} "
                                 f"{' '.join(body[at + 1:])}".strip())
        if body:
            return Rephrased(f"what kinds of {' '.join(body)} are there")

    padded = f" {lower} "
    if (" is to " in padded or " are to " in padded) and " as " in padded:
        return Rephrased(stripped, asking=True)
    if words[0] == "if":
        return Rephrased(stripped, asking=True)
    return Rephrased(stripped)
