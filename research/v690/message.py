"""What a reply has to say: an answer taken apart from the turn it answers.

v689 answers in its own terms -- `yes — nothing was told of the beagle, so
v687's walk passes up to beagle, and v688 answers “can a beagle swim” verified
(corroborated)` -- which is a derivation, not a reply. A message is what a
reply to that turn has to carry, read off the turn as the page gets it:

    stance    what kind of answer it is: yes, no, unknown, which, noted,
              value, social, refused
    heard     what was said, as it was said
    subject   who or what the answer is about, as the conversation describes
              it: `the beagle` for `it`, `Mary`, `a dog`
    claim     what a yes or no settles, or what was noted, as a statement
              about the subject: `the beagle can swim`
    values    what answers a question that asks for something: `the kitchen`,
              `John`, the first items of a listing
    quotes    what was said that the answer rests on, in the words said
    source    where it comes from: told, taught, kind, definition,
              conversation
    trust     how v688 rates an answer it gave
    found     v689's own answer, rules and all: what the decoder reads to say
              why, and never says as it is

`prompt` writes a message as the decoder reads it. `required` is what a reply
is read back for (`roundtrip.py`): its stance, and the words it cannot leave
out -- the subject and what is claimed of it, or the values.

Nothing here is language: the stance is v689's outcome, the claim is the
reading's own slots, and the values and quotes are where every answer of
v689's puts them -- before its first dash, and inside curly quotes.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

#: v689's outcomes, as what kind of answer a reply gives.
STANCES = {"verified": "yes", "denied": "no", "unknown": "unknown",
           "which": "which", "noted": "noted", "retrieved": "value",
           "social": "social"}

#: Every stance a reply can take, in the order the encoder's head says them.
STANCE_LABELS = ("yes", "no", "unknown", "which", "noted", "value", "social",
                 "refused")

#: An answer v688 refuses by name (R18) is not a lack of knowledge.
REFUSED = ("not answerable here",)

#: What a reply lists of a listing, at most.
LISTED = 5

#: What a reply has to name of a listing, at most.
NAMED = 3

#: How much of v689's own account the decoder reads, in characters: past
#: it, E2's and T4's notes go on about what the reply never says.
FOUND = 320

QUOTE = re.compile(r"“([^”]*)”")
#: `(T4, escape-51.1)`, `(R3 at distance 0)`: rules cited in brackets.
CITED = re.compile(r"\s*\((?:[A-Z]\d+[^()]*|[a-z]+-\d[\d.-]*[^()]*)\)")
#: `beagle.n.01`: a sense, said as its word.
SENSE = re.compile(r"\b([a-z_]+)\.[nvasr]\.\d\d\b")
#: `, under beagle.n.01`: where an individual is placed.
UNDER = re.compile(r",?\s*(?:placed )?under [a-z_]+\.[nvasr]\.\d\d")

#: Acts whose reading is a statement about its subject.
STATEMENTS = frozenset({"tell", "introduce", "teach", "name", "compound"})

#: Acts whose reading is a yes or no about its subject.
POLAR = frozenset({"ask", "generic", "why"})

#: Negated auxiliaries, to say a claim that does not hold.
NEGATED = {"can": "can't", "could": "couldn't", "does": "doesn't",
           "do": "don't", "did": "didn't", "is": "isn't", "are": "aren't",
           "was": "wasn't", "were": "weren't", "has": "hasn't",
           "have": "haven't", "had": "hadn't", "will": "won't",
           "would": "wouldn't", "am": "am not", "should": "shouldn't",
           "must": "mustn't", "might": "might not", "may": "may not"}


@dataclass
class Message:
    stance: str
    heard: str
    subject: str = ""
    claim: str = ""
    values: list = field(default_factory=list)
    quotes: list = field(default_factory=list)
    source: str = ""
    trust: str = ""
    found: str = ""
    #: the rules the answer cites, for the page, never for the reply
    rules: list = field(default_factory=list)
    #: a listing's items past the ones named, counted
    more: int = 0
    #: what the social act was, for a social reply
    act: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, found: dict) -> "Message":
        return cls(**{key: value for key, value in found.items()
                      if key in cls.__dataclass_fields__})


def plain(text: str) -> str:
    """v689's words without what only the page reads: senses said as their
    words, placements and bracketed rules left out."""
    text = UNDER.sub("", text or "")
    text = CITED.sub("", text)
    text = SENSE.sub(lambda found: found.group(1).replace("_", " "), text)
    return re.sub(r"\s+", " ", text).strip()


def rules_in(text: str) -> list[str]:
    return sorted(set(re.findall(r"\b([RTSEI]\d{1,2})\b", text or "")))


def referents_of(turn: dict) -> dict:
    """{id: referent} from a kept turn (`conversations.compact`) or from the
    page's, which carries the whole discourse."""
    if turn.get("referents"):
        return turn["referents"]
    return {one["id"]: one for one in
            ((turn.get("discourse") or {}).get("referents") or [])}


def _subject(turn: dict) -> str:
    """Who the answer is about, as the conversation describes them."""
    referents = referents_of(turn)
    resolution = turn.get("resolution") or {}
    found = referents.get(resolution.get("referent") or "")
    if found and found.get("description"):
        return found["description"]
    mention = (turn.get("reading") or {}).get("mention") or {}
    return mention.get("text") or ""


def _claim(turn: dict, subject: str) -> str:
    """What a yes or no settles, or what was noted, said of the subject:
    the reading's auxiliary, whether it holds, and the rest -- with the
    object said as the conversation describes it: `the dog chased it` is
    `the dog chased the cat`."""
    reading = turn.get("reading") or {}
    rest = reading.get("rest") or ""
    aux = reading.get("aux") or ""
    binding = turn.get("object") or {}
    described = referents_of(turn).get(binding.get("referent") or "") or {}
    expression = (binding.get("expression") or "").lower()
    if expression and described.get("description") and rest:
        words = rest.split()
        size = len(expression.split())
        for start in range(len(words) - size, -1, -1):
            if " ".join(words[start:start + size]).lower() == expression:
                words[start:start + size] = [described["description"]]
                rest = " ".join(words)
                break
    if not subject or not (rest or aux):
        return ""
    if not reading.get("holds", True):
        aux = NEGATED.get(aux, f"{aux} not" if aux else "not")
    return " ".join(one for one in (subject, aux, rest) if one)


def _values(text: str, content: dict | None, subject: str) -> list[str]:
    """What answers a question that asks for something: a listing's first
    items, or what v689 says before its first dash -- without a label that
    names who it is about (`Mary: the kitchen`) or a quote, which is kept
    apart."""
    # A definition's items are the kinds above it, not what answers.
    if content and content.get("items") and content.get("kind") != \
            "definition":
        return list(content["items"][:LISTED])
    head = plain(text).split(" — ", 1)[0]
    label, colon, after = head.partition(": ")
    # `Mary: the kitchen`, `yesterday: “...”`, `in the order you told me:
    # “...”`: a label naming who or when, or one followed by nothing but what
    # was said, is not what answers.
    only_quoted = colon and not QUOTE.sub("", after).strip(" ;,:")
    if colon and (label.lower() == subject.lower() or len(label.split()) <= 3
                  or only_quoted):
        head = after if after else label
    head = QUOTE.sub("", head).strip(" ;,:")
    return [one.strip() for one in re.split(r";\s*", head) if one.strip()]


def of_turn(turn: dict) -> Message:
    """A turn, as the page gets it (`Session.say(...).as_dict()`, trimmed),
    taken apart into what a reply to it has to say."""
    answer = turn.get("answer") or {}
    reading = turn.get("reading") or {}
    run = turn.get("run") or {}
    text = answer.get("text") or ""
    outcome = answer.get("outcome") or "unknown"
    stance = STANCES.get(outcome, "unknown")
    if stance == "unknown" and any(text.startswith(one) for one in REFUSED):
        stance = "refused"
    act = reading.get("act") or turn.get("act") or ""
    subject = _subject(turn)
    claim = ""
    if act in STATEMENTS or (act in POLAR and stance in ("yes", "no",
                                                         "unknown")):
        claim = _claim(turn, subject)
    if act in STATEMENTS and not claim and stance == "noted":
        # `there is a beagle`, `my name is Adrian`: nothing is claimed of a
        # subject by an auxiliary and a rest, so what was said is the claim.
        claim = turn.get("said") or ""
    content = run.get("content") if run else None
    values = (_values(text, content, subject)
              if stance == "value" else [])
    if stance == "value" and not values:
        # `a kitten: “young domestic cat”`, `yesterday: “the dog chased the
        # cat”`: what answers is what is quoted.
        values = QUOTE.findall(text)[:LISTED]
    return Message(
        stance=stance, heard=turn.get("said") or "", subject=subject,
        claim=claim, values=values, quotes=QUOTE.findall(text),
        source=answer.get("source") or "", trust=(run or {}).get("trust")
        or "", found=plain(text), rules=rules_in(text),
        more=((content or {}).get("more") or 0)
        + max(0, len((content or {}).get("items") or []) - LISTED),
        act=answer.get("act") or "")


def prompt(message: Message) -> str:
    """A message as the decoder reads it: one field a line, the empty ones
    left out."""
    lines = [f"heard: {message.heard}", f"stance: {message.stance}"]
    if message.act:
        lines.append(f"act: {message.act}")
    if message.subject:
        lines.append(f"about: {message.subject}")
    if message.claim:
        lines.append(f"claim: {message.claim}")
    if message.values:
        lines.append("values: " + "; ".join(message.values)
                     + (f" (and {message.more} more)" if message.more
                        else ""))
    if message.quotes:
        lines.append("quotes: " + " | ".join(message.quotes))
    if message.source:
        lines.append(f"source: {message.source}"
                     + (f" ({message.trust})" if message.trust else ""))
    if message.found:
        found = message.found
        if len(found) > FOUND:
            found = found[:FOUND].rsplit(" ", 1)[0] + " …"
        lines.append(f"found: {found}")
    return "\n".join(lines)


#: What only the page may say: rules, relations, senses, the layers' names.
INTERNAL = re.compile(
    r"\b(?:[RTSEI]\d{1,2}|v68\d|[a-z]+_[a-z_]+|[a-z]+\.[nvasr]\.\d\d|"
    r"VERIFIED|CONTRADICTED|DENIED|LISTING|DISPUTED)\b")


def required(message: Message) -> dict:
    """What a reply cannot leave out: its stance; for a yes, a no, a noted
    statement or an unknown, the subject and what is claimed of it; for a
    value, the first values -- save a value that is v689 naming its own
    workings (`v687's walk: R4`), which a reply is not to say. Phrases,
    compared by their content words (`roundtrip.py`)."""
    need = {"stance": message.stance, "phrases": []}
    if message.stance in ("yes", "no", "unknown", "noted") and message.claim:
        need["phrases"].append(message.claim)
    elif message.stance == "value":
        need["phrases"] += [one for one in message.values
                            if not INTERNAL.search(one)][:NAMED]
    return need
