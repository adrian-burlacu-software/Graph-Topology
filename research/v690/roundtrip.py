"""A reply read back: does it say what its message says?

The decoder writes a reply from a message (`decoder.py`). Before it is said,
the encoder reads it back, as it reads anything said to it
(`research/encoder.py`), with two heads of its own:

    stance   what kind of answer the reply gives (`message.STANCE_LABELS`)
    part     what each word is in it:
                 O       said around what it says: `you told me`, `I think`
                 SUBJ    who or what it is about
                 CLAIM   what is claimed of them
                 NEG     a word that denies the claim
                 VALUE   what answers the question
                 QUOTE   what was said, that the answer rests on

A reply is **traced** when what it is read as is what its message says:

    its stance is the message's
    the claim reads with its subject and every word claimed, and denied
        where the message denies it -- a no to `can the beagle swim` says
        the beagle cannot
    every value the message requires is read as a value
    nothing is read as said that the message does not have: no content word
        outside it, save the words replies say around any content, which
        are counted over the corpus (`teach_decoder.py`), not listed
    nothing internal is said: no rule, relation or sense

The last two keep a fluent reply honest: a reason, a name or an example the
message has no word for is a word read as said that nothing licensed.

Content words are the nouns, verbs, adjectives and numbers that are not
auxiliaries or quantifiers, compared by lemma -- `beagles` is `beagle`,
`went` is `go`. A word is also said by one WordNet puts in the same synset,
and a verb by one it is a kind of, two levels up: `Morissa went to the
cinema` says she travelled there. The same comparison labels the teacher's
replies (`label`), so what the encoder is taught to read and what the round
trip checks are one thing.
"""
from __future__ import annotations

import re

from dataclasses import dataclass, field
from functools import lru_cache

from .message import INTERNAL, Message

PARTS = ("O", "SUBJ", "CLAIM", "NEG", "VALUE", "QUOTE")

#: The encoder's heads for a reply.
REPLY_HEADS = ("stance", "part")

#: Words that deny what they stand next to.
NEGATIONS = frozenset({"not", "never", "no", "nobody", "nothing", "none",
                       "neither", "nor", "nowhere"})

#: Penn tags whose words carry content.
CONTENT_TAGS = ("NN", "VB", "JJ", "CD")

#: Words with those tags that carry none: auxiliaries, and adjectives that
#: only say how many or which. spaCy's stop list is not used: it has
#: `whole`, `empty`, `go`, `give` and `first`, which claims are made of.
FUNCTION = frozenset({"be", "have", "do", "will", "would", "shall", "should",
                      "can", "could", "may", "might", "must", "ought",
                      "other", "such", "own", "same", "many", "much", "few",
                      "more", "most", "several", "each", "every", "all",
                      "any", "some", "'s", "s", "i", "me", "you", "he",
                      "him", "she", "her", "it", "we", "us", "they", "them",
                      # which the tagger calls nouns; `someone` stays, since
                      # `someone told me` is a source nobody gave
                      "anything", "something", "everything", "anyone",
                      "anybody", "everyone", "everybody"})

#: How many of a word's senses its synonyms are read from, and how far up a
#: verb's hypernyms go.
SENSES = 3
UP = 2


class Words:
    """spaCy over a text's words, as `reading.tokens_of` splits them: each
    word's tag, dependency and lemma, read in context."""

    def __init__(self, nlp=None) -> None:
        if nlp is None:
            import spacy
            nlp = spacy.load("en_core_web_sm", disable=["ner"])
        self.nlp = nlp
        self._seen: dict = {}

    def read(self, text: str) -> tuple[list[str], list[str], list[str],
                                       list[str]]:
        """(words as typed, tags, dependencies, lemmas)."""
        from research.v689.reading import tokens_of

        if text in self._seen:
            return self._seen[text]
        from spacy.tokens import Doc

        typed = tokens_of(text)[1]
        if not typed:
            found = ([], [], [], [])
        else:
            doc = Doc(self.nlp.vocab, words=list(typed))
            for _, component in self.nlp.pipeline:
                doc = component(doc)
            found = (typed, [token.tag_ for token in doc],
                     [token.dep_ for token in doc],
                     [token.lemma_.lower() for token in doc])
        if len(self._seen) > 20000:
            self._seen.clear()
        self._seen[text] = found
        return found

    def content(self, text: str) -> list[str]:
        """The lemmas of a text's content words, in order."""
        typed, tags, _, lemmas = self.read(text)
        return content_of(typed, tags, lemmas)


def content_at(word: str, tag: str, lemma: str) -> bool:
    lower = word.lower()
    # `one` is content as a number (`one object`), and none as a pronoun
    # (`this one`, `the first one`), which the tagger calls a noun.
    if lower in ("one", "ones") and not tag.startswith("CD"):
        return False
    return (bool(tag) and tag.startswith(CONTENT_TAGS)
            and lemma not in FUNCTION and lower not in FUNCTION
            and lower not in NEGATIONS
            and (lower.isalnum() or lower.replace("-", "").isalnum()))


def content_of(typed, tags, lemmas) -> list[str]:
    return [lemma for word, tag, lemma in zip(typed, tags, lemmas)
            if content_at(word, tag, lemma)]


@lru_cache(maxsize=50000)
def broader(lemma: str) -> frozenset:
    """What else says a word: its synonyms in its first senses, and for a
    verb the verbs it is a kind of, two levels up (`journey` is said by
    `travel` and by `go`). Empty without WordNet."""
    try:
        from nltk.corpus import wordnet
    except Exception:                               # noqa: BLE001
        return frozenset()
    found: set = set()
    try:
        senses = wordnet.synsets(lemma.replace(" ", "_"))
    except Exception:                               # noqa: BLE001
        return frozenset()
    for pos in ("n", "v", "a", "s"):
        for synset in [one for one in senses if one.pos() == pos][:SENSES]:
            found.update(name.lower().replace("_", " ")
                         for name in synset.lemma_names())
            if pos == "v":
                for above in synset.closure(lambda one: one.hypernyms(),
                                            depth=UP):
                    found.update(name.lower().replace("_", " ")
                                 for name in above.lemma_names())
    found.discard(lemma)
    return frozenset(found)


def says(wanted: str, said) -> bool:
    """Whether a lemma is said among `said`: as itself, or by a word that
    says it (`broader`)."""
    return wanted in said or bool(broader(wanted) & set(said))


def negated(text: str) -> bool:
    from research.v689.reading import tokens_of
    return any(word in NEGATIONS for word in tokens_of(text)[0])


#: What a denial governs when it denies what a reply is framed by rather
#: than what it claims: `you haven't told me`, `I'm not sure`, `no
#: information`, `it hasn't come up`.
FRAMED = frozenset({"know", "sure", "certain", "tell", "say", "mention",
                    "teach", "hear", "find", "remember", "information",
                    "idea", "record", "knowledge", "come", "clear"})

#: How many words after a denial are looked at for what it governs.
GOVERNS = 3


def sentences_of(text: str, count: int) -> list[int]:
    """Which sentence each of a text's words is in, as `reading.tokens_of`
    splits them; one sentence for all when the two splits disagree."""
    import re

    from research.v689.reading import tokens_of

    found: list[int] = []
    for index, chunk in enumerate(re.split(r"(?<=[.!?;:])\s+", text or "")):
        found += [index] * len(tokens_of(chunk)[1])
    return found if len(found) == count else [0] * count


def denying(words: list, tags: list, lemmas: list, parts: list,
            sentence: list[int]) -> list[int]:
    """Where a reply denies what it claims: a denial in a sentence that says
    its subject, claim or a value, or in a sentence that says nothing of its
    own right after one (`Fred went to the bar. You told me he did not.`) --
    and not one governing a word of telling or knowing, nor a `No` opening a
    sentence, which is the answer's stance."""
    said = {sentence[at] for at, part in enumerate(parts)
            if part in ("SUBJ", "CLAIM", "VALUE")}
    # A sentence says something of its own when it has a content word that is
    # not only telling or knowing: `You told me he did not` says nothing but
    # that it was told.
    content = {sentence[at] for at in range(len(words))
               if content_at(words[at], tags[at], lemmas[at])
               and parts[at] != "NEG" and lemmas[at] not in FRAMED}
    empty_after = {one for one in set(sentence)
                   if one not in content and one - 1 in said}
    found = []
    for at, word in enumerate(words):
        lower = str(word).lower()
        if lower not in NEGATIONS:
            continue
        if sentence[at] not in said and sentence[at] not in empty_after:
            continue
        if lower == "no" and (at == 0 or sentence[at - 1] != sentence[at]):
            continue
        # What it governs is in its own sentence: `can't swim. You told me`
        # denies swimming, not telling.
        governed = [lemmas[one] for one in range(at + 1, min(
            len(words), at + 1 + GOVERNS)) if sentence[one] == sentence[at]]
        if FRAMED & set(governed):
            continue
        found.append(at)
    return found


@dataclass
class Read:
    """What a reply is read as: its stance and each word's part."""

    stance: str
    words: list
    parts: list
    tags: list = field(default_factory=list)
    lemmas: list = field(default_factory=list)
    chance: float = 1.0

    def lemmas_in(self, *wanted: str) -> list[str]:
        return [lemma for word, part, tag, lemma
                in zip(self.words, self.parts, self.tags, self.lemmas)
                if part in wanted and content_at(word, tag, lemma)]

    def as_dict(self) -> dict:
        return {"stance": self.stance, "chance": round(self.chance, 3),
                "words": list(self.words), "parts": list(self.parts)}


@dataclass
class Trace:
    """Whether a reply says what its message says, and where it does not."""

    stance: str
    expected: str
    missing: list = field(default_factory=list)
    added: list = field(default_factory=list)
    denial: str = ""
    internal: list = field(default_factory=list)
    #: what the reply says the answer rests on, where that is not so: `you
    #: told me` of what the store says of a kind
    source: str = ""
    #: a reply cut off before its sentence ends
    unfinished: str = ""

    @property
    def traced(self) -> bool:
        return (self.stance == self.expected and not self.missing
                and not self.added and not self.denial
                and not self.internal and not self.source
                and not self.unfinished)

    def why(self) -> str:
        if self.traced:
            return "reads back to its message"
        found = []
        if self.stance != self.expected:
            found.append(f"reads as {self.stance}, not {self.expected}")
        if self.missing:
            found.append("leaves out " + ", ".join(self.missing))
        if self.added:
            found.append("adds " + ", ".join(self.added))
        if self.denial:
            found.append(self.denial)
        if self.internal:
            found.append("says " + ", ".join(self.internal))
        if self.source:
            found.append(self.source)
        if self.unfinished:
            found.append(self.unfinished)
        return "; ".join(found)

    def as_dict(self) -> dict:
        return {"traced": self.traced, "stance": self.stance,
                "expected": self.expected, "missing": self.missing,
                "added": self.added, "denial": self.denial,
                "internal": self.internal, "source": self.source,
                "unfinished": self.unfinished,
                "why": self.why()}


def vocabulary(message: Message, words: Words) -> set:
    """Every content lemma the message has."""
    found: set = set()
    for text in ([message.heard, message.subject, message.claim,
                  message.found, message.trust] + list(message.values)
                 + list(message.quotes)):
        if text:
            found.update(words.content(text))
    if message.more:
        found.add(str(message.more))            # `and 35 more`
    return found


def claimed(message: Message, words: Words) -> tuple[list, list]:
    """(the subject's content lemmas, the rest of the claim's)."""
    subject = words.content(message.subject) if message.subject else []
    rest = [one for one in words.content(message.claim)
            if one not in subject] if message.claim else []
    return subject, rest


def denies(message: Message) -> bool | None:
    """Whether a reply to this message denies its claim: a claim that does
    not hold, said as it is, or one that does, answered no. None where a
    reply's denial says nothing of the claim -- a value, an unknown."""
    if not message.claim or message.stance not in ("yes", "no", "noted"):
        return None
    return negated(message.claim) != (message.stance == "no")


def framing_for(framing, stance: str) -> frozenset:
    """The words said around content in a reply of this stance: `framing`
    kept per stance (`teach_decoder.framing`), or one set for all."""
    if isinstance(framing, dict):
        return frozenset(framing.get(stance, ())) | frozenset(
            framing.get("", ()))
    return frozenset(framing or ())


def trace(message: Message, read: Read, words: Words,
          framing=frozenset(), text: str = "") -> Trace:
    """What a reply, as read, says against what its message says."""
    from .message import NAMED, required

    framing = framing_for(framing, message.stance)
    need = required(message)
    found = Trace(read.stance, need["stance"])
    said = read.lemmas_in("SUBJ", "CLAIM", "NEG")
    wanted: list = []
    if message.claim and need["stance"] in ("yes", "no", "unknown", "noted"):
        subject, rest = claimed(message, words)
        wanted = list(dict.fromkeys(subject + rest))
        found.missing += [one for one in wanted if not says(one, said)]
    if need["stance"] == "value":
        valued = read.lemmas_in("VALUE", "CLAIM")
        for value in need["phrases"][:NAMED]:
            content = words.content(value)
            wanted += content
            found.missing += [one for one in content
                              if not says(one, valued)]
    known = vocabulary(message, words)
    # A word said for one the message has is licensed by it: `she went
    # there`, of `Mary and John journeyed to the kitchen`.
    licensed = set().union(*(broader(one) for one in known)) \
        if known else set()

    def unlicensed(lemma: str) -> bool:
        return lemma not in known and lemma not in licensed

    parted = read.lemmas_in("SUBJ", "CLAIM", "VALUE", "QUOTE")
    around = [lemma for word, part, tag, lemma
              in zip(read.words, read.parts, read.tags, read.lemmas)
              if part == "O" and content_at(word, tag, lemma)]
    found.added = list(dict.fromkeys(
        [one for one in parted if unlicensed(one)]
        + [one for one in around if unlicensed(one)
           and one not in framing]))
    denial = denies(message)
    if denial is not None and need["stance"] != "unknown":
        read_denied = "NEG" in read.parts
        if read_denied != denial:
            found.denial = ("denies the claim" if read_denied
                            else "does not deny the claim")
    # A value that is none -- `nothing`, `nobody` -- has no content word to
    # be read back by, so it is read back by its denial: `Orrin is carrying
    # anything` does not say he carries nothing.
    if (need["stance"] == "value" and not found.denial
            and any(negated(one) for one in message.values[:NAMED])
            and not any(str(word).lower() in NEGATIONS
                        for word in read.words)):
        found.denial = "does not say there is none"
    # Not knowing is said where the claim is: `I do know if the dog is
    # hungry` is no unknown, whatever else it goes on not to know.
    if need["stance"] in ("unknown", "refused") and not found.denial:
        sentence = sentences_of(text or " ".join(map(str, read.words)),
                                len(read.words))
        said = {sentence[at] for at, part in enumerate(read.parts)
                if part in ("SUBJ", "CLAIM", "VALUE")}
        if not any(str(word).lower() in NEGATIONS
                   and (not said or sentence[at] in said)
                   for at, word in enumerate(read.words)):
            found.denial = "does not say it is not known"
    found.internal = sorted(set(INTERNAL.findall(text or " ".join(
        read.words))))
    # Where the answer came from is part of what it says: `you told me` of
    # what the store says of dogs is a derivation that did not happen.
    if message.source and message.source not in FROM_YOU:
        lower = [one.lower() for one in read.words]
        sentence = sentences_of(text or " ".join(read.words), len(read.words))
        for at, lemma in enumerate(read.lemmas):
            # `you haven't told me` says nothing came from you; the denial has
            # to be in its own sentence, or `No, a penguin can't fly. You said
            # so.` is excused by the answer's own `can't`.
            before = {lower[one] for one in range(max(0, at - 3), at)
                      if sentence[one] == sentence[at]}
            if (lemma in SAYING and "you" in lower[max(0, at - 2):at]
                    and not NEGATIONS & before):
                found.source = "says you told it, and you did not"
                break
    # The decoder stops at its longest reply, and what it wrote by then can
    # read back word for word: `... and what someone is like. I can even help`.
    if text and not re.search(r"[.!?…\"'”’)]\s*$", text.strip()):
        found.unfinished = "stops before its sentence ends"
    return found


#: The sources an answer rests on that are something you said.
FROM_YOU = frozenset({"told", "taught", "learned", "conversation"})

#: What a reply says you did when it says an answer came from you.
SAYING = frozenset({"tell", "teach", "say", "mention"})


def label(message: Message, text: str, words: Words) -> Read:
    """What a reply's words are, from its message: the teacher's reading of
    a reply, which the encoder is taught to give. Values first, then the
    subject and the claim, then what is quoted, each content word found by
    lemma -- or by a word that says it -- where nothing has claimed it yet;
    a denial where it denies what the reply claims (`denying`)."""
    from .message import LISTED

    typed, tags, _, lemmas = words.read(text)
    parts = ["O"] * len(typed)

    def free(index: int) -> bool:
        return (parts[index] == "O"
                and content_at(typed[index], tags[index], lemmas[index]))

    def claim_words(wanted: list, part: str) -> None:
        for lemma in wanted:
            at = next((index for index, one in enumerate(lemmas)
                       if one == lemma and free(index)), None)
            if at is None:
                related = broader(lemma)
                at = next((index for index, one in enumerate(lemmas)
                           if one in related and free(index)), None)
            if at is not None:
                parts[at] = part

    for value in message.values[:LISTED]:
        claim_words(words.content(value), "VALUE")
    subject, rest = claimed(message, words)
    claim_words(subject, "SUBJ")
    claim_words(rest, "CLAIM")
    for quote in message.quotes:
        claim_words(words.content(quote), "QUOTE")
    for at in denying(typed, tags, lemmas, parts,
                      sentences_of(text, len(typed))):
        if parts[at] == "O":
            parts[at] = "NEG"
    return Read(message.stance, typed, parts, tags, lemmas)


def reading_of(text: str, words: Words) -> Read:
    """What the encoder reads a reply as."""
    from research import encoder

    typed, tags, deps, lemmas = words.read(text)
    if not typed:
        return Read("unknown", [], [], [], [])
    found = encoder.read(typed, tags, deps, REPLY_HEADS)
    stance, chance = found["stance"][0]
    return Read(stance, typed, list(found["part"]), tags, lemmas, chance)


def enabled() -> bool:
    """Is there an encoder that reads replies?"""
    import json

    from research import encoder

    if not encoder.enabled():
        return False
    heads = json.loads((encoder.MODEL / "labels.json").read_text(
        encoding="utf-8")).get("heads", {})
    return all(name in heads for name in REPLY_HEADS)
