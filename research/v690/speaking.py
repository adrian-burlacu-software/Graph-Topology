"""Saying an answer: the decoder writes it, the encoder reads it back, and a
reply is said when it reads back to what it was meant to say.

    a v689 turn   what was read, resolved and answered, rules and all
    a message     what a reply to it has to say (`message.py`)
    candidates    replies to the message, the likeliest first (`decoder.py`)
    read back     each candidate as the encoder reads it (`roundtrip.py`)
    said          the first candidate that traces

This is the trie paper's backward error correction applied to speaking (v690
`DESIGN.md` §3, §4.8): a learned router is not trusted to be right, so what
it routes to is checked by the reverse traversal, which is cheaper. The
decoder is fluent and can be wrong; the encoder reading its reply back is
what makes a wrong one visible.

When no candidate traces, each is read again with one of its sentences left
out -- what a decoder adds once the answer is said, `You said so.` of what the
store said or a sentence it stopped in the middle of, can go unsaid -- and
then more are written, sampled more freely. When none of those does either,
the one with the fewest things wrong with it is said, and the turn says so: it
is marked untraced, with what could not be traced. A reply is never said as
traced when it was not, and a shortened one is read back like any other.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .message import Message, of_turn, prompt

#: How many replies are written first, and how many more when none traces.
FIRST = 4
MORE = 6
HOTTER = 1.0


@dataclass
class Candidate:
    text: str
    trace: dict
    read: dict
    #: the reply this one is, with a sentence left out
    shortened_from: str = ""

    def as_dict(self) -> dict:
        return {"text": self.text, "trace": self.trace, "read": self.read,
                "shortened_from": self.shortened_from}


@dataclass
class Said:
    """What was said, and how it was chosen."""

    text: str
    traced: bool
    message: dict
    prompt: str
    candidates: list = field(default_factory=list)
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {"text": self.text, "traced": self.traced,
                "message": self.message, "prompt": self.prompt,
                "candidates": [one.as_dict() for one in self.candidates],
                "seconds": round(self.seconds, 3)}


def shortened(text: str) -> list[str]:
    """A reply with one of its sentences left out, each way, the last left
    out first; nothing for a reply of one sentence. Leaving out never adds a
    word, and what is left has to read back as a reply of its own."""
    sentences = [one for one in re.split(r"(?<=[.!?])\s+", text.strip())
                 if one]
    if len(sentences) < 2:
        return []
    return [" ".join(sentences[:at] + sentences[at + 1:])
            for at in range(len(sentences) - 1, -1, -1)]


def wrongness(trace: dict) -> int:
    """How much is wrong with a reply: a stance or an internal word most,
    then a denial the wrong way, then each word left out or added."""
    return (3 * (trace["stance"] != trace["expected"])
            + 3 * bool(trace["internal"]) + 2 * bool(trace["denial"])
            + len(trace["missing"]) + len(trace["added"]))


class Speaker:
    """The decoder and the encoder's reading back, for one process."""

    def __init__(self, nlp=None, decoder=None) -> None:
        from . import decoder as decoders
        from .roundtrip import Words, enabled

        # Nothing is said that is not read back, so there is no speaking
        # without an encoder that reads replies.
        if not enabled():
            raise RuntimeError(
                "the reader has no heads for reading replies back: teach it "
                "(python -m research.v690.teach_decoder label, then python -m "
                "research.v689.teach_reader train), or run the page --silent")
        self.words = Words(nlp)
        self.decoder = decoder or decoders.LOADED.get()
        self.framing = self.decoder.framing

    def check(self, message: Message, texts: list[str],
              shortened_from: str = "") -> list[Candidate]:
        from .roundtrip import reading_of, trace

        found = []
        for text in texts:
            read = reading_of(text, self.words)
            checked = trace(message, read, self.words, self.framing, text)
            found.append(Candidate(text, checked.as_dict(), read.as_dict(),
                                   shortened_from))
        return found

    def shorten(self, message: Message,
                candidates: list[Candidate]) -> list[Candidate]:
        """The untraced candidates with a sentence left out, until one of
        them traces; the least wrong candidates are shortened first."""
        seen = {one.text for one in candidates}
        found: list[Candidate] = []
        for one in sorted(candidates, key=lambda c: wrongness(c.trace)):
            if one.trace["traced"] or one.shortened_from:
                continue
            texts = [text for text in shortened(one.text) if text not in seen]
            seen.update(texts)
            for made in self.check(message, texts, one.text):
                found.append(made)
                if made.trace["traced"]:
                    return found
        return found

    def speak(self, turn: dict) -> Said:
        """A reply to a turn (`Session.say(...).as_dict()`)."""
        message = of_turn(turn)
        return self.say(message)

    def say(self, message: Message) -> Said:
        started = time.time()
        shown = prompt(message)
        candidates = self.check(message, self.decoder.say([shown],
                                                         FIRST)[0])
        chosen = next((one for one in candidates if one.trace["traced"]),
                      None)
        if chosen is None:
            candidates += self.shorten(message, candidates)
            chosen = next((one for one in candidates
                           if one.trace["traced"]), None)
        if chosen is None:
            more = self.decoder.say([shown], MORE, temperature=HOTTER,
                                    greedy=False)[0]
            more = [one for one in more
                    if one not in {c.text for c in candidates}]
            more = self.check(message, more)
            candidates += more
            candidates += self.shorten(message, more)
            chosen = next((one for one in candidates
                           if one.trace["traced"]), None)
        traced = chosen is not None
        if chosen is None and candidates:
            chosen = min(candidates, key=lambda one: wrongness(one.trace))
        text = chosen.text if chosen is not None else ""
        return Said(text, traced, message.as_dict(), shown, candidates,
                    time.time() - started)
