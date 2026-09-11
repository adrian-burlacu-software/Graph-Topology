"""What a conversation needs from the layers below it.

Everything here is v687: its parser reads the words, its reasoner holds the
taxonomy and the store, its matcher decides whether a fact answers a question.
The one thing left abstract is `run` -- v688's loop on a question about a
kind -- because a test supplies a stand-in and the server supplies the loop.
"""
from __future__ import annotations


class Asker:
    """One v687 reasoner and parser, read through the questions v689 asks."""

    def __init__(self, reasoner, parser, matcher=None) -> None:
        self.reasoner = reasoner
        self.parser = parser
        self.matcher = matcher or parser.matcher()

    def subject(self, question: str):
        """What v687's parser reads a question as being about."""
        return self.parser.parse(question).subject

    def parse(self, question: str):
        """The relation and target v687 would put this question to."""
        return self.parser.parse(question)

    def lemma(self, word: str) -> str:
        nlp = getattr(self.parser, "nlp", None)
        if nlp is None:
            return word
        read = nlp(word)
        return read[0].lemma_.lower() if len(read) else word

    def progressive(self, word: str):
        """The verb in `it is flying`; None for `it is boring`.

        Read in a frame rather than alone, because a bare `flying` tags as a
        noun and a bare lemma cannot tell a participle from an adjective.
        """
        nlp = getattr(self.parser, "nlp", None)
        if nlp is None or not word.endswith("ing"):
            return None
        token = nlp(f"it is {word}")[-1]
        lemma = token.lemma_.lower()
        return lemma if token.pos_ == "VERB" and lemma != word else None

    def known(self, phrase: str) -> bool:
        return phrase in (self.parser.nouns or self.parser.vocabulary or ())

    def sense(self, kind: str):
        """The synset a kind word is placed under: the reasoner's first noun
        reading, which is the one v687 itself would take."""
        senses = self.reasoner.senses_of(kind, "n") or []
        return senses[0]["id"] if senses else None

    def run(self, question: str) -> dict:
        raise NotImplementedError("v688's loop is supplied by the caller")
