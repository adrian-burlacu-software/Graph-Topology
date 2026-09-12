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

    def _parsed(self, words: list[str]):
        """spaCy over the words exactly as given, read as one sentence.

        Handed over already split, so token `i` is word `i`: `reading.py`
        expands contractions and strips punctuation first, and letting spaCy
        tokenize again would misalign the two.
        """
        nlp = getattr(self.parser, "nlp", None)
        if nlp is None or not words:
            return None
        from spacy.tokens import Doc

        doc = Doc(nlp.vocab, words=list(words))
        for _, component in nlp.pipeline:
            doc = component(doc)
        return doc

    def tags(self, words: list[str]) -> list[str] | None:
        """Penn tags, in the context of the whole sentence."""
        doc = self._parsed(words)
        return [token.tag_ for token in doc] if doc is not None else None

    def analyse(self, words: list[str]) -> list[tuple] | None:
        """(tag, dependency, head index) per word: what `clauses.py` reads."""
        doc = self._parsed(words)
        return ([(token.tag_, token.dep_, token.head.i) for token in doc]
                if doc is not None else None)

    def words_of(self, text: str) -> list | None:
        """spaCy's own reading of a text, tokenized by spaCy: for glosses,
        where `short-legged` has to come apart at the hyphen."""
        nlp = getattr(self.parser, "nlp", None)
        if nlp is None or not (text or "").strip():
            return None
        from .clauses import Word

        return [Word(token.i, token.text, token.tag_, token.dep_,
                     token.head.i, token.head.text, token.lemma_.lower())
                for token in nlp(text)]

    def antonyms(self, lemma: str) -> frozenset:
        """WordNet's antonyms of a word in any of its senses: `shrink` gives
        expand and stretch. Empty where WordNet is not installed."""
        cache = self.__dict__.setdefault("_antonyms", {})
        if lemma not in cache:
            try:
                from nltk.corpus import wordnet

                cache[lemma] = frozenset(
                    antonym.name().replace("_", " ")
                    for synset in wordnet.synsets(lemma)
                    for word in synset.lemmas()
                    for antonym in word.antonyms())
            except Exception:                   # noqa: BLE001
                cache[lemma] = frozenset()
        return cache[lemma]

    def known(self, phrase: str) -> bool:
        return phrase in (self.parser.nouns or self.parser.vocabulary or ())

    def sense(self, kind: str):
        """The synset a kind word is placed under: the reasoner's first noun
        reading, which is the one v687 itself would take."""
        senses = self.reasoner.senses_of(kind, "n") or []
        return senses[0]["id"] if senses else None

    def judge(self, question: str):
        """(supports, confidence) from v688's teacher, asked bare; None when
        there is no teacher. The server supplies one."""
        return None

    def run(self, question: str) -> dict:
        raise NotImplementedError("v688's loop is supplied by the caller")
