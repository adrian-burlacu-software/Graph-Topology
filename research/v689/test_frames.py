"""Normal form: a sentence said back in the order the reader reads.

`frames.py`'s rules taught it; the encoder says it back (`reader.place`),
reading spaCy's small English model's tags beside each word. No store and no
engine. Skipped without spaCy's model, NLTK's WordNet or `llm/reader`.
"""
from __future__ import annotations

import unittest

from research.v689 import change, frames, reader
from research.v689.reading import tokens_of


def _nlp():
    try:
        import spacy
        return spacy.load("en_core_web_sm", disable=["ner"])
    except Exception:                               # noqa: BLE001
        return None


NLP = _nlp()


class Lexicon:
    """spaCy alone: the tags, dependencies and lemmas read beside each
    word."""

    def _parsed(self, words: list[str]):
        from spacy.tokens import Doc

        doc = Doc(NLP.vocab, words=list(words))
        for _, component in NLP.pipeline:
            doc = component(doc)
        return doc

    def analyse(self, words: list[str]) -> list[tuple]:
        return [(token.tag_, token.dep_, token.head.i)
                for token in self._parsed(words)]

    def tags(self, words: list[str]) -> list[str]:
        return [token.tag_ for token in self._parsed(words)]

    def lemma(self, word: str) -> str:
        read = NLP(word)
        return read[0].lemma_.lower() if len(read) else word


def normal(text: str) -> str:
    """The words `reader.place` says back, or nothing when it says them as
    they were said."""
    found = reader.place(text, Lexicon(), frozenset())
    return ("" if found.tokens == tokens_of(text)[0]
            else " ".join(found.tokens))


def framed(text: str) -> str:
    """The words `frames.py`'s rules say a question back as, or nothing.
    Only a statement is read in normal form (`reading.normal`): a question
    is read as said, and the encoder was taught its wordings by the
    grammar's own transports."""
    lexicon = Lexicon()
    lower, typed = tokens_of(text)
    found = frames.normal(lower, typed, lexicon.analyse(typed), lexicon.lemma,
                          change.theme_subject, lexicon.tags, change.meaning)
    return " ".join(found[0]) if found else ""


@unittest.skipUnless(NLP is not None and frames.derived_adverb("really"),
                     "needs spaCy's small model and WordNet")
class QuestionTests(unittest.TestCase):
    def test_an_adverb_wordnet_derives_is_an_adjunct(self):
        self.assertEqual(framed("where is Mary really"), "where is mary")
        self.assertEqual(framed("where exactly is Mary"), "where is mary")
        self.assertEqual(framed("is Mary really in the kitchen"),
                         "is mary in the kitchen")

    def test_a_question_inside_a_request(self):
        self.assertEqual(framed("can you tell me where Mary is"),
                         "where is mary")
        self.assertEqual(framed("any idea where Mary is"), "where is mary")
        self.assertEqual(framed("I wonder where Mary is"), "where is mary")

    def test_a_place_asked_by_its_kind(self):
        self.assertEqual(framed("which room is Mary in"), "where is mary")
        self.assertEqual(framed("in which room is Mary"), "where is mary")

    def test_a_time_phrase_and_a_discourse_word(self):
        self.assertEqual(framed("where is Mary at the moment"),
                         "where is mary")
        self.assertEqual(framed("so where is Mary"), "where is mary")

    @unittest.skipUnless(change.frames(), "needs data/verbnet3.3")
    def test_a_verb_asked_as_what_verbnet_says_it_means(self):
        self.assertEqual(framed("Where is Shanda hiding?"), "where is shanda")
        self.assertEqual(framed("Who is holding the pear?"),
                         "who has the pear")
        self.assertEqual(framed("Where will the other one go?"), "")
        self.assertEqual(framed("What did Shanda buy?"), "")

    def test_a_question_the_grammar_reads_is_left_alone(self):
        self.assertEqual(framed("who is in the kitchen"), "")


@unittest.skipUnless(NLP is not None and frames.derived_adverb("quickly")
                     and reader.enabled(),
                     "needs spaCy's small model, WordNet and llm/reader")
class StatementTests(unittest.TestCase):
    def test_manner_and_fronted_phrases(self):
        self.assertEqual(normal("Mary quickly went to the kitchen"),
                         "mary went to the kitchen")
        self.assertEqual(normal("in the end Mary went to the kitchen"),
                         "mary went to the kitchen")

    def test_a_passive_with_its_agent(self):
        self.assertEqual(normal("the apple was put in the kitchen by Mary"),
                         "mary put the apple in the kitchen")
        self.assertEqual(normal("the apple was given to Fred by Bill"),
                         "bill gave the apple to fred")

    @unittest.skipUnless(change.frames(), "needs data/verbnet3.3")
    def test_a_passive_without_one_where_verbnet_allows(self):
        self.assertEqual(normal("the apple was moved to the kitchen"),
                         "the apple moved to the kitchen")

    def test_what_is_not_an_adjunct_stays(self):
        for text in ("Mary went back to the kitchen", "it can not swim",
                     "the dog is really big", "maybe the dog is hungry",
                     "Mary went to the kitchen where she slept",
                     "is a rock hard"):
            self.assertEqual(normal(text), "", text)


@unittest.skipUnless(change.frames(), "needs data/verbnet3.3")
class DestinationTests(unittest.TestCase):
    def test_a_place_entered_is_where_the_subject_ends_up(self):
        found = change.effects("enter", has_object=True)
        self.assertIn(("location", "subject", "object", True),
                      {(one.kind, one.position, one.at, one.after)
                       for one in found})

    def test_a_theme_as_subject(self):
        self.assertTrue(change.theme_subject("roll"))
        self.assertFalse(change.theme_subject("put"))


if __name__ == "__main__":
    unittest.main()
