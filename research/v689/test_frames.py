"""Frames: a sentence read off its parse and said back in normal form.

spaCy's small English model parses; no store and no engine. Skipped without
spaCy's model, and the adjunct tests without NLTK's WordNet.
"""
from __future__ import annotations

import unittest

from research.v689 import change, frames
from research.v689.reading import tokens_of


def _nlp():
    try:
        import spacy
        return spacy.load("en_core_web_sm", disable=["ner"])
    except Exception:                               # noqa: BLE001
        return None


NLP = _nlp()


def normal(text: str) -> str:
    from spacy.tokens import Doc

    lower, typed = tokens_of(text)
    doc = Doc(NLP.vocab, words=list(typed))
    for _, component in NLP.pipeline:
        doc = component(doc)
    analysis = [(token.tag_, token.dep_, token.head.i) for token in doc]

    def lemma(word: str) -> str:
        read = NLP(word)
        return read[0].lemma_.lower() if len(read) else word

    def tags(words: list[str]) -> list[str]:
        read = Doc(NLP.vocab, words=list(words))
        for _, component in NLP.pipeline:
            read = component(read)
        return [token.tag_ for token in read]

    found = frames.normal(lower, typed, analysis, lemma, change.theme_subject,
                          tags, change.meaning)
    return " ".join(found[0]) if found else ""


@unittest.skipUnless(NLP is not None and frames.derived_adverb("really"),
                     "needs spaCy's small model and WordNet")
class QuestionTests(unittest.TestCase):
    def test_an_adverb_wordnet_derives_is_an_adjunct(self):
        self.assertEqual(normal("where is Mary really"), "where is mary")
        self.assertEqual(normal("where exactly is Mary"), "where is mary")
        self.assertEqual(normal("is Mary really in the kitchen"),
                         "is mary in the kitchen")

    def test_a_question_inside_a_request(self):
        self.assertEqual(normal("can you tell me where Mary is"),
                         "where is mary")
        self.assertEqual(normal("any idea where Mary is"), "where is mary")
        self.assertEqual(normal("I wonder where Mary is"), "where is mary")

    def test_a_place_asked_by_its_kind(self):
        self.assertEqual(normal("which room is Mary in"), "where is mary")
        self.assertEqual(normal("in which room is Mary"), "where is mary")

    def test_a_time_phrase_and_a_discourse_word(self):
        self.assertEqual(normal("where is Mary at the moment"),
                         "where is mary")
        self.assertEqual(normal("so where is Mary"), "where is mary")

    @unittest.skipUnless(change.frames(), "needs data/verbnet3.3")
    def test_a_verb_asked_as_what_verbnet_says_it_means(self):
        self.assertEqual(normal("Where is Shanda hiding?"), "where is shanda")
        self.assertEqual(normal("Who is holding the pear?"),
                         "who has the pear")
        self.assertEqual(normal("Where will the other one go?"), "")
        self.assertEqual(normal("What did Shanda buy?"), "")

    def test_a_question_the_grammar_reads_is_left_alone(self):
        self.assertEqual(normal("who is in the kitchen"), "")


@unittest.skipUnless(NLP is not None and frames.derived_adverb("quickly"),
                     "needs spaCy's small model and WordNet")
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
