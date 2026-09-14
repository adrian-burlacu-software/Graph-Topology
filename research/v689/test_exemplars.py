"""Exemplars: a question read like the nearest one the grammar understood.

No engine: the choice of a cell, the noun phrases put in place, and the
documented exemplars. The encoder test needs `llm/MiniLM-L6-v2` and torch; the
phrase tests spaCy's small English model.
"""
from __future__ import annotations

import unittest

from research.v689 import exemplars
from research.v689.exemplars import Exemplar, choose


def _nlp():
    try:
        import spacy
        return spacy.load("en_core_web_sm", disable=["ner"])
    except Exception:                               # noqa: BLE001
        return None


NLP = _nlp()

PLACE = ("place", "located")
GONE = ("place", "occurrence")


class ChoiceTests(unittest.TestCase):
    def test_near_and_clear_of_every_other_cell(self):
        ranked = [(0.91, Exemplar("where did mary go", GONE)),
                  (0.85, Exemplar("where is mary", PLACE))]
        cell, similarity, tried = choose(ranked)
        self.assertEqual(cell, GONE)
        self.assertEqual([one.text for one in tried], ["where did mary go"])

    def test_not_near_enough(self):
        self.assertIsNone(choose([(0.69, Exemplar("who can swim", GONE))]))

    def test_two_cells_as_near_is_no_choice(self):
        ranked = [(0.85, Exemplar("where will mary go", ("place", "motive"))),
                  (0.84, Exemplar("where is mary", PLACE))]
        self.assertIsNone(choose(ranked))


@unittest.skipUnless(NLP is not None, "needs spaCy's small model")
class PhraseTests(unittest.TestCase):
    def words(self, text: str):
        from spacy.tokens import Doc

        from research.v689.clauses import parse
        from research.v689.reading import tokens_of

        typed = tokens_of(text)[1]
        doc = Doc(NLP.vocab, words=list(typed))
        for _, component in NLP.pipeline:
            doc = component(doc)
        return typed, parse(typed, [(t.tag_, t.dep_, t.head.i) for t in doc])

    def test_the_phrases_that_fill_slots(self):
        typed, words = self.words("who has got the pear")
        self.assertEqual([typed[a:b] for a, b in exemplars.phrases(words)],
                         [["the", "pear"]])

    def test_never_i_or_you_or_a_question_word(self):
        typed, words = self.words("where can I find Mary")
        self.assertEqual([typed[a:b] for a, b in exemplars.phrases(words)],
                         [["Mary"]])

    def test_a_key_is_what_is_asked_not_of_whom(self):
        typed, words = self.words("where did Shanda end up")
        lower = [word.lower() for word in typed]
        self.assertEqual(exemplars.key(lower, exemplars.phrases(words)),
                         "where did it end up")

    def test_a_question_by_the_tagger(self):
        for text, asked in (("where did Shanda end up", True),
                            ("is Mary in the kitchen", False),
                            ("why is Mary in the kitchen", False),
                            ("Mary went to the kitchen", False)):
            typed, words = self.words(text)
            analysis = [(word.tag, word.dep, word.head) for word in words]
            self.assertEqual(exemplars.asking([w.lower() for w in typed],
                                              analysis, text), asked, text)

    def test_an_exemplar_s_own_phrases_by_position(self):
        tokens = "who did fred give the ball to".split()
        self.assertEqual(exemplars._spans(tokens, ["the ball", "fred"]),
                         [(2, 3), (4, 6)])
        self.assertIsNone(exemplars._spans(tokens, ["the cat"]))


class DocumentedTests(unittest.TestCase):
    def test_the_grammar_documents_its_questions(self):
        found = exemplars.documented()
        self.assertIn("where did Mary go", found)
        self.assertIn("who has the football", found)
        self.assertTrue(all("(" not in one and "\n" not in one
                            for one in found))


@unittest.skipUnless(exemplars.enabled(), "needs llm/MiniLM-L6-v2")
class EncoderTests(unittest.TestCase):
    def test_what_means_alike_is_near(self):
        encode = exemplars.Encoder()
        vectors = encode(["where did Mary end up", "where did Mary go",
                          "what is a hammer"])
        near = float(vectors[0] @ vectors[1])
        far = float(vectors[0] @ vectors[2])
        self.assertGreater(near, exemplars.SIMILARITY)
        self.assertLess(far, exemplars.SIMILARITY)


if __name__ == "__main__":
    unittest.main()
