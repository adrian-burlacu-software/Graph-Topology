"""The bAbI harness: reading the files and reading an answer out of a reply.

No engines and no model: what is tested is that both systems are scored by
the same reader, and that the reader neither misses a plain answer nor finds
one in a listing.
"""
from __future__ import annotations

import unittest

from research.v689.babi import (Story, base, gold, kind_of, outcome, parse,
                                predict, prompt)

SAMPLE = """1 Mary moved to the bathroom.
2 John went to the hallway.
3 Where is Mary? \tbathroom\t1
4 Daniel went back to the hallway.
5 Where is Daniel? \thallway\t4
1 Sandra journeyed to the bedroom.
2 Is Sandra in the hallway?\tno\t1
"""

PLACES = frozenset({"bathroom", "hallway", "kitchen", "garden", "office",
                    "bedroom"})


def _wordnet() -> bool:
    try:
        from nltk.corpus import wordnet
        return wordnet.morphy("wolves", wordnet.NOUN) == "wolf"
    except Exception:                               # noqa: BLE001
        return False


class ParseTests(unittest.TestCase):
    def test_a_line_numbered_one_starts_a_story(self):
        found = parse(SAMPLE, task=1)
        self.assertEqual([len(story.lines) for story in found], [5, 2])
        self.assertEqual([story.index for story in found], [0, 1])

    def test_a_question_keeps_its_answer_and_supports(self):
        line = parse(SAMPLE, task=1)[0].lines[2]
        self.assertEqual(line.text, "Where is Mary?")
        self.assertEqual(line.question.answer, "bathroom")
        self.assertEqual(line.question.supports, (1,))
        self.assertIsNone(parse(SAMPLE, task=1)[0].lines[0].question)

    def test_the_model_is_told_statements_not_earlier_questions(self):
        story = parse(SAMPLE, task=1)[0]
        text = prompt(story, 4)
        self.assertIn("Daniel went back to the hallway.", text)
        self.assertNotIn("Where is Mary?", text)
        self.assertTrue(text.endswith("Answer with one word only."))

    def test_the_form_is_the_task_s(self):
        self.assertEqual(kind_of(6), "yesno")
        self.assertEqual(kind_of(10), "maybe")
        self.assertEqual(kind_of(1), "word")
        story = Story(19, 0, parse(SAMPLE, task=19)[0].lines)
        self.assertIn("n, s, e and w", prompt(story, 2))


class PredictTests(unittest.TestCase):
    def test_one_word(self):
        self.assertEqual(predict("word", "bathroom", "Where is Mary?", PLACES),
                         "bathroom")
        self.assertEqual(predict("word", "Mary is in the bathroom.",
                                 "Where is Mary?", PLACES), "bathroom")

    def test_a_word_from_the_question_is_not_the_answer(self):
        self.assertEqual(predict("word", "before the school she was in the "
                                 "office", "Where was Julie before the "
                                 "school?", PLACES | {"school"}), "office")

    def test_a_listing_is_no_answer(self):
        self.assertIsNone(predict("word", "found in the kitchen, the garden, "
                                  "the office", "Where is Mary?", PLACES))
        self.assertIsNone(predict("word", "UNKNOWN (absent, not false)",
                                  "Where is Mary?", PLACES))
        # one word of the answer space, but in a listing of other things
        self.assertIsNone(predict("word", "mary is found in the hospital, "
                                  "prison, the kitchen, labor and 32 more",
                                  "Where is Mary?", PLACES))

    def test_only_the_headline_is_read(self):
        # v689's reasons come after a dash; the model's after its first line.
        self.assertEqual(predict("word", "the kitchen — before that, the "
                                 "garden", "Where is Mary?", PLACES),
                         "kitchen")
        self.assertEqual(predict("word", "Garden\n\nStep 1: the bedroom is "
                                 "north", "What is south of the office?",
                                 PLACES), "garden")

    def test_repeating_the_question_is_a_wrong_answer(self):
        self.assertEqual(predict("word", "bedroom", "Where was Julie before "
                                 "the bedroom?", PLACES), "bedroom")
        self.assertIsNone(predict("word", "Julie was in the bedroom before "
                                  "that, I think", "Where was Julie before "
                                  "the bedroom?", PLACES))

    def test_yes_or_no_is_the_first_word(self):
        self.assertEqual(predict("yesno", "No.", "Is Mary here?", PLACES),
                         "no")
        self.assertEqual(predict("yesno", "no — she moved to the kitchen",
                                 "Is Mary here?", PLACES), "no")
        self.assertIsNone(predict("yesno", "not known of Mary; no answer",
                                  "Is Mary here?", PLACES))
        self.assertIsNone(predict("yesno", "maybe", "Is Mary here?", PLACES))
        self.assertEqual(predict("maybe", "maybe", "Is Mary here?", PLACES),
                         "maybe")

    def test_a_count(self):
        self.assertEqual(predict("count", "Two.", "How many?", PLACES), "two")
        self.assertEqual(predict("count", "she carries 1 thing", "How many?",
                                 PLACES), "one")
        self.assertEqual(predict("count", "none", "How many?", PLACES),
                         "none")

    def test_a_list_is_a_set(self):
        space = frozenset({"football", "milk", "apple", "nothing"})
        self.assertEqual(predict("list", "milk, football", "What is Mary "
                                 "carrying?", space), "football,milk")
        self.assertEqual(gold("list", "football,milk"), "football,milk")
        self.assertEqual(predict("list", "Nothing.", "What is Mary "
                                 "carrying?", space), "nothing")
        self.assertIsNone(predict("list", "mary is happy", "What is Mary "
                                  "carrying?", space))

    def test_a_path_in_order(self):
        self.assertEqual(predict("path", "s,e", "How do you go?", PLACES),
                         "s,e")
        self.assertEqual(predict("path", "go south, then east", "How do you "
                                 "go?", PLACES), "s,e")
        self.assertIsNone(predict("path", "the dog's bowl", "How do you go?",
                                  PLACES))

    @unittest.skipUnless(_wordnet(), "needs nltk's WordNet")
    def test_a_plural_answers_its_singular(self):
        self.assertEqual(base("wolves"), "wolf")
        self.assertEqual(predict("word", "wolves", "What is Gertrude afraid "
                                 "of?", frozenset({"wolf", "cat"})), "wolf")

    def test_outcomes(self):
        self.assertEqual(outcome(None, "bathroom"), "none")
        self.assertEqual(outcome("bathroom", "bathroom"), "right")
        self.assertEqual(outcome("kitchen", "bathroom"), "wrong")


if __name__ == "__main__":
    unittest.main()
