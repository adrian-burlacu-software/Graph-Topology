"""Requests put as their questions, the openers v688 asks as they stand, and
the constructions R18 refuses by name. No engine is loaded."""
from __future__ import annotations

import unittest

from research.v687 import logic
from research.v688.loop import seed_questions
from research.v688.rephrase import rephrase


class RephraseTests(unittest.TestCase):

    def said(self, text):
        return rephrase(text).text

    def test_an_embedded_question_is_the_question(self):
        self.assertEqual(self.said("do you know if a dog can swim"),
                         "can a dog swim")
        self.assertEqual(self.said("tell me whether a cat has fur"),
                         "does a cat have fur")
        self.assertEqual(self.said("is it true that dogs bark"),
                         "do dogs bark")
        self.assertEqual(self.said("can you tell me whether a dog barks?"),
                         "does a dog bark")

    def test_likelihood_is_asked_as_whether_it_holds_and_says_so(self):
        found = rephrase("is it likely that a bird can fly")
        self.assertEqual(found.text, "can a bird fly")
        self.assertIn("likely", found.note)

    def test_a_negative_question_asks_the_positive_one(self):
        found = rephrase("can't a dog swim")
        self.assertEqual((found.text, bool(found.note)), ("can a dog swim", True))
        self.assertEqual(self.said("Doesn't a cat have fur?"),
                         "does a cat have fur")
        self.assertEqual(self.said("does not a cat have fur"),
                         "does a cat have fur")

    def test_requests(self):
        self.assertEqual(self.said("describe a cat"), "tell me about a cat")
        self.assertEqual(self.said("define hammer"), "what is a hammer")
        self.assertEqual(self.said("what does bark mean"), "what is a bark")
        self.assertEqual(self.said("name three birds"),
                         "what kinds of birds are there")
        self.assertEqual(self.said("list animals that fly"),
                         "which animals fly")
        self.assertEqual(self.said("please describe a dog"),
                         "tell me about a dog")

    def test_you_is_anyone_when_the_question_names_the_thing(self):
        """`can you cut bread with a knife` was asked of a computer program."""
        for said, asked in (
                ("can you cut bread with a knife",
                 "is a knife used for cutting bread"),
                ("can you drink from a cup", "is a cup used for drinking"),
                ("can you sit on a chair", "is a chair used for sitting"),
                ("can you write with a pencil", "is a pencil used for writing"),
                ("can you eat an apple", "can an apple be eaten"),
                ("can you eat rocks", "can rocks be eaten"),
                ("what do you use to cut paper", "what is used to cut paper"),
                ("what do people use for writing", "what is used for writing")):
            found = rephrase(said)
            self.assertEqual(found.text, asked, said)
            self.assertIn("anyone", found.note, said)
        # Nothing named, or no thing for the doing: still the program.
        for said in ("can you swim", "can you see me", "can you eat",
                     "can you eat it", "can you see in the dark",
                     "can you help me with my homework"):
            self.assertEqual(self.said(said), said)

    def test_being_in_a_place_is_where_it_is_found(self):
        self.assertEqual(self.said("is a fridge in a kitchen"),
                         "is a fridge found in a kitchen")
        for said in ("is a dog in danger", "is it in a box",
                     "is a fridge found in a kitchen"):
            self.assertEqual(self.said(said), said)

    def test_purpose_existence_and_how(self):
        self.assertEqual(self.said("what is the purpose of a hammer"),
                         "what is a hammer used for")
        self.assertEqual(self.said("what is a hammer good for"),
                         "what is a hammer used for")
        self.assertEqual(self.said("is there such a thing as a flying fish"),
                         "what is a flying fish")
        found = rephrase("how does a bird fly")
        self.assertEqual((found.text, found.why), ("does a bird fly", True))
        self.assertFalse(rephrase("how do you make bread").why)

    def test_are_there_asks_which_ones(self):
        found = rephrase("are there any birds that cannot fly")
        self.assertEqual(found.text, "which birds cannot fly")
        self.assertTrue(found.note)
        self.assertEqual(self.said("are there two dogs"), "are there two dogs")

    def test_analogies_and_conditionals_are_left_whole(self):
        for text in ("fins are to fish as what are to birds",
                     "if a dog had wings could it fly"):
            found = rephrase(text)
            self.assertEqual((found.text, found.asking), (text, True))

    def test_why_about_a_kind_asks_the_yes_or_no_inside_it(self):
        found = rephrase("why can a bird fly")
        self.assertEqual((found.text, found.why, found.negative),
                         ("can a bird fly", True, False))
        found = rephrase("why can't a penguin fly")
        self.assertEqual((found.text, found.why, found.negative),
                         ("can a penguin fly", True, True))
        found = rephrase("why do birds fly?")
        self.assertEqual((found.text, found.why), ("do birds fly", True))
        found = rephrase("how come a penguin can't fly")
        self.assertEqual((found.text, found.why, found.negative),
                         ("can a penguin fly", True, True))

    def test_why_about_one_thing_or_none_is_left(self):
        for text in ("why can't it fly", "why does the dog bark",
                     "why does it rain", "why"):
            found = rephrase(text)
            self.assertEqual((found.text, found.why), (text, False), text)

    def test_a_plain_question_is_untouched(self):
        found = rephrase("can a dog swim?")
        self.assertEqual((found.text, found.note, found.asking),
                         ("can a dog swim", "", False))


class SeedTests(unittest.TestCase):

    def test_modal_questions_are_asked_as_they_stand(self):
        for text in ("should a dog eat chocolate", "must a bird have wings",
                     "might a dog bite"):
            self.assertEqual(seed_questions(text), [text])

    def test_a_long_unreadable_utterance_is_not_wrapped_in_what_is(self):
        self.assertEqual(seed_questions("cna a dog swim"), ["cna a dog swim"])
        self.assertEqual(seed_questions("wemble"), ["what is a wemble"])


class ConnectiveTests(unittest.TestCase):

    def test_but_joins_two_claims(self):
        query = logic.parse("fur but not feathers", frozenset())
        self.assertEqual((query.tree.op, len(query.tree.children)),
                         ("and", 2))
        self.assertEqual(query.tree.children[1].op, "not")
        self.assertNotIn("but", query.tree.terms())


class RefusedByName(unittest.TestCase):

    def refused(self, text, word):
        reason = logic.unsupported(text)
        self.assertIsNotNone(reason, text)
        self.assertIn(word, reason)

    def test_advice_and_obligation(self):
        self.refused("should a dog eat chocolate", "ought")
        self.refused("is it safe to eat a mushroom", "ought")
        self.refused("must a bird have wings", "ought")

    def test_conditionals(self):
        self.refused("could a pig fly if it had wings", "counterfactual")
        self.refused("if a dog had wings could it fly", "counterfactual")
        self.refused("what would happen if the sun disappeared",
                     "counterfactual")

    def test_words_arithmetic_and_a_comparative_choice(self):
        self.refused("what is another word for big", "word")
        self.refused("what is the plural of mouse", "word")
        self.refused("what is two plus two", "arithmetic")
        self.refused("is seven a prime number", "arithmetic")
        self.refused("which is heavier, a feather or a brick", "comparative")

    def test_durations_methods_and_prices(self):
        self.refused("how long do dogs live", "duration")
        self.refused("how often do cats sleep", "duration")
        self.refused("how do you make bread", "method")
        self.refused("how much does a car cost", "price")
        for text in ("how much does an elephant weigh", "how long is a snake",
                     "how big is an elephant", "how fast can a cheetah run"):
            self.assertIsNone(logic.unsupported(text), text)

    def test_what_is_answerable_is_not_refused(self):
        for text in ("can a dog swim", "what is a hammer made of",
                     "how many kinds of dog are there",
                     "is a penguin a typical bird", "does a dog have a tail",
                     "do you know if a dog can swim", "might a dog bite"):
            self.assertIsNone(logic.unsupported(text), text)


if __name__ == "__main__":
    unittest.main()
