"""The rated norms (THINGSplus, NEWTON), R31 and R32, and the common-sense
questions they were built for."""
from __future__ import annotations

import unittest

from research.v687 import build, corpora

STORE = build.DEFAULT_STORE
HAVE_RATINGS = (STORE.exists()
                and (corpora.THINGSPLUS_DIR / "property-ratings.tsv").exists()
                and (corpora.NEWTON_DIR / "confident_questions.csv").exists())
requires_ratings = unittest.skipUnless(
    HAVE_RATINGS, "no store, or THINGSplus and NEWTON are not downloaded")


@unittest.skipUnless(corpora.VERBNET_DIR.exists(), "VerbNet is not downloaded")
class VerbNetTests(unittest.TestCase):
    """Which verbs only something alive can be the subject of."""

    @classmethod
    def setUpClass(cls):
        cls.verbs = corpora.load_animate_only_verbs()

    def test_breathing_eating_and_thinking_want_a_living_doer(self):
        for verb in ("breathe", "eat", "drink", "think"):
            self.assertIn(verb, self.verbs, verb)

    def test_one_class_that_lets_a_thing_do_it_is_enough_to_leave_it(self):
        """run-51.3.2 lets a machine run, and a hotel sleeps a hundred."""
        for verb in ("run", "fly", "move", "swim", "sleep", "grow"):
            self.assertNotIn(verb, self.verbs, verb)


@requires_ratings
class RatingsTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from research.v687.rated import Ratings
        from research.v687.reason import Reasoner

        cls.reasoner = Reasoner(STORE)
        cls.ratings = Ratings(cls.reasoner)

    @classmethod
    def tearDownClass(cls):
        cls.reasoner.close()

    def senses(self, word):
        return [sense["id"] for sense in self.reasoner.senses_of(word)
                if sense.get("pos") == "n"]

    def rated(self, word, term):
        senses = self.senses(word)
        return self.ratings.settle(senses[0], senses, term)

    def test_the_join_reaches_nearly_every_object(self):
        report = self.ratings.report
        self.assertGreaterEqual(report["thingsplus: joined"], 1700)
        self.assertGreaterEqual(report["newton: joined by LVIS gloss"]
                                + report["newton: joined by its physical sense"],
                                700)

    def test_a_low_rating_is_a_no_and_a_high_one_a_yes(self):
        for word, term, verdict in (
                ("chair", "alive", "DENIED"), ("tree", "alive", "HELD"),
                ("ant", "big", "DENIED"), ("ant", "tiny", "HELD"),
                ("elephant", "big", "HELD"), ("feather", "heavy", "DENIED"),
                ("car", "man-made", "HELD"), ("pillow", "soft", "HELD"),
                ("pillow", "sharp", "DENIED"), ("knife", "sharp", "HELD"),
                ("cup", "fragile", "HELD")):
            found = self.rated(word, term)
            self.assertIsNotNone(found, (word, term))
            self.assertEqual(found.verdict, verdict, (word, term, found))

    def test_readings_that_disagree_say_nothing(self):
        """THINGS rates the animal mouse alive and the device not, and the
        store's first mouse is the device."""
        self.assertIsNone(self.rated("mouse", "alive"))

    def test_a_rating_is_borrowed_within_a_word_but_not_across_alive(self):
        found = self.rated("rock", "alive")
        self.assertEqual((found.concept, found.verdict), ("rock.n.01", "DENIED"))
        # The store's first turkey is the meat, which nothing rated, and the
        # bird THINGSplus rated alive is on the other side of that line.
        self.assertIn("is alive", self.ratings.held.get("turkey.n.01", {}))
        self.assertFalse(self.ratings.living("turkey.n.02"))
        self.assertIsNone(self.rated("turkey", "alive"))

    def test_r32_joins_verbnet_to_the_rating(self):
        senses = self.senses("rock")
        found = self.ratings.unable(senses[0], senses, "breathe")
        self.assertIsNotNone(found)
        self.assertEqual(found.verdict, "DENIED")
        self.assertIn("breathe-40.1.2", found.detail)

    def test_r32_stands_down_where_the_store_says_it_does(self):
        for word, target in (("computer", "talk"), ("robot", "think"),
                             ("pillow", "breathe")):
            senses = self.senses(word)
            self.assertIsNone(self.ratings.unable(senses[0], senses, target),
                              (word, target))

    def test_r32_leaves_anything_that_might_be_alive_alone(self):
        for word in ("dog", "tree", "mouse", "bat"):
            senses = self.senses(word)
            self.assertIsNone(self.ratings.unable(senses[0], senses,
                                                  "breathe"), word)


@requires_ratings
class EngineTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from research.v687.reasoning import ReasoningEngine

        cls.engine = ReasoningEngine(STORE)

    def test_a_no_the_crawl_could_not_give(self):
        for question in ("is a chair alive", "does a rock breathe",
                         "can a table think", "is a pillow sharp"):
            self.assertEqual(self.engine.ask(question)["verdict"],
                             "CONTRADICTED", question)
        self.assertEqual(self.engine.ask("does a rock breathe")["parse"]
                         ["relation"], "R32")

    def test_r32_answers_the_verb_alone(self):
        """An object can choose a sense VerbNet does not list -- a cup holds
        water by containing it -- and the norms' conjunction splits a verb out
        of a question about something else: `can dogs eat chocolate` is routed
        to chocolate."""
        for question in ("can a cup hold water", "can dogs eat chocolate"):
            self.assertNotEqual(self.engine.ask(question)["verdict"],
                                "CONTRADICTED", question)

    def test_a_rated_answer_names_the_word_it_was_asked_about(self):
        """v688 ranks a reading among the senses of the question's word, and
        given `rock.n.01` for the word it headlined a sense mismatch."""
        answer = self.engine.ask("does a rock breathe")
        self.assertEqual(answer["parse"]["subject"], "rock")
        self.assertTrue(answer["senses"])

    def test_r31_compares_size_and_weight(self):
        for question, verdict in (
                ("is an elephant bigger than a mouse", "VERIFIED"),
                ("is a cat bigger than a horse", "CONTRADICTED"),
                ("is a mouse heavier than an elephant", "CONTRADICTED"),
                ("is a car heavier than a bicycle", "VERIFIED")):
            answer = self.engine.ask(question)
            self.assertEqual((answer["verdict"], answer["parse"]["relation"]),
                             (verdict, "R31"), question)

    def test_r31_answers_a_choice_and_the_page_can_say_which(self):
        from research.v688 import content

        answer = self.engine.ask("which is heavier, a feather or a brick")
        self.assertEqual(answer["verdict"], "LISTING")
        self.assertEqual(answer["comparison"]["winner"], "brick")
        said = content.digest(answer)
        self.assertEqual(said["kind"], "comparison")
        self.assertTrue(said["text"].startswith("brick is heavier"), said)

    def test_r18_still_refuses_a_comparative_with_no_scale(self):
        answer = self.engine.ask("is a cheetah faster than a turtle")
        self.assertEqual((answer["verdict"], answer["parse"]["relation"]),
                         ("UNSUPPORTED", "R18"))

    def test_a_folk_category_answers_only_after_the_taxonomy(self):
        answer = self.engine.ask("is a tomato a fruit")
        self.assertEqual(answer["verdict"], "VERIFIED")
        self.assertEqual(answer["folk"]["category"], "fruit")
        self.assertEqual(answer["folk"]["taxonomy"], "UNKNOWN")
        # The taxonomy answers what it can, and a category never overrules it.
        self.assertNotIn("folk", self.engine.ask("is a dog an animal"))

    def test_being_found_somewhere_is_a_location(self):
        """`found in a kitchen` was a property nothing has; the rows say
        `electric refrigerator at_location kitchen`."""
        answer = self.engine.ask("is a fridge found in a kitchen")
        self.assertEqual((answer["verdict"], answer["parse"]["relation"],
                          answer["parse"]["target"]),
                         ("VERIFIED", "at_location", "kitchen"))
        # One crawled row about fish in trees is still not a fish in a tree.
        self.assertNotEqual(self.engine.ask("is a fish found in a tree")
                            ["verdict"], "VERIFIED")

    def test_the_norms_denominators_are_untouched(self):
        """Merging 2,000 rated objects into `stated` would have put
        THINGSplus's birds in R19's and the quantifier's denominators."""
        note = self.engine.ask("do all birds fly")["note"]
        self.assertIn("of 27 recorded kinds of bird", note)
        for kind in ("chicken", "emu", "penguin"):
            self.assertIn(kind, note)


if __name__ == "__main__":
    unittest.main()
