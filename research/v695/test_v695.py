"""v695: common-sense actions -- how a goal is reached, and who can.

Needs the store, VerbNet and the mined actions (`python -m regenerate
--only actions`), and skips itself without them.
"""
from __future__ import annotations

import unittest

from research.v687 import build
from research.v687.corpora import VERBNET_DIR
from research.v695 import mined

HAVE = (VERBNET_DIR.exists() and build.DEFAULT_STORE.exists()
        and mined.PATH.exists())
needs_data = unittest.skipUnless(HAVE, "VerbNet, the store or the mined "
                                       "actions missing")


class NormTests(unittest.TestCase):

    def test_phrases_meet_whatever_their_inflection(self):
        self.assertEqual(mined.norm("jumping over the fences"),
                         "jump over fence")


@needs_data
class MinedTests(unittest.TestCase):

    def test_a_doing_is_kept_as_said(self):
        heads = [head for head, _, _ in
                 mined.edges("MotivatedByGoal", tail="get over a fence")]
        self.assertIn("climb", heads)


@needs_data
class CanTests(unittest.TestCase):

    def can(self, kind, verb):
        from research.v695.can import can
        return can(kind, verb).verdict

    def test_what_its_kind_is_said_to_do(self):
        self.assertEqual(self.can("person", "jump"), "yes")
        self.assertEqual(self.can("hen", "fly"), "yes")

    def test_verbnet_rules_out_what_cannot_be_the_doer(self):
        for kind in ("computer program", "rock", "car"):
            self.assertEqual(self.can(kind, "jump"), "no", kind)
        self.assertEqual(self.can("computer program", "eat"), "no")

    def test_flying_something_is_not_flying(self):
        # People fly helicopters and fly with machines: nothing unaided.
        self.assertNotEqual(self.can("person", "fly"), "yes")

    def test_what_some_kinds_above_can_do_is_not_inherited(self):
        # A carnivore can climb trees: cats can, and dogs are not cats.
        self.assertNotEqual(self.can("dog", "climb"), "yes")
        self.assertNotEqual(self.can("fish", "climb"), "yes")

    def test_the_shape_of_doing_it_oneself(self):
        from research.v695.can import unaided
        self.assertTrue(unaided("jump", "jump over puddle"))
        self.assertTrue(unaided("climb", "climb tree"))
        self.assertTrue(unaided("swim", "swim in water"))
        self.assertFalse(unaided("fly", "fly with machines"))
        self.assertFalse(unaided("fall", "fall in love"))

    def test_the_designer_asks_about_its_own_doer(self):
        from research.v694 import knowing as K
        self.assertGreater(K.person_can("open", "door"), 0)
        self.assertEqual(K.person_can("open", "door", "computer program"),
                         0)


@needs_data
class AchievingTests(unittest.TestCase):

    def answer(self, text):
        from research.v695 import achieving
        return achieving.answer(achieving.read(text))

    def test_a_goal_is_a_doing_that_goes_somewhere(self):
        from research.v695 import achieving
        goal = achieving.read("how would I get over a fence?")
        self.assertEqual((goal.verb, goal.place, goal.thing, goal.subject),
                         ("get", "over", "fence", "person"))
        goal = achieving.read("how would a cat get down from a tree")
        self.assertEqual((goal.place, goal.subject), ("down from", "cat"))
        self.assertIsNone(achieving.read("how do i open a jar"))

    def test_over_a_fence(self):
        said = self.answer("how would I get over a fence?")
        self.assertTrue(said.startswith("You could"), said)
        self.assertIn("jump", said)
        self.assertIn("climb", said)

    def test_only_what_the_subject_can_do(self):
        self.assertIn("fly", self.answer("how would a hen get over a fence"))
        self.assertNotIn("fly", self.answer("how would I get over a "
                                            "fence?"))
        self.assertNotIn("climb", self.answer("how can a dog get over a "
                                              "fence"))


@needs_data
class PageTests(unittest.TestCase):

    def test_whether_you_or_i_can_by_what_we_are(self):
        from research.v695 import page
        heard = type("Heard", (), {"said": "could you jump over a fence"})
        self.assertTrue(page.able(None, heard).startswith("No, I can't"))
        heard.said = "can i jump"
        self.assertTrue(page.able(None, heard).startswith("Yes, you can"))
        self.assertIsNone(page.asked_able("can a dog swim"))
        self.assertIsNone(page.asked_able("can you tell me the time"))


@needs_data
class PiqaTests(unittest.TestCase):

    def test_words_and_doings(self):
        from research.v695 import piqa
        self.assertIn("paper", piqa.words("bedding made of ripped paper "
                                          "strips"))
        self.assertIn(("crush", "petal"),
                      piqa.doings("To crush the petals of a flower"))

    def test_the_same_ways_abstain(self):
        from research.v695 import piqa
        item = piqa.Item("to open a door", "turn the knob",
                         "turn the knob", 0)
        self.assertIsNone(piqa.score(item)[0])


if __name__ == "__main__":
    unittest.main()
