"""What an utterance says about a world, read off its parse (`hearing.py`)."""
from __future__ import annotations

import unittest

from research.v687 import build
from research.v687.corpora import VERBNET_DIR
from research.v691 import hearing, learned as L, openworld, page, verbs
from research.v691.scene import Scene

needs_parser = unittest.skipUnless(hearing.nlp() is not None, "no spaCy")
needs_verbnet = unittest.skipUnless(VERBNET_DIR.exists(), "no VerbNet")
needs_store = unittest.skipUnless(build.DEFAULT_STORE.exists(), "no store")


def heard(text: str, it: str = "") -> hearing.Heard:
    return hearing.hear(text, verbs.stated, it=it)


@needs_parser
@needs_verbnet
class StatementTests(unittest.TestCase):

    def test_the_shapes_a_fact_is_stated_in(self):
        for said, facts in (
                ("the book is in the shop", ["at book shop"]),
                ("john has the book", ["with book john"]),
                ("there is a red block on the table", ["at block table"]),
                ("the door is locked", ["locked door"]),
                ("the keys are on the table and the cup is in the sink",
                 ["at keys table", "at cup sink"])):
            with self.subTest(said=said):
                self.assertEqual(heard(said).facts, facts)

    def test_still_is_not_a_state(self):
        """The regular expressions read `the door is still closed` as a
        door in the state *still*."""
        found = heard("the door is still closed, it is locked", it="door")
        self.assertEqual(found.facts, ["closed door", "locked door"])
        self.assertEqual(found.still, ["closed door"])

    def test_a_denial_is_not_a_fact(self):
        self.assertEqual(heard("the door did not open").denied,
                         ["open door"])
        self.assertEqual(heard("the door is not open").denied,
                         ["open door"])
        self.assertEqual(heard("the door is not open").facts, [])

    def test_a_question_states_nothing(self):
        for said in ("what steps are required to make a pig fly",
                     "does a table have legs", "is the door open?",
                     "is the book in the shop"):
            with self.subTest(said=said):
                self.assertEqual(heard(said).facts, [])


@needs_parser
@needs_verbnet
class OrderTests(unittest.TestCase):

    def test_where_an_order_puts_a_thing_is_the_sentence_s(self):
        """No list of the verbs that move things: any verb with its object
        and a place asks for the object there."""
        for said, wants in (
                ("get the book to the kitchen", ["at book kitchen"]),
                ("carry the box into the garden", ["at box garden"]),
                ("please take the book to the office", ["at book office"]),
                ("could you put the cup in the sink", ["at cup sink"])):
            with self.subTest(said=said):
                found = heard(said)
                self.assertEqual(found.wants, wants)
                self.assertTrue(found.order)

    def test_a_verb_that_hands_a_thing_over_asks_that_they_have_it(self):
        """give-13.1's Theme leaves the Agent: VerbNet, not a list."""
        self.assertEqual(heard("give the cup to mary").wants,
                         ["with cup mary"])

    def test_a_doing_asked_about(self):
        found = heard("how would a pig fly?")
        self.assertEqual(found.wants, ["fly pig"])
        self.assertIn("fly", found.doings)
        self.assertTrue(found.asked)
        self.assertFalse(found.order)

    def test_a_question_of_whether_asks_for_no_plan(self):
        for said in ("does a table have legs", "can a penguin fly",
                     "i put the key in the drawer"):
            with self.subTest(said=said):
                found = heard(said)
                self.assertFalse(found.asked or found.order)
                self.assertEqual(found.wants, [])


@needs_parser
@needs_verbnet
@needs_store
class OnThePageTests(unittest.TestCase):

    def scene(self) -> Scene:
        scene = Scene(openworld.Open(openworld.resolver(), L.Learned(None)))
        self.addCleanup(scene.domain.learned.close)
        return scene

    def test_how_would_a_pig_fly_is_read(self):
        scene = self.scene()
        self.assertEqual(page.hear("how would a pig fly?", scene).act, "how")

    def test_a_verb_off_the_old_list_is_an_order(self):
        scene = self.scene()
        page.say_to(scene, "the box is in the kitchen")
        self.assertEqual(page.hear("carry the box into the garden",
                                   scene).act, "want")


@needs_parser
@needs_verbnet
@needs_store
class OneConversationTests(unittest.TestCase):
    """Two questions in one conversation: what the first left in the scene
    must not do the second's work."""

    def test_the_piano_after_the_pig(self):
        memory = L.Learned(None)
        self.addCleanup(memory.close)
        memory.carry("fly", "hog.n.03", "airplane.n.01", "plane", "put on",
                     False, "i put my pig on a plane")
        scene = Scene(openworld.Open(openworld.resolver(), memory))
        scene.domain.can = lambda kind, verb: kind not in ("pig", "piano",
                                                           "person")
        scene.domain.fits = lambda kind, carrier: True
        page.say_to(scene, "how would a pig fly?")
        said = page.say_to(scene, "how would a piano fly?")
        # The one asking loads it, not the pig named a turn ago; and on the
        # plane that was seen, not a jet an earlier answer named.
        self.assertEqual([one.name for one in scene.last_plan],
                         ["put you piano plane", "fly piano plane"])
        self.assertIn("Put the piano on the plane", said)


if __name__ == "__main__":
    unittest.main()
