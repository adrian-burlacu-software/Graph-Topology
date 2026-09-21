"""What the agent works out about acting, and keeps.

The store is in memory here, so a test never writes to what a running
server is using.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research.v687 import build
from research.v687.corpora import VERBNET_DIR
from research.v691 import learned as L, openworld, page, world as W
from research.v691.scene import Scene

needs_verbnet = unittest.skipUnless(VERBNET_DIR.exists(), "no VerbNet")
needs_store = unittest.skipUnless(build.DEFAULT_STORE.exists(), "no store")


class StoreTests(unittest.TestCase):
    """Three things, remembered and taken back."""

    def setUp(self):
        self.learned = L.Learned(None)
        self.addCleanup(self.learned.close)

    def test_exclusion_is_symmetric(self):
        """Kept both ways round, because a lookup that had to try both
        orders would eventually forget to."""
        self.assertTrue(self.learned.exclude("open", "closed", "said"))
        self.assertIn("closed", self.learned.excluded("open"))
        self.assertIn("open", self.learned.excluded("closed"))

    def test_the_same_thing_twice_is_not_news(self):
        """A compound utterance is acted on once per claim, and a reply
        that reported it twice would be telling the person what they just
        said."""
        self.assertTrue(self.learned.exclude("open", "closed"))
        self.assertFalse(self.learned.exclude("open", "closed"))
        self.assertFalse(self.learned.exclude("closed", "open"))

    def test_a_requirement_is_kept_per_verb(self):
        self.assertTrue(self.learned.require("drop", "with ?object ?subject"))
        self.assertEqual(self.learned.required("drop"),
                         ["with ?object ?subject"])
        self.assertEqual(self.learned.required("open"), [])

    def test_it_can_be_taken_back(self):
        """Being corrected is how most of this is learned, and a correction
        can be wrong in its turn."""
        self.learned.exclude("open", "closed")
        self.assertEqual(self.learned.forget("excludes", "closed", "open"), 2)
        self.assertEqual(self.learned.excluded("open"), frozenset())

    def test_it_survives_the_conversation_it_was_said_in(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "learned.sqlite"
            one = L.Learned(path)
            one.exclude("full", "empty", "you told me")
            one.close()
            two = L.Learned(path)
            try:
                self.assertIn("empty", two.excluded("full"))
                self.assertEqual(two.exclusions(),
                                 [("empty", "full", "you told me")])
            finally:
                # Before the directory goes: Windows will not remove a file
                # sqlite still has open.
                two.close()


class TeachingTests(unittest.TestCase):
    """What a sentence teaches about acting."""

    def test_the_shapes_that_teach_an_exclusion(self):
        for said in ("a door cannot be open and closed",
                     "open and closed are opposites",
                     "nothing can be open and closed"):
            with self.subTest(said=said):
                self.assertEqual(L.teaching(said),
                                 [("exclude", "open", "closed")])

    def test_the_shapes_that_teach_a_requirement(self):
        for said, verb, literal in (
                ("you can only drop it if you are holding it",
                 "drop", "with ?object ?subject"),
                ("you must be holding it to put it down",
                 "put", "with ?object ?subject"),
                ("it must be closed before you can open it",
                 "open", "closed ?object")):
            with self.subTest(said=said):
                self.assertEqual(L.teaching(said), [("require", verb,
                                                     literal)])

    def test_an_ordinary_sentence_teaches_nothing(self):
        for said in ("the book is in the shop", "a whale is a mammal",
                     "can a penguin fly", "get the book to the kitchen",
                     "what is a whale"):
            with self.subTest(said=said):
                self.assertEqual(L.teaching(said), [])


class ApplyingTests(unittest.TestCase):
    """What is learned, folded into the actions."""

    def setUp(self):
        self.learned = L.Learned(None)
        self.addCleanup(self.learned.close)

    def action(self, name, needs=(), adds=(), deletes=()):
        return W.Action(name, frozenset(needs), frozenset(adds),
                        frozenset(deletes))

    def test_an_exclusion_becomes_a_precondition_that_can_be_said(self):
        """**This is the point of learning exclusion first.** `needs` is a
        list of slots that must be present, so *the door is not already
        open* cannot be said -- but once `open` and `closed` are known to
        exclude each other, *the door is closed* says it and is positive.
        """
        self.learned.exclude("open", "closed")
        one = L.applied([self.action("open door", adds=["open door"])],
                        self.learned)[0]
        self.assertIn("closed door", one.needs)
        self.assertIn("closed door", one.deletes)

    def test_a_requirement_is_ground_against_the_action(self):
        self.learned.require("drop", "with ?object ?subject")
        one = L.applied([self.action("drop john book")], self.learned)[0]
        self.assertIn("with book john", one.needs)

    def test_a_requirement_leaves_an_action_without_the_position_alone(self):
        """`you must be holding it` says nothing about a verb with no
        object."""
        self.learned.require("go", "with ?object ?subject")
        one = L.applied([self.action("go john")], self.learned)[0]
        self.assertEqual(one.needs, frozenset())

    def test_nothing_learned_changes_nothing(self):
        before = self.action("open door", adds=["open door"])
        self.assertEqual(L.applied([before], self.learned)[0], before)
        self.assertEqual(L.applied([before], None)[0], before)


@needs_verbnet
class GuardTests(unittest.TestCase):
    """What the open world may take, now that it is the default."""

    def scene(self) -> Scene:
        scene = Scene(openworld.Open(None, L.Learned(None)))
        self.addCleanup(scene.domain.learned.close)
        return scene

    def test_a_conversation_starts_in_the_open_world(self):
        class Fake:
            conversation = "guard-test"

        page.SCENES.pop("guard-test", None)
        scene = page.scene_for(Fake())
        self.assertTrue(scene.open)
        self.assertEqual(scene.domain.name, "open")

    def test_a_statement_about_a_kind_is_not_a_scene(self):
        """`a whale is a mammal` reads as `mammal whale` and is v688's
        question. `mammal` is not an adjective and no whale is being
        talked about, so this layer leaves it alone."""
        scene = self.scene()
        self.assertFalse(page.scene_ish(["mammal whale"], scene))
        self.assertEqual(page.hear("a whale is a mammal", scene).act, "")

    def test_a_state_and_a_place_are_a_scene(self):
        scene = self.scene()
        self.assertTrue(page.scene_ish(["closed door"], scene))
        self.assertTrue(page.scene_ish(["at book shop"], scene))

    def test_a_declared_domain_needs_no_guard(self):
        """Its reader only ever matched its own `say` lines."""
        from research.v691.domains import DOMAINS
        blocks = Scene(DOMAINS["blocks"])
        self.assertTrue(page.scene_ish(["on red green"], blocks))


@needs_verbnet
@needs_store
class LearningInAConversationTests(unittest.TestCase):
    """Being corrected, and being told, while it is running."""

    def scene(self) -> Scene:
        scene = Scene(openworld.Open(openworld.resolver(), L.Learned(None)))
        self.addCleanup(scene.domain.learned.close)
        return scene

    def test_a_correction_teaches_which_states_exclude(self):
        scene = self.scene()
        page.say_to(scene, "the door is closed")
        said = page.say_to(scene, "actually the door is open now")
        self.assertIn("cannot be open and closed", said)
        self.assertIn("closed", scene.domain.learned.excluded("open"))
        self.assertNotIn("closed door", scene.world.facts)

    def test_what_it_learned_applies_to_something_else(self):
        """Learned of a door and used on a window, which is what makes it
        knowledge rather than a note about one thing."""
        scene = self.scene()
        page.say_to(scene, "the door is closed")
        page.say_to(scene, "actually the door is open now")
        page.say_to(scene, "the window is closed")
        page.say_to(scene, "open the window")
        self.assertIn("open window", scene.world.facts)
        self.assertNotIn("closed window", scene.world.facts)

    def test_being_told_is_remembered_and_said_back(self):
        scene = self.scene()
        said = page.say_to(scene, "a door cannot be open and closed")
        self.assertIn("I will remember", said)
        self.assertIn("closed", scene.domain.learned.excluded("open"))
        self.assertEqual(page.say_to(scene, "a door cannot be open and "
                                            "closed"),
                         "I already knew that")

    def test_a_taught_requirement_reaches_the_planner(self):
        scene = self.scene()
        page.say_to(scene, "you can only drop it if you are holding it")
        self.assertEqual(scene.domain.learned.required("drop"),
                         ["with ?object ?subject"])
        page.say_to(scene, "john is in the kitchen and the book is in "
                           "the kitchen")
        scene.domain.toward({"at book kitchen"})
        for one in scene.domain.ground({}):
            if one.name.startswith("drop ") and len(one.name.split()) > 2:
                with self.subTest(action=one.name):
                    self.assertTrue([need for need in one.needs
                                     if need.startswith("with ")])


if __name__ == "__main__":
    unittest.main()
