"""Choosing the verb: who acts, and the verbs people use (roadmap item 5)."""
from __future__ import annotations

import unittest

from research.v687 import build
from research.v687.corpora import VERBNET_DIR
from research.v691 import (errands, hearing, learned as L, openworld, page,
                           verbs)
from research.v691.scene import Scene

needs_verbnet = unittest.skipUnless(VERBNET_DIR.exists(), "no VerbNet")
needs_store = unittest.skipUnless(build.DEFAULT_STORE.exists(), "no store")
needs_parser = unittest.skipUnless(hearing.nlp() is not None, "no spaCy")


class BodyTests(unittest.TestCase):
    """Who acts is a question about the thing, by its most common sense."""

    def test_the_living_act_and_the_rest_do_not(self):
        things = verbs.Things()
        for name, kind, acts in (("john", "man", True), ("rex", "dog", True),
                                 ("book", "book", False),
                                 ("cup", "cup", False),
                                 ("plane", "plane", False)):
            with self.subTest(name=name):
                things.add(name, kind)
                self.assertIs(things.acts(name), acts)

    def test_a_name_is_someone(self):
        """WordNet makes `john` a toilet; a name said as one is a person."""
        things = verbs.Things()
        things.add("john", "john")
        self.assertIs(things.acts("john"), False)
        things.agents.add("john")
        self.assertIs(things.acts("john"), True)

    @needs_verbnet
    @needs_store
    def test_a_book_does_not_go_by_itself(self):
        found = errands.work(errands.Errand(**{
            **errands.ERRANDS[0].__dict__}), openworld.resolver())
        self.assertTrue(found.reached)
        self.assertTrue(all(one.split()[1] == "john" for one in found.plan),
                        found.plan)


class StoreTests(unittest.TestCase):

    def test_preferences_count_and_order(self):
        memory = L.Learned(None)
        self.addCleanup(memory.close)
        memory.prefer("at", "go", "sam went to the park")
        memory.prefer("at", "carry", "no, carry it", 2.0)
        memory.prefer("at", "go", "the girl went home")
        memory.prefer("at", "take", "mary took the cup")
        self.assertEqual(memory.preferred("at"), ["go", "carry", "take"])
        self.assertEqual(memory.forget("prefers", "at", "take"), 1)
        self.assertEqual(memory.preferred("at"), ["go", "carry"])


@needs_parser
@needs_verbnet
class SeenTests(unittest.TestCase):

    def test_what_people_say_they_did(self):
        for said, done in (("sam went to the park", [("at", "go")]),
                           ("he put the plate on the shelf", [("at", "put")]),
                           ("the dog barked", []),
                           ("carry the box into the garden", [])):
            with self.subTest(said=said):
                self.assertEqual(hearing.hear(said, verbs.stated).done, done)

    @needs_store
    def test_seen_verbs_make_errands_sensible(self):
        """Five sentences about other people, other things and other
        places: only the verbs carry over."""
        before = sum(one.sensible for one in errands.measure(
            openworld.resolver()))
        after = sum(one.sensible for one in errands.measure(
            openworld.resolver(), prefer=errands.preferences()))
        self.assertGreaterEqual(before, 10)
        self.assertGreater(after, before)


@needs_parser
@needs_verbnet
@needs_store
class CorrectionTests(unittest.TestCase):

    def scene(self) -> Scene:
        scene = Scene(openworld.Open(openworld.resolver(), L.Learned(None)))
        self.addCleanup(scene.domain.learned.close)
        return scene

    def test_a_correction_is_how_it_is_done_next_time(self):
        scene = self.scene()
        page.say_to(scene, "john is in the kitchen")
        page.say_to(scene, "the box is in the kitchen")
        page.say_to(scene, "get the box to the garden")
        self.assertEqual(page.hear("no, carry it", scene).act, "prefer")
        self.assertIn("next time I will carry it",
                      page.say_to(scene, "no, carry it"))
        page.say_to(scene, "the cup is in the kitchen")
        page.say_to(scene, "get the cup to the garden")
        self.assertEqual(scene.last_plan[0].name.split()[0], "carry")

    def test_the_verb_said_is_tried_first(self):
        scene = self.scene()
        page.say_to(scene, "john is in the kitchen")
        page.say_to(scene, "the box is in the kitchen")
        page.say_to(scene, "carry the box into the garden")
        self.assertEqual(scene.last_plan[0].name.split()[0], "carry")

    def test_a_statement_of_what_was_done_is_kept(self):
        scene = self.scene()
        self.assertEqual(page.seen_done(scene, "sam went to the park"),
                         [("at", "go")])
        self.assertEqual(scene.domain.learned.preferred("at"), ["go"])


if __name__ == "__main__":
    unittest.main()
