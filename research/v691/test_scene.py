"""Talking to the agent: the reader, the scene, and the acts on the page.

Fast for the same reason `test_v691` is -- no store and no spaCy. What is
read here is the domains' own `say` and `reads` lines, and the point of
keeping the reader that thin is that a failure in this suite is the
planner's or the domain's and not a grammar's.
"""
from __future__ import annotations

import unittest

from research.v691 import acting, page, talking
from research.v691.domains import DOMAINS
from research.v691.scene import Reader, Scene


def consistent(scene: Scene) -> list:
    """Every way a scene's facts can be nonsense, as complaints.

    General: it reads the domain for what a placement is (`goalish`) and
    what a `clear`-like fact is (`taken`), so it checks errands and
    delivery as readily as blocks -- which is how it found that asserting
    a fact by hand left a block in two places at once.
    """
    domain, facts = scene.domain, set(scene.world.facts)
    wrong = []
    for name in scene.objects:
        places = [one for one in facts if one.split()[0] in domain.goalish
                  and len(one.split()) > 1 and one.split()[1] == name]
        if len(places) > 1:
            wrong.append(f"{name} is in {len(places)} places: {places}")
    for predicate, where in domain.taken.items():
        claimed = {one.split()[where] for one in facts
                   if one.split()[0] in domain.goalish
                   and len(one.split()) > where}
        for name in scene.objects:
            has = f"{predicate} {name}" in facts
            if has == (name in claimed):
                wrong.append(f"`{predicate} {name}` is {has} "
                             f"with claimed={name in claimed}")
    return wrong


class ReadingTests(unittest.TestCase):
    """The `say` lines, backwards."""

    def setUp(self):
        self.blocks = Reader(DOMAINS["blocks"])
        self.errands = Reader(DOMAINS["errands"])

    def test_one_template_reads_a_statement_and_an_order(self):
        for said in ("there is a red block on a green block",
                     "the red block is on the green block",
                     "put the red block on the green block",
                     "move red onto green"):
            with self.subTest(said=said):
                self.assertIn("on red green", self.blocks.facts_in(said))

    def test_the_most_specific_reading_wins_its_span(self):
        """`on the table` has three literal words and `{0} on {1}` has one,
        so the table is not read as a block called `table`."""
        self.assertEqual(
            self.blocks.facts_in("put the red block on the table"),
            ["table red"])

    def test_a_copula_is_english_and_not_a_domain_fact(self):
        """`the keys **are** at the office` needs no line of its own."""
        self.assertEqual(
            self.errands.facts_in("the keys are at the office"),
            ["in keys office"])

    def test_things_named_without_a_place_are_still_named(self):
        found = self.blocks.mentions(
            "a red block, a green block and a blue block")
        self.assertEqual(sorted(found),
                         [("blue", "block"), ("green", "block"),
                          ("red", "block")])

    def test_another_domain_reads_with_the_same_machinery(self):
        self.assertEqual(self.errands.facts_in("the book is at the shop"),
                         ["in book shop"])
        self.assertEqual(self.errands.facts_in("get the book to the library"),
                         ["in book library"])


class HearingTests(unittest.TestCase):
    """What an utterance is taken to be, and what it must never take."""

    def test_nothing_is_heard_until_a_world_is_opened(self):
        """`the dog is on the mat` reads as `on dog mat` without trouble,
        and taking it would break every question v687 to v690 answer."""
        shut = Scene()
        for said in ("the dog is on the mat", "can a penguin fly",
                     "put the kettle on"):
            with self.subTest(said=said):
                self.assertEqual(page.hear(said, shut).act, "")

    def test_a_world_can_be_asked_for_and_opened(self):
        shut = Scene()
        self.assertEqual(page.hear("what worlds do you have", shut).act,
                         "which worlds")
        heard = page.hear("use the errands world", shut)
        self.assertEqual((heard.act, heard.domain), ("use a world", "errands"))

    def test_each_act_is_found_once_a_world_is_open(self):
        scene = Scene(DOMAINS["blocks"])
        page.say_to(scene, "there is a red block on a green block")
        for said, act in (
                ("there is a blue block on the table", "tell"),
                ("put the red block on the table", "want"),
                ("what is on the green block", "upon"),
                ("where is the red block", "where"),
                ("what do you see", "look"),
                ("actually the red block is on the table now", "meddle")):
            with self.subTest(said=said):
                self.assertEqual(page.hear(said, scene).act, act)

    def test_an_order_must_name_something_the_domain_can_be_asked_for(self):
        """`goalish` is what an order may mean: you can ask for a block to
        be somewhere, not for it to be clear."""
        scene = Scene(DOMAINS["blocks"])
        page.say_to(scene, "there is a red block on a green block")
        self.assertEqual(page.hear("make the red block clear", scene).act, "")


class SceneTests(unittest.TestCase):
    """A scene, and the acts over it."""

    def blocks(self) -> Scene:
        scene = Scene(DOMAINS["blocks"])
        page.say_to(scene, "there is a red block on a green block, and a "
                           "blue block on the table")
        return scene

    def test_the_sussman_anomaly_in_english(self):
        """The whole of v691 in two sentences: a scene, and a conjunctive
        goal whose parts interfere."""
        scene = self.blocks()
        said = page.say_to(scene, "put the green block on the blue block "
                                  "and the red block on the green block")
        self.assertEqual(said,
                         "I took the red block off the green block and put "
                         "it on the table, put the green block on the blue "
                         "block, then put the red block on the green block")
        self.assertTrue(scene.world.solved({"on green blue", "on red green"}))
        self.assertEqual(consistent(scene), [])

    def test_it_answers_about_the_scene_it_has(self):
        scene = self.blocks()
        self.assertEqual(page.say_to(scene, "where is the red block"),
                         "the red block is on the green block")
        self.assertEqual(page.say_to(scene, "what is on the green block"),
                         "the red block is on the green block")
        self.assertEqual(page.say_to(scene, "what is on the blue block"),
                         "nothing is on the blue block")

    def test_why_reads_the_means_ends_subgoal_back(self):
        """The planner's subgoals are already named `achieve clear green
        for take green`, so nothing is invented."""
        scene = self.blocks()
        page.say_to(scene, "put the green block on the blue block")
        self.assertEqual(
            page.say_to(scene, "why did you move the red block"),
            "I took the red block off the green block because I could not "
            "pick up the green block until nothing is on the green block")

    def test_why_before_anything_was_done(self):
        scene = self.blocks()
        self.assertEqual(page.hear("why did you do that", scene).act, "")

    def test_telling_it_something_leaves_a_scene_that_makes_sense(self):
        scene = self.blocks()
        for said in ("actually the red block is on the table now",
                     "actually the blue block is on the red block now",
                     "now the green block is on the blue block"):
            with self.subTest(said=said):
                page.say_to(scene, said)
                self.assertEqual(consistent(scene), [])

    def test_it_plans_again_from_where_the_world_now_is(self):
        scene = self.blocks()
        page.say_to(scene, "put the green block on the blue block and the "
                           "red block on the green block")
        page.say_to(scene, "actually the red block is on the table now")
        page.say_to(scene, "put the red block on the green block")
        self.assertTrue(scene.world.solved({"on red green"}))
        self.assertEqual(consistent(scene), [])


class OtherDomainTests(unittest.TestCase):
    """The same conversation, in worlds it was not written for."""

    def test_errands_are_talked_about_and_done(self):
        scene = Scene(DOMAINS["errands"])
        page.say_to(scene, "the book is at the shop and the keys are at "
                           "the office")
        said = page.say_to(scene, "get the book to the library and the keys "
                                  "to the shop")
        self.assertTrue(scene.world.solved({"in book library",
                                            "in keys shop"}))
        self.assertIn("picked up the book", said)
        self.assertEqual(page.say_to(scene, "where is the book"),
                         "the book is at library")

    def test_a_delivery_is_talked_about_and_done(self):
        scene = Scene(DOMAINS["delivery"])
        page.say_to(scene, "the blue van is in york and parcel1 is in leeds")
        page.say_to(scene, "get parcel1 to hull")
        self.assertTrue(scene.world.solved({"at parcel1 hull"}))
        self.assertEqual(consistent(scene), [])

    def test_why_works_wherever_the_planner_does(self):
        scene = Scene(DOMAINS["errands"])
        page.say_to(scene, "the book is at the shop")
        page.say_to(scene, "get the book to the library")
        said = page.say_to(scene, "why did you go to the shop")
        self.assertIn("because I could not", said)


class SurpriseTests(unittest.TestCase):
    """Something moving while the agent is working, in the REPL's world."""

    def setUp(self):
        self.scene = Scene(DOMAINS["blocks"])
        self.scene.world = talking.Gremlin(
            self.scene.world.facts, talking.undo_one(self.scene.domain))
        page.say_to(self.scene, "there is a red block, a green block and a "
                                "blue block on the table")

    def test_it_notices_replans_and_says_so(self):
        said = page.say_to(self.scene, "put the red block on the green "
                                       "block and the green block on the "
                                       "blue block")
        self.assertIn("something moved while I was working", said)
        self.assertIn("planned again", said)
        self.assertTrue(self.scene.world.solved({"on red green",
                                                 "on green blue"}))
        self.assertEqual(self.scene.last.surprises, 1)

    def test_the_story_is_told_in_stretches(self):
        """After a surprise the agent covers ground it has already covered,
        and one flat list reads as a stutter."""
        said = page.say_to(self.scene, "put the red block on the green "
                                       "block and the green block on the "
                                       "blue block")
        self.assertEqual(said.count("Then something moved"), 1)
        self.assertEqual(said.split(".")[0],
                         "I put the green block on the blue block")

    def test_the_gap_is_kept_as_a_prediction_that_failed(self):
        page.say_to(self.scene, "put the red block on the green block and "
                                "the green block on the blue block")
        gap = self.scene.last.gaps[0]
        self.assertIsInstance(gap, acting.Gap)
        self.assertIn("on green blue", gap.missing)
        self.assertIn("table green", gap.extra)


if __name__ == "__main__":
    unittest.main()
