"""v694: designing a way to a goal in the open world.

Most of this needs the store and VerbNet, and skips itself without them,
as v691's open-world tests do.
"""
from __future__ import annotations

import unittest

from research.v687 import build
from research.v687.corpora import VERBNET_DIR

HAVE = VERBNET_DIR.exists() and build.DEFAULT_STORE.exists()
needs_data = unittest.skipUnless(HAVE, "VerbNet or the store missing")


@needs_data
class KnowingTests(unittest.TestCase):
    """What the store is asked, and how its rows are read."""

    @classmethod
    def setUpClass(cls):
        from research.v694 import knowing
        cls.K = knowing

    def row(self, said, relation="used_for"):
        words = tuple((word, self.K.lemmas(word)) for word in said.split())
        return self.K.Row("x.n.01", relation, said, 0.35, words)

    def test_a_state_is_its_own_words_not_its_neighbours_verbs(self):
        words = self.K.state_words("cold")
        self.assertIn("chilly", words)
        self.assertNotIn("snap", words)
        self.assertNotIn("bake", self.K.state_words("dry"))

    def test_a_row_about_the_thing_counts_more(self):
        self.assertEqual(self.K.about(self.row("kill flies"), "fly",
                                      self.K.lemmas("kill")), 3.0)
        self.assertEqual(self.K.about(self.row("keep food cold"), "milk",
                                      self.K.state_words("cold")
                                      | self.K.KEEPING), 2.0)
        self.assertLess(self.K.about(self.row("cut hair"), "rope",
                                     self.K.lemmas("cut")), 1.0)
        self.assertEqual(self.K.about(self.row("wake people up"), "person",
                                      self.K.lemmas("wake")), 2.0)

    def test_what_was_done_to_a_thing_is_not_doing_it(self):
        self.assertFalse(self.K.does(self.row("fixed to the vehicle",
                                              "capable_of"), "fix"))
        self.assertTrue(self.K.does(self.row("cutting paper"), "cut"))

    def test_a_verb_can_name_its_tool(self):
        self.assertIn("mower", self.K.named_tools("mow"))
        self.assertIn("iron", self.K.named_tools("iron"))
        self.assertNotIn("instrument", self.K.named_tools("cut"))

    def test_a_door_has_a_knob(self):
        self.assertIn("doorknob", self.K.has_parts("door"))

    def test_a_category_is_not_a_thing_to_get(self):
        self.assertTrue(self.K.category("kitchen utensil"))
        self.assertFalse(self.K.category("knife"))


@needs_data
class ClauseTests(unittest.TestCase):

    def test_a_result_is_designed_as_its_doing(self):
        from research.v694.goals import Clause
        self.assertEqual((Clause("fixed car").kind,
                          Clause("fixed car").verbs()), ("done", ["fix"]))
        self.assertEqual(Clause("unlocked door").verbs(), ["unlock"])

    def test_a_state_is_a_state(self):
        from research.v694.goals import Clause
        self.assertEqual(Clause("cold milk").kind, "state")
        self.assertEqual(Clause("cold milk").verbs(), [])
        self.assertEqual(Clause("at book kitchen").kind, "place")


@needs_data
class DesigningTests(unittest.TestCase):
    """Ways designed, and the simplest kept."""

    @classmethod
    def setUpClass(cls):
        from research.v694 import designing, knowing
        from research.v694.goals import Goal
        knowing.index()
        cls.designing, cls.Goal = designing, Goal

    def design(self, wants, facts=(), names=(), **options):
        return self.designing.design(self.Goal(
            tuple(wants), frozenset(facts), frozenset(names)), **options)

    def test_a_rope_is_cut_with_something_that_cuts(self):
        found = self.design(["cut rope"])
        best = found.parts[0].best
        self.assertEqual(best.form, "tool")
        self.assertIn(best.means, {"knife", "scissors", "saw", "blade"})
        self.assertEqual(len(best.supposed), 1)

    def test_what_is_at_hand_is_used_before_anything_is_supposed(self):
        best = self.design(["cut rope"], {"at scissors table"}).parts[0].best
        self.assertEqual(best.means, "scissors")
        self.assertEqual(best.supposed, [])

    def test_a_kind_of_what_serves_is_at_hand_too(self):
        best = self.design(["cut rope"],
                           {"at carving-knife drawer"}).parts[0].best
        self.assertEqual(best.means, "carving-knife")

    def test_milk_is_made_cold_in_a_fridge(self):
        best = self.design(["cold milk"]).parts[0].best
        self.assertEqual(best.form, "place")
        self.assertEqual(best.means, "refrigerator")
        best = self.design(["cold milk"], {"at fridge kitchen",
                                           "at milk table"}).parts[0].best
        self.assertEqual((best.means, len(best.steps)), ("fridge", 1))

    def test_a_door_is_opened_by_hand_because_its_knob_comes_with_it(self):
        self.assertEqual(self.design(["open door"]).parts[0].best.form,
                         "unaided")

    def test_a_car_is_fixed_by_someone_whose_trade_it_is(self):
        best = self.design(["fixed car"]).parts[0].best
        self.assertEqual(best.form, "helper")
        self.assertEqual(len(best.helpers), 1)

    def test_a_person_is_warmed_with_something_warm(self):
        best = self.design(["warm john"], names={"john"}).parts[0].best
        self.assertIn(best.form, {"wear", "tool"})
        self.assertIn(best.means, {"blanket", "coat", "jacket", "heater",
                                   "sweater", "quilt"})

    def test_what_one_clause_fetched_the_next_uses(self):
        found = self.design(["cut rope", "cut string"])
        self.assertTrue(found.done)
        self.assertEqual(len(found.steps), 3)
        self.assertEqual(len(found.supposed()), 1)

    def test_a_weak_tool_in_hand_does_not_beat_a_good_one_to_get(self):
        best = self.design(["dig hole"], {"with knife me"}).parts[0].best
        self.assertNotEqual(best.means, "knife")

    def test_a_step_that_cannot_happen_does_not_check(self):
        from research.v691 import world as W
        from research.v694 import ways
        from research.v694.goals import Clause
        clause = Clause("cut rope")
        one = ways.Candidate("tool", clause, steps=[W.Action(
            "cut rope knife", frozenset({"with knife me"}),
            frozenset({"cut rope"}), frozenset())])
        self.assertIsNone(self.designing.checked(one, frozenset()))
        self.assertIsNotNone(self.designing.checked(
            one, frozenset({"with knife me"})))


@needs_data
class CarryingTests(unittest.TestCase):
    """Act, be surprised, learn, design again -- and remember."""

    @classmethod
    def setUpClass(cls):
        from research.v694 import knowing
        knowing.index()

    def broken_world(self, facts):
        from research.v691 import world as W

        class Broken(W.World):
            def can(self, action):
                parts = action.name.split()
                if parts[0] == "put" and len(parts) == 3 and \
                        f"broken {parts[2]}" in self.facts:
                    return False
                return super().can(action)
        return Broken(facts)

    def test_a_broken_fridge_is_learned_and_designed_around(self):
        from research.v691 import learned, lessons
        from research.v694 import carrying
        from research.v694.goals import Goal
        learner = lessons.Learner(learned.Learned(None))
        memory = carrying.Remembered()
        world = self.broken_world({"at fridge kitchen", "at milk table",
                                   "broken fridge"})
        out = carrying.carry_out(Goal(("cold milk",)), world, learner,
                                 memory, told=lambda step, w: {
                                     "broken fridge"})
        self.assertTrue(out.done)
        self.assertEqual(len(out.surprises), 1)
        self.assertEqual([(one.kind, one.literal) for one in
                          out.surprises[0][1]], [("blocks",
                                                  "broken ?object")])
        # The next goal does not try the broken fridge at all.
        world = self.broken_world({"at fridge kitchen", "broken fridge",
                                   "at beer table"})
        again = carrying.carry_out(Goal(("cold beer",)), world, learner,
                                   memory)
        self.assertTrue(again.done)
        self.assertEqual(again.surprises, [])

    def test_a_taught_way_is_used(self):
        from research.v694 import carrying, designing
        from research.v694.goals import Goal
        memory = carrying.Remembered()
        memory.remember("cut", "rope", "tool", "saw", said="you told me")
        best = designing.design(Goal(("cut rope",), memory=memory)).parts[
            0].best
        self.assertEqual((best.means, best.why), ("saw", "you told me"))


@needs_data
class SimilarTests(unittest.TestCase):
    """What worked for one thing, carried to things like it."""

    @classmethod
    def setUpClass(cls):
        from research.v694 import knowing
        knowing.index()
        cls.K = knowing

    def test_close_siblings_are_alike_and_strangers_are_not(self):
        self.assertEqual(self.K.similar("rope", "cord"), 0.5)
        self.assertEqual(self.K.similar("carving-knife", "knife"), 1.0)
        self.assertEqual(self.K.similar("knife", "fork"), 0.0)

    def test_a_way_of_doing_a_thing_is_like_doing_it(self):
        self.assertEqual(self.K.similar_verb("slice", "cut"), 0.5)
        self.assertEqual(self.K.similar_verb("fix", "repair"), 1.0)
        self.assertEqual(self.K.similar_verb("cut", "dig"), 0.0)

    def test_a_row_about_a_sibling_is_weaker_evidence(self):
        words = tuple((word, self.K.lemmas(word)) for word in
                      "cut cord".split())
        row = self.K.Row("x.n.01", "used_for", "cut cord", 0.35, words)
        self.assertEqual(self.K.about(row, "rope", self.K.lemmas("cut")),
                         self.K.SIBLING)

    def test_a_taught_way_is_tried_on_a_similar_thing(self):
        from research.v694 import carrying, designing
        from research.v694.goals import Goal
        memory = carrying.Remembered()
        memory.remember("cut", "rope", "tool", "axe", said="you told me")
        best = designing.design(Goal(("cut cord",), frozenset(
            {"at axe shed"}), memory=memory)).parts[0].best
        self.assertEqual(best.means, "axe")


@needs_data
class BuildingTests(unittest.TestCase):
    """A tool made from what is at hand, by a recipe taught or seen."""

    @classmethod
    def setUpClass(cls):
        from research.v694 import knowing
        knowing.index()

    def design(self, facts, recipe=True):
        from research.v694 import carrying, designing
        from research.v694.goals import Goal
        memory = carrying.Remembered()
        if recipe:
            memory.learn_recipe("torch", ["stick", "cloth"])
        return designing.design(Goal(("light room",), frozenset(facts),
                                     memory=memory)).parts[0].best

    def test_a_torch_is_made_from_a_stick_and_something_like_cloth(self):
        best = self.design({"at stick shed", "at rag drawer"})
        self.assertTrue(any(step.name.startswith("make torch")
                            for step in best.steps))
        self.assertEqual(best.supposed, [])

    def test_nothing_is_made_without_every_part(self):
        best = self.design({"at stick shed"})
        self.assertFalse(any(step.name.startswith("make ")
                             for step in best.steps))

    def test_nothing_is_made_without_a_recipe(self):
        best = self.design({"at stick shed", "at rag drawer"}, recipe=False)
        self.assertFalse(any(step.name.startswith("make ")
                             for step in best.steps))


@needs_data
class InsideTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from research.v694 import knowing
        knowing.index()

    def test_bread_is_baked_in_an_oven(self):
        from research.v694 import designing
        from research.v694.goals import Goal
        best = designing.design(Goal(("bake bread",))).parts[0].best
        self.assertEqual((best.form, best.means), ("inside", "oven"))

    def test_a_person_is_not_put_inside_anything(self):
        from research.v694 import designing
        from research.v694.goals import Goal
        best = designing.design(Goal(("warm john",), names=frozenset(
            {"john"}))).parts[0].best
        self.assertNotEqual(best.form, "inside")


@needs_data
class TeachingTests(unittest.TestCase):

    def read(self, text):
        from research.v694 import teaching
        return teaching.read(text)

    def test_a_way_said_in_general_or_as_advice(self):
        for text in ("you can cut the rope with a saw",
                     "use a saw to cut the rope"):
            taught = self.read(text)
            self.assertEqual((taught.kind, taught.predicate, taught.patient,
                              taught.means), ("way", "cut", "rope", "saw"))

    def test_a_recipe_said_either_way(self):
        for text in ("you can make a torch from a stick and a cloth",
                     "a torch is made from a stick and a cloth"):
            taught = self.read(text)
            self.assertEqual((taught.product, taught.parts),
                             ("torch", ("stick", "cloth")))

    def test_what_was_done_is_seen_not_told(self):
        self.assertTrue(self.read("i made a raft from the logs and the "
                                  "rope").seen)

    def test_an_order_or_a_request_teaches_nothing(self):
        self.assertIsNone(self.read("cut the rope with a saw"))
        self.assertIsNone(self.read("can you cut the rope with a saw"))


@needs_data
class AskingTests(unittest.TestCase):
    """How a thing is made, asked for (`asking.py`)."""

    def read(self, text):
        from research.v694 import asking
        return asking.read(text)

    def test_a_recipe_asked_for_by_name(self):
        for text in ("do you have any good recipes for cake?",
                     "do you know a good cake recipe",
                     "is there a recipe for cake"):
            asked = self.read(text)
            self.assertEqual((asked.product, asked.by_name), ("cake", True),
                             text)
        self.assertEqual(self.read("give me a recipe for making cookies")
                         .product, "cookie")

    def test_asked_as_how_one_is_made(self):
        self.assertEqual(self.read("how do you make a cake").product, "cake")
        self.assertEqual(self.read("how are cookies made").product,
                         "cookie")
        self.assertFalse(self.read("how do i bake bread").by_name)

    def test_what_is_not_asking_for_a_recipe(self):
        for text in ("how can i make the milk cold", "what is a recipe",
                     "how do you make friends", "is there a recipe for "
                     "success", "the recipe for soup is on the table",
                     "you can make a torch from a stick and a cloth",
                     "how do you open a jar",
                     "what are the directions to the station"):
            self.assertIsNone(self.read(text), text)

    def test_the_store_says_what_it_can_be_made_of_not_how(self):
        from research.v694 import asking
        said = asking.answer(self.read("do you have a recipe for cake"))
        self.assertTrue(said.startswith("No, I have no recipe"), said)
        self.assertIn("flour", said)
        self.assertNotIn("ingredients", said)

    def test_a_taught_recipe_is_one_and_a_like_one_is_offered(self):
        from research.v694 import asking
        from research.v694.carrying import Remembered
        memory = Remembered()
        memory.learn_recipe("torch", ["stick", "cloth"], said="you told me")
        said = asking.answer(self.read("is there a recipe for a torch"),
                             memory)
        self.assertTrue(said.startswith("Yes: a torch can be made from"),
                        said)
        said = asking.answer(self.read("how do you make a lamp"), memory)
        self.assertIn("like a lamp", said)

    def test_nothing_known_of_how_leaves_it_to_the_others(self):
        from research.v694 import asking
        self.assertFalse(asking.knows(self.read("how is a raft built")))


@needs_data
class ExamplesTests(unittest.TestCase):
    """Examples asked for (`examples.py`)."""

    def read(self, text):
        from research.v694 import examples
        return examples.read(text)

    def test_examples_asked_for(self):
        wanted = self.read("do you have any tools for gardening")
        self.assertEqual((wanted.kind, wanted.purpose, wanted.verb),
                         ("tool", "gardening", "have"))
        self.assertEqual(self.read("name three birds").many, 3)
        self.assertEqual(self.read("do you know any jokes").kind, "joke")

    def test_what_i_have_is_not_asking_for_examples(self):
        for text in ("do you have any pets", "do you have a dog",
                     "do dogs have fleas", "name the dog rex"):
            self.assertIsNone(self.read(text), text)

    def test_a_purpose_picks_among_the_kinds(self):
        from research.v694 import examples
        found = [name for name, _ in examples.examples(
            self.read("do you know any tools for cutting"))]
        self.assertIn("scissors", found)

    def test_the_best_known_come_first_and_only_kinds_of_it(self):
        from research.v694 import examples
        found = [name for name, _ in examples.examples(
            self.read("name some birds"))]
        self.assertTrue({"robin", "duck", "owl"} & set(found), found)
        found = [name for name, _ in examples.examples(
            self.read("do you know any pets"))]
        self.assertIn("dog", found)
        self.assertNotIn("tenant", found)


@needs_data
class HearingTests(unittest.TestCase):
    """v691's reader, for what a designer is asked."""

    def hear(self, text):
        from research.v691 import hearing
        return hearing.hear(text)

    def test_the_state_a_thing_is_to_be_left_in(self):
        self.assertEqual(self.hear("make the milk cold").wants,
                         ["cold milk"])
        self.assertEqual(self.hear("keep the food cold").wants,
                         ["cold food"])
        self.assertEqual(self.hear("make the milk and the beer cold").wants,
                         ["cold milk", "cold beer"])

    def test_things_joined_by_and(self):
        self.assertEqual(self.hear("cut the rope and the string").wants,
                         ["cut rope", "cut string"])

    def test_a_noun_is_a_thing_whatever_else_it_can_be(self):
        self.assertIn("can", self.hear("open the can").nouns)


@needs_data
class PageTests(unittest.TestCase):
    """The designer on v691's page, before the planner."""

    @classmethod
    def setUpClass(cls):
        from research.v694 import knowing, page  # noqa: F401
        knowing.index()

    def scene(self):
        from research.v691 import learned, openworld
        from research.v691.scene import Scene
        return Scene(openworld.Open(openworld.resolver(),
                                    learned.Learned(None)))

    def test_an_order_that_takes_a_tool_is_designed_and_done(self):
        from research.v691.page import say_to
        scene = self.scene()
        say_to(scene, "the rope is in the kitchen")
        said = say_to(scene, "cut the rope")
        self.assertIn(" with the ", said)
        self.assertIn("cut rope", scene.world.facts)
        self.assertTrue(scene.last_plan)

    def test_why_a_thing_was_used_is_the_design_that_used_it(self):
        from research.v691.page import say_to
        scene = self.scene()
        say_to(scene, "cut the rope")
        said = say_to(scene, "why did you get the knife")
        self.assertIn("cut the rope", said)

    def test_a_recipe_taught_on_the_page_is_used(self):
        from research.v691.page import say_to
        scene = self.scene()
        say_to(scene, "the stick is in the shed and the rag is in the "
                      "drawer")
        said = say_to(scene, "you can make a torch from a stick and a cloth")
        self.assertTrue(said.startswith("I will remember"), said)
        said = say_to(scene, "light the room")
        self.assertIn("made a torch", said)

    def test_what_was_seen_done_is_learned_quietly(self):
        from research.v694 import page
        scene = self.scene()
        page.observe(scene, "i made a raft from the logs and the rope")
        self.assertEqual([(one.product, one.parts) for one in
                          page._memory(scene).recipes],
                         [("raft", ("log", "rope"))])

    def test_asking_how_designs_without_doing(self):
        from research.v691.page import say_to
        scene = self.scene()
        said = say_to(scene, "how can i make the milk cold")
        self.assertTrue(said.startswith("I would"), said)
        self.assertNotIn("cold milk", scene.world.facts)

    def test_a_recipe_asked_for_is_answered_on_the_page(self):
        from research.v691.page import say_to
        scene = self.scene()
        said = say_to(scene, "do you have any good recipes for cake?")
        self.assertTrue(said.startswith("No, I have no recipe for cake"),
                        said)
        say_to(scene, "you can make a torch from a stick and a cloth")
        said = say_to(scene, "is there a recipe for a torch")
        self.assertIn("stick", said)

    def test_moving_a_thing_is_still_the_planners(self):
        from research.v691.page import say_to
        scene = self.scene()
        say_to(scene, "the book is in the shop")
        said = say_to(scene, "get the book to the kitchen")
        self.assertNotIn(" -- ", said)
        self.assertIn("at book kitchen", scene.world.facts)


if __name__ == "__main__":
    unittest.main()
