"""Actions read off what verbs mean, and a world with nothing declared.

The tests that need VerbNet or the store skip themselves without them, as
the norms suites do. What does not need either -- the translation from
VerbNet's semantics to a schema -- is tested on frames built here, so the
reading is pinned even where the data is not installed.
"""
from __future__ import annotations

import unittest

from research.v687 import build
from research.v687.corpora import VERBNET_DIR
from research.v689.change import Frame
from research.v691 import acting, errands, openworld, verbs, world as W
from research.v691.scene import Scene

HAVE_VERBNET = VERBNET_DIR.exists()
HAVE_STORE = build.DEFAULT_STORE.exists()
needs_verbnet = unittest.skipUnless(HAVE_VERBNET, "VerbNet not downloaded")
needs_store = unittest.skipUnless(HAVE_STORE, "store not built")


class ReadingTests(unittest.TestCase):
    """VerbNet's phases, as a precondition and an effect."""

    def ability(self, semantics, positions, verb="do"):
        frame = Frame("test-1", tuple(positions), frozenset(), tuple(
            semantics))
        return verbs._ability(verb, frame, {})

    def test_an_end_is_an_effect_and_a_start_is_a_precondition(self):
        one = self.ability(
            [("path_rel", False, "start", ("Theme", "Source", "ch_of_loc")),
             ("path_rel", False, "end", ("Theme", "Goal", "ch_of_loc"))],
            [("Theme", "object"), ("Source", "place"), ("Goal", "place")])
        self.assertEqual(one.needs, ("at ?Theme ?Source",))
        self.assertEqual(one.adds, ("at ?Theme ?Goal",))

    def test_a_negated_predicate_is_a_delete(self):
        one = self.ability(
            [("alive", False, "start", ("Patient",)),
             ("alive", True, "result", ("Patient",))],
            [("Agent", "subject"), ("Patient", "object")], verb="kill")
        self.assertIn("alive ?Patient", one.deletes)
        self.assertIn("alive ?Patient", one.needs)

    def test_a_change_of_state_is_written_the_other_way_round(self):
        """`path_rel(result(E), Result, Patient, ch_of_state)` is (state,
        thing) where a change of place is (thing, place). Reading both the
        same way made every state a fact about the state."""
        one = self.ability(
            [("path_rel", False, "result",
              ("?Initial_State", "Patient", "ch_of_state"))],
            [("Agent", "subject"), ("Patient", "object")], verb="open")
        self.assertEqual(one.adds, ("open ?Patient",))

    def test_an_unexpressed_role_is_dropped_and_not_guessed(self):
        """`?Initial_Location` is VerbNet saying the frame does not say
        where it was. Filling it with a participant is how bAbI's `took it
        there` put Daniel at the football."""
        one = self.ability(
            [("path_rel", False, "start",
              ("Theme", "?Initial_Location", "ch_of_loc")),
             ("path_rel", False, "end", ("Theme", "Goal", "ch_of_loc"))],
            [("Theme", "object"), ("Goal", "place")])
        self.assertEqual(one.needs, ())
        self.assertEqual(one.adds, ("at ?Theme ?Goal",))

    def test_how_it_happened_is_not_a_fact(self):
        """`cause`, `motion`, `manner` say that it happened and how, which
        a planner can neither bring about nor check."""
        self.assertEqual(
            verbs._literals("cause", ("Agent",), "push"), [])
        self.assertEqual(
            verbs._literals("motion", ("during", "Theme"), "push"), [])

    def test_a_frame_that_changes_nothing_is_not_an_action(self):
        self.assertIsNone(self.ability(
            [("cause", False, "", ("Agent",))],
            [("Agent", "subject")]))


class RestrictionTests(unittest.TestCase):
    """Who may fill a role."""

    def test_or_restrictions_are_a_choice_not_a_demand(self):
        """`bring`'s Destination is `+animate` **or** `+location`, because
        you can bring a thing to a place or to a person. Read as a
        conjunction it refuses every destination there is."""
        things = verbs.Things()
        things.add("kitchen", "room")
        self.assertTrue(things.allows(
            "kitchen", ("or", (("+", "animate"), ("+", "location")))))
        self.assertFalse(things.allows(
            "kitchen", ("and", (("+", "animate"), ("+", "location")))))

    def test_a_thing_the_store_knows_nothing_of_may_do_anything(self):
        """Refusing it would make an empty graph an agent that can do
        nothing, and the world refuses an action that does not apply."""
        things = verbs.Things()
        things.add("zzz", "zzz")
        self.assertTrue(things.allows("zzz", ("and", (("+", "animate"),))))

    def test_a_negative_restriction_is_enforced(self):
        things = verbs.Things()
        things.add("kitchen", "room")
        self.assertFalse(things.allows(
            "kitchen", ("and", (("-", "location"),))))

    @needs_store
    def test_the_store_says_what_a_thing_is(self):
        things = verbs.Things(openworld.resolver())
        for name, kind, wanted in (("john", "man", "animate"),
                                   ("kitchen", "kitchen", "location"),
                                   ("vase", "vase", "concrete")):
            things.add(name, kind)
            with self.subTest(thing=name):
                self.assertIn(wanted, things.categories(name))


@needs_verbnet
class AbilityTests(unittest.TestCase):
    """What VerbNet comes to, over the whole of it."""

    @classmethod
    def setUpClass(cls):
        cls.abilities = verbs.abilities()

    def test_most_of_verbnet_becomes_operators(self):
        """4,569 verbs have frames; 2,749 of them change something a
        planner can bring about, in 7,796 ways. The rest say that
        something happened, or how."""
        self.assertGreater(len(self.abilities), 2500)
        self.assertGreater(sum(len(one) for one in self.abilities.values()),
                           7000)

    def test_the_everyday_verbs_are_there(self):
        for verb in ("put", "take", "give", "carry", "open", "break", "go",
                     "kill", "wash", "cook", "fill", "lock"):
            with self.subTest(verb=verb):
                self.assertIn(verb, self.abilities)

    def test_carrying_needs_the_carrier_where_the_thing_is(self):
        """The one operator that shows the whole idea works: VerbNet says
        the agent and the theme are both at the initial location and both
        at the destination, which is a move-while-holding."""
        found = [one for one in self.abilities["carry"]
                 if one.klass.startswith("carry-") and len(one.needs) == 2]
        self.assertTrue(found)
        one = found[0]
        self.assertEqual(set(one.needs), {"at ?Theme ?Initial_Location",
                                          "at ?Agent ?Initial_Location"})
        self.assertEqual(set(one.adds), {"at ?Theme ?Destination",
                                         "at ?Agent ?Destination"})

    def test_being_in_one_place_is_learned_and_not_declared(self):
        """One frame, one thing, two different places: that is what it is
        for a relation to be a function of its first argument."""
        self.assertIn("at", verbs.functional())

    def test_centrality_is_counted_rather_than_chosen(self):
        central = verbs.central()
        self.assertGreater(central.get("take", 0), central.get("ferry", 99))
        self.assertGreater(central.get("go", 0), central.get("barge", 99))


@needs_verbnet
class GroundingTests(unittest.TestCase):
    """Backwards from the goal, because 4,569 verbs do not ground."""

    def things(self):
        found = verbs.Things()
        for name, kind in (("john", "man"), ("book", "book"),
                           ("kitchen", "kitchen"), ("shop", "shop")):
            found.add(name, kind)
        return found

    def test_it_grounds_something_that_achieves_the_goal(self):
        actions = verbs.useful({"at book kitchen"}, self.things())
        self.assertTrue(actions)
        self.assertTrue([one for one in actions
                         if "at book kitchen" in one.adds])

    def test_a_thing_moved_leaves_where_it_was(self):
        """The functional predicate's other values are deleted at grounding,
        because only there is it known what the other places are."""
        actions = verbs.useful({"at book kitchen"}, self.things())
        moved = [one for one in actions if "at book kitchen" in one.adds][0]
        self.assertIn("at book shop", moved.deletes)

    def test_nothing_is_ground_that_the_goal_does_not_reach(self):
        few = verbs.useful({"at book kitchen"}, self.things(), per_verb=2)
        many = verbs.useful({"at book kitchen"}, self.things(), per_verb=8)
        self.assertLess(len(few), len(many))


@needs_verbnet
@needs_store
class ErrandTests(unittest.TestCase):
    """The numbers `DESIGN.md` §8 quotes, pinned."""

    @classmethod
    def setUpClass(cls):
        cls.rows = errands.measure(openworld.resolver())

    def test_every_errand_is_reached(self):
        missed = [one.name for one in self.rows if not one.reached]
        self.assertEqual(missed, [])

    def test_and_a_third_of_them_choose_a_verb_nobody_would(self):
        """Recorded rather than asserted away, and the more interesting of
        the two numbers: the control is general and the choice of verb is
        a judgement about meaning that the store does not make."""
        sensible = sum(one.sensible for one in self.rows)
        self.assertGreaterEqual(sensible, 7)
        self.assertLess(sensible, len(self.rows))


@needs_verbnet
class OpenReaderTests(unittest.TestCase):
    """Facts out of English with no templates to read them by."""

    def setUp(self):
        self.reader = openworld.Open().reader()

    def test_the_shapes_a_fact_is_stated_in(self):
        for said, fact in (
                ("the book is in the shop", "at book shop"),
                ("john has the book", "with book john"),
                ("the door is closed", "closed door")):
            with self.subTest(said=said):
                self.assertIn(fact, self.reader.facts_in(said))

    def test_the_shapes_something_is_asked_for_in(self):
        for said, fact in (
                ("get the book to the kitchen", "at book kitchen"),
                ("give the book to mary", "with book mary"),
                ("open the door", "open door")):
            with self.subTest(said=said):
                self.assertIn(fact,
                              self.reader.facts_in(said, wanting=True))

    def test_a_verb_may_be_a_predicate_even_though_it_is_never_a_thing(self):
        """`open the door` has a verb in the predicate position, which is
        what a predicate is allowed to be -- only the arguments have to be
        things."""
        self.assertEqual(self.reader.facts_in("open the door", wanting=True),
                         ["open door"])


@needs_verbnet
@needs_store
class OpenWorldTests(unittest.TestCase):
    """A conversation about a world nobody declared."""

    def scene(self) -> Scene:
        return Scene(openworld.Open(openworld.resolver()))

    def test_it_is_told_a_situation_and_does_something_about_it(self):
        scene = self.scene()
        from research.v691.page import say_to
        say_to(scene, "john is in the kitchen and the book is in the shop")
        self.assertIn("at book shop", scene.world.facts)
        say_to(scene, "get the book to the kitchen")
        self.assertIn("at book kitchen", scene.world.facts)
        self.assertNotIn("at book shop", scene.world.facts)

    def test_it_says_why_from_the_goal_stack(self):
        scene = self.scene()
        from research.v691.page import say_to
        say_to(scene, "john is in the kitchen and the book is in the shop")
        say_to(scene, "get the book to the kitchen")
        said = say_to(scene, "why did you move the book")
        self.assertIn("because", said)

    def test_anything_a_verb_brings_about_can_be_asked_for(self):
        """A declared domain lists what an order may mean. Here the list is
        the verbs': nothing said `open` was a thing to want."""
        scene = self.scene()
        self.assertIn("open", scene.domain.goalish)
        self.assertIn("cook", scene.domain.goalish)


if __name__ == "__main__":
    unittest.main()
