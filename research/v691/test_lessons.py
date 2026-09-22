"""What a surprise teaches (`lessons.py`), and the negative preconditions
it teaches (`acting.negated`)."""
from __future__ import annotations

import unittest

from research.v687 import build
from research.v687.corpora import VERBNET_DIR
from research.v687.executive import Working
from research.v691 import (acting, domains, hidden, learned as L, lessons,
                           openworld, page, world as W)
from research.v691.scene import Scene

needs_verbnet = unittest.skipUnless(VERBNET_DIR.exists(), "no VerbNet")
needs_store = unittest.skipUnless(build.DEFAULT_STORE.exists(), "no store")


def act(name, needs=(), adds=(), deletes=(), forbids=()):
    return W.Action(name, frozenset(needs), frozenset(adds),
                    frozenset(deletes), frozenset(forbids))


class NegativeTests(unittest.TestCase):
    """A negative precondition, planned around."""

    def test_the_world_refuses_what_is_forbidden(self):
        world = W.World({"locked door"})
        self.assertFalse(world.do(act("open door", adds=["open door"],
                                      forbids=["locked door"])))

    def test_the_planner_removes_a_blocker_first(self):
        """`needs` can only ask for presence, so the blocker is compiled
        into a `not` slot, and means-ends plans to bring *that* about."""
        actions = [act("open door", adds=["open door"],
                       forbids=["locked door"]),
                   act("unlock door", adds=["unlocked door"],
                       deletes=["locked door"])]
        found = acting.think(actions, {"locked door"}, {"open door"})
        self.assertEqual([one.name for one in found.plan],
                         ["unlock door", "open door"])
        # And the plan is the world's own actions, not the compiled ones.
        self.assertIs(found.plan[1], actions[0])

    def test_a_domain_can_say_forbids(self):
        self.assertIn("wrapped book", next(
            one for one in hidden.REAL.ground({"book": "thing",
                                               "shop": "place"})
            if one.name == "fetch book shop").forbids)


class LearnerTests(unittest.TestCase):
    """The evidence rule: one explanation, or nothing."""

    def setUp(self):
        self.memory = L.Learned(None)
        self.addCleanup(self.memory.close)
        self.learner = lessons.Learner(self.memory)

    def gap(self, action, before):
        return acting.Gap(action, frozenset(action.adds), frozenset(), 0,
                          frozenset(before), True)

    def test_what_held_every_time_it_worked_is_required(self):
        go = act("go home shop", needs=["at home"], adds=["at shop"],
                 deletes=["at home"])
        self.learner.worked(go, {"at home", "lit shop"})
        found = self.learner.failed(self.gap(go, {"at home"}), {"at home"})
        self.assertEqual([(one.kind, one.literal) for one in found],
                         [("requires", "lit ?it")])
        self.assertEqual(self.memory.required("go"), ["lit ?it"])

    def test_what_held_only_when_it_failed_blocks(self):
        fetch = act("fetch book shop", adds=["carrying book"])
        self.learner.worked(fetch, {"in book shop"})
        found = self.learner.failed(
            self.gap(fetch, {"in book shop", "wrapped book"}), set())
        self.assertEqual([(one.kind, one.literal) for one in found],
                         [("blocks", "wrapped ?it")])

    def test_two_explanations_teach_nothing(self):
        fetch = act("fetch book shop", adds=["carrying book"])
        self.learner.worked(fetch, {"in book shop"})
        found = self.learner.failed(
            self.gap(fetch, {"in book shop", "wrapped book", "red book"}),
            set())
        self.assertEqual(found, [])
        self.assertEqual(self.memory.blockers("fetch"), [])

    def test_with_no_success_to_compare_nothing_is_learned(self):
        fetch = act("fetch book shop", adds=["carrying book"])
        self.assertEqual(self.learner.failed(
            self.gap(fetch, {"wrapped book"}), set()), [])

    def test_a_reason_said_is_enough(self):
        """*It is locked*, said with the failure, is the explanation."""
        opening = act("open door", adds=["open door"])
        found = self.learner.failed(self.gap(opening, {"closed door"}),
                                    {"closed door", "locked door"},
                                    revealed={"locked door"}, said="it is "
                                                                   "locked")
        self.assertEqual([(one.kind, one.literal) for one in found],
                         [("blocks", "locked ?it")])

    def test_a_success_takes_a_lesson_back(self):
        """Nothing is final: a blocker that held while it worked was not
        the reason."""
        fetch = act("fetch book shop", adds=["carrying book"])
        self.learner.worked(fetch, {"in book shop"})
        self.learner.failed(self.gap(fetch, {"in book shop", "wrapped book"}),
                            set())
        other = act("fetch cup park", adds=["carrying cup"])
        found = self.learner.worked(other, {"in cup park", "wrapped cup"})
        self.assertEqual([(one.kind, one.was) for one in found],
                         [("forgot", "blocks")])
        self.assertEqual(self.memory.blockers("fetch"), [])

    def test_a_taught_requirement_is_not_taken_back_by_trying(self):
        self.memory.require("drop", "with ?object ?subject", "you told me")
        drop = act("drop john book", adds=["at book floor"])
        self.learner.worked(drop, set())
        self.assertEqual(self.memory.required("drop"),
                         ["with ?object ?subject"])

    def test_an_effect_nobody_predicted_is_brought(self):
        fill = act("fill glass", adds=["full glass"])
        found = self.learner.failed(
            acting.Gap(fill, frozenset(), frozenset({"wet glass"}), 0,
                       frozenset(), False), {"full glass", "wet glass"})
        self.assertEqual([(one.kind, one.literal) for one in found],
                         [("brings", "wet ?it")])


class KindTests(unittest.TestCase):
    """A lesson about a kind of thing: scoped, narrowed and widened."""

    KINDS = {"book": "thing", "cup": "thing", "hat": "garment",
             "scarf": "garment", "vase": "pot"}

    def setUp(self):
        self.memory = L.Learned(None)
        self.addCleanup(self.memory.close)
        self.learner = lessons.Learner(
            self.memory, kind_of=self.KINDS.get,
            is_a=lambda thing, kind: self.KINDS.get(thing) == kind)

    def fetch(self, thing):
        return act(f"fetch {thing} shop", adds=[f"carrying {thing}"])

    def fail(self, thing, before):
        action = self.fetch(thing)
        return self.learner.failed(acting.Gap(
            action, frozenset(action.adds), frozenset(), 0,
            frozenset(before), True), set())

    def test_compared_within_the_kind_when_the_kinds_disagree(self):
        """A wrapped hat was picked up, so across everything wrapping is
        not the reason a wrapped book was not; among books, it is."""
        self.learner.worked(self.fetch("hat"), {"wrapped hat"})
        self.learner.worked(self.fetch("cup"), set())
        found = self.fail("book", {"wrapped book"})
        self.assertEqual([(one.kind, one.literal) for one in found],
                         [("blocks", "wrapped ?it")])
        self.assertEqual(self.memory.scoped("blocks", "fetch",
                                            "wrapped ?it"), ("thing", ()))
        applied = self.learner.applied([self.fetch("cup"),
                                        self.fetch("scarf")])
        self.assertEqual(applied[0].forbids, frozenset({"wrapped cup"}))
        self.assertEqual(applied[1].forbids, frozenset())

    def test_a_counterexample_of_another_kind_narrows(self):
        self.learner.worked(self.fetch("cup"), set())
        self.fail("book", {"wrapped book"})
        found = self.learner.worked(self.fetch("hat"), {"wrapped hat"})
        self.assertEqual([one.kind for one in found], ["narrowed"])
        self.assertEqual(self.memory.blockers("fetch"), ["wrapped ?it"])
        self.assertEqual(self.memory.scoped("blocks", "fetch",
                                            "wrapped ?it")[1], ("garment",))

    def test_a_counterexample_of_the_same_kind_refutes(self):
        self.learner.worked(self.fetch("cup"), set())
        self.fail("book", {"wrapped book"})
        found = self.learner.worked(self.fetch("cup"), {"wrapped cup"})
        self.assertEqual([one.kind for one in found], ["forgot"])
        self.assertEqual(self.memory.blockers("fetch"), [])

    def test_a_second_kind_widens_it(self):
        self.learner.worked(self.fetch("hat"), {"wrapped hat"})
        self.learner.worked(self.fetch("cup"), set())
        self.fail("book", {"wrapped book"})
        found = self.fail("vase", {"wrapped vase"})
        self.assertEqual([one.kind for one in found], ["widened"])
        self.assertEqual(self.memory.scoped("blocks", "fetch",
                                            "wrapped ?it")[0], "")


class HiddenTests(unittest.TestCase):
    """Rules nobody told it, learned by acting (`hidden.py`)."""

    def test_it_learns_both_rules_and_stops_being_surprised(self):
        memory = L.Learned(None)
        self.addCleanup(memory.close)
        found = hidden.summary(hidden.run(hidden.problems(30, 0),
                                          lessons.Learner(memory)))
        self.assertEqual(found["surprises last"], 0)
        self.assertEqual(found["solved last"], 10)
        self.assertEqual(memory.required("go"), ["lit ?it"])
        self.assertEqual(memory.blockers("fetch"), ["wrapped ?it"])
        self.assertEqual(memory.count()["brings"], 0)

    def test_a_rule_about_a_kind_needs_kinds(self):
        """Seed 2 of `--kinds`: compared across everything, wrapped hats
        that were picked up hide the rule for good; compared within the
        kind, it is found and held for things only."""
        pairs = hidden.problems(30, 2, kinds=True)
        plain = L.Learned(None)
        by_kind = L.Learned(None)
        self.addCleanup(plain.close)
        self.addCleanup(by_kind.close)
        found = hidden.summary(hidden.run(pairs, lessons.Learner(plain)))
        self.assertGreater(found["surprises last"], 0)
        learner = lessons.Learner(by_kind)
        learner.by_kind = True
        found = hidden.summary(hidden.run(pairs, learner))
        self.assertEqual(found["surprises last"], 0)
        self.assertEqual(found["solved last"], 10)
        self.assertEqual(by_kind.scoped("blocks", "fetch", "wrapped ?it"),
                         ("thing", ()))

    def test_without_learning_it_keeps_failing(self):
        found = hidden.summary(hidden.run(hidden.problems(30, 0)))
        self.assertLess(found["solved last"], 5)
        self.assertGreater(found["surprises last"], 10)


@needs_verbnet
@needs_store
class ReportedTests(unittest.TestCase):
    """Told that what it did did not happen, on the page."""

    def scene(self) -> Scene:
        scene = Scene(openworld.Open(openworld.resolver(), L.Learned(None)))
        self.addCleanup(scene.domain.learned.close)
        return scene

    def test_a_locked_door_is_learned_and_planned_around(self):
        scene = self.scene()
        page.say_to(scene, "the door is closed")
        page.say_to(scene, "open the door")
        said = page.say_to(scene, "the door is still closed, it is locked")
        self.assertIn("could not open the door", said)
        self.assertIn("nothing locked can be opened", said)
        self.assertNotIn("open door", scene.world.facts)
        self.assertIn("locked door", scene.world.facts)
        page.say_to(scene, "open the door")
        self.assertEqual([one.name for one in scene.last_plan],
                         ["unlock door", "open door"])

    def test_what_a_door_taught_holds_of_a_window(self):
        scene = self.scene()
        page.say_to(scene, "the door is closed")
        page.say_to(scene, "open the door")
        page.say_to(scene, "the door is still closed, it is locked")
        page.say_to(scene, "the window is locked")
        page.say_to(scene, "open the window")
        self.assertEqual([one.name for one in scene.last_plan],
                         ["unlock window", "open window"])

    def test_a_denial_is_a_report_too(self):
        scene = self.scene()
        page.say_to(scene, "the box is closed")
        page.say_to(scene, "open the box")
        said = page.say_to(scene, "the box did not open")
        self.assertIn("could not open the box", said)
        self.assertIn("do not know why", said)

    def test_a_state_word_is_not_a_thing(self):
        scene = self.scene()
        page.say_to(scene, "the door is closed")
        self.assertEqual(set(scene.objects), {"door"})

    def test_an_order_for_a_state_asks_for_the_state(self):
        scene = self.scene()
        page.say_to(scene, "the door is locked")
        page.say_to(scene, "unlock the door")
        self.assertIn("unlocked door", scene.world.facts)
        self.assertNotIn("locked door", scene.world.facts)


@needs_verbnet
@needs_store
class AskingTests(unittest.TestCase):
    """An order it cannot plan is suspended, and what is missing is asked
    for; the answer resumes it."""

    def scene(self) -> Scene:
        scene = Scene(openworld.Open(openworld.resolver(), L.Learned(None)))
        self.addCleanup(scene.domain.learned.close)
        return scene

    def test_a_thing_nobody_placed_is_asked_after(self):
        scene = self.scene()
        page.say_to(scene, "john is in the kitchen")
        said = page.say_to(scene, "get the book to the kitchen")
        self.assertIn("Where is the book?", said)
        self.assertEqual(scene.last_plan, [])
        self.assertEqual(page.hear("the book is in the garden", scene).act,
                         "resume")
        said = page.say_to(scene, "the book is in the garden")
        self.assertTrue(said.startswith("then "))
        self.assertIn("at book kitchen", scene.world.facts)
        self.assertIsNone(scene.pending)

    def test_a_state_nothing_makes_is_asked_whether(self):
        """In a world nobody declared, a fact not said is not known to be
        false: asked, not called impossible."""
        scene = self.scene()
        page.say_to(scene, "you can only open it if it is safe")
        page.say_to(scene, "the box is closed")
        said = page.say_to(scene, "open the box")
        self.assertIn("Is the box safe?", said)
        said = page.say_to(scene, "yes")
        self.assertIn("opened the box", said)
        self.assertIn("safe box", scene.world.facts)

    def test_no_turns_whether_into_how_and_it_can_be_dropped(self):
        scene = self.scene()
        page.say_to(scene, "you can only open it if it is safe")
        page.say_to(scene, "the box is closed")
        page.say_to(scene, "open the box")
        self.assertIn("how would I make it so that the box is safe",
                      page.say_to(scene, "no"))
        self.assertEqual(page.say_to(scene, "never mind"),
                         "all right, I will leave it")
        self.assertIsNone(scene.pending)

    def test_a_requirement_something_makes_is_just_done(self):
        """Asked only when nothing it knows can do it: `unlocked` is
        `unlock`'s, so it unlocks."""
        scene = self.scene()
        page.say_to(scene, "you can only open it if it is unlocked")
        page.say_to(scene, "the box is closed")
        page.say_to(scene, "open the box")
        self.assertEqual([one.name for one in scene.last_plan],
                         ["unlock box", "open box"])

    def test_a_new_order_is_not_an_answer(self):
        scene = self.scene()
        page.say_to(scene, "john is in the kitchen")
        page.say_to(scene, "get the book to the kitchen")
        self.assertEqual(page.hear("open the door", scene).act, "want")


if __name__ == "__main__":
    unittest.main()
