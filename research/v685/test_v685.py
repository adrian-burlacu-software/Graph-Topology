"""Regression suite. Run: python -m unittest research.v685.test_v685 -v

Store-dependent tests skip themselves when the v684 store is absent, so the
suite runs on a fresh clone before anything is built.
"""
from __future__ import annotations

import unittest

from research.v685 import ask as ask_module
from research.v685.bridge import Bridge, Route
from research.v685.graph import FactGraph, Hop

STORE = ask_module.DEFAULT_STORE
requires_store = unittest.skipUnless(STORE.exists(), f"no store at {STORE}")


@requires_store
class FactGraphTests(unittest.TestCase):
    """Reading v684's free-text objects as edges."""

    @classmethod
    def setUpClass(cls):
        cls.graph = FactGraph(STORE)

    @classmethod
    def tearDownClass(cls):
        cls.graph.close()

    def test_an_object_phrase_names_a_concept(self):
        self.assertEqual(self.graph.denotes("owner"), "owner.n.01")
        self.assertTrue(self.graph.denotes("carpenter's toolbox"))

    def test_an_unresolvable_phrase_is_not_forced(self):
        self.assertIsNone(self.graph.denotes("zzzqqq wumpus"))

    def test_the_edge_that_makes_the_dog_question_answerable(self):
        hops = {(h.relation, h.target) for h in self.graph.hops("dog.n.01")}
        self.assertIn(("has_a", "owner.n.01"), hops)

    def test_a_hop_carries_the_fact_that_licenses_it(self):
        hop = next(h for h in self.graph.hops("dog.n.01")
                   if h.relation == "has_a" and h.target == "owner.n.01")
        self.assertTrue(hop.text)
        self.assertGreater(hop.confidence, 0)

    def test_the_taxonomy_is_in_the_same_graph(self):
        hops = {(h.relation, h.target) for h in self.graph.hops("dog.n.01")}
        self.assertIn(("is_a", "canine.n.02"), hops)

    def test_one_edge_per_relation_and_target(self):
        """Many phrasings say the same thing; the graph keeps one edge."""
        hops = self.graph.hops("dog.n.01")
        keys = [(h.relation, h.target) for h in hops]
        self.assertEqual(len(keys), len(set(keys)))

    def test_the_branching_this_whole_module_exists_for(self):
        """v684 ascends 15 concepts; one fact hop from dog reaches hundreds."""
        self.assertGreater(len(self.graph.hops("dog.n.01")), 200)


@requires_store
class BridgeTests(unittest.TestCase):
    """Bounded search for the route between two concepts."""

    @classmethod
    def setUpClass(cls):
        cls.graph = FactGraph(STORE)
        cls.bridge = Bridge(cls.graph, depth=3, breadth=60)

    @classmethod
    def tearDownClass(cls):
        cls.graph.close()

    def test_it_finds_the_route_and_shows_the_evidence(self):
        report = self.bridge.search("dog.n.01", "owner.n.01")
        self.assertTrue(report.routes)
        route = report.routes[0]
        self.assertEqual(route.target, "owner.n.01")
        self.assertEqual(route.hops[0].source, "dog.n.01")
        self.assertTrue(route.hops[0].text)

    def test_the_shortest_route_is_offered_first(self):
        report = self.bridge.search("violin.n.01", "musician.n.01")
        lengths = [len(r.hops) for r in report.routes]
        self.assertEqual(lengths, sorted(lengths))

    def test_a_route_never_revisits_its_own_start(self):
        report = self.bridge.search("dog.n.01", "food.n.01")
        for route in report.routes:
            concepts = route.concepts()
            self.assertEqual(len(concepts), len(set(concepts)))

    def test_the_budget_is_reported_not_assumed(self):
        report = self.bridge.search("dog.n.01", "veterinarian.n.01")
        self.assertGreater(report.expanded, 0)
        self.assertGreater(report.generated, report.expanded)
        self.assertGreaterEqual(report.depth_reached, 1)

    def test_widening_the_breadth_does_not_change_the_best_route(self):
        """What the breadth budget costs, stated exactly.

        Measured from 20 to 1000: the route that gets used is identical every
        time, which is what makes the budget safe. The *alternatives* below it
        do change -- at 20 the third route for violin runs through `mozart`,
        at 400 through `music` -- so this pins the best route only. Claiming
        the whole list was stable was wrong, and this test is what caught it.
        """
        narrow = Bridge(self.graph, depth=3, breadth=20)
        wide = Bridge(self.graph, depth=3, breadth=400)
        for source, target in (("dog.n.01", "owner.n.01"),
                               ("violin.n.01", "musician.n.01"),
                               ("dog.n.01", "food.n.01")):
            a = narrow.search(source, target)
            b = wide.search(source, target)
            self.assertTrue(a.routes and b.routes, f"{source}->{target}")
            self.assertEqual(str(a.routes[0]), str(b.routes[0]),
                             f"{source}->{target}")

    def test_an_absent_link_is_reported_as_absent(self):
        report = self.bridge.search("hammer.n.02", "carpenter.n.01")
        self.assertEqual(report.routes, [])

    def test_a_concept_does_not_bridge_to_itself(self):
        self.assertEqual(self.bridge.search("dog.n.01", "dog.n.01").routes, [])

    def test_kinship_prefers_the_specific_over_the_universal(self):
        """Everything shares `entity`; only some things share `canine`."""
        near = self.bridge.kinship("dog.n.01", "wolf.n.01")
        far = self.bridge.kinship("dog.n.01", "hammer.n.02")
        self.assertGreater(near, far)


@requires_store
class BridgedQuestionTests(unittest.TestCase):
    """The two-part question, end to end."""

    @classmethod
    def setUpClass(cls):
        cls.engine = ask_module.BridgedReasoner()
        if cls.engine.parser.nlp is None:
            cls.engine.close()
            raise unittest.SkipTest("possessive parsing needs spaCy")

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def test_the_possessive_is_read_from_the_dependency_not_the_apostrophe(self):
        for question, expected in (
                ("what does a dog's owner need", ("dog", "owner")),
                ("what can a violin's player do", ("violin", "player")),
                ("where does a dog's owner live", ("dog", "owner"))):
            self.assertEqual(self.engine.possessive(question), expected, question)

    def test_a_question_with_no_possessive_is_handed_back(self):
        answer = self.engine.ask("can a dog fall into a hole")
        self.assertEqual(answer.verdict, "NOT_BRIDGED")

    def test_the_target_question(self):
        answer = self.engine.ask("what does a dog's owner need")
        self.assertEqual(answer.verdict, "BRIDGED")
        self.assertEqual(answer.anchor, "dog.n.01")
        self.assertEqual(answer.role, "owner.n.01")
        self.assertEqual(answer.relation, "has_prerequisite")
        self.assertEqual(answer.route.hops[0].target, "owner.n.01")
        self.assertTrue(answer.answer.evidence)

    def test_the_graph_picks_the_sense_the_anchor_can_reach(self):
        """`player` defaults to the sports sense; a violin's does not."""
        answer = self.engine.ask("what can a violin's player do")
        self.assertEqual(answer.role, "musician.n.01")
        self.assertEqual(answer.verdict, "BRIDGED")

    def test_the_relation_still_comes_from_the_question(self):
        need = self.engine.ask("what does a dog's owner need")
        does = self.engine.ask("what can a dog's owner do")
        where = self.engine.ask("where does a dog's owner live")
        self.assertEqual(need.relation, "has_prerequisite")
        self.assertEqual(does.relation, "capable_of")
        self.assertEqual(where.relation, "at_location")

    def test_it_serialises_for_a_ui(self):
        payload = self.engine.ask("what does a dog's owner need").as_dict()
        for key in ("verdict", "route", "answer", "search", "relation"):
            self.assertIn(key, payload)
        self.assertIn("hops", payload["route"])


if __name__ == "__main__":
    unittest.main()
