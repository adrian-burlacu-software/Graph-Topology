"""Regression suite. Run: python -m unittest research.v687.test_v685 -v

Store-dependent tests skip themselves when the v684 store is absent, so the
suite runs on a fresh clone before anything is built.
"""
from __future__ import annotations

import unittest

from research.v687 import ask as ask_module
from research.v687.bridge import Bridge, Route
from research.v687.graph import FactGraph, Hop
from research.v687.relevance import Relevance, SIBLING_LIMIT

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
class RelevanceTests(unittest.TestCase):
    """R14: a sibling of the anchor is not the anchor."""

    @classmethod
    def setUpClass(cls):
        cls.graph = FactGraph(STORE)
        cls.judge = Relevance(cls.graph)

    @classmethod
    def tearDownClass(cls):
        cls.graph.close()

    def verdict(self, phrase):
        return self.judge.judge(phrase, "violin.n.01").verdict

    def test_a_rival_instrument_is_a_sibling(self):
        """The reported bug: a violin player does not play the drums."""
        for phrase in ("play drum", "play piano", "play trumpet", "play cello",
                       "play saxophone", "play accordion", "play guitar",
                       "play flute", "play clarinet"):
            self.assertEqual(self.verdict(phrase), "sibling", phrase)

    def test_the_instrument_sense_is_found_even_when_it_is_not_the_default(self):
        """`drum` resolves to a barrel, `bass` to a fish, `brass` to management."""
        for phrase in ("play drum", "play bass", "play brass"):
            judgement = self.judge.judge(phrase, "violin.n.01")
            self.assertEqual(judgement.verdict, "sibling", phrase)
            self.assertIn("instrument", judgement.shared, phrase)

    def test_what_a_violin_player_actually_does_survives(self):
        for phrase in ("make living", "give concert", "write song",
                       "create music", "read music", "tune instrument",
                       "use bow", "join orchestra", "take turn"):
            self.assertNotEqual(self.verdict(phrase), "sibling", phrase)

    def test_the_anchor_itself_is_marked_so_it_can_lead(self):
        self.assertEqual(self.verdict("play violin"), "anchor")

    def test_an_ancestor_transfers_and_is_never_a_sibling(self):
        """`string` names violin's own parent; R2 already says it descends."""
        self.assertEqual(self.judge.compare("musical instrument.n.01",
                                            "violin.n.01").verdict, "kin")
        self.assertEqual(self.judge.compare("bowed stringed instrument.n.01",
                                            "violin.n.01").verdict, "kin")

    def test_the_threshold_sits_in_the_gap_that_was_measured(self):
        """Rivals join violin at <=163 descendants, the rest at >=2,764."""
        rival = self.judge.judge("play drum", "violin.n.01")
        self.assertLess(rival.shared_size, SIBLING_LIMIT)
        for phrase in ("use bow", "give concert"):
            other = self.judge.judge(phrase, "violin.n.01")
            self.assertNotEqual(other.verdict, "sibling", phrase)

    def test_an_unrelated_anchor_rules_nothing_out(self):
        """R14 must not fire where the anchor has no class in common."""
        for phrase in ("play drum", "give concert", "make living"):
            self.assertNotEqual(
                self.judge.judge(phrase, "dog.n.01").verdict, "sibling", phrase)


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

    def test_a_compound_is_two_subjects_when_the_ontology_has_no_name_for_it(self):
        """`violin player` is two concepts; `fire truck` is one."""
        for question, expected in (
                ("what can a violin player do", ("violin", "player")),
                ("what does a dog owner need", ("dog", "owner")),
                ("what does a car driver need", ("car", "driver"))):
            self.assertEqual(self.engine.two_subjects(question), expected, question)

    def test_a_compound_that_names_one_concept_is_never_split(self):
        for question in ("what can a fire truck do", "what can a police dog do",
                         "what can a musical instrument do"):
            self.assertEqual(self.engine.two_subjects(question), (None, None),
                             question)

    def test_the_compound_form_reaches_the_same_sense_as_the_possessive(self):
        """`violin player` and `violin's player` are the same question."""
        compound = self.engine.ask("what can a violin player do")
        possessive = self.engine.ask("what can a violin's player do")
        self.assertEqual(compound.role, "musician.n.01")
        self.assertEqual(compound.role, possessive.role)
        self.assertEqual(compound.verdict, "BRIDGED")

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

    def test_the_anchor_filters_the_roles_facts(self):
        """A musician plays every instrument; a violin player does not."""
        answer = self.engine.ask("what can a violin player do")
        kept = {f.object for f in answer.answer.evidence}
        aside = {f.object for f in answer.excluded}
        self.assertTrue(aside)
        self.assertIn("play drum", aside)
        self.assertIn("play piano", aside)
        self.assertNotIn("play drum", kept)
        self.assertIn("play violin", kept)
        self.assertEqual(answer.excluded_class, "musical instrument.n.01")

    def test_the_anchors_own_facts_lead(self):
        answer = self.engine.ask("what can a violin player do")
        self.assertIn(answer.answer.evidence[0].object,
                      {"play violin", "play fiddle"})

    def test_nothing_is_set_aside_when_the_anchor_shares_no_class(self):
        for question in ("what does a dog owner need",
                         "what does a car driver need"):
            self.assertEqual(self.engine.ask(question).excluded, [], question)

    def test_set_aside_facts_are_returned_not_deleted(self):
        payload = self.engine.ask("what can a violin player do").as_dict()
        self.assertIn("excluded", payload)
        self.assertTrue(payload["excluded"])
        self.assertIn("excluded_class", payload)

    def test_it_serialises_for_a_ui(self):
        payload = self.engine.ask("what does a dog's owner need").as_dict()
        for key in ("verdict", "route", "answer", "search", "relation"):
            self.assertIn(key, payload)
        self.assertIn("hops", payload["route"])


if __name__ == "__main__":
    unittest.main()
