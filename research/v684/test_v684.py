"""Regression suite. Run: python -m unittest research.v684.test_v684 -v

Tests that need the built store skip themselves when it is absent, so the suite
runs on a fresh clone before `python -m research.v684.build`.
"""
from __future__ import annotations

import unittest

from research.v684 import build, rules
from research.v684.language import Parser
from research.v684.reason import Reasoner

STORE = build.DEFAULT_STORE
HAVE_STORE = STORE.exists()
requires_store = unittest.skipUnless(HAVE_STORE, f"no store at {STORE}")


class RuleTests(unittest.TestCase):
    def test_related_to_never_participates(self):
        """R7. It is 1,678,150 of 3.9M edges and says only 'co-occurs'."""
        self.assertIn("related_to", rules.GATED)
        self.assertFalse(rules.inheritable("related_to"))

    def test_inheritable_and_not_are_disjoint_and_reasoned(self):
        """R2. Every non-inheritable relation states why."""
        self.assertFalse(rules.INHERITABLE & set(rules.NOT_INHERITABLE))
        for relation in rules.NOT_INHERITABLE:
            self.assertTrue(rules.why_not_inheritable(relation))
            self.assertFalse(rules.inheritable(relation))

    def test_made_of_does_not_descend(self):
        """A chair is furniture; furniture is not therefore made of wood."""
        self.assertFalse(rules.inheritable("made_of"))
        self.assertTrue(rules.inheritable("capable_of"))

    def test_confidence_decays_with_distance(self):
        """R5."""
        self.assertAlmostEqual(rules.confidence_at(1.0, 0), 1.0)
        self.assertLess(rules.confidence_at(1.0, 5), rules.confidence_at(1.0, 1))
        self.assertAlmostEqual(rules.confidence_at(1.0, 2), rules.DECAY ** 2)

    def test_negation_blocks_its_positive(self):
        """R3."""
        self.assertTrue(rules.blocks("not_capable_of", "capable_of"))
        self.assertTrue(rules.blocks("capable_of", "not_capable_of"))
        self.assertFalse(rules.blocks("capable_of", "at_location"))

    def test_relation_families_are_symmetric(self):
        """R9. has_a and has_part must answer for each other, both ways."""
        self.assertEqual(rules.family("has_a"), rules.family("has_part"))
        self.assertIn("has_part", rules.family("has_a"))
        self.assertEqual(rules.family("capable_of"), ["capable_of"])

    def test_part_of_is_not_in_the_has_part_family(self):
        """It is the inverse, not a synonym; conflating them reverses facts."""
        self.assertNotIn("part_of", rules.family("has_part"))

    def test_every_rule_has_text_for_the_ui(self):
        for key in ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"):
            self.assertIn(key, rules.RULE_TEXT)


class ParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = Parser()

    def test_polar_question_yields_subject_relation_target(self):
        parse = self.parser.parse("can a dog fall into a hole")
        self.assertEqual(parse.subject, "dog")
        self.assertEqual(parse.relation, "capable_of")
        self.assertIn("fall", parse.target)
        self.assertTrue(parse.polar)

    def test_copula_plus_determiner_is_a_taxonomy_question(self):
        """`is a dog an animal` asks about kinds, not properties."""
        parse = self.parser.parse("is a dog an animal")
        self.assertEqual(parse.relation, "is_a")
        self.assertEqual(parse.target, "animal")

    def test_copula_without_determiner_stays_a_property_question(self):
        parse = self.parser.parse("is a dog friendly")
        self.assertEqual(parse.relation, "has_property")

    def test_auxiliary_verb_is_stripped_from_the_target(self):
        """`does a dog have a tail` is about a tail, not about having."""
        parse = self.parser.parse("does a dog have a tail")
        self.assertEqual(parse.relation, "has_part")
        self.assertNotIn("have", (parse.target or "").split())

    def test_open_question_has_no_target(self):
        parse = self.parser.parse("what can a violin do")
        self.assertEqual(parse.subject, "violin")
        self.assertIsNone(parse.target)
        self.assertFalse(parse.polar)

    def test_relation_cues_cover_the_documented_shapes(self):
        for question, relation in (
            ("what is a violin made of", "made_of"),
            ("what is a hammer used for", "used_for"),
            ("where do you find a hammer", "at_location"),
            ("what does a dog want", "desires"),
        ):
            self.assertEqual(self.parser.parse(question).relation, relation, question)

    def test_matcher_requires_real_overlap(self):
        matches = self.parser.matcher()
        self.assertTrue(matches("fall into hole", "fall into a hole"))
        self.assertFalse(matches("fall in love", "fall into a hole"))
        self.assertTrue(matches("anything", None))

    def test_unparseable_input_does_not_raise(self):
        parse = self.parser.parse("???")
        self.assertIsNone(parse.subject)
        self.assertTrue(parse.note)


@requires_store
class ReasonerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reasoner = Reasoner(STORE)
        cls.parser = Parser()
        # staticmethod: a bare function on a class binds as a method,
        # which would pass `self` as the first argument to the matcher.
        cls.match = staticmethod(cls.parser.matcher())

    @classmethod
    def tearDownClass(cls):
        cls.reasoner.close()

    def test_word_resolves_to_several_senses_eponymous_first(self):
        """R6. `dog` names eight synsets; dog.n.01 must lead."""
        senses = self.reasoner.senses_of("dog")
        self.assertGreater(len(senses), 1)
        self.assertEqual(senses[0]["id"], "dog.n.01")

    def test_facts_landed_on_the_eponymous_sense(self):
        """The build bug this guards: all of dog's facts went to andiron.n.01."""
        self.assertGreater(self.reasoner.fact_count("dog.n.01"), 100)

    def test_taxonomy_walk_reaches_animal_from_dog(self):
        answer = self.reasoner.classify("dog.n.01", "animal")
        self.assertEqual(answer.verdict, "VERIFIED")
        self.assertEqual(answer.evidence[0].distance, 2)

    def test_absent_classification_is_unknown_not_false(self):
        """R8. The ontology does not assert negatives by omission."""
        answer = self.reasoner.classify("dog.n.01", "vehicle")
        self.assertEqual(answer.verdict, "UNKNOWN")
        self.assertIn("Absent, not false", answer.note)

    def test_inheritance_finds_a_fact_from_an_ancestor(self):
        answer = self.reasoner.verify("dog.n.01", "capable_of",
                                      "fall into a hole", self.match)
        self.assertEqual(answer.verdict, "VERIFIED")
        self.assertGreater(answer.evidence[0].distance, 0)

    def test_inherited_confidence_is_below_direct(self):
        """R5, observable end to end."""
        answer = self.reasoner.verify("dog.n.01", "capable_of",
                                      "fall into a hole", self.match)
        self.assertLess(answer.evidence[0].confidence, 0.95)

    def test_relation_family_finds_has_a_when_asked_has_part(self):
        """R9. Ascent++ files the tail under has_a, WordNet under has_part."""
        answer = self.reasoner.verify("dog.n.01", "has_part", "tail", self.match)
        self.assertEqual(answer.verdict, "VERIFIED")

    def test_non_inheritable_relation_stops_the_walk(self):
        """R2. made_of must not climb the taxonomy."""
        answer = self.reasoner.verify("dog.n.01", "made_of", "wood", self.match)
        stops = [s for s in answer.steps if s.kind == "stop"]
        self.assertTrue(stops)
        self.assertEqual(stops[0].rule, "R2")

    def test_every_step_names_the_rule_that_produced_it(self):
        answer = self.reasoner.verify("dog.n.01", "capable_of",
                                      "fall into a hole", self.match)
        self.assertTrue(answer.steps)
        for step in answer.steps:
            self.assertIn(step.rule, rules.RULE_TEXT, step.detail)

    def test_ascent_terminates_on_the_acyclic_taxonomy(self):
        visited = [node for node, _, _ in self.reasoner.ascend("dog.n.01")]
        self.assertEqual(len(visited), len(set(visited)))
        self.assertIn("entity.n.01", visited)

    def test_describe_ranks_direct_facts_above_inherited(self):
        answer = self.reasoner.describe("violin.n.01", "capable_of")
        self.assertEqual(answer.verdict, "LISTING")
        self.assertTrue(answer.evidence)
        self.assertEqual(answer.evidence[0].distance, 0)

    def test_describe_never_repeats_a_fact_from_higher_up(self):
        """R4. The nearest statement wins; duplicates are dropped."""
        answer = self.reasoner.describe("dog.n.01", None)
        seen = [(f.relation, f.object.lower()) for f in answer.evidence]
        self.assertEqual(len(seen), len(set(seen)))

    def test_answer_serialises_for_the_ui(self):
        payload = self.reasoner.classify("dog.n.01", "animal").as_dict()
        for key in ("verdict", "steps", "evidence", "chain", "rules"):
            self.assertIn(key, payload)


@requires_store
class EngineTests(unittest.TestCase):
    """The path the browser actually takes."""

    @classmethod
    def setUpClass(cls):
        from research.v684.server import Engine
        cls.engine = Engine(STORE)

    @classmethod
    def tearDownClass(cls):
        cls.engine.reasoner.close()

    def test_end_to_end_question(self):
        payload = self.engine.ask("is a dog an animal")
        self.assertEqual(payload["verdict"], "VERIFIED")
        self.assertTrue(payload["steps"])
        self.assertTrue(payload["senses"])

    def test_unknown_word_is_reported_not_raised(self):
        payload = self.engine.ask("is a zzzqqq an animal")
        self.assertIn(payload["verdict"], ("UNKNOWN_WORD", "UNPARSED"))

    def test_sense_can_be_overridden(self):
        payload = self.engine.ask("what can a dog do", concept="cad.n.01")
        self.assertEqual(payload["concept"], "cad.n.01")


if __name__ == "__main__":
    unittest.main()
