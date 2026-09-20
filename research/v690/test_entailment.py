"""EntailmentBank: the tree built by the executive, and the score that
says whether it found anything.

The data-dependent tests skip themselves when EntailmentBank has not been
pulled in, as the norms suites do.
"""
from __future__ import annotations

import unittest

from research.v690 import entailment

HAVE = entailment.DATASET.exists()
requires_data = unittest.skipUnless(HAVE, "EntailmentBank not downloaded")


class ProofTests(unittest.TestCase):
    """Reading a proof, and naming its steps by what is beneath them."""

    def test_steps_are_parsed(self):
        steps = entailment.steps_of(
            "sent1 & sent2 -> int1: a star is a source of light; "
            "int1 & sent3 -> hypothesis; ")
        self.assertEqual(steps, [(["sent1", "sent2"], "int1"),
                                 (["int1", "sent3"], "hypothesis")])

    def test_a_step_is_named_by_the_sentences_beneath_it(self):
        """So two trees that agree in structure score the same however
        their intermediate conclusions happen to be numbered."""
        one = entailment.signature("sent1 & sent2 -> int1; "
                                   "int1 & sent3 -> hypothesis; ")
        other = entailment.signature("sent2 & sent1 -> intA; "
                                     "sent3 & intA -> hypothesis; ")
        self.assertEqual(one, other)
        self.assertEqual(one, {frozenset({"sent1", "sent2"}),
                               frozenset({"sent1", "sent2", "sent3"})})

    def test_the_root_step_is_dropped(self):
        """Task 1 needs every sentence it gives, so the root is always all
        of them and any method gets it for nothing."""
        tree = entailment.signature("sent1 & sent2 -> int1; "
                                    "int1 & sent3 -> hypothesis; ")
        self.assertEqual(entailment.inner(tree),
                         {frozenset({"sent1", "sent2"})})

    def test_one_step_recovers_no_structure_at_all(self):
        tree = entailment.signature(
            entailment.one_step(["sent1", "sent2", "sent3"]))
        self.assertEqual(entailment.inner(tree), set())


class OperatorTests(unittest.TestCase):
    """The four ways two nodes belong together."""

    SENTENCES = {"sent1": "a planet rotating causes cycles of day and night "
                          "on that planet",
                 "sent2": "earth is a kind of planet",
                 "sent3": "the earth rotates on its tilted axis"}
    HYPOTHESIS = ("earth rotating on its axis causes the cycle of day and "
                  "night on earth")

    def test_taxonomy_is_split_into_thing_and_class(self):
        thing, klass = entailment.taxonomy("earth is a kind of planet")
        self.assertEqual(thing, {"earth"})
        self.assertEqual(klass, {"planet"})
        self.assertEqual(entailment.taxonomy("the earth rotates"),
                         (None, None))

    def test_a_taxonomic_sentence_grounds_the_thing_not_the_class(self):
        """`earth is a kind of planet` shares `planet` with the general
        rule and `earth` with the specific fact, and it is the specific
        fact it combines with."""
        proof, _ = entailment.build(self.SENTENCES, self.HYPOTHESIS)
        first = entailment.steps_of(proof)[0][0]
        self.assertEqual(set(first), {"sent2", "sent3"})

    def test_every_sentence_ends_up_in_the_tree(self):
        proof, _ = entailment.build(self.SENTENCES, self.HYPOTHESIS)
        tree = entailment.signature(proof)
        self.assertEqual(max(tree, key=len), frozenset(self.SENTENCES))

    def test_the_operators_compete(self):
        """What bAbI never supplied: more than one operator proposing for
        the same step, so a conflict set is a choice and not a pipeline."""
        _, trace = entailment.build(self.SENTENCES, self.HYPOTHESIS)
        self.assertIsNotNone(trace)
        widest = max(len(step.candidates or ()) for step in trace.fired)
        self.assertGreater(widest, 1)

    def test_a_repeating_operator_stops_when_its_pattern_is_gone(self):
        """A condition that outlives its action fires for ever (V2). The
        run has to end on its own."""
        proof, trace = entailment.build(self.SENTENCES, self.HYPOTHESIS)
        self.assertTrue(proof)
        self.assertLess(len(trace.fired), 20)


@requires_data
class DataTests(unittest.TestCase):
    """The published counts, and the bar the executive has to clear."""

    @classmethod
    def setUpClass(cls):
        cls.rows = entailment.load("task_1", "dev")

    def test_the_split_is_the_published_size(self):
        self.assertEqual(len(self.rows), 187)

    def test_task_one_gives_exactly_the_sentences_the_proof_needs(self):
        """Which is what makes it a test of control and not of knowledge."""
        for row in self.rows[:40]:
            with self.subTest(row=row["id"]):
                self.assertEqual(row["meta"]["distractors"], [])

    def test_the_executive_beats_both_baselines_on_inner_steps(self):
        """The number that means anything: the root step is free, so this
        is structure actually found. Chaining left to right scores 10.6%
        and one step for everything scores nothing."""
        _, chained = entailment.measure(self.rows, entailment.chain)
        _, flat = entailment.measure(self.rows, entailment.one_step)
        _, mine = entailment.measure(self.rows, entailment.by_executive)
        self.assertEqual(flat.f1, 0.0)
        self.assertGreater(mine.f1, chained.f1)
        self.assertGreater(mine.f1, 0.15)

    def test_depth_is_where_it_fails(self):
        """Recorded rather than asserted away: deep trees are long binary
        chains and a local choice compounds. Nothing reaches depth 4."""
        whole, _ = entailment.measure(self.rows, entailment.by_executive)
        deep = sum(count for depth, (count, _) in whole.by_depth.items()
                   if depth >= 4)
        self.assertEqual(deep, 0)


if __name__ == "__main__":
    unittest.main()
