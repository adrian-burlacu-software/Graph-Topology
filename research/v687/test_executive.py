"""The executive, with no engine: the cycle keeps a cascade's order,
conditions are read against what earlier operators wrote, and utilities move
only when they are allowed to."""
from __future__ import annotations

import unittest

from research.v687.executive import (ANSWERED, CONTINUE, DECLINED, Executive,
                                     Operator, attempt)


class ExecutiveTests(unittest.TestCase):
    def cascade(self, answers: dict, rate: float = 0.0) -> Executive:
        return Executive([
            Operator("first", attempt(lambda: answers.get("first"))),
            Operator("second", attempt(lambda: answers.get("second"))),
            Operator("last", attempt(lambda: answers.get("last")))],
            rate=rate)

    def test_the_order_written_is_the_order_fired(self):
        memory = {}
        trace = self.cascade({"second": "yes", "last": "no"}).run(memory)
        self.assertEqual(memory["answer"], "yes")
        self.assertEqual(trace.answered_by, "second")
        self.assertEqual([(one.operator, one.outcome) for one in trace.fired],
                         [("first", DECLINED), ("second", ANSWERED)])

    def test_nothing_answering_is_an_impasse(self):
        trace = self.cascade({}).run({})
        self.assertTrue(trace.impasse)
        self.assertEqual(len(trace.fired), 3)

    def test_conditions_read_what_earlier_operators_wrote(self):
        def relation(memory):
            memory["relation"] = "is_a"
            return CONTINUE

        executive = Executive([
            Operator("relation", relation),
            Operator("taxonomy", attempt(lambda: "R1"),
                     proposes=lambda memory: memory.get("relation") == "is_a"),
            Operator("facts", attempt(lambda: "R2"),
                     proposes=lambda memory: memory.get("relation") != "is_a")])
        memory = {}
        trace = executive.run(memory)
        self.assertEqual((memory["answer"], trace.answered_by), ("R1",
                                                                 "taxonomy"))

    def test_utilities_move_only_when_allowed(self):
        still = self.cascade({"last": "x"})
        before = [one.utility for one in still.operators]
        still.reward(still.run({}), 10.0)
        self.assertEqual([one.utility for one in still.operators], before)
        learning = self.cascade({"last": "x"}, rate=0.5)
        trace = learning.run({})
        learning.reward(trace, 10.0)
        last = learning.operators[-1]
        self.assertAlmostEqual(last.utility, 1.0 + 0.5 * (10.0 - 1.0))
        # a declined operator learned nothing
        self.assertEqual(learning.operators[0].utility, 3.0)

    def test_a_learned_utility_changes_what_fires_first(self):
        executive = self.cascade({"first": "a", "last": "c"})
        executive.operators[-1].utility = 99.0
        memory = {}
        executive.run(memory)
        self.assertEqual(memory["answer"], "c")


if __name__ == "__main__":
    unittest.main()
