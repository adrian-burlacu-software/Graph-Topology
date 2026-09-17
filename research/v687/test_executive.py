"""The executive, with no engine: the cycle keeps a cascade's order,
conditions are read against what earlier operators wrote, and utilities move
only when they are allowed to."""
from __future__ import annotations

import unittest

from research.v687.executive import (ANSWERED, CONTINUE, DECLINED, Executive,
                                     Operator, Working, attempt)


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


class WorkingTests(unittest.TestCase):
    """Working memory as a stack of goals (E1): one frame deep it is the dict
    it replaced, and a subgoal reads what is beneath it, writes only its own
    frame, and hands back what it wrote."""

    def cascade(self) -> Executive:
        def relation(memory):
            memory["relation"] = "is_a"
            return CONTINUE
        return Executive([
            Operator("relation", relation),
            Operator("taxonomy", attempt(lambda: "R1"),
                     proposes=lambda memory: memory.get("relation") == "is_a"),
            Operator("facts", attempt(lambda: "R2"))])

    def test_one_frame_deep_it_is_the_dict_it_replaced(self):
        plain, working = {"why": False}, Working({"why": False}, goal="ask")
        before = self.cascade().run(plain)
        after = self.cascade().run(working)
        self.assertEqual(after.as_dict(), before.as_dict())
        self.assertEqual(dict(working), plain)
        self.assertEqual((after.goal, after.depth), ("ask", 1))

    def test_a_subgoal_reads_beneath_and_writes_only_its_own_frame(self):
        memory = Working({"referent": "rex", "relation": "is_a"}, goal="ask")
        memory.push("which sense", relation="has_part")
        self.assertEqual(memory["referent"], "rex")          # read beneath
        self.assertEqual(memory["relation"], "has_part")     # its own, first
        self.assertIn("referent", memory)
        self.assertEqual(memory.get("missing", "none"), "none")
        memory["sense"] = "dog.n.01"
        self.assertEqual((memory.depth, memory.open_goals),
                         (2, ["ask", "which sense"]))
        result = memory.pop()
        self.assertEqual(result, {"relation": "has_part",
                                  "sense": "dog.n.01"})
        # the goal beneath is as it was: nothing the subgoal wrote leaked
        self.assertEqual(dict(memory), {"referent": "rex",
                                        "relation": "is_a"})
        self.assertEqual((memory.goal, memory.depth), ("ask", 1))

    def test_the_outermost_goal_cannot_be_popped(self):
        with self.assertRaises(IndexError):
            Working(goal="ask").pop()

    def test_a_subgoal_that_fails_is_still_closed(self):
        memory = Working({"referent": "rex"}, goal="ask")
        with self.assertRaises(RuntimeError):
            with memory.subgoal("which sense"):
                memory["sense"] = "half-written"
                raise RuntimeError("the subgoal broke")
        self.assertEqual((memory.goal, dict(memory)),
                         ("ask", {"referent": "rex"}))

    def test_an_executive_runs_on_a_subgoal_and_returns_its_answer(self):
        memory = Working({"relation": "is_a"}, goal="ask")
        with memory.subgoal("resolve the relation") as result:
            trace = self.cascade().run(memory)
        self.assertEqual((trace.goal, trace.depth, trace.answered_by),
                         ("resolve the relation", 2, "taxonomy"))
        self.assertEqual(result["answer"], "R1")
        self.assertNotIn("answer", memory)


if __name__ == "__main__":
    unittest.main()
