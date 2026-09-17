"""Counterfactual credit, scored on made-up runs: what an operator was
worth is measured against the others on the same question.

Run: python -m unittest research.v688.test_counterfactual -v
"""
from __future__ import annotations

import unittest

from research.v687.executive import Executive, Operator, attempt, record
from research.v688 import counterfactual


def node(outcome: str, did=(), without=()) -> dict:
    return {"suppressed": [list(one) for one in without],
            "outcome": outcome, "did": [list(one) for one in did]}


NORMS, FACTS = ("v687", "the norms"), ("v687", "the fact graph")


class ScoreTests(unittest.TestCase):

    def test_the_marginal_is_with_less_without_on_the_same_question(self):
        # the norms deny a penguin flies; without them the fact graph says
        # yes -- and the claim is false
        rows = [{"key": "penguin", "nodes": [
            node("denied", [NORMS]),
            node("verified", [FACTS], without=[NORMS])]}]
        found = counterfactual.score(rows, {"penguin": "negative"})
        marginal = {one["operator"]: one for one in found["marginal"]}
        self.assertEqual(marginal["the norms"]["mean_marginal"], 2.0)
        self.assertEqual(marginal["the norms"]["helped"], 1)

    def test_a_passed_over_operator_is_compared_where_it_would_decide(self):
        rows = [{"key": "penguin", "nodes": [
            node("denied", [NORMS]),
            node("verified", [FACTS], without=[NORMS])]}]
        found = counterfactual.score(rows, {"penguin": "negative"})
        self.assertEqual(found["competition"], [
            {"executive": "v687", "chosen": "the norms",
             "instead": "the fact graph", "questions": 1,
             "chosen_mean": 1.0, "instead_mean": -1.0}])

    def test_nothing_answering_without_it_is_needed_not_a_loss(self):
        rows = [{"key": "dog", "nodes": [
            node("verified", [FACTS]),
            {"suppressed": [list(FACTS)], "error": "RuntimeError"}]}]
        found = counterfactual.score(rows, {"dog": "positive"})
        self.assertEqual(found["marginal"][0]["needed"], 1)
        self.assertEqual(found["competition"], [])

    def test_the_oracle_is_the_best_explored_on_each_question(self):
        rows = [{"key": "tomato", "nodes": [
            node("unknown", [FACTS]),
            node("verified", [FACTS, ("v687", "a folk category")],
                 without=[("v687", "comparative")])]},
                {"key": "gold unknown", "nodes": [node("verified")]}]
        found = counterfactual.score(rows, {"tomato": "positive"})
        self.assertEqual((found["questions"], found["base_mean"],
                          found["oracle_mean"]), (1, 0.0, 1.0))


class ExploreTests(unittest.TestCase):
    """The same question asked again with what did something suppressed."""

    class Engine:
        def ask(self, question: str) -> dict:
            memory: dict = {}
            trace = Executive([
                Operator("the norms", attempt(lambda: "denied")),
                Operator("the fact graph", attempt(lambda: "verified"))],
                name="v687").run(memory)
            if trace.impasse:
                raise RuntimeError("nothing answered")
            return {"verdict": memory["answer"].upper(),
                    "executed": [record("v687", trace)]}

    def test_each_one_that_did_something_is_suppressed_in_turn(self):
        nodes = counterfactual.explore(self.Engine(), "can a penguin fly")
        self.assertEqual([one["suppressed"] for one in nodes],
                         [[], [NORMS], [FACTS, NORMS]])
        self.assertEqual(nodes[1]["did"], [FACTS])
        self.assertIn("error", nodes[2])


if __name__ == "__main__":
    unittest.main()
