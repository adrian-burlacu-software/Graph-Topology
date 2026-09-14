"""One retrieval: preferences that narrow without emptying, activation, and
the lead a pronoun and a description each need. No engine."""
from __future__ import annotations

import unittest

from research.v688 import retrieval


class RetrievalTests(unittest.TestCase):
    SALIENCE = {"first beagle": 0.36, "second beagle": 0.6, "cat": 0.1}
    ORDER = {"first beagle": 1, "second beagle": 2, "cat": 3}

    def find(self, pool, lead=None, preferences=()):
        return retrieval.retrieve(pool, self.SALIENCE.get, self.ORDER.get,
                                  lead, preferences)

    def test_a_pronoun_needs_only_to_lead(self):
        found = self.find(["first beagle", "second beagle"])
        self.assertEqual(found.chosen, "second beagle")

    def test_a_description_needs_a_clear_lead(self):
        found = self.find(["first beagle", "second beagle"], lead=2.0)
        self.assertIsNone(found.chosen)
        self.assertTrue(found.ambiguous)
        self.assertEqual(found.ranked, ["second beagle", "first beagle"])

    def test_one_left_is_recalled(self):
        found = self.find(["cat"], lead=2.0)
        self.assertEqual(found.chosen, "cat")

    def test_a_preference_narrows_and_never_empties(self):
        dogs = ("a dog", lambda one: "beagle" in one)
        found = self.find(["cat", "first beagle"], preferences=[dogs])
        self.assertEqual((found.chosen, found.kept_by),
                         ("first beagle", ["a dog"]))
        horses = ("a horse", lambda one: "horse" in one)
        found = self.find(["cat", "first beagle"], preferences=[horses])
        self.assertEqual(found.kept_by, [])
        self.assertEqual(found.chosen, "first beagle")

    def test_nothing_active_is_nothing_recalled(self):
        found = retrieval.retrieve(["a", "b"], lambda one: 0.0)
        self.assertIsNone(found.chosen)

    def test_ties_go_to_the_later(self):
        found = retrieval.retrieve(["a", "b"], lambda one: 1.0,
                                   {"a": 1, "b": 2}.get)
        self.assertEqual(found.ranked, ["b", "a"])

    def test_the_latest_is_retrieval_by_recency(self):
        told = [("white", 3), ("gray", 7), ("white", 5)]
        self.assertEqual(retrieval.latest(told, lambda one: one[1]),
                         ("gray", 7))
        self.assertIsNone(retrieval.latest([], lambda one: one))

    def test_base_level_falls_with_age_and_rises_with_use(self):
        once_recent = retrieval.base_level([9.0], 10.0)
        once_old = retrieval.base_level([1.0], 10.0)
        twice = retrieval.base_level([1.0, 9.0], 10.0)
        self.assertGreater(once_recent, once_old)
        self.assertGreater(twice, once_recent)


if __name__ == "__main__":
    unittest.main()
