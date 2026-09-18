"""The proving ground's gold: drawn only from what is held, and never from
XCSLB's zeros.

Run: python -m unittest research.v688.test_depth -v
"""
from __future__ import annotations

import json
import unittest

from research.v687 import corpora
from research.v688 import depth


class ProvingTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.asked = depth.proving()
        cls.holders = corpora.xcslb_holders()
        cls.screened = set()
        with (corpora.XCSLB_DIR / "comps_screened.jsonl").open(
                encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                cls.screened.add(f"can {row['prefix_unacceptable']} "
                                 f"{row['property'].split()[-1]}")

    def test_it_is_balanced(self):
        gold = [one for _, one in self.asked]
        self.assertEqual(gold.count("positive"), gold.count("negative"))

    def test_a_negative_is_one_a_judge_screened_false(self):
        for utterance, gold in self.asked:
            if gold == "negative":
                self.assertIn(utterance, self.screened)

    def test_a_positive_is_one_the_matrix_lists(self):
        for utterance, gold in self.asked:
            if gold == "positive":
                words = utterance.split()
                concept, verb = " ".join(words[2:-1]), words[-1]
                self.assertIn(concept, self.holders[f"can {verb}"])


if __name__ == "__main__":
    unittest.main()
