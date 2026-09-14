"""Truth as evidence, with no store: the arithmetic, and that the thresholds
written as sample sizes read the same said as confidences."""
from __future__ import annotations

import unittest

from research.v687 import truth
from research.v687.truth import FALSE, TRUE, UNKNOWN, Evidence


class EvidenceTests(unittest.TestCase):
    def test_frequency_and_confidence(self):
        birds = truth.counted(28, 29)
        self.assertAlmostEqual(birds.frequency, 28 / 29)
        self.assertAlmostEqual(birds.confidence(), 29 / 30)
        self.assertIsNone(Evidence().frequency)
        self.assertEqual(Evidence().confidence(), 0.0)

    def test_revision_pools(self):
        pooled = truth.counted(3, 4).revise(truth.counted(1, 4))
        self.assertEqual((pooled.positive, pooled.negative), (4.0, 4.0))

    def test_a_sample_size_said_as_a_confidence(self):
        speaks = truth.speaks_at(8)
        self.assertGreaterEqual(truth.counted(0, 8).confidence(), speaks)
        self.assertLess(truth.counted(0, 7).confidence(), speaks)

    def test_three_answers_are_read_off(self):
        speaks = truth.speaks_at(8)
        self.assertEqual(truth.judge(truth.counted(28, 29), 0.6, speaks), TRUE)
        self.assertEqual(truth.judge(truth.counted(0, 24), 0.6, speaks), FALSE)
        # four whales are not evidence
        self.assertEqual(truth.judge(truth.counted(0, 4), 0.6, speaks),
                         UNKNOWN)
        # a family split down the middle settles nothing
        self.assertEqual(truth.judge(truth.counted(5, 10), 0.6, speaks),
                         UNKNOWN)

    def test_deduction_spends_confidence(self):
        self.assertAlmostEqual(truth.deduced(1.0, 0, 0.85), 1.0)
        self.assertAlmostEqual(truth.deduced(1.0, 2, 0.85), 0.85 ** 2)


if __name__ == "__main__":
    unittest.main()
