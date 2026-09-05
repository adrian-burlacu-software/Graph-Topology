import importlib.util
import unittest

from attention_benchmark import decision_boundary_episodes


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class EvaluationTests(unittest.TestCase):
    def test_rollout_confusion_matrix_reconciles_with_decision_count(self):
        from attention_evaluate import evaluate_rollouts
        from attention_student import NeuralAttentionPolicy
        metrics = evaluate_rollouts(decision_boundary_episodes(5), NeuralAttentionPolicy())
        for report in metrics.values():
            self.assertEqual(sum(sum(row.values()) for row in report["confusion_matrix"].values()),
                             report["decisions"])
            self.assertIn("correct_candidate_top3", report)
            self.assertIn("premature_stop", report)


if __name__ == "__main__":
    unittest.main()
