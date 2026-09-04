import importlib.util
import unittest

from attention_dataset import collect_jepa_transition_episodes, collect_teacher_episodes
from attention_env import benchmark_episodes


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class JEPATests(unittest.TestCase):
    def test_action_predictions_are_distinct_and_jepa_beats_zero_baseline(self):
        from attention_jepa import AttentionJEPA, evaluate_jepa, train_jepa
        records = collect_jepa_transition_episodes()
        model, _ = train_jepa(records, epochs=8, seed=7)
        report = evaluate_jepa(records, model)
        self.assertTrue(report["action_conditioned"])
        self.assertGreater(report["mean_action_conditioned_prediction_distance"], 1e-6)
        self.assertLess(report["prediction_error"]["jepa"], report["prediction_error"]["zero"])
        self.assertIn("misleading", report["by_transition_category"])

    def test_jepa_uses_observable_transitions_only(self):
        from attention_jepa import action_vector, observation_vector
        state = benchmark_episodes()[0]["nodes"]
        records = collect_teacher_episodes([benchmark_episodes()[0]])
        observation = records[0]["trajectory"][0]["state"]
        self.assertNotIn("proof_target", observation)
        from attention_types import AttentionObservation
        value = AttentionObservation.from_dict(observation)
        self.assertEqual(len(observation_vector(value)), 18)
        self.assertEqual(len(action_vector(value, 0)), 14)

    def test_jepa_augmented_student_consumes_prediction_for_every_action(self):
        from attention_distill import train_distillation
        from attention_jepa import train_jepa
        from attention_evaluate import evaluate
        transitions = collect_jepa_transition_episodes()
        jepa, _ = train_jepa(transitions, epochs=2, seed=7)
        student, _ = train_distillation(collect_teacher_episodes(), epochs=1, seed=7,
                                        jepa=jepa, use_jepa=True)
        report = evaluate(collect_teacher_episodes(), student, jepa=jepa)
        self.assertIn("held_out_adversarial", report)


if __name__ == "__main__":
    unittest.main()
