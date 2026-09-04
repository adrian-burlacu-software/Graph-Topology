import importlib.util
import unittest

from attention_dataset import collect_teacher_episodes
from attention_types import AttentionObservation, audit_model_input
from attention_ablation import ablate


class SchemaTests(unittest.TestCase):
    def test_observation_rejects_oracle_leakage(self):
        state = collect_teacher_episodes()[0]["trajectory"][0]["state"]
        state["proof_target"] = "leak"
        with self.assertRaisesRegex(ValueError, "oracle fields"):
            AttentionObservation.from_dict(state)

    def test_no_history_removes_all_history_derived_inputs(self):
        records = ablate(collect_teacher_episodes(), "no_history")
        for step in [s for episode in records for s in episode["trajectory"]]:
            state = step["state"]
            self.assertFalse(state["visited_nodes"]); self.assertFalse(state["visited_relations"])
            self.assertFalse(state["attention_history"]); self.assertFalse(state["relation_activation"])
            self.assertFalse(state["candidate_activation"])
            self.assertTrue(all(not item["already_visited"] and not item["relation_activation"]
                                and not item["candidate_activation"] for item in state["candidate_features"]))

    def test_recursive_audit_rejects_teacher_future_and_oracle_metadata(self):
        for forbidden in ("teacher_action", "proof_exists", "future_state", "future_reward", "terminal_outcome"):
            with self.subTest(forbidden=forbidden), self.assertRaisesRegex(ValueError, "forbidden"):
                audit_model_input({"candidate": {forbidden: True}})


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class DistillationTests(unittest.TestCase):
    def test_distillation_consumes_direct_candidate_feature_schema(self):
        from attention_distill import train_distillation
        records = collect_teacher_episodes()
        model, _ = train_distillation(records, epochs=2, seed=7)
        self.assertTrue(model.state_dict())


if __name__ == "__main__":
    unittest.main()
