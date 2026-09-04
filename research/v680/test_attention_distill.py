import importlib.util
import unittest

from attention_dataset import collect_teacher_episodes
from attention_types import AttentionObservation


class SchemaTests(unittest.TestCase):
    def test_observation_rejects_oracle_leakage(self):
        state = collect_teacher_episodes()[0]["trajectory"][0]["state"]
        state["proof_target"] = "leak"
        with self.assertRaisesRegex(ValueError, "oracle fields"):
            AttentionObservation.from_dict(state)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class DistillationTests(unittest.TestCase):
    def test_distillation_consumes_direct_candidate_feature_schema(self):
        from attention_distill import train_distillation
        records = collect_teacher_episodes()
        model, _ = train_distillation(records, epochs=2, seed=7)
        self.assertTrue(model.state_dict())


if __name__ == "__main__":
    unittest.main()
