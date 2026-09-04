import unittest
import importlib.util

from attention_dataset import collect_teacher_episodes


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class DistillationTests(unittest.TestCase):
    def test_student_reproduces_teacher_on_ordinary_and_adversarial_states(self):
        from attention_distill import train
        from attention_evaluate import evaluate
        records = collect_teacher_episodes()
        report = evaluate(records, train(records, epochs=120))
        self.assertGreaterEqual(report["ordinary"]["teacher_action_accuracy"], .95)
        self.assertGreaterEqual(report["adversarial"]["abstention_accuracy"], .95)


if __name__ == "__main__":
    unittest.main()
