import unittest
import importlib.util

from attention_dataset import collect_teacher_episodes


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class DaggerTests(unittest.TestCase):
    def test_student_induced_states_are_teacher_labeled(self):
        from attention_dagger import collect_dagger
        from attention_distill import train
        records = collect_dagger(train(collect_teacher_episodes(), epochs=20), rounds=2)
        self.assertTrue(records)
        self.assertTrue(all(step["teacher"]["logits"] for row in records for step in row["trajectory"]))


if __name__ == "__main__":
    unittest.main()
