import importlib.util
import tempfile
import unittest

from attention_dataset import collect_teacher_episodes, write_jsonl


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class DaggerTests(unittest.TestCase):
    def test_each_round_retrains_and_aggregates_teacher_labeled_states(self):
        from attention_dagger import run_dagger
        with tempfile.TemporaryDirectory() as root:
            dataset = f"{root}/teacher.jsonl"; write_jsonl(dataset, collect_teacher_episodes())
            _, aggregate, stats = run_dagger(dataset, rounds=2, epochs=1, seed=7, checkpoint_dir=root)
        self.assertEqual([item["round"] for item in stats], [0, 1])
        self.assertTrue(all(item["states_collected"] > 0 for item in stats))
        self.assertTrue(all(item["teacher_labels"] == item["states_collected"] for item in stats))
        self.assertTrue(all("false_positive_attention_events" in item for item in stats))
        self.assertTrue(all("held_out_no_proof" in item for item in stats))
        self.assertGreater(len(aggregate), len(collect_teacher_episodes()))


if __name__ == "__main__":
    unittest.main()
