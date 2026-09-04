import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "v680"))

from experience import (Experience, ExperienceQuality, ExperienceSource, ExperienceStore,
                        attention_step_experience, chat_trace_experience, worker_batch_experience)


class ExperienceTests(unittest.TestCase):
    def test_replay_preserves_provenance_and_teacher_outcome_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExperienceStore(Path(directory) / "experience.sqlite")
            value = Experience(ExperienceSource.DAGGER, "episode", split="train",
                               teacher_action={"kind": "traverse"},
                               outcome={"kind": "verified_answer"}, quality=ExperienceQuality.TEACHER_LABELLED,
                               provenance={"teacher_version": "frozen"})
            store.append(value)
            replayed = store.sample_episodes()[0][0]
            self.assertEqual(replayed.provenance["teacher_version"], "frozen")
            self.assertEqual(replayed.teacher_action["kind"], "traverse")
            self.assertEqual(replayed.outcome["kind"], "verified_answer")
            store.close()

    def test_evaluation_cannot_be_training_experience(self):
        with self.assertRaises(ValueError):
            Experience(ExperienceSource.ATTENTION_EVAL, "evaluation", split="train")

    def test_chat_and_worker_adapters_keep_distinct_provenance(self):
        chat = chat_trace_experience({"timestamp": 1, "route": {"success": False}, "candidate_evidence": []})
        worker = worker_batch_experience({"worker_id": 2, "batch": 3, "lane": "synonym_structure"})
        self.assertEqual(chat.source, ExperienceSource.CHAT)
        self.assertEqual(chat.quality, ExperienceQuality.UNVERIFIED)
        self.assertEqual(worker.source, ExperienceSource.OFFLINE_WORKER)
        self.assertIn("lane", worker.evidence_acquired[0])

    def test_dagger_adapter_replays_sequential_attention_step(self):
        from attention_dataset import collect_teacher_episodes
        from attention_env import benchmark_episodes
        step = collect_teacher_episodes([benchmark_episodes()[0]])[0]["trajectory"][0]
        item = attention_step_experience(step)
        self.assertEqual(item.source, ExperienceSource.DAGGER)
        self.assertIsNotNone(item.state)
        self.assertIsNotNone(item.next_state)
        self.assertIsNotNone(item.teacher_action)
        self.assertIsNotNone(item.outcome)


if __name__ == "__main__":
    unittest.main()
