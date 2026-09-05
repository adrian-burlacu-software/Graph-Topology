import importlib.util
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class PPOTests(unittest.TestCase):
    def test_batched_ppo_stores_old_log_probs_and_checkpoint(self):
        from attention_ppo import run_ppo
        with tempfile.TemporaryDirectory() as root:
            _, trajectories, metrics = run_ppo(episode_count=2, seed=7, checkpoint=f"{root}/ppo.pt",
                                               ppo_epochs=1, minibatch_size=2)
        self.assertTrue(trajectories)
        self.assertTrue(all("old_log_probability" in item and "value" in item for item in trajectories))
        self.assertIn("loss", metrics)


if __name__ == "__main__":
    unittest.main()
