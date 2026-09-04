import unittest
import importlib.util

from attention_dataset import collect_teacher_episodes


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class PPOTests(unittest.TestCase):
    def test_teacher_regularized_ppo_produces_a_usable_policy(self):
        from attention_distill import train
        from attention_ppo import train_ppo
        model = train(collect_teacher_episodes(), epochs=10)
        self.assertTrue(train_ppo(model, episodes=1, beta=.5).state_dict())


if __name__ == "__main__":
    unittest.main()
