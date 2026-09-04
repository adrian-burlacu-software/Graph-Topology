import unittest
import sqlite3
import tempfile
from pathlib import Path

from attention_env import AttentionEnv, benchmark_episodes, episodes_from_database
from attention_types import AttentionAction, AttentionActionKind


class EnvironmentTests(unittest.TestCase):
    def test_oracle_rewards_correct_abstention_without_exposing_oracle_state(self):
        env = AttentionEnv(benchmark_episodes()[-1])
        state = env.reset()
        self.assertNotIn("proof_target", state.as_dict())
        _, reward, done, info = env.step(AttentionAction(AttentionActionKind.ABSTAIN))
        self.assertTrue(done); self.assertEqual(info["terminal_outcome"], "no_verified_evidence")
        self.assertEqual(reward, 4.0)

    def test_frozen_sqlite_graph_becomes_bounded_direct_proof_episode(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "graph.sqlite"
            con = sqlite3.connect(path)
            con.execute("CREATE TABLE edges(subject,relation,object)")
            con.executemany("INSERT INTO edges VALUES(?,?,?)", [
                ("en:dog", "has_part", "en:tail"),
                ("en:dog", "related_to", "en:animal"),
            ])
            con.commit(); con.close()
            episodes = episodes_from_database(path)
        self.assertEqual(len(episodes), 2)
        self.assertTrue(all(any(c.verified for c in episode["candidates"]) for episode in episodes))


if __name__ == "__main__":
    unittest.main()
