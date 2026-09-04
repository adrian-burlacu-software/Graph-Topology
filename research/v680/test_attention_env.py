import sqlite3
import tempfile
import unittest
from pathlib import Path

from attention_env import AttentionEnv, benchmark_episodes, episodes_from_database
from attention_types import AttentionAction, AttentionActionKind


class EnvironmentTests(unittest.TestCase):
    def test_branching_traversal_changes_focus_and_candidate_set(self):
        env = AttentionEnv(benchmark_episodes()[0])
        state = env.reset()
        next_state, _, done, _ = env.step(AttentionAction(AttentionActionKind.TRAVERSE, 0))
        self.assertFalse(done); self.assertNotEqual(state.current_focus, next_state.current_focus)
        self.assertNotEqual([item.target for item in state.candidate_features],
                            [item.target for item in next_state.candidate_features])
        self.assertEqual(next_state.visited_nodes, ["en:animal"])

    def test_oracle_never_appears_in_policy_observation(self):
        env = AttentionEnv(benchmark_episodes()[-1])
        state = env.reset()
        self.assertFalse({"proof_target", "oracle", "valid_paths", "terminal_answer"} & state.as_dict().keys())
        _, reward, done, info = env.step(AttentionAction(AttentionActionKind.ABSTAIN))
        self.assertTrue(done); self.assertEqual(info["terminal_outcome"], "no_verified_evidence")
        self.assertEqual(reward, 4.0)

    def test_frozen_sqlite_graph_becomes_bounded_episode(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "graph.sqlite"
            connection = sqlite3.connect(path)
            connection.execute("CREATE TABLE edges(subject,relation,object)")
            connection.executemany("INSERT INTO edges VALUES(?,?,?)", [
                ("en:dog", "has_part", "en:tail"), ("en:dog", "related_to", "en:animal")])
            connection.commit(); connection.close()
            episodes = episodes_from_database(path)
        self.assertEqual(len(episodes), 2)
        self.assertTrue(all(episode["nodes"] for episode in episodes))


if __name__ == "__main__":
    unittest.main()
