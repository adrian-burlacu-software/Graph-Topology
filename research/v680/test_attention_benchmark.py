import unittest

from attention_benchmark import CATEGORIES, decision_boundary_episodes
from attention_dataset import collect_teacher_episodes
from attention_types import DATASET_VERSION, JEPA_VERSION, STUDENT_VERSION, TEACHER_VERSION


class DecisionBoundaryBenchmarkTests(unittest.TestCase):
    def test_each_boundary_category_has_one_hundred_matched_structural_cases(self):
        episodes = decision_boundary_episodes()
        self.assertEqual(len(episodes), len(CATEGORIES) * 100)
        for category in CATEGORIES:
            cases = [item for item in episodes if item["category"] == category]
            self.assertEqual(len(cases), 100)
            self.assertEqual({item["partition"] for item in cases}, {"train", "validation", "heldout"})

    def test_frozen_teacher_matches_initial_boundary_action_contract(self):
        episodes = decision_boundary_episodes(samples_per_category=10)
        traces = collect_teacher_episodes(episodes)
        for episode, trace in zip(episodes, traces):
            step = trace["trajectory"][0]
            count = len(step["state"]["candidate_features"])
            action = ("traverse" if step["teacher"]["selected_action"] < count else
                      "stop" if step["teacher"]["selected_action"] == count else "abstain")
            self.assertEqual(action, episode["expected_initial_action"], episode["episode_id"])
            self.assertEqual(step["teacher_version"], TEACHER_VERSION)
            self.assertEqual(step["dataset_version"], DATASET_VERSION)
            self.assertEqual(step["student_version"], STUDENT_VERSION)
            self.assertEqual(step["jepa_version"], JEPA_VERSION)


if __name__ == "__main__":
    unittest.main()
