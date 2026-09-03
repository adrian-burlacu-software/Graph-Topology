import unittest
from types import SimpleNamespace

from v679_attention import (
    AttentionController,
    AttentionState,
    DistilledAttentionPolicy,
)
from v679_semantic_core import Hypothesis


class AttentionControllerTests(unittest.TestCase):
    def test_verified_specific_evidence_wins_arbitration(self):
        controller = AttentionController()
        vague = Hypothesis(
            "en:dog", "related_to", "relation_lookup", 0.9, {}
        )
        verified = Hypothesis(
            "en:dog", "has_part", "relation_lookup", 0.2,
            {"target_terms": ["tail"]},
        )
        decision = controller.arbitrate([
            (3.6, vague, {"success": False, "path": [], "target": None}),
            (1.8, verified, {
                "success": True,
                "direct_proof": True,
                "path": ["has_part"],
                "target": "en:tail",
            }),
        ])
        self.assertEqual(decision["selected_candidate_index"], 1)
        self.assertEqual(decision["outcome"], "verified")
        self.assertEqual(
            decision["candidates"][0]["features"]["provenance"], 1.0
        )

    def test_adversarial_candidates_abstain_without_verified_evidence(self):
        controller = AttentionController()
        candidates = [
            (9.0, Hypothesis("en:dog", "related_to", "relation_lookup", .95, {}),
             {"success": False, "path": [], "target": "en:tail"}),
            (8.0, Hypothesis("en:dog", "has_part", "relation_lookup", .85, {}),
             {"success": False, "path": ["is_a", "has_part"], "target": "en:tail"}),
            (7.0, Hypothesis("en:dog", "is_a", "relation_lookup", .80,
                             {"argument_unverified": True}),
             {"success": False, "path": [], "target": "en:cat"}),
            (6.0, Hypothesis("en:dog", "has_part", "relation_lookup", .20, {}),
             {"success": False, "path": [], "target": "en:tail"}),
            (10.0, Hypothesis("en:dog", "has_property", "relation_lookup", 1.0, {}),
             {"success": False, "path": [], "target": "en:unrelated"}),
        ]
        decision = controller.arbitrate(candidates)
        self.assertEqual(decision["outcome"], "abstain")
        self.assertEqual(decision["reason"], "no_verified_evidence")
        self.assertIsNone(decision["selected_candidate_index"])

    def test_attention_state_decays_and_distilled_policy_replays_targets(self):
        state = AttentionState(decay=.5)
        controller = AttentionController(state=state)
        hypothesis = Hypothesis(
            "en:dog", "has_part", "relation_lookup", .5,
            {"target_terms": ["tail"]},
        )
        controller.begin_turn(hypothesis.subject)
        controller.begin_hypothesis(hypothesis)
        edges = [
            SimpleNamespace(relation="related_to", object="en:dog_house"),
            SimpleNamespace(relation="has_part", object="en:tail"),
        ]
        ranked = controller.select_traversal_targets(hypothesis, (), edges)
        controller.record_traversal_target(ranked[0].relation, ranked[0].object)
        first_activation = state.relation_activation["has_part"]
        controller.begin_turn(hypothesis.subject)
        self.assertEqual(state.relation_activation["has_part"], first_activation * .5)
        learned = DistilledAttentionPolicy.fit(controller.policy_examples)
        self.assertEqual(
            learned.score_traversal({
                "relation": "has_part", "goal_relation_match": 1.0,
                "relation_activation": 0.0, "candidate_activation": 0.0,
                "target_term_match": 1.0, "specificity": 1.0,
            }),
            controller.policy.score_traversal({
                "relation": "has_part", "goal_relation_match": 1.0,
                "relation_activation": 0.0, "candidate_activation": 0.0,
                "target_term_match": 1.0, "specificity": 1.0,
            }),
        )

if __name__ == "__main__":
    unittest.main()
