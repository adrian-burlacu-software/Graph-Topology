import unittest

from v679_attention import AttentionController
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
            decision["candidates"][0]["components"]["provenance"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
