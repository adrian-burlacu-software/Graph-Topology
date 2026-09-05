"""Exercise V679 against adversarial ambiguity and emit a distilled policy."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from v679_attention import AttentionController, DistilledAttentionPolicy
from v679_semantic_core import Hypothesis


def adversarial_candidates():
    return [
        ("strong_association", Hypothesis("en:dog", "related_to", "relation_lookup", .95, {}),
         {"success": False, "path": [], "target": "en:tail"}),
        ("plausible_derived_path", Hypothesis("en:dog", "has_part", "relation_lookup", .85, {}),
         {"success": False, "path": ["is_a", "has_part"], "target": "en:tail"}),
        ("direct_contradiction", Hypothesis("en:dog", "is_a", "relation_lookup", .80,
                                             {"argument_unverified": True}),
         {"success": False, "path": [], "target": "en:cat"}),
        ("weak_direct_evidence", Hypothesis("en:dog", "has_part", "relation_lookup", .20, {}),
         {"success": False, "path": [], "target": "en:tail"}),
        ("unrelated_lexical_match", Hypothesis("en:dog", "has_property", "relation_lookup", 1.0, {}),
         {"success": False, "path": [], "target": "en:unrelated"}),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark V679 temporal attention against adversarial ambiguity."
    )
    parser.add_argument("--output", default="./results/v679/attention_benchmark.jsonl")
    parser.add_argument("--policy-output", default="./results/v679/distilled_attention_policy.json")
    args = parser.parse_args()
    controller = AttentionController()
    traversal_hypothesis = Hypothesis(
        "en:dog", "has_part", "relation_lookup", .5, {"target_terms": ["tail"]}
    )
    controller.begin_turn(traversal_hypothesis.subject)
    controller.begin_hypothesis(traversal_hypothesis)
    controller.select_traversal_targets(traversal_hypothesis, (), [
        SimpleNamespace(relation="related_to", object="en:dog_house"),
        SimpleNamespace(relation="has_part", object="en:tail"),
    ])
    ranked = [
        (score, hypothesis, result)
        for _, hypothesis, result in adversarial_candidates()
        for score in [hypothesis.lexical_score]
    ]
    decision = controller.arbitrate(ranked)
    learned = DistilledAttentionPolicy.fit(controller.policy_examples)
    policy_path = Path(args.policy_output)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    learned.save(policy_path)
    record = {
        "benchmark": "v679_adversarial_attention",
        "passed": decision["outcome"] == "abstain",
        "expected_outcome": "abstain",
        "decision": decision,
        "attention": controller.trace(),
        "distilled_policy": str(policy_path),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{'PASS' if record['passed'] else 'FAIL'} adversarial no-valid-answer")


if __name__ == "__main__":
    main()
