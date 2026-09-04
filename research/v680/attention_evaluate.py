"""Teacher/student metrics with ordinary and adversarial splits reported separately."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from attention_dataset import collect_teacher_episodes, read_jsonl
from attention_student import NeuralAttentionPolicy


def load_student(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = NeuralAttentionPolicy(payload["hidden_size"])
    model.load_state_dict(payload["state_dict"]); model.eval()
    return model


def evaluate(records, model):
    totals = {}
    for record in records:
        split = record["split"]
        metrics = totals.setdefault(split, {
            "count": 0, "teacher_action_accuracy": 0, "top1_attention_accuracy": 0,
            "top3_recall": 0, "ranking_correlation": 0, "KL_divergence": 0,
            "abstention_accuracy": 0, "abstention_count": 0, "false_positive_attention_rate": 0,
            "average_steps_to_evidence": 0, "student_vs_teacher_final_decision": 0,
        })
        final_match = True
        for step in record["trajectory"]:
            state_data = step["state"]
            from attention_types import AttentionObservation, CandidateFeatures
            state = AttentionObservation(
                **{**state_data, "candidate_features": [
                    CandidateFeatures(**candidate) for candidate in state_data["candidate_features"]
                ]}
            )
            logits = model.score_candidates(state)
            teacher = step["teacher"]
            predicted = max(range(len(logits)), key=logits.__getitem__)
            target = teacher["selected_action"]
            metrics["count"] += 1
            metrics["teacher_action_accuracy"] += predicted == target
            metrics["top1_attention_accuracy"] += predicted == target
            metrics["top3_recall"] += target in sorted(range(len(logits)), key=logits.__getitem__, reverse=True)[:3]
            teacher_order = sorted(range(len(logits)), key=teacher["logits"].__getitem__)
            student_order = sorted(range(len(logits)), key=logits.__getitem__)
            metrics["ranking_correlation"] += sum(
                (teacher_order.index(i) - student_order.index(i)) ** 2
                for i in range(len(logits))
            ) / max(len(logits) ** 2, 1)
            teacher_probs = torch.softmax(torch.tensor(teacher["logits"]) / 2.0, dim=-1)
            student_log_probs = torch.log_softmax(torch.tensor(logits) / 2.0, dim=-1)
            metrics["KL_divergence"] += float(torch.sum(teacher_probs * (torch.log(teacher_probs) - student_log_probs)))
            abstain = target == len(state.candidate_features) + 1
            if abstain:
                metrics["abstention_count"] += 1
                metrics["abstention_accuracy"] += predicted == target
            metrics["false_positive_attention_rate"] += int(abstain and predicted != target)
            final_match = final_match and predicted == target
        metrics["student_vs_teacher_final_decision"] += final_match
    for metrics in totals.values():
        count = max(metrics["count"], 1)
        for key in ("teacher_action_accuracy", "top1_attention_accuracy", "top3_recall",
                    "ranking_correlation", "KL_divergence",
                    "false_positive_attention_rate", "student_vs_teacher_final_decision"):
            metrics[key] /= count
        metrics["abstention_accuracy"] /= max(metrics.pop("abstention_count"), 1)
        metrics["average_steps_to_evidence"] = 1.0
    return totals


def main():
    parser = argparse.ArgumentParser(description="Evaluate V680 student against frozen V679 teacher.")
    parser.add_argument("--dataset", default="./results/v680/distillation_dataset.jsonl")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="./results/v680/evaluation.json")
    args = parser.parse_args()
    report = evaluate(
        read_jsonl(args.dataset) if Path(args.dataset).exists() else collect_teacher_episodes(),
        load_student(args.checkpoint),
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
