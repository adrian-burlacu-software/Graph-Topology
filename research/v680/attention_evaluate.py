"""Teacher/student attention metrics, retaining split boundaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from attention_dataset import collect_teacher_episodes, read_jsonl
from attention_student import NeuralAttentionPolicy
from attention_types import AttentionObservation


def load_student(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = NeuralAttentionPolicy(payload.get("hidden_size", 32), jepa_dim=payload.get("jepa_dim", 0))
    model.load_state_dict(payload.get("model", payload.get("state_dict")))
    model.eval()
    return model


def _correlations(left, right):
    size = len(left)
    if size < 2:
        return 1.0, 1.0
    ranks_left = {item: rank for rank, item in enumerate(sorted(range(size), key=left.__getitem__))}
    ranks_right = {item: rank for rank, item in enumerate(sorted(range(size), key=right.__getitem__))}
    delta = sum((ranks_left[index] - ranks_right[index]) ** 2 for index in range(size))
    spearman = 1 - 6 * delta / (size * (size ** 2 - 1))
    concordant = discordant = 0
    for first in range(size):
        for second in range(first + 1, size):
            pair = (left[first] - left[second]) * (right[first] - right[second])
            concordant += pair > 0; discordant += pair < 0
    denominator = concordant + discordant
    return spearman, (concordant - discordant) / denominator if denominator else 0.0


def _decision_name(index, candidate_count):
    return "traverse" if index < candidate_count else ("stop" if index == candidate_count else "abstain")


def evaluate(records, model, recurrent=True, jepa=None, shuffled_jepa=False):
    if bool(model.jepa_dim) != bool(jepa):
        raise ValueError("evaluation JEPA configuration must match the student checkpoint")
    totals = {}
    for episode in records:
        metrics = totals.setdefault(episode["split"], {"count": 0, "teacher_action_accuracy": 0,
            "top1_attention_accuracy": 0, "top3_attention_recall": 0,
            "spearman_rank_correlation": 0, "kendall_tau": 0, "mean_rank_position_error": 0,
            "KL_divergence": 0, "abstention_accuracy": 0, "abstention_count": 0,
            "stop_accuracy": 0, "stop_count": 0, "false_positive_attention_events": 0,
            "false_negative_attention_events": 0, "abstain_true_positive": 0,
            "abstain_false_positive": 0, "abstain_false_negative": 0,
            "mean_attention_steps": 0, "episode_count": 0,
            "teacher_final_decision": [], "student_final_decision": [],
            "teacher_student_final_decision_agreement": 0})
        final_student = final_teacher = None
        hidden = None
        for step in episode["trajectory"]:
            state = AttentionObservation.from_dict(step["state"])
            selected = model.select_action(state, deterministic=True, hidden=hidden, jepa=jepa,
                                           shuffled_jepa=shuffled_jepa)
            logits = selected["logits"]
            hidden = selected["hidden"] if recurrent else None
            teacher_logits = step["teacher"]["logits"]; target = step["teacher"]["selected_action"]
            predicted = max(range(len(logits)), key=logits.__getitem__)
            top3 = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)[:3]
            spearman, kendall = _correlations(teacher_logits, logits)
            teacher_order = sorted(range(len(logits)), key=teacher_logits.__getitem__)
            student_order = sorted(range(len(logits)), key=logits.__getitem__)
            rank_error = sum(abs(teacher_order.index(i) - student_order.index(i)) for i in range(len(logits))) / len(logits)
            teacher_probs = torch.softmax(torch.tensor(teacher_logits) / 2, -1)
            student_log_probs = torch.log_softmax(torch.tensor(logits) / 2, -1)
            metrics["count"] += 1; metrics["teacher_action_accuracy"] += predicted == target
            metrics["top1_attention_accuracy"] += predicted == target; metrics["top3_attention_recall"] += target in top3
            metrics["spearman_rank_correlation"] += spearman; metrics["kendall_tau"] += kendall
            metrics["mean_rank_position_error"] += rank_error
            metrics["KL_divergence"] += float(torch.sum(teacher_probs * (torch.log(teacher_probs) - student_log_probs)))
            if target == len(state.candidate_features) + 1:
                metrics["abstention_count"] += 1
                metrics["abstention_accuracy"] += predicted == target
                metrics["abstain_true_positive"] += predicted == target
                metrics["abstain_false_negative"] += predicted != target
                metrics["false_positive_attention_events"] += predicted < len(state.candidate_features)
            elif target < len(state.candidate_features):
                metrics["false_negative_attention_events"] += predicted == len(state.candidate_features) + 1
                metrics["abstain_false_positive"] += predicted == len(state.candidate_features) + 1
            elif target == len(state.candidate_features):
                metrics["stop_count"] += 1
                metrics["stop_accuracy"] += predicted == target
            final_student = _decision_name(predicted, len(state.candidate_features))
            final_teacher = _decision_name(target, len(state.candidate_features))
        metrics["mean_attention_steps"] += len(episode["trajectory"])
        metrics["episode_count"] += 1
        metrics["teacher_final_decision"].append(final_teacher)
        metrics["student_final_decision"].append(final_student)
        metrics["teacher_student_final_decision_agreement"] += final_student == final_teacher
    for metrics in totals.values():
        count = max(1, metrics["count"])
        for key in ("teacher_action_accuracy", "top1_attention_accuracy", "top3_attention_recall",
                    "spearman_rank_correlation", "kendall_tau", "mean_rank_position_error",
                    "KL_divergence"):
            metrics[key] /= count
        metrics["mean_attention_steps"] /= metrics.pop("episode_count")
        metrics["abstention_accuracy"] /= max(1, metrics.pop("abstention_count"))
        metrics["stop_accuracy"] /= max(1, metrics.pop("stop_count"))
        metrics["false_positive_attention_rate"] = (
            metrics["false_positive_attention_events"] / max(1, metrics["abstain_true_positive"]
                                                               + metrics["abstain_false_negative"]))
        metrics["false_negative_attention_rate"] = (
            metrics["false_negative_attention_events"] / max(1, count))
        precision = metrics["abstain_true_positive"] / max(
            1, metrics["abstain_true_positive"] + metrics["abstain_false_positive"])
        recall = metrics["abstain_true_positive"] / max(
            1, metrics["abstain_true_positive"] + metrics["abstain_false_negative"])
        metrics["abstain_precision"] = precision
        metrics["abstain_recall"] = recall
        metrics["abstain_f1"] = 2 * precision * recall / max(1e-12, precision + recall)
        metrics["teacher_student_final_decision_agreement"] /= max(1, len(metrics["teacher_final_decision"]))
    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./results/v680/distillation_dataset.jsonl")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", default="./results/v680/evaluation.json")
    parser.add_argument("--use-jepa", action="store_true"); parser.add_argument("--jepa-checkpoint")
    args = parser.parse_args()
    if args.use_jepa != bool(args.jepa_checkpoint):
        parser.error("--use-jepa and --jepa-checkpoint must be supplied together")
    jepa = None
    if args.use_jepa:
        from attention_jepa import load_jepa
        jepa = load_jepa(args.jepa_checkpoint)
    report = evaluate(read_jsonl(args.dataset) if Path(args.dataset).exists() else collect_teacher_episodes(),
                      load_student(args.checkpoint), jepa=jepa)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
