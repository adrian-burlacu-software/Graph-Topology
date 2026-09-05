# V681-owned learning implementation; derived from V680.
"""Teacher/student attention metrics, retaining split boundaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .dataset import collect_teacher_episodes, read_jsonl
from .student import NeuralAttentionPolicy
from .types import AttentionObservation
from .environment import AttentionEnv
from .teacher import V679AttentionTeacher

ACTION_KINDS = ("traverse", "stop", "abstain")


def load_student(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = NeuralAttentionPolicy(payload.get("hidden_size", 32), jepa_dim=payload.get("jepa_dim", 0))
    model.load_state_dict(payload.get("model", payload.get("state_dict")))
    model.eval()
    return model


class TeacherPolicyAdapter:
    """Evaluation-only adapter so teacher rows use the same rollout metrics."""
    jepa_dim = 0

    def __init__(self):
        self.teacher = V679AttentionTeacher()

    def select_action(self, state, **_):
        decision = self.teacher.select_action(state, deterministic=True)
        return {"logits": decision["logits"], "selected_action": decision["selected_action"],
                "action": decision["action"], "hidden": None}


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


def _action_distribution(metric, source, teacher_name, student_name):
    counts = metric["action_distribution_by_source"].setdefault(
        source, {"teacher": {kind: 0 for kind in ACTION_KINDS}, "student": {kind: 0 for kind in ACTION_KINDS}})
    counts["teacher"][teacher_name] += 1
    counts["student"][student_name] += 1


def _finalize_stop_metrics(metric):
    true_positive = metric["stop_true_positive"]
    false_positive = metric["stop_false_positive"]
    false_negative = metric["stop_false_negative"]
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    metric["stop_confusion"] = {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": metric.get("decisions", metric.get("count", 0))
        - true_positive - false_positive - false_negative,
    }
    metric["stop_precision"] = precision
    metric["stop_recall"] = recall
    metric["stop_f1"] = 2 * precision * recall / max(1e-12, precision + recall)


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
            "stop_true_positive": 0, "stop_false_positive": 0, "stop_false_negative": 0,
            "confusion_matrix": {action: {prediction: 0 for prediction in ACTION_KINDS} for action in ACTION_KINDS},
            "action_distribution_by_source": {},
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
            teacher_name = _decision_name(target, len(state.candidate_features))
            student_name = _decision_name(predicted, len(state.candidate_features))
            metrics["count"] += 1; metrics["teacher_action_accuracy"] += predicted == target
            metrics["top1_attention_accuracy"] += predicted == target; metrics["top3_attention_recall"] += target in top3
            metrics["spearman_rank_correlation"] += spearman; metrics["kendall_tau"] += kendall
            metrics["mean_rank_position_error"] += rank_error
            metrics["KL_divergence"] += float(torch.sum(teacher_probs * (torch.log(teacher_probs) - student_log_probs)))
            metrics["confusion_matrix"][teacher_name][student_name] += 1
            _action_distribution(metrics, step.get("source", episode.get("source", "unknown")),
                                 teacher_name, student_name)
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
                metrics["stop_true_positive"] += predicted == target
                metrics["stop_false_negative"] += predicted != target
            elif predicted == len(state.candidate_features):
                metrics["stop_false_positive"] += 1
            final_student = student_name
            final_teacher = teacher_name
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
        _finalize_stop_metrics(metrics)
        metrics["teacher_student_final_decision_agreement"] /= max(1, len(metrics["teacher_final_decision"]))
    return totals


def evaluate_rollouts(episodes, model, jepa=None, shuffled_jepa=False, random_jepa=False,
                      failure_path=None, policy_name="student"):
    """Evaluate decisions on the policy-induced sequence, never teacher trajectories."""
    if bool(model.jepa_dim) != bool(jepa):
        raise ValueError("evaluation JEPA configuration must match the student checkpoint")
    teacher = V679AttentionTeacher()
    totals, failures = {}, []
    for spec in episodes:
        split = spec["split"]
        metric = totals.setdefault(split, {
            "episodes": 0, "decisions": 0, "overall_action_accuracy": 0, "traverse_accuracy": 0,
            "traverse_count": 0, "stop_accuracy": 0, "stop_count": 0, "abstain_accuracy": 0,
            "abstain_count": 0, "correct_candidate_top1": 0, "correct_candidate_top3": 0,
            "candidate_mrr": 0,
            "false_positive_traverse": 0, "false_negative_traverse": 0, "false_positive_abstain": 0,
            "false_negative_abstain": 0, "premature_stop": 0, "premature_abstain": 0,
            "stop_true_positive": 0, "stop_false_positive": 0, "stop_false_negative": 0,
            "confusion_matrix": {a: {b: 0 for b in ACTION_KINDS} for a in ACTION_KINDS},
            "action_distribution_by_source": {},
            "episode_success": 0, "final_decision_accuracy": 0, "proof_completion_rate": 0,
            "unnecessary_steps": 0, "redundant_steps": 0, "average_steps_to_success": 0,
        })
        env, state, hidden, teacher_final, student_final = AttentionEnv(spec), None, None, None, None
        state = env.reset(); steps = 0; proof_steps = 0
        while not env.done:
            teacher_action = teacher.select_action(state, deterministic=True)
            student = model.select_action(state, deterministic=True, hidden=hidden, jepa=jepa,
                                          shuffled_jepa=shuffled_jepa, random_jepa=random_jepa)
            teacher_name = _decision_name(teacher_action["selected_action"], len(state.candidate_features))
            student_name = _decision_name(student["selected_action"], len(state.candidate_features))
            metric["decisions"] += 1; metric["overall_action_accuracy"] += teacher_name == student_name
            metric["confusion_matrix"][teacher_name][student_name] += 1
            _action_distribution(metric, spec.get("source", "heldout_benchmark"), teacher_name, student_name)
            if teacher_name == "traverse":
                metric["traverse_count"] += 1
                metric["traverse_accuracy"] += teacher_action["selected_action"] == student["selected_action"]
                ranked = sorted(range(len(student["logits"])), key=student["logits"].__getitem__, reverse=True)
                metric["correct_candidate_top1"] += teacher_action["selected_action"] == ranked[0]
                metric["correct_candidate_top3"] += teacher_action["selected_action"] in ranked[:3]
                metric["candidate_mrr"] += 1 / (ranked.index(teacher_action["selected_action"]) + 1)
                metric["false_negative_traverse"] += student_name != "traverse"
                metric["premature_stop"] += student_name == "stop"
                metric["premature_abstain"] += student_name == "abstain"
            if teacher_name == "abstain":
                metric["abstain_count"] += 1; metric["abstain_accuracy"] += student_name == "abstain"
                metric["false_positive_traverse"] += student_name == "traverse"
                metric["false_negative_abstain"] += student_name != "abstain"
            if teacher_name == "stop":
                metric["stop_count"] += 1; metric["stop_accuracy"] += student_name == "stop"
                metric["stop_true_positive"] += student_name == "stop"
                metric["stop_false_negative"] += student_name != "stop"
            elif student_name == "stop":
                metric["stop_false_positive"] += 1
            if student_name == "abstain" and teacher_name != "abstain":
                metric["false_positive_abstain"] += 1
            next_state, _, done, oracle = env.step(student["action"])
            if student_name == "traverse":
                proof_steps += bool(oracle.get("valid_proof_edge"))
                metric["redundant_steps"] += bool(oracle.get("already_visited"))
            if teacher_name != student_name:
                failures.append({
                    "policy": policy_name, "episode_id": spec["episode_id"], "category": spec.get("category", ""),
                    "state_id": steps, "teacher_action": teacher_name, "student_action": student_name,
                    "teacher_scores": teacher_action["logits"], "student_scores": student["logits"],
                    "state": state.as_dict(), "oracle_outcome": oracle["terminal_outcome"],
                })
            steps += 1; state, hidden = next_state, student["hidden"]
            teacher_final, student_final = teacher_name, student_name
        metric["episodes"] += 1
        success = env.valid_proof_seen and student_final == "stop"
        metric["episode_success"] += success
        metric["proof_completion_rate"] += success
        metric["final_decision_accuracy"] += teacher_final == student_final
        metric["average_steps_to_success"] += steps if success else 0
        metric["unnecessary_steps"] += max(0, steps - proof_steps - int(spec.get("initial_proof", False)))
    for metric in totals.values():
        decisions = max(1, metric["decisions"]); episodes_count = max(1, metric["episodes"])
        for key in ("overall_action_accuracy", "false_positive_traverse", "false_negative_traverse",
                    "false_positive_abstain", "false_negative_abstain", "premature_stop",
                    "premature_abstain", "redundant_steps"):
            metric[key] /= decisions
        for key, count_key in (("traverse_accuracy", "traverse_count"), ("stop_accuracy", "stop_count"),
                               ("abstain_accuracy", "abstain_count"), ("correct_candidate_top1", "traverse_count"),
                               ("correct_candidate_top3", "traverse_count"), ("candidate_mrr", "traverse_count")):
            metric[key] /= max(1, metric[count_key])
        for key in ("episode_success", "proof_completion_rate", "final_decision_accuracy", "unnecessary_steps",
                    "average_steps_to_success"):
            metric[key] /= episodes_count
        _finalize_stop_metrics(metric)
    if failure_path:
        Path(failure_path).write_text("\n".join(json.dumps(item, sort_keys=True) for item in failures) + "\n")
    return totals


def jepa_action_swap_diagnostic(episodes, model, jepa):
    """Measures whether action-specific predictions materially affect policy logits/rankings."""
    if not model.jepa_dim:
        raise ValueError("JEPA action-swap diagnostic requires a JEPA-augmented policy")
    deltas, rank_changes = [], 0
    for spec in episodes:
        state = AttentionEnv(spec).reset()
        original = model.select_action(state, deterministic=True, jepa=jepa)["logits"]
        swapped = model.select_action(state, deterministic=True, jepa=jepa, shuffled_jepa=True)["logits"]
        deltas.append(sum(abs(left - right) for left, right in zip(original, swapped)) / len(original))
        rank_changes += sorted(range(len(original)), key=original.__getitem__) != sorted(
            range(len(swapped)), key=swapped.__getitem__)
    return {"states": len(deltas), "mean_action_swap_logit_delta": sum(deltas) / max(1, len(deltas)),
            "ranking_change_rate": rank_changes / max(1, len(deltas)),
            "coupled": bool(deltas and any(value > 1e-6 for value in deltas))}


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
        from .jepa import load_jepa
        jepa = load_jepa(args.jepa_checkpoint)
    report = evaluate(read_jsonl(args.dataset) if Path(args.dataset).exists() else collect_teacher_episodes(),
                      load_student(args.checkpoint), jepa=jepa)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
