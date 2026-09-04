"""Deterministic iterative DAgger: retrain, roll out, label, aggregate."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch

from attention_dataset import collect_teacher_episodes, dataset_stats, read_jsonl, step_record, write_jsonl
from attention_distill import train_distillation, training_records
from attention_env import AttentionEnv, benchmark_episodes, no_proof_episodes
from attention_evaluate import evaluate_rollouts
from attention_teacher import V679AttentionTeacher
from attention_types import TEACHER_VERSION


def _action_name(index, candidate_count):
    return "traverse" if index < candidate_count else ("stop" if index == candidate_count else "abstain")


def _error_type(state, teacher_index, student_index, oracle):
    teacher_action = _action_name(teacher_index, len(state.candidate_features))
    student_action = _action_name(student_index, len(state.candidate_features))
    if teacher_action == student_action == "traverse":
        return "wrong_candidate"
    if student_action == "stop":
        return "premature_stop" if teacher_action == "traverse" else "other"
    if student_action == "abstain":
        return "premature_abstain" if teacher_action == "traverse" else "other"
    if teacher_action == "traverse" and student_action != "traverse":
        return "missed_useful_traversal"
    candidate = state.candidate_features[student_index] if student_action == "traverse" else None
    if candidate and candidate.already_visited:
        return "redundant_traversal"
    if candidate and candidate.contradiction:
        return "contradiction_error"
    if candidate and candidate.relation == "related_to":
        return "association_trap"
    if candidate and candidate.lexical_score >= .9:
        return "lexical_trap"
    return "irrelevant_traversal" if student_action == "traverse" else "other"


def _failure_record(round_number, spec, state_id, state, teacher, student, oracle, jepa):
    teacher_index = teacher["selected_action"]; student_index = student["selected_action"]
    if teacher_index == student_index:
        return None
    jepa_summary = {}
    if jepa is not None:
        with torch.no_grad():
            predictions = jepa.predict_actions(state)
            jepa_summary = {"prediction_norms": predictions.norm(dim=1).tolist(),
                            "selected_prediction_norm": float(predictions[student_index].norm())}
    return {
        "round": round_number, "episode_id": spec["episode_id"], "state_id": state_id,
        "teacher_version": TEACHER_VERSION, "teacher_action": _action_name(teacher_index, len(state.candidate_features)),
        "student_action": _action_name(student_index, len(state.candidate_features)),
        "teacher_scores": teacher["logits"], "student_scores": student["logits"],
        "current_focus": state.current_focus, "goal": {"relation": state.goal_relation, "terms": state.goal_terms},
        "candidate_ids": list(range(len(state.candidate_features))),
        "candidate_features": [item.__dict__ for item in state.candidate_features],
        "visited_nodes": state.visited_nodes, "visited_relations": state.visited_relations,
        "jepa_prediction_summary": jepa_summary,
        "error_type": _error_type(state, teacher_index, student_index, oracle),
    }


def student_labeled_rollouts(model, episodes, round_number, seed, jepa=None):
    teacher = V679AttentionTeacher()
    records, failures = [], []
    random.seed(seed + (round_number + 1) * 1000); torch.manual_seed(seed + (round_number + 1) * 1000)
    for spec in episodes:
        env = AttentionEnv(spec)
        state, hidden, trajectory = env.reset(), None, []
        while not env.done:
            student = model.select_action(state, deterministic=False, hidden=hidden, jepa=jepa)
            teacher_label = teacher.select_action(state, deterministic=True)
            next_state, reward, _, oracle = env.step(student["action"])
            failure = _failure_record(round_number, spec, len(trajectory), state, teacher_label, student, oracle, jepa)
            if failure:
                failures.append(failure)
            trajectory.append(step_record(
                spec["episode_id"], spec["split"], len(trajectory), state, teacher_label,
                student["action"], next_state, reward, oracle["terminal_outcome"], oracle,
                {"generator": "student_rollout_teacher_label", "round": round_number,
                 "student_action": student["action"].as_dict(),
                 "student_logits": student["logits"]},
                spec.get("partition", ""), spec.get("category", ""), spec.get("no_proof", False),
            ))
            state, hidden = next_state, student["hidden"]
        records.append({"episode_id": f"{spec['episode_id']}_dagger_{round_number}",
                        "split": spec["split"], "trajectory": trajectory,
                        "partition": spec.get("partition", ""), "category": spec.get("category", ""),
                        "no_proof": spec.get("no_proof", False),
                        "terminal_outcome": trajectory[-1]["terminal_outcome"],
                        "provenance": {"generator": "student_rollout_teacher_label",
                                       "round": round_number}})
    return records, failures


def _round_metrics(new_records, checkpoint, failures, prior_state_ids=()):
    steps = [step for episode in new_records for step in episode["trajectory"]]
    agreement = sum(step["action"] == step["candidates"][step["teacher"]["selected_action"]]["action"]
                    for step in steps)
    top3 = sum(step["teacher"]["selected_action"] in sorted(
        range(len(step["provenance"]["student_logits"])),
        key=lambda i: step["provenance"]["student_logits"][i], reverse=True)[:3]
               for step in steps)
    teacher_abstentions = [s for s in steps
                           if s["teacher"]["selected_action"] == len(s["state"]["candidate_features"]) + 1]
    teacher_traversals = [s for s in steps
                          if s["teacher"]["selected_action"] < len(s["state"]["candidate_features"])]
    false_positives = [s for s in teacher_abstentions if s["action"]["kind"] == "traverse"]
    false_negatives = [s for s in teacher_traversals if s["action"]["kind"] != "traverse"]
    action_distribution = Counter(_action_name(step["teacher"]["selected_action"],
                                                len(step["state"]["candidate_features"])) for step in steps)
    student_distribution = Counter(step["action"]["kind"] for step in steps)
    state_ids = {json.dumps(step["state"], sort_keys=True) for step in steps}
    return {"student_checkpoint": str(checkpoint), "states_collected": len(steps),
            "teacher_labels": len(steps), "teacher_action_agreement": agreement / max(1, len(steps)),
            "top3_recall": top3 / max(1, len(steps)),
            "abstention_accuracy": sum(s["action"]["kind"] == "abstain" for s in teacher_abstentions)
            / max(1, len(teacher_abstentions)),
            "false_positive_attention_events": len(false_positives),
            "false_positive_attention_rate": len(false_positives) / max(1, len(teacher_abstentions)),
            "false_negative_attention_events": len(false_negatives),
            "false_negative_attention_rate": len(false_negatives) / max(1, len(teacher_traversals)),
            "unique_states": len(state_ids), "new_states": len(state_ids - set(prior_state_ids)),
            "revisited_states": len(state_ids & set(prior_state_ids)),
            "total_states": len(steps), "teacher_action_distribution": dict(action_distribution),
            "student_action_distribution": dict(student_distribution),
            "failure_distribution": dict(Counter(failure["error_type"] for failure in failures))}


def run_dagger(dataset, rounds=2, epochs=8, seed=0, checkpoint_dir="checkpoints", episodes=None,
               jepa=None, use_jepa=False, class_balance=True):
    input_records = read_jsonl(dataset) if isinstance(dataset, (str, Path)) else list(dataset)
    aggregate = training_records(input_records)
    source = episodes or benchmark_episodes()
    specs = [spec for spec in source if spec.get("partition", "train") == "train"]
    heldout = collect_teacher_episodes([spec for spec in source if spec.get("partition") == "heldout"])
    heldout_no_proof = collect_teacher_episodes([spec for spec in source if spec.get("partition") == "heldout"
                                                 and spec.get("no_proof")])
    output = Path(checkpoint_dir); output.mkdir(parents=True, exist_ok=True)
    stats = []
    prior_state_ids = set()
    model, optimizer = train_distillation(aggregate, epochs=epochs, seed=seed, jepa=jepa, use_jepa=use_jepa,
                                          class_balance=class_balance)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "round": -1, "seed": seed, "epochs": epochs, "role": "pre_dagger_baseline",
                "hidden_size": model.hidden_size, "jepa_dim": model.jepa_dim},
               output / "dagger_round_-1_baseline.pt")
    for round_number in range(int(rounds)):
        checkpoint = output / f"dagger_round_{round_number}.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "round": round_number, "seed": seed, "epochs": epochs,
                    "hidden_size": model.hidden_size, "jepa_dim": model.jepa_dim}, checkpoint)
        collected, failures = student_labeled_rollouts(model, specs, round_number, seed, jepa)
        aggregate.extend(collected)
        post_model, post_optimizer = train_distillation(aggregate, epochs=epochs, seed=seed + (round_number + 1) * 1000,
                                                         jepa=jepa, use_jepa=use_jepa, class_balance=class_balance)
        metrics = {"round": round_number, "class_balance": class_balance,
                   **_round_metrics(collected, checkpoint, failures, prior_state_ids),
                   "aggregate": dataset_stats(aggregate),
                   "held_out_before_dagger": evaluate_rollouts(
                       [spec for spec in source if spec.get("partition") == "heldout"], model, jepa=jepa),
                   "held_out_no_proof_before_dagger": evaluate_rollouts(
                       [spec for spec in source if spec.get("partition") == "heldout" and spec.get("no_proof")],
                       model, jepa=jepa),
                   "held_out": evaluate_rollouts(
                       [spec for spec in source if spec.get("partition") == "heldout"], post_model, jepa=jepa),
                   "held_out_no_proof": evaluate_rollouts(
                       [spec for spec in source if spec.get("partition") == "heldout" and spec.get("no_proof")],
                       post_model, jepa=jepa)}
        stats.append(metrics)
        prior_state_ids.update(json.dumps(step["state"], sort_keys=True)
                               for episode in collected for step in episode["trajectory"])
        write_jsonl(output / f"dagger_aggregate_round_{round_number}.jsonl", aggregate)
        (output / f"dagger_round_{round_number}.json").write_text(json.dumps(metrics, indent=2))
        write_jsonl(output / f"dagger_failures_round_{round_number}.jsonl", failures)
        (output / f"dagger_failures_round_{round_number}.txt").write_text(
            "\n".join(f"{kind}: {count}" for kind, count in metrics["failure_distribution"].items()) + "\n")
        model, optimizer = post_model, post_optimizer
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "round": int(rounds), "seed": seed, "epochs": epochs,
                "hidden_size": model.hidden_size, "jepa_dim": model.jepa_dim},
               output / "dagger_final.pt")
    return model, aggregate, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True); parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--use-jepa", action="store_true"); parser.add_argument("--jepa-checkpoint")
    parser.add_argument("--raw-class-loss", action="store_true")
    args = parser.parse_args()
    if args.use_jepa != bool(args.jepa_checkpoint):
        parser.error("--use-jepa and --jepa-checkpoint must be supplied together")
    jepa = None
    if args.use_jepa:
        from attention_jepa import load_jepa
        jepa = load_jepa(args.jepa_checkpoint)
    _, _, stats = run_dagger(args.dataset, args.rounds, args.epochs, args.seed, args.checkpoint_dir,
                             jepa=jepa, use_jepa=args.use_jepa,
                             class_balance=not args.raw_class_loss)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
