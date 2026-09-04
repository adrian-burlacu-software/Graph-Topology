"""Deterministic iterative DAgger: retrain, roll out, label, aggregate."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from attention_dataset import collect_teacher_episodes, dataset_stats, read_jsonl, step_record, write_jsonl
from attention_distill import train_distillation
from attention_env import AttentionEnv, benchmark_episodes, no_proof_episodes
from attention_evaluate import evaluate
from attention_teacher import V679AttentionTeacher


def student_labeled_rollouts(model, episodes, round_number, seed, jepa=None):
    teacher = V679AttentionTeacher()
    records = []
    random.seed(seed + round_number); torch.manual_seed(seed + round_number)
    for spec in episodes:
        env = AttentionEnv(spec)
        state, hidden, trajectory = env.reset(), None, []
        while not env.done:
            student = model.select_action(state, deterministic=False, hidden=hidden, jepa=jepa)
            teacher_label = teacher.select_action(state, deterministic=True)
            next_state, reward, _, oracle = env.step(student["action"])
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
    return records


def _round_metrics(new_records, checkpoint):
    steps = [step for episode in new_records for step in episode["trajectory"]]
    agreement = sum(step["action"] == step["candidates"][step["teacher"]["selected_action"]]["action"]
                    for step in steps)
    top3 = sum(step["teacher"]["selected_action"] in sorted(
        range(len(step["provenance"]["student_logits"])),
        key=lambda i: step["provenance"]["student_logits"][i], reverse=True)[:3]
               for step in steps)
    teacher_abstentions = [s for s in steps
                           if s["teacher"]["selected_action"] == len(s["state"]["candidate_features"]) + 1]
    false_positives = [s for s in teacher_abstentions if s["action"]["kind"] == "traverse"]
    return {"student_checkpoint": str(checkpoint), "states_collected": len(steps),
            "teacher_labels": len(steps), "teacher_action_agreement": agreement / max(1, len(steps)),
            "top3_recall": top3 / max(1, len(steps)),
            "abstention_accuracy": sum(s["action"]["kind"] == "abstain" for s in teacher_abstentions)
            / max(1, len(teacher_abstentions)),
            "false_positive_attention_events": len(false_positives),
            "false_positive_attention_rate": len(false_positives) / max(1, len(teacher_abstentions))}


def run_dagger(dataset, rounds=2, epochs=8, seed=0, checkpoint_dir="checkpoints", episodes=None,
               jepa=None, use_jepa=False):
    input_records = read_jsonl(dataset) if isinstance(dataset, (str, Path)) else list(dataset)
    aggregate = [record for record in input_records if record["split"] != "held_out_adversarial"
                 and record.get("partition") != "validation"]
    source = episodes or benchmark_episodes()
    specs = [spec for spec in source if spec["split"] != "held_out_adversarial"
             and spec.get("partition") != "validation"]
    heldout = collect_teacher_episodes([spec for spec in source if spec["split"] == "held_out_adversarial"])
    heldout_no_proof = collect_teacher_episodes(no_proof_episodes("heldout"))
    output = Path(checkpoint_dir); output.mkdir(parents=True, exist_ok=True)
    stats = []
    model, optimizer = train_distillation(aggregate, epochs=epochs, seed=seed, jepa=jepa, use_jepa=use_jepa)
    for round_number in range(int(rounds)):
        checkpoint = output / f"dagger_round_{round_number}.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "round": round_number, "seed": seed, "epochs": epochs,
                    "hidden_size": model.hidden_size, "jepa_dim": model.jepa_dim}, checkpoint)
        collected = student_labeled_rollouts(model, specs, round_number, seed, jepa)
        aggregate.extend(collected)
        post_model, post_optimizer = train_distillation(aggregate, epochs=epochs, seed=seed + round_number + 1,
                                                         jepa=jepa, use_jepa=use_jepa)
        metrics = {"round": round_number, **_round_metrics(collected, checkpoint),
                   "aggregate": dataset_stats(aggregate),
                   "held_out_adversarial_before_dagger": evaluate(heldout, model, jepa=jepa).get("held_out_adversarial", {}),
                   "held_out_no_proof_before_dagger": evaluate(heldout_no_proof, model, jepa=jepa).get("held_out_adversarial", {}),
                   "held_out_adversarial": evaluate(heldout, post_model, jepa=jepa).get("held_out_adversarial", {}),
                   "held_out_no_proof": evaluate(heldout_no_proof, post_model, jepa=jepa).get("held_out_adversarial", {})}
        stats.append(metrics)
        write_jsonl(output / f"dagger_aggregate_round_{round_number}.jsonl", aggregate)
        (output / f"dagger_round_{round_number}.json").write_text(json.dumps(metrics, indent=2))
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
    args = parser.parse_args()
    if args.use_jepa != bool(args.jepa_checkpoint):
        parser.error("--use-jepa and --jepa-checkpoint must be supplied together")
    jepa = None
    if args.use_jepa:
        from attention_jepa import load_jepa
        jepa = load_jepa(args.jepa_checkpoint)
    _, _, stats = run_dagger(args.dataset, args.rounds, args.epochs, args.seed, args.checkpoint_dir,
                             jepa=jepa, use_jepa=args.use_jepa)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
