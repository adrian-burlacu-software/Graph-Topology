"""Deterministic iterative DAgger: retrain, roll out, label, aggregate."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from attention_dataset import dataset_stats, read_jsonl, step_record, write_jsonl
from attention_distill import train_distillation
from attention_env import AttentionEnv, benchmark_episodes
from attention_teacher import V679AttentionTeacher


def student_labeled_rollouts(model, episodes, round_number, seed):
    teacher = V679AttentionTeacher()
    records = []
    random.seed(seed + round_number); torch.manual_seed(seed + round_number)
    for spec in episodes:
        env = AttentionEnv(spec)
        state, hidden, trajectory = env.reset(), None, []
        while not env.done:
            student = model.select_action(state, deterministic=False, hidden=hidden)
            teacher_label = teacher.select_action(state, deterministic=True)
            next_state, reward, _, oracle = env.step(student["action"])
            trajectory.append(step_record(
                spec["episode_id"], spec["split"], len(trajectory), state, teacher_label,
                student["action"], next_state, reward, oracle["terminal_outcome"], oracle,
                {"generator": "student_rollout_teacher_label", "round": round_number,
                 "student_action": student["action"].as_dict(),
                 "student_logits": student["logits"]},
            ))
            state, hidden = next_state, student["hidden"]
        records.append({"episode_id": f"{spec['episode_id']}_dagger_{round_number}",
                        "split": spec["split"], "trajectory": trajectory,
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
    abstentions = [s for s in steps if s["action"]["kind"] == "abstain"]
    return {"student_checkpoint": str(checkpoint), "states_collected": len(steps),
            "teacher_labels": len(steps), "teacher_action_agreement": agreement / max(1, len(steps)),
            "top3_recall": top3 / max(1, len(steps)),
            "abstention_accuracy": sum(s["terminal_outcome"] == "no_verified_evidence"
                                       for s in abstentions) / max(1, len(abstentions))}


def run_dagger(dataset, rounds=2, epochs=8, seed=0, checkpoint_dir="checkpoints", episodes=None):
    aggregate = read_jsonl(dataset) if isinstance(dataset, (str, Path)) else list(dataset)
    specs = episodes or benchmark_episodes()
    output = Path(checkpoint_dir); output.mkdir(parents=True, exist_ok=True)
    stats = []
    for round_number in range(int(rounds)):
        model, optimizer = train_distillation(aggregate, epochs=epochs, seed=seed + round_number)
        checkpoint = output / f"dagger_round_{round_number}.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "round": round_number, "seed": seed, "epochs": epochs}, checkpoint)
        collected = student_labeled_rollouts(model, specs, round_number, seed)
        aggregate.extend(collected)
        metrics = {"round": round_number, **_round_metrics(collected, checkpoint),
                   "aggregate": dataset_stats(aggregate)}
        stats.append(metrics)
        write_jsonl(output / f"dagger_aggregate_round_{round_number}.jsonl", aggregate)
        (output / f"dagger_round_{round_number}.json").write_text(json.dumps(metrics, indent=2))
    return model, aggregate, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True); parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()
    _, _, stats = run_dagger(args.dataset, args.rounds, args.epochs, args.seed, args.checkpoint_dir)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
