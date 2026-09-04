"""Reproducible V680 phase runner; PPO remains an explicit, separate phase."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from attention_dataset import collect_jepa_transition_episodes, collect_teacher_episodes, read_jsonl, write_jsonl
from attention_dagger import run_dagger
from attention_distill import train_distillation
from attention_evaluate import evaluate
from attention_jepa import AttentionJEPA, evaluate_jepa, load_jepa, train_jepa
from attention_ppo import run_ppo
from attention_env import no_proof_episodes


def revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8); parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--phases", default="distillation,dagger,jepa,evaluation")
    parser.add_argument("--use-jepa", action="store_true"); parser.add_argument("--ppo-episodes", type=int, default=0)
    parser.add_argument("--ppo-smoke", action="store_true",
                        help="run PPO only as an explicitly labelled smoke test")
    args = parser.parse_args()
    phases = set(args.phases.split(","))
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    dataset = output / "teacher.jsonl"
    if not dataset.exists():
        write_jsonl(dataset, collect_teacher_episodes())
    records = read_jsonl(dataset)
    jepa_dataset = output / "jepa_transitions.jsonl"
    if not jepa_dataset.exists():
        write_jsonl(jepa_dataset, collect_jepa_transition_episodes())
    jepa_records = read_jsonl(jepa_dataset)
    jepa_path = output / "jepa.pt"
    student_path = output / ("student_jepa.pt" if args.use_jepa else "student.pt")
    manifest = {
        "git_revision": revision(), "seed": args.seed, "dataset": str(dataset),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "jepa_dataset": str(jepa_dataset),
        "jepa_dataset_sha256": hashlib.sha256(jepa_dataset.read_bytes()).hexdigest(),
        "configuration": {"epochs": args.epochs, "rounds": args.rounds, "use_jepa": args.use_jepa,
                          "model": {"hidden_size": 32, "recurrent": True},
                          "teacher": "frozen_v679", "jepa": {"representation_dim": 24, "target_momentum": .99},
                          "dagger": {"iterative": True, "rounds": args.rounds},
                          "ppo": {"episodes": args.ppo_episodes, "ppo_epochs": 1}},
    }
    if "jepa" in phases:
        jepa, optimizer = train_jepa(jepa_records, args.epochs, seed=args.seed, model=AttentionJEPA())
        torch.save({"model": jepa.state_dict(), "optimizer": optimizer.state_dict(), "seed": args.seed,
                    "configuration": {"representation_dim": 24, "hidden_size": 48, "target_momentum": .99}}, jepa_path)
        manifest["jepa"] = evaluate_jepa(jepa_records, jepa)
    jepa = load_jepa(jepa_path) if args.use_jepa else None
    if "distillation" in phases:
        student, optimizer = train_distillation(records, args.epochs, seed=args.seed, jepa=jepa, use_jepa=args.use_jepa)
        torch.save({"model": student.state_dict(), "optimizer": optimizer.state_dict(),
                    "hidden_size": student.hidden_size, "jepa_dim": student.jepa_dim}, student_path)
        if args.use_jepa:
            baseline, baseline_optimizer = train_distillation(records, args.epochs, seed=args.seed)
            torch.save({"model": baseline.state_dict(), "optimizer": baseline_optimizer.state_dict(),
                        "hidden_size": baseline.hidden_size, "jepa_dim": 0}, output / "student_baseline.pt")
    if "evaluation" in phases:
        from attention_evaluate import load_student
        model = load_student(student_path)
        manifest["evaluation"] = evaluate(records, model, jepa=jepa)
        if args.use_jepa:
            baseline = load_student(output / "student_baseline.pt")
            manifest["jepa_ablation"] = {
                "baseline": evaluate(records, baseline),
                "jepa": manifest["evaluation"],
                "shuffled_jepa": evaluate(records, model, jepa=jepa, shuffled_jepa=True),
            }
    if "dagger" in phases:
        _, _, manifest["dagger"] = run_dagger(records, args.rounds, args.epochs, args.seed,
                                               output / "dagger", jepa=jepa, use_jepa=args.use_jepa)
        from attention_evaluate import load_student
        heldout_no_proof = collect_teacher_episodes(no_proof_episodes("heldout"))
        pre_dagger = (load_student(student_path) if student_path.exists() else
                      train_distillation(records, args.epochs, seed=args.seed, jepa=jepa,
                                         use_jepa=args.use_jepa)[0])
        manifest["no_proof_generalization"] = {
            "before_dagger": evaluate(heldout_no_proof, pre_dagger, jepa=jepa)
            .get("held_out_adversarial", {}),
            **{f"after_dagger_round_{item['round']}": item["held_out_no_proof"]
               for item in manifest["dagger"]},
        }
    if "ppo" in phases:
        adversarial = manifest.get("evaluation", {}).get("held_out_adversarial", {})
        dagger_no_proof = (manifest.get("dagger") or [{}])[-1].get("held_out_no_proof", {})
        readiness_checks = {
            "distillation": adversarial.get("abstain_f1", 0.0) >= .8,
            "dagger": dagger_no_proof.get("abstain_f1", 0.0) >= .8,
            "jepa": (manifest.get("jepa", {}).get("action_conditioned", False)
                     and manifest.get("jepa", {}).get("prediction_error", {}).get("jepa", float("inf"))
                     < manifest.get("jepa", {}).get("prediction_error", {}).get("zero", float("-inf"))),
            "jepa_attention_gain": (
                manifest.get("jepa_ablation", {}).get("jepa", {}).get("held_out_adversarial", {})
                .get("top1_attention_accuracy", 0.0)
                > manifest.get("jepa_ablation", {}).get("baseline", {}).get("held_out_adversarial", {})
                .get("top1_attention_accuracy", float("inf"))),
            "dagger_failure_curve": (
                len(manifest.get("dagger", [])) >= 2
                and manifest["dagger"][-1]["false_positive_attention_rate"]
                <= manifest["dagger"][0]["false_positive_attention_rate"]),
        }
        readiness = all(readiness_checks.values())
        manifest["ppo_readiness"] = {"ready": readiness, "checks": readiness_checks}
        if not readiness and not args.ppo_smoke:
            raise RuntimeError("PPO readiness gate failed; use --ppo-smoke only for a labelled smoke test")
        _, trajectories, ppo = run_ppo(episode_count=args.ppo_episodes or 2, seed=args.seed,
                                        initial_checkpoint=student_path, checkpoint=output / "ppo.pt",
                                        jepa=jepa, use_jepa=args.use_jepa, ppo_epochs=1)
        manifest["ppo"] = {"transitions": len(trajectories), **ppo, "smoke_only": args.ppo_smoke or not readiness}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
