"""Single reproducible V680.1 experiment matrix entry point."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from attention_benchmark import DECISION_BOUNDARY_VERSION, decision_boundary_episodes
from attention_dataset import collect_jepa_transition_episodes, collect_teacher_episodes, read_jsonl, write_jsonl
from attention_dagger import run_dagger
from attention_distill import train_distillation
from attention_evaluate import TeacherPolicyAdapter, evaluate, evaluate_rollouts, jepa_action_swap_diagnostic, load_student
from attention_jepa import AttentionJEPA, evaluate_jepa, load_jepa, train_jepa
from attention_ppo import run_ppo
from attention_types import DATASET_VERSION, JEPA_VERSION, STUDENT_VERSION, TEACHER_VERSION


def revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_report(path, result):
    readiness = result["ppo_readiness"]
    lines = [
        "V680.1 attention learning report",
        f"dataset={result['dataset_version']} teacher={result['teacher_version']} seed={result['seed']}",
        f"PPO ready: {readiness['ready']} ({', '.join(key for key, value in readiness['checks'].items() if not value) or 'all checks passed'})",
        "This report is descriptive for one seed unless a multi-seed run is requested.",
    ]
    for name, matrix in result.get("attention_matrix", {}).items():
        heldout = matrix.get("held_out_structural", {})
        lines.append(f"{name}: held-out structural action accuracy={heldout.get('overall_action_accuracy', 0):.3f}")
    Path(path).write_text("\n".join(lines) + "\n")


def _first_metric(matrix, name, default=0.0):
    return next((value[name] for value in matrix.values() if isinstance(value, dict) and name in value), default)


def main():
    parser = argparse.ArgumentParser(description="Run reproducible V680.1 attention experiments.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--samples-per-category", type=int, default=100)
    parser.add_argument("--phases", default="distillation,dagger,jepa,evaluation")
    parser.add_argument("--ppo-episodes", type=int, default=0)
    parser.add_argument("--ppo-smoke", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="use five examples per category; never publish as robust evidence")
    args = parser.parse_args()
    phases = set(args.phases.split(","))
    sample_count = 5 if args.smoke else args.samples_per_category
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    specs = decision_boundary_episodes(sample_count)
    teacher_dataset = output / "decision_boundary_teacher.jsonl"
    if not teacher_dataset.exists():
        write_jsonl(teacher_dataset, collect_teacher_episodes(specs))
    records = read_jsonl(teacher_dataset)
    jepa_dataset = output / "decision_boundary_transitions.jsonl"
    if not jepa_dataset.exists():
        write_jsonl(jepa_dataset, collect_jepa_transition_episodes(specs))
    jepa_records = read_jsonl(jepa_dataset)
    jepa_path = output / "jepa.pt"
    baseline_path = output / "student_baseline.pt"
    jepa_student_path = output / "student_jepa.pt"
    result = {
        "experiment_version": "v680.1",
        "git_revision": revision(), "seed": args.seed, "smoke": args.smoke,
        "dataset_version": DATASET_VERSION, "benchmark_version": DECISION_BOUNDARY_VERSION,
        "teacher_version": TEACHER_VERSION, "student_version": STUDENT_VERSION, "jepa_version": JEPA_VERSION,
        "dataset": str(teacher_dataset), "dataset_sha256": _hash(teacher_dataset),
        "jepa_dataset": str(jepa_dataset), "jepa_dataset_sha256": _hash(jepa_dataset),
        "configuration": {"epochs": args.epochs, "rounds": args.rounds, "samples_per_category": sample_count,
                          "student": {"hidden_size": 32, "recurrent": True},
                          "jepa": {"representation_dim": 24, "target_momentum": .99},
                          "dagger": {"iterative": True, "raw_and_balanced": True},
                          "ppo": {"episodes": args.ppo_episodes, "separate_from_imitation": True}},
    }
    if "jepa" in phases:
        jepa, jepa_optimizer = train_jepa(jepa_records, args.epochs, seed=args.seed, model=AttentionJEPA())
        torch.save({"model": jepa.state_dict(), "optimizer": jepa_optimizer.state_dict(), "seed": args.seed,
                    "version": JEPA_VERSION, "configuration": {"representation_dim": 24, "hidden_size": 48,
                    "target_momentum": .99}}, jepa_path)
        result["jepa"] = evaluate_jepa(jepa_records, jepa)
    jepa = load_jepa(jepa_path) if jepa_path.exists() else None
    if "distillation" in phases:
        baseline, baseline_optimizer = train_distillation(records, args.epochs, seed=args.seed, class_balance=False)
        torch.save({"model": baseline.state_dict(), "optimizer": baseline_optimizer.state_dict(),
                    "hidden_size": baseline.hidden_size, "jepa_dim": 0, "version": STUDENT_VERSION}, baseline_path)
        jepa_student, jepa_optimizer = train_distillation(records, args.epochs, seed=args.seed, jepa=jepa, use_jepa=True)
        torch.save({"model": jepa_student.state_dict(), "optimizer": jepa_optimizer.state_dict(),
                    "hidden_size": jepa_student.hidden_size, "jepa_dim": jepa_student.jepa_dim,
                    "version": STUDENT_VERSION}, jepa_student_path)
    baseline = load_student(baseline_path) if baseline_path.exists() else None
    jepa_student = load_student(jepa_student_path) if jepa_student_path.exists() else None
    if "evaluation" in phases:
        if baseline is None or jepa_student is None or jepa is None:
            raise RuntimeError("evaluation requires existing baseline, JEPA, and JEPA-student artifacts")
        test_specs = [spec for spec in specs if spec["partition"] != "train"]
        result["attention_matrix"] = {
            "baseline": evaluate_rollouts(test_specs, baseline, policy_name="baseline",
                                          failure_path=output / "failures_baseline.jsonl"),
            "jepa": evaluate_rollouts(test_specs, jepa_student, jepa=jepa, policy_name="jepa",
                                      failure_path=output / "failures_jepa.jsonl"),
            "shuffled_jepa": evaluate_rollouts(test_specs, jepa_student, jepa=jepa, shuffled_jepa=True,
                                               policy_name="shuffled_jepa", failure_path=output / "failures_shuffled_jepa.jsonl"),
            "random_jepa": evaluate_rollouts(test_specs, jepa_student, jepa=jepa, random_jepa=True,
                                             policy_name="random_jepa", failure_path=output / "failures_random_jepa.jsonl"),
            "teacher": evaluate_rollouts(test_specs, TeacherPolicyAdapter(), policy_name="teacher"),
        }
        heldout_no_proof = [spec for spec in test_specs if spec.get("no_proof")]
        result["no_proof_generalization"] = {
            "baseline_before_dagger": evaluate_rollouts(heldout_no_proof, baseline, policy_name="baseline"),
            "jepa_before_dagger": evaluate_rollouts(heldout_no_proof, jepa_student, jepa=jepa, policy_name="jepa"),
        }
        result["jepa_attention_coupling"] = jepa_action_swap_diagnostic(test_specs, jepa_student, jepa)
        heldout = "held_out_structural"
        scores = {name: matrix.get(heldout, {}).get("overall_action_accuracy", 0.0)
                  for name, matrix in result["attention_matrix"].items()
                  if name != "teacher"}
        result["jepa_attention_conclusion"] = {
            "metric": "held_out_structural.overall_action_accuracy", "scores": scores,
            "action_specific_gain": (scores["jepa"] > scores["baseline"] and
                                     scores["jepa"] > scores["shuffled_jepa"] and
                                     scores["jepa"] > scores["random_jepa"]),
            "conclusion": ("action-conditioned JEPA improves all controls" if
                           scores["jepa"] > scores["baseline"] and scores["jepa"] > scores["shuffled_jepa"]
                           and scores["jepa"] > scores["random_jepa"] else
                           "no demonstrated action-conditioned JEPA attention gain; retain as negative result"),
        }
    if "dagger" in phases:
        if jepa is None:
            raise RuntimeError("DAgger requires an existing JEPA artifact")
        result["dagger_raw"] = run_dagger(records, args.rounds, args.epochs, args.seed,
                                           output / "dagger_raw", episodes=specs, jepa=jepa, use_jepa=True,
                                           class_balance=False)[2]
        result["dagger_balanced"] = run_dagger(records, args.rounds, args.epochs, args.seed,
                                                output / "dagger_balanced", episodes=specs, jepa=jepa, use_jepa=True,
                                                class_balance=True)[2]
        no_proof = result.setdefault("no_proof_generalization", {})
        no_proof["after_dagger_raw"] = [
            item["held_out_no_proof"] for item in result["dagger_raw"]]
        no_proof["after_dagger_balanced"] = [
            item["held_out_no_proof"] for item in result["dagger_balanced"]]
    checks = {
        "distillation_passes": bool(result.get("attention_matrix", {}).get("baseline")),
        "dagger_iterative": len(result.get("dagger_balanced", [])) >= 4,
        "dagger_failure_not_worsening": (len(result.get("dagger_balanced", [])) >= 2 and
            result["dagger_balanced"][-1]["false_positive_attention_rate"]
            <= result["dagger_balanced"][0]["false_positive_attention_rate"]),
        "held_out_structural_meaningful": bool(result.get("attention_matrix", {}).get("jepa", {}).get("held_out_structural")),
        "held_out_no_proof_passes": _first_metric(
            result.get("no_proof_generalization", {}).get("jepa_before_dagger", {}),
            "abstain_accuracy") >= .8,
        "jepa_prediction": result.get("jepa", {}).get("prediction_error", {}).get("jepa", float("inf"))
                           < result.get("jepa", {}).get("prediction_error", {}).get("random", float("-inf")),
        "jepa_action_conditioned": result.get("jepa", {}).get("action_conditioned", False),
        "jepa_policy_coupled": result.get("jepa_attention_coupling", {}).get("coupled", False),
        "jepa_attention_gain": result.get("jepa_attention_conclusion", {}).get("action_specific_gain", False),
    }
    ready = all(value is True for value in checks.values() if isinstance(value, bool))
    result["ppo_readiness"] = {"ready": ready, "checks": checks}
    if "ppo" in phases:
        if jepa_student is None or jepa is None:
            raise RuntimeError("PPO requires an existing JEPA student and JEPA artifact")
        if not ready and not args.ppo_smoke:
            raise RuntimeError("PPO readiness gate failed; use --ppo-smoke only for an explicitly labelled smoke")
        _, transitions, metrics = run_ppo(episode_count=args.ppo_episodes or 2, seed=args.seed,
                                           initial_checkpoint=jepa_student_path, checkpoint=output / "ppo.pt",
                                           jepa=jepa, use_jepa=True, ppo_epochs=1)
        result["ppo"] = {"transitions": len(transitions), **metrics, "smoke_only": not ready or args.ppo_smoke}
    (output / "v680_1_results.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    _write_report(output / "v680_1_report.txt", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
