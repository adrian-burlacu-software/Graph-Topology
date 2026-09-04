"""Frozen-recipe causal investigation of JEPA feature controls."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from attention_benchmark import decision_boundary_episodes
from attention_dataset import collect_jepa_transition_episodes, collect_teacher_episodes, read_jsonl, write_jsonl
from attention_dagger import run_dagger
from attention_distill import train_distillation
from attention_evaluate import evaluate_rollouts
from attention_jepa import AttentionJEPA, JEPAFeatureControl, evaluate_jepa, representation_statistics, train_jepa
from attention_types import DATASET_VERSION, JEPA_VERSION, STUDENT_VERSION, TEACHER_VERSION, AttentionObservation


CONDITIONS = ("zero", "fixed_random", "per_state_random", "real", "action_shuffled", "dimension_permuted")


def summary(values):
    mean = sum(values) / max(1, len(values))
    return {"n": len(values), "mean": mean,
            "std": (sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)) ** .5,
            "min": min(values) if values else 0.0, "max": max(values) if values else 0.0}


def _rank(values, target):
    return sorted(range(len(values)), key=values.__getitem__, reverse=True).index(target) + 1


def rank_utility(records, baseline, variants, providers):
    """Teacher-action ranking shifts by condition/category; teacher labels remain targets only."""
    buckets = defaultdict(lambda: defaultdict(list))
    for episode in records:
        if episode.get("partition") != "heldout":
            continue
        for step in episode["trajectory"]:
            state = AttentionObservation.from_dict(step["state"])
            target = step["teacher"]["selected_action"]
            base_rank = _rank(baseline.select_action(state, deterministic=True)["logits"], target)
            action_type = ("traverse" if target < len(state.candidate_features) else
                           "stop" if target == len(state.candidate_features) else "abstain")
            for name, model in variants.items():
                rank = _rank(model.select_action(state, deterministic=True, jepa=providers[name])["logits"], target)
                change = base_rank - rank
                bucket = buckets[(episode.get("category", ""), action_type)][name]
                bucket.append(change)
    report = {}
    for (category, action_type), variants_report in buckets.items():
        report.setdefault(category, {})[action_type] = {
            name: {
                "mean_rank_improvement": sum(changes) / len(changes),
                "rank_improvement_rate": sum(change > 0 for change in changes) / len(changes),
                "rank_degradation_rate": sum(change < 0 for change in changes) / len(changes),
                "teacher_action_mean_rank": None,
            } for name, changes in variants_report.items()
        }
    return report


def usefulness_matrix(records, baseline, jepa_model, jepa_provider):
    report = defaultdict(lambda: {"helps": 0, "hurts": 0, "neutral": 0})
    for episode in records:
        if episode.get("partition") != "heldout":
            continue
        for step in episode["trajectory"]:
            state = AttentionObservation.from_dict(step["state"]); target = step["teacher"]["selected_action"]
            base_logits = baseline.select_action(state, deterministic=True)["logits"]
            jepa_logits = jepa_model.select_action(state, deterministic=True, jepa=jepa_provider)["logits"]
            base_rank, jepa_rank = _rank(base_logits, target), _rank(jepa_logits, target)
            base_correct = max(range(len(base_logits)), key=base_logits.__getitem__) == target
            jepa_correct = max(range(len(jepa_logits)), key=jepa_logits.__getitem__) == target
            category = episode.get("category", "")
            if (jepa_rank < base_rank) or (jepa_correct and not base_correct):
                report[category]["helps"] += 1
            elif (jepa_rank > base_rank) or (base_correct and not jepa_correct):
                report[category]["hurts"] += 1
            else:
                report[category]["neutral"] += 1
    for values in report.values():
        total = sum(values.values())
        values.update({f"{name}_rate": count / total for name, count in list(values.items())})
    return dict(report)


def main():
    parser = argparse.ArgumentParser(description="Run V680.1 frozen JEPA causal comparison.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--samples-per-category", type=int, default=100)
    parser.add_argument("--dagger-rounds", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    samples = 5 if args.smoke else args.samples_per_category
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    specs = decision_boundary_episodes(samples)
    teacher_path, transitions_path = output / "teacher.jsonl", output / "transitions.jsonl"
    write_jsonl(teacher_path, collect_teacher_episodes(specs))
    write_jsonl(transitions_path, collect_jepa_transition_episodes(specs))
    records, transitions = read_jsonl(teacher_path), read_jsonl(transitions_path)
    test_specs = [spec for spec in specs if spec["partition"] == "heldout"]
    per_seed, matrices, all_representation_stats = {}, defaultdict(list), {}
    for seed in seeds:
        jepa, _ = train_jepa(transitions, args.epochs, seed=seed, model=AttentionJEPA())
        calibration = representation_statistics(transitions, jepa)
        providers = {name: JEPAFeatureControl(jepa, name, seed, calibration["mean"], calibration["std"])
                     for name in CONDITIONS}
        models = {
            name: train_distillation(records, args.epochs, seed=seed, jepa=provider, use_jepa=True)[0]
            for name, provider in providers.items()
        }
        matrix = {
            name: evaluate_rollouts(test_specs, model, jepa=providers[name], policy_name=name)
            for name, model in models.items()
        }
        for name, report in matrix.items():
            matrices[name].append(report.get("held_out_structural", {}).get("overall_action_accuracy", 0.0))
        control_stats = {name: representation_statistics(
            [record for record in transitions], jepa) if name == "real" else
            _provider_statistics(transitions, providers[name]) for name in CONDITIONS}
        all_representation_stats[str(seed)] = control_stats
        per_seed[str(seed)] = {
            "representation_prediction": evaluate_jepa(transitions, jepa),
            "representation_statistics": control_stats, "attention_matrix": matrix,
            "ranking_utility": rank_utility(records, models["zero"], models, providers),
            "usefulness": usefulness_matrix(records, models["zero"], models["real"], providers["real"]),
        }
    # DAgGER uses the same frozen dataset, teacher, JEPA architecture and seed recipe as causal comparison.
    dagger_jepa, _ = train_jepa(transitions, args.epochs, seed=seeds[0], model=AttentionJEPA())
    calibration = representation_statistics(transitions, dagger_jepa)
    dagger_provider = JEPAFeatureControl(dagger_jepa, "real", seeds[0], calibration["mean"], calibration["std"])
    dagger_raw = run_dagger(records, args.dagger_rounds, args.epochs, seeds[0], output / "dagger_raw",
                            episodes=specs, jepa=dagger_provider, use_jepa=True, class_balance=False)[2]
    dagger_balanced = run_dagger(records, args.dagger_rounds, args.epochs, seeds[0], output / "dagger_balanced",
                                 episodes=specs, jepa=dagger_provider, use_jepa=True, class_balance=True)[2]
    causal = {
        "experiment": "v680.1-jepa-causal-1", "seeds": seeds, "smoke": args.smoke,
        "teacher_version": TEACHER_VERSION, "dataset_version": DATASET_VERSION,
        "student_version": STUDENT_VERSION, "jepa_version": JEPA_VERSION,
        "frozen_recipe": {"epochs": args.epochs, "samples_per_category": samples,
                          "student_architecture": "same 24D future feature pathway for every condition",
                          "conditions": CONDITIONS, "held_out_partition": "heldout"},
        "attention_accuracy_summary": {name: summary(values) for name, values in matrices.items()},
        "per_seed": per_seed,
        "random_anomaly_explained": (
            "Random controls are standardized to the learned JEPA mean/std and trained through the same "
            "24-dimensional pathway; compare five-seed summary before attributing any gain to randomness."),
        "ppo_ready": False,
        "ppo_blockers": ["JEPA causal utility comparison requires review", "PPO is outside this investigation"],
    }
    dagger_report = {"experiment": "v680.1-dagger-stability-1", "seed": seeds[0],
                     "raw": dagger_raw, "balanced": dagger_balanced}
    (output / "v680_1_jepa_causal_results.json").write_text(json.dumps(causal, indent=2, sort_keys=True))
    (output / "v680_1_dagger_report.json").write_text(json.dumps(dagger_report, indent=2, sort_keys=True))
    report = [
        "# V680.1 JEPA causal report",
        "", "## Frozen comparison", f"Seeds: {seeds}; examples/category: {samples}; epochs: {args.epochs}.",
        "All variants use the same student architecture, initialization seed, optimizer, data split, ordering, and training steps.",
        "Only the normalized 24D future-representation provider changes.",
        "", "## Held-out structural decision accuracy",
        *[f"- {name}: {values['mean']:.4f} ± {values['std']:.4f} (n={values['n']})"
          for name, values in causal["attention_accuracy_summary"].items()],
        "", "## Conclusion", causal["random_anomaly_explained"],
        "PPO remains gated pending causal-result review.",
    ]
    (output / "v680_1_jepa_causal_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(causal["attention_accuracy_summary"], indent=2, sort_keys=True))


@torch.no_grad()
def _provider_statistics(records, provider):
    values = []
    from attention_jepa import transition_records
    for state, _, _, _ in transition_records(records):
        values.append(provider.predict_actions(state))
    matrix = torch.cat(values)
    return {"mean": float(matrix.mean()), "std": float(matrix.std(unbiased=False)),
            "min": float(matrix.min()), "max": float(matrix.max()),
            "mean_l2_norm": float(matrix.norm(dim=1).mean()),
            "per_dimension_variance": matrix.var(dim=0, unbiased=False).tolist()}


if __name__ == "__main__":
    main()
