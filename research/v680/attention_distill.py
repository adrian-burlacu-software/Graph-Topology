"""Train the student from serialized frozen-teacher distributions and actions."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

from attention_dataset import read_jsonl
from attention_student import NeuralAttentionPolicy, tensors_from_observation
from attention_types import AttentionObservation, validate_step_record


def flatten_steps(records):
    return [validate_step_record(step) for episode in records for step in episode["trajectory"]]


def training_records(records):
    """Keep validation and structurally held-out records out of all student fitting."""
    return [record for record in records if record["split"] != "held_out_structural"
            and record["split"] != "held_out_adversarial" and record.get("partition") != "validation"]


def teacher_action_kind(step):
    count = len(step["state"]["candidate_features"])
    selected = step["teacher"]["selected_action"]
    return "traverse" if selected < count else ("stop" if selected == count else "abstain")


def metadata(dataset, seed, **hyperparameters):
    path = Path(dataset)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    return {
        "git_revision": revision, "seed": seed, "dataset": str(path),
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "",
        "model_hyperparameters": hyperparameters,
        "optimizer_hyperparameters": {"name": "Adam", "lr": hyperparameters["learning_rate"]},
        "environment_configuration": {"bounded_actions": ["traverse", "stop", "abstain"]},
        "teacher_configuration": {"source": "frozen_v679"},
    }


def train_distillation(records, epochs=8, learning_rate=1e-3, temperature=2.0,
                       lambda_soft=1.0, lambda_rank=.2, lambda_hard=1.0,
                       seed=0, model=None, jepa=None, use_jepa=False):
    torch.manual_seed(seed); random.seed(seed)
    if use_jepa and jepa is None:
        raise ValueError("--use-jepa requires a trained JEPA model")
    model = model or NeuralAttentionPolicy(jepa_dim=jepa.representation_dim if use_jepa else 0)
    if bool(model.jepa_dim) != bool(use_jepa):
        raise ValueError("student JEPA feature dimension does not match use_jepa")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    records = training_records(records)
    steps = flatten_steps(records)
    if not steps:
        raise ValueError("distillation requires at least one teacher step")
    counts = {kind: sum(teacher_action_kind(step) == kind for step in steps)
              for kind in ("traverse", "stop", "abstain")}
    class_weights = {kind: len(steps) / (len(counts) * max(1, count))
                     for kind, count in counts.items()}
    for _ in range(int(epochs)):
        for episode in records:
            hidden, losses = None, []
            for step in episode["trajectory"]:
                step = validate_step_record(step)
                state = AttentionObservation.from_dict(step["state"])
                vector, candidates = tensors_from_observation(state)
                future = jepa.predict_actions(state).detach() if use_jepa else None
                logits, _, hidden = model(vector, candidates, hidden=hidden,
                                          action_mask=model.action_mask(state), future_representations=future)
                teacher_logits = torch.tensor(step["teacher"]["logits"], dtype=torch.float32)
                if len(teacher_logits) != len(logits):
                    raise ValueError("teacher logits do not match serialized candidate action space")
                selected = int(step["teacher"]["selected_action"])
                soft = F.kl_div(F.log_softmax(logits / temperature, -1),
                                F.softmax(teacher_logits / temperature, -1),
                                reduction="batchmean") * temperature ** 2
                hard = F.cross_entropy(logits.unsqueeze(0), torch.tensor([selected]))
                teacher_order = torch.argsort(teacher_logits, descending=True)
                rank = sum((F.softplus(-(logits[high] - logits[low]))
                            for high, low in zip(teacher_order[:-1], teacher_order[1:])), torch.tensor(0.0))
                losses.append(lambda_soft * soft + lambda_rank * rank
                              + lambda_hard * class_weights[teacher_action_kind(step)] * hard)
            optimizer.zero_grad(); torch.stack(losses).mean().backward(); optimizer.step()
    return model, optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lambda-soft", type=float, default=1.0)
    parser.add_argument("--lambda-rank", type=float, default=.2)
    parser.add_argument("--lambda-hard", type=float, default=1.0)
    parser.add_argument("--use-jepa", action="store_true")
    parser.add_argument("--jepa-checkpoint")
    args = parser.parse_args()
    records = read_jsonl(args.dataset)
    jepa = None
    if args.use_jepa:
        if not args.jepa_checkpoint:
            parser.error("--use-jepa requires --jepa-checkpoint")
        from attention_jepa import load_jepa
        jepa = load_jepa(args.jepa_checkpoint)
    model, optimizer = train_distillation(records, args.epochs, args.learning_rate,
                                          args.temperature, args.lambda_soft, args.lambda_rank,
                                          args.lambda_hard, args.seed, jepa=jepa, use_jepa=args.use_jepa)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "hidden_size": model.hidden_size, "jepa_dim": model.jepa_dim,
                "metadata": metadata(args.dataset, args.seed, epochs=args.epochs,
                                     learning_rate=args.learning_rate, temperature=args.temperature,
                                     lambda_soft=args.lambda_soft, lambda_rank=args.lambda_rank,
                                     lambda_hard=args.lambda_hard, use_jepa=args.use_jepa,
                                     jepa_checkpoint=args.jepa_checkpoint or "")}, args.checkpoint)


if __name__ == "__main__":
    main()
