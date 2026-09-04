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
                       seed=0, model=None):
    torch.manual_seed(seed); random.seed(seed)
    model = model or NeuralAttentionPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    steps = flatten_steps(records)
    if not steps:
        raise ValueError("distillation requires at least one teacher step")
    for _ in range(int(epochs)):
        for step in steps:
            state = AttentionObservation.from_dict(step["state"])
            vector, candidates = tensors_from_observation(state)
            logits, _, _ = model(vector, candidates, action_mask=model.action_mask(state))
            teacher_logits = torch.tensor(step["teacher"]["logits"], dtype=torch.float32)
            if len(teacher_logits) != len(logits):
                raise ValueError("teacher logits do not match serialized candidate action space")
            selected = int(step["teacher"]["selected_action"])
            soft = F.kl_div(F.log_softmax(logits / temperature, -1),
                            F.softmax(teacher_logits / temperature, -1),
                            reduction="batchmean") * temperature ** 2
            hard = F.cross_entropy(logits.unsqueeze(0), torch.tensor([selected]))
            teacher_order = torch.argsort(teacher_logits, descending=True)
            rank = torch.tensor(0.0)
            for high, low in zip(teacher_order[:-1], teacher_order[1:]):
                rank = rank + F.softplus(-(logits[high] - logits[low]))
            loss = lambda_soft * soft + lambda_rank * rank + lambda_hard * hard
            optimizer.zero_grad(); loss.backward(); optimizer.step()
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
    args = parser.parse_args()
    records = read_jsonl(args.dataset)
    model, optimizer = train_distillation(records, args.epochs, args.learning_rate,
                                          args.temperature, args.lambda_soft, args.lambda_rank,
                                          args.lambda_hard, args.seed)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "metadata": metadata(args.dataset, args.seed, epochs=args.epochs,
                                     learning_rate=args.learning_rate, temperature=args.temperature,
                                     lambda_soft=args.lambda_soft, lambda_rank=args.lambda_rank,
                                     lambda_hard=args.lambda_hard)}, args.checkpoint)


if __name__ == "__main__":
    main()
