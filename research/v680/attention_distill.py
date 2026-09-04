"""KL, pairwise ranking, hard action, and invalid-action-masked distillation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from attention_dataset import collect_teacher_episodes, read_jsonl, write_jsonl
from attention_student import NeuralAttentionPolicy


def losses(student_logits, teacher_logits, selected_action, temperature=2.0,
           soft_weight=1.0, rank_weight=1.0, hard_weight=.25, mask_weight=1.0):
    teacher = torch.tensor(teacher_logits, dtype=torch.float32)
    valid = torch.isfinite(student_logits)
    student = student_logits[valid]
    teacher = teacher[valid]
    soft = F.kl_div(
        F.log_softmax(student / temperature, dim=-1),
        F.softmax(teacher / temperature, dim=-1), reduction="batchmean",
    ) * temperature ** 2
    differences = teacher[:, None] - teacher[None, :]
    pairs = differences > 0
    rank = F.softplus(-(student[:, None] - student[None, :]))[pairs].mean() if pairs.any() else student.sum() * 0
    hard = F.cross_entropy(student.unsqueeze(0), torch.tensor([selected_action]))
    invalid = student_logits[~valid].exp().sum() if (~valid).any() else student.sum() * 0
    total = soft_weight * soft + rank_weight * rank + hard_weight * hard + mask_weight * invalid
    return total, {"soft": float(soft), "rank": float(rank), "hard": float(hard), "invalid": float(invalid)}


def train(records, epochs=80, learning_rate=.01, **weights):
    torch.manual_seed(680)
    model = NeuralAttentionPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    states = [step for record in records for step in record["trajectory"]]
    for _ in range(int(epochs)):
        for step in states:
            state = step["state"]
            vectors = [item["features"] for item in state["candidate_features"]]
            state_vector = torch.tensor([[
                state["step"], state["remaining_budget"], len(vectors),
                len(state["visited_nodes"]), len(state["visited_relations"]),
                state["relation_activation"].get(state["goal_relation"], 0.0),
            ]], dtype=torch.float32)
            candidates = torch.tensor([[item[key] for key in (
                "path_length", "goal_relation_match", "target_term_match", "specificity",
                "lexical_score", "relation_activation", "candidate_activation", "provenance",
                "verified", "contradiction", "direct_proof", "already_visited",
            )] for item in vectors], dtype=torch.float32)
            logits, _, _ = model(state_vector, candidates)
            loss, _ = losses(logits, step["teacher"]["logits"], step["teacher"]["selected_action"], **weights)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model


def main():
    parser = argparse.ArgumentParser(description="Train V680 behavioral-cloned attention policy.")
    parser.add_argument("--dataset", default="./results/v680/distillation_dataset.jsonl")
    parser.add_argument("--database", default="", help="Frozen V679 semantic graph SQLite database.")
    parser.add_argument("--teacher-output", default="./results/v680/teacher_trajectories.jsonl")
    parser.add_argument("--checkpoint", default="./results/v680/student_checkpoint.pt")
    parser.add_argument("--trace-output", default="./results/v680/attention_traces.jsonl")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--soft-weight", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--hard-weight", type=float, default=.25)
    args = parser.parse_args()
    records = read_jsonl(args.dataset) if Path(args.dataset).exists() else collect_teacher_episodes(database=args.database)
    write_jsonl(args.dataset, records); write_jsonl(args.teacher_output, records)
    write_jsonl(
        args.trace_output,
        [
            {
                "episode_id": record["episode_id"], "split": record["split"],
                "state": step["state"], "candidates": step["candidates"],
                "teacher": step["teacher"], "selected_action": step["action"],
                "resulting_state": step["resulting_state"], "reward": step["reward"],
                "terminal_outcome": step["terminal_outcome"],
            }
            for record in records for step in record["trajectory"]
        ],
    )
    model = train(records, args.epochs, soft_weight=args.soft_weight, rank_weight=args.rank_weight, hard_weight=args.hard_weight)
    path = Path(args.checkpoint); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hidden_size": model.hidden_size}, path)
    print(f"trained {len(records)} teacher episodes -> {path}")


if __name__ == "__main__":
    main()
