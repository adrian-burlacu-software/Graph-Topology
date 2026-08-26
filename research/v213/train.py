from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_RESEARCH_ROOT = _HERE.parent
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

from dataset import generate_dataset, save_dataset
from model import TeacherDistilledController
from state import ACTIONS, ACTION_TO_ID, Edge, Node, State


def state_from_json(p):
    return State(
        nodes=[
            Node(
                str(n["concept"]),
                float(n.get("activation", 0)),
                int(n.get("role", 0)),
                bool(n.get("persistent", False)),
            )
            for n in p.get("nodes", [])
        ],
        edges=[
            Edge(
                str(e["source"]),
                str(e["relation"]),
                str(e["target"]),
                float(e.get("activation", 0)),
                bool(e.get("persistent", False)),
            )
            for e in p.get("edges", [])
        ],
    )


def prepare_rows(rows):
    """Precompute everything that does not depend on model parameters."""
    prepared = []

    # Relation IDs are stable and only needed once per record.
    from v200_graph_transformer_cognitive.long_term_memory import RELATION_TO_ID

    for r in rows:
        state = state_from_json(r["initial_state"])
        node_names = [n.concept for n in state.nodes]
        relevant = set(r["attention_target"])

        attention_target = [
            1.0 if name in relevant else 0.0
            for name in node_names
        ]

        action = r["action"]
        source = action.get("source")
        target = action.get("target")
        relation = action.get("relation")

        source_index = node_names.index(source) if source in node_names else -1
        target_index = node_names.index(target) if target in node_names else -1

        prepared.append(
            {
                "record": r,
                "state": state,
                "goal": r["goal"],
                "node_names": node_names,
                "attention_target": attention_target,
                "action_id": ACTION_TO_ID[r["final_action"]],
                "source_index": source_index,
                "target_index": target_index,
                "relation_id": RELATION_TO_ID.get(relation, 0),
                "chain_depth": r["chain_depth"],
            }
        )

    return prepared


def split_indices(rows, valid_fraction, seed):
    groups = {action: [] for action in ACTIONS}
    for i, row in enumerate(rows):
        groups[row["final_action"]].append(i)

    rng = random.Random(seed)
    train = []
    valid = []

    for ids in groups.values():
        rng.shuffle(ids)
        valid_count = max(1, int(len(ids) * valid_fraction))
        valid.extend(ids[:valid_count])
        train.extend(ids[valid_count:])

    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def run_epoch(
    model,
    prepared,
    indices,
    device,
    optimizer=None,
    train=False,
    progress_every=50,
    epoch=1,
    phase="train",
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    count = 0

    action_correct = 0
    source_correct = 0
    target_correct = 0
    relation_correct = 0

    att_tp = 0
    att_fp = 0
    att_fn = 0

    started = time.perf_counter()

    for position, idx in enumerate(indices, 1):
        item = prepared[idx]
        state = item["state"]
        goal = item["goal"]

        attention_target = torch.tensor(
            item["attention_target"],
            dtype=torch.float32,
            device=device,
        )

        action_target = torch.tensor(
            [item["action_id"]],
            dtype=torch.long,
            device=device,
        )

        with torch.set_grad_enabled(train):
            out = model(state, goal, device)

            action_loss = F.cross_entropy(
                out["action_logits"].unsqueeze(0),
                action_target,
            )

            attention_loss = F.binary_cross_entropy_with_logits(
                out["attention_logits"],
                attention_target,
            )

            if item["source_index"] >= 0:
                source_target = torch.tensor(
                    [item["source_index"]],
                    dtype=torch.long,
                    device=device,
                )
                source_loss = F.cross_entropy(
                    out["source_logits"].unsqueeze(0),
                    source_target,
                )
            else:
                source_loss = out["source_logits"].sum() * 0.0

            if item["target_index"] >= 0:
                target_target = torch.tensor(
                    [item["target_index"]],
                    dtype=torch.long,
                    device=device,
                )
                target_loss = F.cross_entropy(
                    out["target_logits"].unsqueeze(0),
                    target_target,
                )
            else:
                target_loss = out["target_logits"].sum() * 0.0

            relation_target = torch.tensor(
                [item["relation_id"]],
                dtype=torch.long,
                device=device,
            )
            relation_loss = F.cross_entropy(
                out["relation_logits"].unsqueeze(0),
                relation_target,
            )

            relevant_indices = [
                i
                for i, value in enumerate(item["attention_target"])
                if value > 0.5
            ]

            if relevant_indices:
                oracle_attention = out["node_state"][
                    relevant_indices
                ].mean(
                    dim=0,
                    keepdim=True,
                )

                next_loss = F.mse_loss(
                    out["attended_graph"],
                    oracle_attention,
                )
            else:
                next_loss = out["attended_graph"].sum() * 0.0

            loss = (
                action_loss
                + 0.7 * attention_loss
                + 0.5 * source_loss
                + 0.5 * target_loss
                + 0.4 * relation_loss
                + 0.15 * next_loss
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )
                optimizer.step()

        with torch.no_grad():
            predicted_action = out["action_logits"].argmax().item()
            action_correct += int(
                predicted_action == item["action_id"]
            )

            if item["source_index"] >= 0:
                source_correct += int(
                    out["source_logits"].argmax().item()
                    == item["source_index"]
                )
            else:
                source_correct += 1

            if item["target_index"] >= 0:
                target_correct += int(
                    out["target_logits"].argmax().item()
                    == item["target_index"]
                )
            else:
                target_correct += 1

            relation_correct += int(
                out["relation_logits"].argmax().item()
                == item["relation_id"]
            )

            predicted_attention = (
                torch.sigmoid(out["attention_logits"]) > 0.5
            )
            true_attention = attention_target > 0.5

            att_tp += int(
                (predicted_attention & true_attention)
                .sum()
                .item()
            )
            att_fp += int(
                (predicted_attention & ~true_attention)
                .sum()
                .item()
            )
            att_fn += int(
                (~predicted_attention & true_attention)
                .sum()
                .item()
            )

        total_loss += float(loss.item())
        count += 1

        if progress_every > 0 and (
            position == 1
            or position % progress_every == 0
            or position == len(indices)
        ):
            elapsed = time.perf_counter() - started
            rate = position / max(elapsed, 1e-9)
            remaining = (len(indices) - position) / max(rate, 1e-9)

            print(
                f"  {phase} [{position}/{len(indices)}] "
                f"loss={total_loss / count:.4f} "
                f"rate={rate:.2f}/s "
                f"eta={remaining:.1f}s",
                flush=True,
            )

    precision = att_tp / max(1, att_tp + att_fp)
    recall = att_tp / max(1, att_tp + att_fn)
    f1 = (
        2 * precision * recall / max(1e-9, precision + recall)
    )

    return {
        "loss": total_loss / max(1, count),
        "action": action_correct / max(1, count),
        "att_f1": f1,
        "att_p": precision,
        "att_r": recall,
        "src": source_correct / max(1, count),
        "tgt": target_correct / max(1, count),
        "rel": relation_correct / max(1, count),
        "seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(
        description="V213 deterministic multi-hop cognitive training"
    )
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=213)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/v213_multihop_oracle_dataset.jsonl"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "results/v213_multihop_cognitive.pt"
        ),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        "=== V213 DETERMINISTIC MULTI-HOP ORACLE TRAINING ===",
        flush=True,
    )
    print(f"device: {device}", flush=True)

    if device.type == "cuda":
        print(
            f"gpu: {torch.cuda.get_device_name(0)}",
            flush=True,
        )

    print("generating_dataset...", flush=True)
    rows = generate_dataset(
        samples=args.samples,
        seed=args.seed,
    )

    save_dataset(rows, args.output)

    counts = Counter(
        row["final_action"] for row in rows
    )

    depths = [
        row["chain_depth"]
        for row in rows
    ]

    print(
        f"dataset_size: {len(rows)}",
        flush=True,
    )
    print(
        f"action_counts: {dict(counts)}",
        flush=True,
    )
    print(
        f"chain_depth: {min(depths)} - {max(depths)}",
        flush=True,
    )
    print(
        f"dataset_saved: {args.output}",
        flush=True,
    )

    print("preparing_dataset...", flush=True)
    prepared = prepare_rows(rows)
    print(
        f"prepared_records: {len(prepared)}",
        flush=True,
    )

    train_idx, valid_idx = split_indices(
        rows,
        valid_fraction=0.15,
        seed=args.seed,
    )

    print(
        f"train_size: {len(train_idx)}",
        flush=True,
    )
    print(
        f"valid_size: {len(valid_idx)}",
        flush=True,
    )

    print("creating_model...", flush=True)

    model = TeacherDistilledController(
        hidden_size=args.hidden_size,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_loss = float("inf")
    best_epoch = 0

    print("starting_training...", flush=True)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        print(
            f"\nEPOCH {epoch}/{args.epochs}",
            flush=True,
        )

        train_metrics = run_epoch(
            model,
            prepared,
            train_idx,
            device,
            optimizer=optimizer,
            train=True,
            progress_every=args.progress_every,
            epoch=epoch,
            phase="train",
        )

        print(
            f"  train_complete "
            f"time={train_metrics['seconds']:.1f}s",
            flush=True,
        )

        valid_metrics = run_epoch(
            model,
            prepared,
            valid_idx,
            device,
            optimizer=None,
            train=False,
            progress_every=args.progress_every,
            epoch=epoch,
            phase="valid",
        )

        print(
            f"  valid_complete "
            f"time={valid_metrics['seconds']:.1f}s",
            flush=True,
        )

        print(
            f"EPOCH {epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_action={train_metrics['action']:.4f} "
            f"train_att_f1={train_metrics['att_f1']:.4f} "
            f"train_att_p={train_metrics['att_p']:.4f} "
            f"train_att_r={train_metrics['att_r']:.4f} "
            f"train_src={train_metrics['src']:.4f} "
            f"train_tgt={train_metrics['tgt']:.4f} "
            f"train_rel={train_metrics['rel']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_action={valid_metrics['action']:.4f} "
            f"valid_att_f1={valid_metrics['att_f1']:.4f} "
            f"valid_att_p={valid_metrics['att_p']:.4f} "
            f"valid_att_r={valid_metrics['att_r']:.4f} "
            f"valid_src={valid_metrics['src']:.4f} "
            f"valid_tgt={valid_metrics['tgt']:.4f} "
            f"valid_rel={valid_metrics['rel']:.4f} "
            f"epoch_time={time.perf_counter() - epoch_start:.1f}s",
            flush=True,
        )

        if valid_metrics["loss"] < best_loss:
            best_loss = valid_metrics["loss"]
            best_epoch = epoch

            args.checkpoint.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                {
                    "version": "v213",
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "valid": valid_metrics,
                    "args": vars(args),
                },
                args.checkpoint,
            )

            print(
                f"checkpoint_saved: {args.checkpoint} "
                f"epoch={epoch} "
                f"valid_loss={best_loss:.4f}",
                flush=True,
            )

    print(
        f"\nBEST_EPOCH: {best_epoch}",
        flush=True,
    )
    print(
        f"BEST_VALID_LOSS: {best_loss:.4f}",
        flush=True,
    )
    print(
        f"FINAL_CHECKPOINT: {args.checkpoint}",
        flush=True,
    )


if __name__ == "__main__":
    main()
