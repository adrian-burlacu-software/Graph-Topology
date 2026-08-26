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
from model import HardAttentionController
from state import ACTIONS, ACTION_TO_ID, Edge, Node, State

from v200_graph_transformer_cognitive.long_term_memory import RELATION_TO_ID


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
    out = []
    for r in rows:
        state = state_from_json(r["initial_state"])
        names = [n.concept for n in state.nodes]
        relevant = set(r["attention_target"])
        action = r["action"]

        out.append({
            "state": state,
            "goal": r["goal"],
            "names": names,
            "attention": torch.tensor(
                [1.0 if n in relevant else 0.0 for n in names],
                dtype=torch.float32,
            ),
            "action_id": ACTION_TO_ID[r["final_action"]],
            "source": names.index(action["source"]) if action["source"] in names else -1,
            "target": names.index(action["target"]) if action["target"] in names else -1,
            "relation": RELATION_TO_ID.get(action["relation"], 0),
            "depth": r["chain_depth"],
        })
    return out


def split(rows, valid_fraction, seed):
    groups = {a: [] for a in ACTIONS}
    for i, r in enumerate(rows):
        groups[r["final_action"]].append(i)

    rng = random.Random(seed)
    train, valid = [], []
    for ids in groups.values():
        rng.shuffle(ids)
        n = max(1, int(len(ids) * valid_fraction))
        valid.extend(ids[:n])
        train.extend(ids[n:])

    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def run_epoch(
    model,
    data,
    indices,
    device,
    optimizer=None,
    train=False,
    progress_every=25,
):
    model.train(train)
    if not train:
        model.eval()

    sums = Counter()
    total_loss = 0.0
    start = time.perf_counter()

    for pos, idx in enumerate(indices, 1):
        item = data[idx]

        with torch.set_grad_enabled(train):
            out = model(item["state"], item["goal"], device)

            action_target = torch.tensor(
                [item["action_id"]], dtype=torch.long, device=device
            )
            att_target = item["attention"].to(device)

            action_loss = F.cross_entropy(
                out["action_logits"].unsqueeze(0),
                action_target,
            )

            # Supervise soft attention, but do not let this dominate.
            attention_loss = F.binary_cross_entropy_with_logits(
                out["attention_logits"],
                att_target,
            )

            if item["source"] >= 0:
                src_loss = F.cross_entropy(
                    out["source_logits"].unsqueeze(0),
                    torch.tensor([item["source"]], device=device),
                )
            else:
                src_loss = out["source_logits"].sum() * 0

            if item["target"] >= 0:
                tgt_loss = F.cross_entropy(
                    out["target_logits"].unsqueeze(0),
                    torch.tensor([item["target"]], device=device),
                )
            else:
                tgt_loss = out["target_logits"].sum() * 0

            rel_loss = F.cross_entropy(
                out["relation_logits"].unsqueeze(0),
                torch.tensor([item["relation"]], device=device),
            )

            # Encourage the selected representation to equal the oracle
            # reasoning representation, without bypassing the hard mask.
            relevant = torch.nonzero(
                att_target > 0.5,
                as_tuple=False,
            ).squeeze(-1)

            oracle = out["node_state"][relevant].mean(
                dim=0, keepdim=True
            )

            attended_loss = F.mse_loss(
                out["attended_graph"],
                oracle,
            )

            # Keep action as the primary objective.
            loss = (
                1.0 * action_loss
                + 0.35 * attention_loss
                + 0.35 * src_loss
                + 0.35 * tgt_loss
                + 0.25 * rel_loss
                + 0.20 * attended_loss
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()

        with torch.no_grad():
            pred_action = out["action_logits"].argmax().item()
            sums["action"] += int(pred_action == item["action_id"])

            if item["source"] >= 0:
                sums["src"] += int(
                    out["source_logits"].argmax().item()
                    == item["source"]
                )
            else:
                sums["src"] += 1

            if item["target"] >= 0:
                sums["tgt"] += int(
                    out["target_logits"].argmax().item()
                    == item["target"]
                )
            else:
                sums["tgt"] += 1

            sums["rel"] += int(
                out["relation_logits"].argmax().item()
                == item["relation"]
            )

            # Evaluate both soft attention and the actual hard top-k mask.
            hard = out["attention_hard"] > 0.5
            truth = att_target > 0.5

            sums["soft_tp"] += int(
                ((out["attention_soft"] > 0.5) & truth).sum()
            )
            sums["soft_fp"] += int(
                ((out["attention_soft"] > 0.5) & ~truth).sum()
            )
            sums["soft_fn"] += int(
                ((out["attention_soft"] <= 0.5) & truth).sum()
            )

            sums["hard_tp"] += int((hard & truth).sum())
            sums["hard_fp"] += int((hard & ~truth).sum())
            sums["hard_fn"] += int((~hard & truth).sum())

            total_loss += loss.item()

        if (
            pos == 1
            or pos % progress_every == 0
            or pos == len(indices)
        ):
            elapsed = time.perf_counter() - start
            rate = pos / max(elapsed, 1e-9)
            eta = (len(indices) - pos) / max(rate, 1e-9)
            print(
                f"  [{'train' if train else 'valid'} "
                f"{pos}/{len(indices)}] "
                f"loss={total_loss/pos:.4f} "
                f"rate={rate:.2f}/s eta={eta:.1f}s",
                flush=True,
            )

    def prf(prefix):
        p = sums[f"{prefix}_tp"] / max(
            1, sums[f"{prefix}_tp"] + sums[f"{prefix}_fp"]
        )
        r = sums[f"{prefix}_tp"] / max(
            1, sums[f"{prefix}_tp"] + sums[f"{prefix}_fn"]
        )
        f = 2 * p * r / max(1e-9, p + r)
        return p, r, f

    soft_p, soft_r, soft_f1 = prf("soft")
    hard_p, hard_r, hard_f1 = prf("hard")

    n = len(indices)
    return {
        "loss": total_loss / max(1, n),
        "action": sums["action"] / n,
        "src": sums["src"] / n,
        "tgt": sums["tgt"] / n,
        "rel": sums["rel"] / n,
        "soft_p": soft_p,
        "soft_r": soft_r,
        "soft_f1": soft_f1,
        "hard_p": hard_p,
        "hard_r": hard_r,
        "hard_f1": hard_f1,
    }


def main():
    p = argparse.ArgumentParser(
        description="V214 hard attention cognitive training"
    )
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=214)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/v214_hard_attention_dataset.jsonl"),
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/v214_hard_attention_cognitive.pt"),
    )
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=== V214 HARD ATTENTION COGNITIVE TRAINING ===", flush=True)
    print("device:", device, flush=True)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0), flush=True)

    print("generating_dataset...", flush=True)
    rows = generate_dataset(args.samples, args.seed)
    save_dataset(rows, args.output)

    counts = Counter(r["final_action"] for r in rows)
    depths = [r["chain_depth"] for r in rows]

    print("dataset_size:", len(rows), flush=True)
    print("action_counts:", dict(counts), flush=True)
    print(
        "chain_depth:", min(depths), "-", max(depths),
        flush=True,
    )
    print("dataset_saved:", args.output, flush=True)

    print("preparing_dataset...", flush=True)
    data = prepare_rows(rows)

    train_idx, valid_idx = split(
        rows, valid_fraction=0.15, seed=args.seed
    )

    print("train_size:", len(train_idx), flush=True)
    print("valid_size:", len(valid_idx), flush=True)

    model = HardAttentionController(
        hidden_size=args.hidden_size,
        topk=args.topk,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEPOCH {epoch}/{args.epochs}", flush=True)

        train_m = run_epoch(
            model,
            data,
            train_idx,
            device,
            optimizer=optimizer,
            train=True,
            progress_every=args.progress_every,
        )

        valid_m = run_epoch(
            model,
            data,
            valid_idx,
            device,
            optimizer=None,
            train=False,
            progress_every=args.progress_every,
        )

        print(
            f"EPOCH {epoch} "
            f"train_loss={train_m['loss']:.4f} "
            f"train_action={train_m['action']:.4f} "
            f"train_hard_att_f1={train_m['hard_f1']:.4f} "
            f"train_soft_att_f1={train_m['soft_f1']:.4f} "
            f"train_src={train_m['src']:.4f} "
            f"train_tgt={train_m['tgt']:.4f} "
            f"train_rel={train_m['rel']:.4f} "
            f"valid_loss={valid_m['loss']:.4f} "
            f"valid_action={valid_m['action']:.4f} "
            f"valid_hard_att_f1={valid_m['hard_f1']:.4f} "
            f"valid_hard_att_p={valid_m['hard_p']:.4f} "
            f"valid_hard_att_r={valid_m['hard_r']:.4f} "
            f"valid_soft_att_f1={valid_m['soft_f1']:.4f} "
            f"valid_src={valid_m['src']:.4f} "
            f"valid_tgt={valid_m['tgt']:.4f} "
            f"valid_rel={valid_m['rel']:.4f}",
            flush=True,
        )

        # We care about the downstream task first. Attention is a constraint,
        # not the score we are trying to maximize in isolation.
        selection_score = (
            valid_m["action"]
            + 0.25 * valid_m["hard_f1"]
            - 0.05 * valid_m["loss"]
        )

        if valid_m["loss"] < best_loss:
            best_loss = valid_m["loss"]
            best_epoch = epoch

            args.checkpoint.parent.mkdir(
                parents=True, exist_ok=True
            )

            torch.save(
                {
                    "version": "v214",
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "valid": valid_m,
                    "selection_score": selection_score,
                    "args": vars(args),
                },
                args.checkpoint,
            )
            print(
                "checkpoint_saved:",
                args.checkpoint,
                f"epoch={epoch}",
                flush=True,
            )

    print("\nBEST_EPOCH:", best_epoch, flush=True)
    print("BEST_VALID_LOSS:", best_loss, flush=True)
    print("FINAL_CHECKPOINT:", args.checkpoint, flush=True)


if __name__ == "__main__":
    main()
