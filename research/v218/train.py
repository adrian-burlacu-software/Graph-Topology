
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from dataset import generate_dataset, save_dataset
from model import CognitiveModel
from state import ACTIONS, ACTION_TO_ID, Edge, Node, State
from v200_graph_transformer_cognitive.long_term_memory import RELATION_TO_ID


def state_from_json(p):
    return State(
        nodes=[
            Node(
                str(n["concept"]),
                float(n["activation"]),
                int(n["role"]),
                bool(n.get("persistent", False)),
            )
            for n in p["nodes"]
        ],
        edges=[
            Edge(
                str(e["source"]),
                str(e["relation"]),
                str(e["target"]),
                float(e["activation"]),
                bool(e.get("persistent", False)),
            )
            for e in p["edges"]
        ],
    )


def prepare(rows):
    prepared = []

    for row in rows:
        states = [
            state_from_json(x)
            for x in row["trajectory_states"]
        ]

        trajectory_targets = []

        for t, state in enumerate(states):
            names = [n.concept for n in state.nodes]
            action = row["trajectory_actions"][t]
            attention_truth = set(row["trajectory_attention"][t])

            trajectory_targets.append({
                "attention": torch.tensor(
                    [float(name in attention_truth) for name in names],
                    dtype=torch.float32,
                ),
                "action": ACTION_TO_ID[action["action"]],
                "source": (
                    names.index(action["source"])
                    if action["source"] in names else -1
                ),
                "target": (
                    names.index(action["target"])
                    if action["target"] in names else -1
                ),
                "relation": RELATION_TO_ID.get(
                    action["relation"], 0
                ),
            })

        # The depth-only task is explicitly the FINAL decision task.
        final_state = states[-1]
        final_action = row["final_action"]
        final_names = [n.concept for n in final_state.nodes]

        prepared.append({
            "initial_state": states[0],
            "final_state": final_state,
            "goal": row["goal"],
            "trajectory_states": states,
            "trajectory_targets": trajectory_targets,
            "final_target": {
                "action": ACTION_TO_ID[final_action["action"]],
                "source": (
                    final_names.index(final_action["source"])
                    if final_action["source"] in final_names else -1
                ),
                "target": (
                    final_names.index(final_action["target"])
                    if final_action["target"] in final_names else -1
                ),
                "relation": RELATION_TO_ID.get(
                    final_action["relation"], 0
                ),
                "attention": torch.tensor(
                    [
                        float(name in set(row["trajectory_attention"][-1]))
                        for name in final_names
                    ],
                    dtype=torch.float32,
                ),
            },
            "final_action_name": row["final_action"]["action"],
        })

    return prepared


def stratified_split(rows, seed, valid_fraction=0.15):
    groups = {action: [] for action in ACTIONS}
    rng = random.Random(seed)

    for i, row in enumerate(rows):
        groups[row["final_action"]["action"]].append(i)

    train_ids = []
    valid_ids = []

    for ids in groups.values():
        rng.shuffle(ids)
        n = max(1, int(len(ids) * valid_fraction))
        valid_ids.extend(ids[:n])
        train_ids.extend(ids[n:])

    rng.shuffle(train_ids)
    rng.shuffle(valid_ids)

    return train_ids, valid_ids


def metrics_update(counter, out, target):
    # Metrics are bookkeeping only, but the target tensors are deliberately
    # kept on CPU in the dataset. Move them to the prediction device here.
    prediction_device = out["attention_hard"].device

    counter["action"] += int(
        out["action_logits"].argmax().item() == target["action"]
    )

    counter["source"] += int(
        target["source"] < 0
        or out["source_logits"].argmax().item() == target["source"]
    )

    counter["target"] += int(
        target["target"] < 0
        or out["target_logits"].argmax().item() == target["target"]
    )

    counter["relation"] += int(
        out["relation_logits"].argmax().item() == target["relation"]
    )

    # IMPORTANT: target["attention"] originates in prepare() on CPU,
    # while model outputs are on CUDA during training.
    truth = target["attention"].to(
        device=prediction_device,
        dtype=torch.bool,
    )
    prediction = out["attention_hard"].to(
        device=prediction_device
    ) > 0.5

    counter["tp"] += int((prediction & truth).sum().item())
    counter["fp"] += int((prediction & ~truth).sum().item())
    counter["fn"] += int((~prediction & truth).sum().item())


def prf(counter):
    precision = counter["tp"] / max(
        1, counter["tp"] + counter["fp"]
    )
    recall = counter["tp"] / max(
        1, counter["tp"] + counter["fn"]
    )
    f1 = (
        2 * precision * recall / max(1e-9, precision + recall)
    )
    return precision, recall, f1


def loss_for_output(out, target, device):
    truth = target["attention"].to(device)

    action_loss = F.cross_entropy(
        out["action_logits"].unsqueeze(0),
        torch.tensor([target["action"]], device=device),
    )

    attention_loss = F.binary_cross_entropy_with_logits(
        out["attention_logits"],
        truth,
    )

    if target["source"] >= 0:
        source_loss = F.cross_entropy(
            out["source_logits"].unsqueeze(0),
            torch.tensor([target["source"]], device=device),
        )
    else:
        source_loss = out["source_logits"].sum() * 0.0

    if target["target"] >= 0:
        target_loss = F.cross_entropy(
            out["target_logits"].unsqueeze(0),
            torch.tensor([target["target"]], device=device),
        )
    else:
        target_loss = out["target_logits"].sum() * 0.0

    relation_loss = F.cross_entropy(
        out["relation_logits"].unsqueeze(0),
        torch.tensor([target["relation"]], device=device),
    )

    # Action remains the primary objective.
    return (
        action_loss
        + 0.35 * attention_loss
        + 0.35 * source_loss
        + 0.35 * target_loss
        + 0.25 * relation_loss
    )


def run_static_epoch(
    model,
    data,
    ids,
    device,
    optimizer,
    train,
    depth,
    progress_every,
):
    model.train(train)
    if not train:
        model.eval()

    total_loss = 0.0
    counter = Counter()
    start = time.perf_counter()

    for pos, idx in enumerate(ids, 1):
        item = data[idx]

        with torch.set_grad_enabled(train):
            out = model.forward_static(
                item["final_state"],
                item["goal"],
                device,
                transformer_depth=depth,
            )

            loss = loss_for_output(
                out,
                item["final_target"],
                device,
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()

        metrics_update(counter, out, item["final_target"])
        total_loss += loss.item()

        if (
            pos == 1
            or pos % progress_every == 0
            or pos == len(ids)
        ):
            elapsed = time.perf_counter() - start
            rate = pos / max(elapsed, 1e-9)
            eta = (len(ids) - pos) / max(rate, 1e-9)

            print(
                f"  [{'train' if train else 'valid'} "
                f"{pos}/{len(ids)}] "
                f"loss={total_loss / pos:.4f} "
                f"rate={rate:.2f}/s eta={eta:.1f}s",
                flush=True,
            )

    p, r, f1 = prf(counter)
    n = len(ids)

    return {
        "loss": total_loss / n,
        "action": counter["action"] / n,
        "source": counter["source"] / n,
        "target": counter["target"] / n,
        "relation": counter["relation"] / n,
        "hard_att_p": p,
        "hard_att_r": r,
        "hard_att_f1": f1,
    }


def run_iterative_epoch(
    model,
    data,
    ids,
    device,
    optimizer,
    train,
    depth,
    steps,
    progress_every,
):
    model.train(train)
    if not train:
        model.eval()

    if steps < 1:
        raise ValueError(f"iterative steps must be >= 1, got {steps}")

    total_loss = 0.0
    counter = Counter()
    total_steps = 0
    start = time.perf_counter()

    for pos, idx in enumerate(ids, 1):
        item = data[idx]

        with torch.set_grad_enabled(train):
            outputs = model.forward_iterative(
                item["trajectory_states"],
                item["goal"],
                device,
                transformer_depth=depth,
                max_steps=steps,
            )

            loss = torch.zeros((), device=device)

            for step_index, (out, target) in enumerate(
                zip(outputs, item["trajectory_targets"])
            ):
                # Later steps matter slightly more because they depend on
                # successful propagation of the iterative working state.
                step_weight = 1.0 + 0.15 * step_index

                loss = loss + step_weight * loss_for_output(
                    out,
                    target,
                    device,
                )

            loss = loss / len(outputs)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()

        for out, target in zip(
            outputs,
            item["trajectory_targets"],
        ):
            metrics_update(counter, out, target)

        total_steps += len(outputs)
        total_loss += loss.item()

        if (
            pos == 1
            or pos % progress_every == 0
            or pos == len(ids)
        ):
            elapsed = time.perf_counter() - start
            rate = pos / max(elapsed, 1e-9)
            eta = (len(ids) - pos) / max(rate, 1e-9)

            print(
                f"  [{'train' if train else 'valid'} "
                f"{pos}/{len(ids)}] "
                f"loss={total_loss / pos:.4f} "
                f"rate={rate:.2f}/s eta={eta:.1f}s",
                flush=True,
            )

    p, r, f1 = prf(counter)

    return {
        "loss": total_loss / len(ids),
        "action": counter["action"] / max(1, total_steps),
        "source": counter["source"] / max(1, total_steps),
        "target": counter["target"] / max(1, total_steps),
        "relation": counter["relation"] / max(1, total_steps),
        "hard_att_p": p,
        "hard_att_r": r,
        "hard_att_f1": f1,
    }


def run_experiment(
    args,
    family,
    depth,
    steps,
    data,
    train_ids,
    valid_ids,
):
    tag = f"v216_{family}_d{depth}_steps{steps}"

    print(
        f"\n{'=' * 72}\n"
        f"{tag}\n"
        f"{'=' * 72}",
        flush=True,
    )

    model = CognitiveModel(
        hidden_size=args.hidden_size,
        heads=args.heads,
        depth=depth,
        topk=args.topk,
        mode=family,
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_loss = float("inf")
    best_metrics = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        print(
            f"\nEPOCH {epoch}/{args.epochs} [{tag}]",
            flush=True,
        )

        if family == "depth":
            train_metrics = run_static_epoch(
                model, data, train_ids, args.device, optimizer,
                True, depth, args.progress_every,
            )
            valid_metrics = run_static_epoch(
                model, data, valid_ids, args.device, None,
                False, depth, args.progress_every,
            )
        else:
            train_metrics = run_iterative_epoch(
                model, data, train_ids, args.device, optimizer,
                True, depth, steps, args.progress_every,
            )
            valid_metrics = run_iterative_epoch(
                model, data, valid_ids, args.device, None,
                False, depth, steps, args.progress_every,
            )

        print(
            f"EPOCH {epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_action={train_metrics['action']:.4f} "
            f"train_hard_att_f1={train_metrics['hard_att_f1']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_action={valid_metrics['action']:.4f} "
            f"valid_hard_att_f1={valid_metrics['hard_att_f1']:.4f} "
            f"valid_hard_att_p={valid_metrics['hard_att_p']:.4f} "
            f"valid_hard_att_r={valid_metrics['hard_att_r']:.4f} "
            f"valid_source={valid_metrics['source']:.4f} "
            f"valid_target={valid_metrics['target']:.4f} "
            f"valid_relation={valid_metrics['relation']:.4f}",
            flush=True,
        )

        if valid_metrics["loss"] < best_loss:
            best_loss = valid_metrics["loss"]
            best_metrics = valid_metrics
            best_epoch = epoch

            path = args.output_dir / f"{tag}.pt"

            torch.save(
                {
                    "version": "v216",
                    "experiment": tag,
                    "family": family,
                    "transformer_depth": depth,
                    "iterative_steps": steps,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "valid": valid_metrics,
                },
                path,
            )

            print(
                f"checkpoint_saved: {path} epoch={epoch}",
                flush=True,
            )

    return {
        "experiment": tag,
        "family": family,
        "transformer_depth": depth,
        "iterative_steps": steps,
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "best": best_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="V216 controlled cognitive depth vs iterative-state experiment"
    )

    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=216)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=25)

    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[2, 4, 6, 8],
    )

    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[2, 4, 6, 8],
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/v216"),
    )

    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=Path("results/v216_iterative_dataset.jsonl"),
    )

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        "=== V216 TRUE ITERATIVE-STATE EXPERIMENT MATRIX ===",
        flush=True,
    )
    print("device:", args.device, flush=True)

    if args.device.type == "cuda":
        print(
            "gpu:",
            torch.cuda.get_device_name(0),
            flush=True,
        )

    print("generating shared dataset...", flush=True)

    rows = generate_dataset(
        args.samples,
        args.seed,
    )

    save_dataset(
        rows,
        args.dataset_output,
    )

    data = prepare(rows)

    train_ids, valid_ids = stratified_split(
        rows,
        args.seed,
    )

    print("dataset_size:", len(rows), flush=True)
    print(
        "action_counts:",
        dict(Counter(r["final_action"]["action"] for r in rows)),
        flush=True,
    )

    depths = [r["chain_depth"] for r in rows]

    print(
        "chain_depth:",
        min(depths),
        "-",
        max(depths),
        flush=True,
    )

    print("train_size:", len(train_ids), flush=True)
    print("valid_size:", len(valid_ids), flush=True)
    print(
        "dataset_saved:",
        args.dataset_output,
        flush=True,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    # A: Transformer depth only.
    # One static final decision; no iterative working state.
    for depth in args.depths:
        results.append(
            run_experiment(
                args,
                "depth",
                depth,
                1,
                data,
                train_ids,
                valid_ids,
            )
        )

    # B: TRUE iterative state.
    # One Transformer layer per cognitive step; working state persists.
    for steps in args.steps:
        # The model processes the full teacher trajectory. `steps` is recorded
        # as the intended recurrence budget; the dataset trajectory defines the
        # actual available teacher steps.
        results.append(
            run_experiment(
                args,
                "iterative",
                1,
                steps,
                data,
                train_ids,
                valid_ids,
            )
        )

    # C: Deep Transformer + TRUE iterative state.
    max_depth = max(args.depths)

    for steps in args.steps:
        results.append(
            run_experiment(
                args,
                "both",
                max_depth,
                steps,
                data,
                train_ids,
                valid_ids,
            )
        )

    summary_path = args.output_dir / "v216_summary.json"

    summary_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72, flush=True)
    print("V216 SUMMARY", flush=True)
    print("=" * 72, flush=True)

    for result in results:
        best = result["best"]

        print(
            f"{result['experiment']:36s} "
            f"action={best['action']:.4f} "
            f"hard_att={best['hard_att_f1']:.4f} "
            f"loss={result['best_valid_loss']:.4f} "
            f"epoch={result['best_epoch']}",
            flush=True,
        )

    print(
        "summary_saved:",
        summary_path,
        flush=True,
    )


if __name__ == "__main__":
    main()
