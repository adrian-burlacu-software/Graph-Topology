
from __future__ import annotations
import argparse, json, random, sys, time
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
        [Node(str(n["concept"]), float(n["activation"]), int(n["role"]), bool(n.get("persistent", False)))
         for n in p["nodes"]],
        [Edge(str(e["source"]), str(e["relation"]), str(e["target"]),
              float(e["activation"]), bool(e.get("persistent", False)))
         for e in p["edges"]],
    )


def prepare(rows):
    result = []
    for row in rows:
        states = [state_from_json(x) for x in row["trajectory_states"]]
        targets = []
        for t, state in enumerate(states):
            names = [n.concept for n in state.nodes]
            action = row["trajectory_actions"][t]
            attention = set(row["trajectory_attention"][t])
            targets.append({
                "attention": torch.tensor([float(n in attention) for n in names], dtype=torch.float32),
                "action": ACTION_TO_ID[action["action"]],
                "source": names.index(action["source"]) if action["source"] in names else -1,
                "target": names.index(action["target"]) if action["target"] in names else -1,
                "relation": RELATION_TO_ID.get(action["relation"], 0),
            })
        result.append({
            "initial_state": state_from_json(row["initial_state"]),
            "trajectory_states": states,
            "trajectory_targets": targets,
            "goal": row["goal"],
            "final_action": row["final_action"]["action"],
        })
    return result


def split(rows, seed, valid_fraction=0.15):
    groups = {a: [] for a in ACTIONS}
    rng = random.Random(seed)
    for i, row in enumerate(rows):
        groups[row["final_action"]["action"]].append(i)
    train_ids, valid_ids = [], []
    for ids in groups.values():
        rng.shuffle(ids)
        n = max(1, int(len(ids) * valid_fraction))
        valid_ids.extend(ids[:n])
        train_ids.extend(ids[n:])
    rng.shuffle(train_ids)
    rng.shuffle(valid_ids)
    return train_ids, valid_ids


def loss_for_output(out, target, device):
    truth = target["attention"].to(device)
    la = F.cross_entropy(out["action_logits"][None],
                         torch.tensor([target["action"]], device=device))
    lh = F.binary_cross_entropy_with_logits(out["attention_logits"], truth)
    ls = (F.cross_entropy(out["source_logits"][None],
                          torch.tensor([target["source"]], device=device))
          if target["source"] >= 0 else out["source_logits"].sum() * 0)
    lt = (F.cross_entropy(out["target_logits"][None],
                          torch.tensor([target["target"]], device=device))
          if target["target"] >= 0 else out["target_logits"].sum() * 0)
    lr = F.cross_entropy(out["relation_logits"][None],
                         torch.tensor([target["relation"]], device=device))
    return la + .35 * lh + .35 * ls + .35 * lt + .25 * lr


def metrics_update(c, out, target):
    c["action"] += int(out["action_logits"].argmax().item() == target["action"])
    c["source"] += int(target["source"] < 0 or out["source_logits"].argmax().item() == target["source"])
    c["target"] += int(target["target"] < 0 or out["target_logits"].argmax().item() == target["target"])
    c["relation"] += int(out["relation_logits"].argmax().item() == target["relation"])

    truth = target["attention"].to(out["attention_hard"].device, dtype=torch.bool)
    pred = out["attention_hard"] > .5
    c["tp"] += int((pred & truth).sum().item())
    c["fp"] += int((pred & ~truth).sum().item())
    c["fn"] += int((~pred & truth).sum().item())


def finish_metrics(c, steps):
    p = c["tp"] / max(1, c["tp"] + c["fp"])
    r = c["tp"] / max(1, c["tp"] + c["fn"])
    f1 = 2 * p * r / max(1e-9, p + r)
    n = max(1, steps)
    return {
        "action": c["action"] / n,
        "source": c["source"] / n,
        "target": c["target"] / n,
        "relation": c["relation"] / n,
        "hard_att_p": p,
        "hard_att_r": r,
        "hard_att_f1": f1,
    }


def teacher_epoch(model, data, ids, device, optimizer, train, depth, steps, every):
    model.train(train)
    total = 0.0
    c = Counter()
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
            for step_i, (out, target) in enumerate(
                zip(outputs, item["trajectory_targets"][:steps])
            ):
                loss = loss + (1 + .15 * step_i) * loss_for_output(out, target, device)
                metrics_update(c, out, target)
            loss = loss / max(1, len(outputs))
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total += loss.item()
        total_steps += len(outputs)

        if pos == 1 or pos % every == 0 or pos == len(ids):
            elapsed = time.perf_counter() - start
            rate = pos / max(elapsed, 1e-9)
            eta = (len(ids) - pos) / max(rate, 1e-9)
            print(f"  [{'train' if train else 'valid'} {pos}/{len(ids)}] "
                  f"loss={total/pos:.4f} rate={rate:.2f}/s eta={eta:.1f}s", flush=True)

    metrics = finish_metrics(c, total_steps)
    metrics["loss"] = total / len(ids)
    return metrics


@torch.no_grad()
def autonomous_eval(model, data, ids, device, depth, steps):
    model.eval()
    final_correct = 0
    exact = 0
    early_stop = 0
    cases = 0
    confusion = Counter()

    for idx in ids:
        item = data[idx]
        rollout = model.autonomous_rollout(
            item["initial_state"],
            item["goal"],
            device,
            steps=steps,
            transformer_depth=depth,
        )

        predicted = [x["action_id"] for x in rollout["outputs"]]
        truth = [x["action"] for x in item["trajectory_targets"][:steps]]

        pred_final = predicted[-1] if predicted else -1
        true_final = truth[-1]

        final_correct += int(pred_final == true_final)
        exact += int(predicted == truth)
        early_stop += int(len(predicted) < min(steps, len(truth)))

        confusion[(ACTIONS[true_final],
                   ACTIONS[pred_final] if pred_final >= 0 else "NONE")] += 1
        cases += 1

    return {
        "autonomous_final_action": final_correct / max(1, cases),
        "autonomous_exact_trajectory": exact / max(1, cases),
        "autonomous_early_stop": early_stop / max(1, cases),
        "autonomous_cases": cases,
        "confusion": {f"{a}->{b}": n for (a, b), n in confusion.items()},
    }


def run_experiment(args, family, depth, steps, data, train_ids, valid_ids):
    tag = f"v219_{family}_d{depth}_steps{steps}"
    print(f"\n{'='*72}\n{tag}\n{'='*72}", flush=True)

    model = CognitiveModel(
        hidden_size=args.hidden_size,
        heads=args.heads,
        depth=depth,
        topk=args.topk,
        mode=family,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_loss = float("inf")
    best = None
    best_epoch = 0
    checkpoint_path = args.output_dir / f"{tag}.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEPOCH {epoch}/{args.epochs} [{tag}]", flush=True)

        tr = teacher_epoch(model, data, train_ids, args.device, optimizer,
                           True, depth, steps, args.progress_every)
        va = teacher_epoch(model, data, valid_ids, args.device, None,
                           False, depth, steps, args.progress_every)

        print(
            f"EPOCH {epoch} train_loss={tr['loss']:.4f} "
            f"train_action={tr['action']:.4f} "
            f"train_att_f1={tr['hard_att_f1']:.4f} "
            f"valid_loss={va['loss']:.4f} "
            f"valid_action={va['action']:.4f} "
            f"valid_att_f1={va['hard_att_f1']:.4f} "
            f"valid_src={va['source']:.4f} "
            f"valid_tgt={va['target']:.4f} "
            f"valid_rel={va['relation']:.4f}",
            flush=True,
        )

        if va["loss"] < best_loss:
            best_loss = va["loss"]
            best = va
            best_epoch = epoch
            torch.save({
                "version": "v219",
                "experiment": tag,
                "family": family,
                "transformer_depth": depth,
                "iterative_steps": steps,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "valid": va,
            }, checkpoint_path)
            print(f"checkpoint_saved: {checkpoint_path} epoch={epoch}", flush=True)

    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    autonomous = autonomous_eval(
        model, data, valid_ids, args.device, depth, steps
    )

    print(
        f"AUTONOMOUS final_action={autonomous['autonomous_final_action']:.4f} "
        f"exact_trajectory={autonomous['autonomous_exact_trajectory']:.4f} "
        f"early_stop={autonomous['autonomous_early_stop']:.4f}",
        flush=True,
    )

    return {
        "experiment": tag,
        "family": family,
        "transformer_depth": depth,
        "iterative_steps": steps,
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "best": best,
        "autonomous": autonomous,
        "checkpoint": str(checkpoint_path),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=219)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--depths", type=int, nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--steps", type=int, nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--output-dir", type=Path, default=Path("results/v219"))
    p.add_argument("--dataset-output", type=Path,
                   default=Path("results/v219_closed_loop_dataset.jsonl"))
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== V219 CLOSED-LOOP COGNITIVE EXPERIMENT ===", flush=True)
    print("device:", args.device, flush=True)
    if args.device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0), flush=True)

    rows = generate_dataset(args.samples, args.seed)
    save_dataset(rows, args.dataset_output)
    data = prepare(rows)
    train_ids, valid_ids = split(rows, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("dataset_size:", len(rows), flush=True)
    print("action_counts:",
          dict(Counter(r["final_action"]["action"] for r in rows)), flush=True)
    print("train_size:", len(train_ids), "valid_size:", len(valid_ids), flush=True)
    print("dataset_saved:", args.dataset_output, flush=True)

    results = []

    # Baseline: depth only.
    for depth in args.depths:
        results.append(run_experiment(
            args, "depth", depth, 1, data, train_ids, valid_ids
        ))

    # True iterative state.
    for steps in args.steps:
        results.append(run_experiment(
            args, "iterative", 1, steps, data, train_ids, valid_ids
        ))

    # Deep Transformer + true iterative state.
    max_depth = max(args.depths)
    for steps in args.steps:
        results.append(run_experiment(
            args, "both", max_depth, steps, data, train_ids, valid_ids
        ))

    summary = args.output_dir / "v219_summary.json"
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "="*72, flush=True)
    print("V219 SUMMARY", flush=True)
    print("="*72, flush=True)
    for r in results:
        b = r["best"]
        a = r["autonomous"]
        print(
            f"{r['experiment']:36s} "
            f"teacher_action={b['action']:.4f} "
            f"teacher_loss={r['best_valid_loss']:.4f} "
            f"autonomous_action={a['autonomous_final_action']:.4f} "
            f"exact={a['autonomous_exact_trajectory']:.4f}",
            flush=True,
        )
    print("summary_saved:", summary, flush=True)


if __name__ == "__main__":
    main()
