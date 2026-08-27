
from __future__ import annotations

import argparse
import os
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


# ---------------------------------------------------------------------------
# Parsing / dataset
# ---------------------------------------------------------------------------

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
    result = []

    for row in rows:
        states = [state_from_json(x) for x in row["trajectory_states"]]
        initial = state_from_json(row["initial_state"])
        targets = []

        for t, state in enumerate(states):
            names = [n.concept for n in state.nodes]
            action = row["trajectory_actions"][t]
            attention = set(row["trajectory_attention"][t])

            targets.append({
                "attention_concepts": tuple(attention),
                "attention": torch.tensor(
                    [float(n in attention) for n in names],
                    dtype=torch.float32,
                ),
                "action": ACTION_TO_ID[action["action"]],
                "action_name": action["action"],
                "source_concept": action["source"],
                "target_concept": action["target"],
                "relation": RELATION_TO_ID.get(action["relation"], 0),
            })

        final = row["final_action"]

        result.append({
            "case_id": row["case_id"],
            "initial_state": initial,
            "trajectory_states": states,
            "trajectory_targets": targets,
            "goal": row["goal"],
            "final_action": ACTION_TO_ID[final["action"]],
            "final_action_name": final["action"],
        })

    return result


def stratified_split(rows, seed, valid_fraction=0.15):
    groups = {a: [] for a in ACTIONS}
    rng = random.Random(seed)

    for i, row in enumerate(rows):
        groups[row["final_action"]["action"]].append(i)

    train_ids, valid_ids = [], []

    for action, ids in groups.items():
        if not ids:
            raise ValueError(f"Missing action category: {action}")

        rng.shuffle(ids)
        n = max(1, int(len(ids) * valid_fraction))

        valid_ids.extend(ids[:n])
        train_ids.extend(ids[n:])

    rng.shuffle(train_ids)
    rng.shuffle(valid_ids)

    train_cases = {rows[i]["case_id"] for i in train_ids}
    valid_cases = {rows[i]["case_id"] for i in valid_ids}

    overlap = train_cases & valid_cases
    if overlap:
        raise AssertionError(
            f"Train/validation case leakage: {sorted(overlap)[:5]}"
        )

    return train_ids, valid_ids


def sanity_check_dataset(rows, data):
    counts = Counter(r["final_action"]["action"] for r in rows)

    if set(counts) != set(ACTIONS):
        raise AssertionError(
            f"Dataset action coverage failure: {dict(counts)}"
        )

    if sum(counts.values()) != len(rows):
        raise AssertionError("Dataset count mismatch.")

    for row, item in zip(rows, data):
        if row["final_action"]["action"] != item["final_action_name"]:
            raise AssertionError(
                f"Final-action mismatch: {row['case_id']}"
            )

        actions = row["trajectory_actions"]

        if actions[-1]["action"] != row["final_action"]["action"]:
            raise AssertionError(
                f"Final target is not trajectory terminal action: {row['case_id']}"
            )

        if len(row["trajectory_states"]) != len(actions):
            raise AssertionError(
                f"Trajectory length mismatch: {row['case_id']}"
            )

        if len(row["trajectory_attention"]) != len(actions):
            raise AssertionError(
                f"Attention trajectory length mismatch: {row['case_id']}"
            )

    # The most important V219 regression check:
    # step 0 must NOT be treated as the final action for non-NOOP examples.
    non_noop = [
        r for r in rows
        if len(r["trajectory_actions"]) > 1
    ]
    if non_noop and all(
        r["trajectory_actions"][0]["action"] == r["final_action"]["action"]
        for r in non_noop
    ):
        raise AssertionError(
            "Regression: trajectory step 0 is being used as final target."
        )


# ---------------------------------------------------------------------------
# Metrics / losses
# ---------------------------------------------------------------------------

def aligned_attention_target(out, target, state, device):
    """Create the attention target against the CURRENT state's node list."""
    names = [n.concept for n in state.nodes]
    concepts = set(target["attention_concepts"])
    truth = torch.tensor(
        [float(name in concepts) for name in names],
        dtype=torch.float32,
        device=device,
    )
    if truth.shape != out["attention_logits"].shape:
        raise AssertionError(
            "Attention alignment invariant violated: "
            f"state_nodes={len(names)} "
            f"target_shape={tuple(truth.shape)} "
            f"logit_shape={tuple(out['attention_logits'].shape)}"
        )
    return truth


def metrics_update(counter, out, target, state):
    counter["action"] += int(
        out["action_logits"].argmax().item() == target["action"]
    )

    names = [n.concept for n in state.nodes]
    truth = aligned_attention_target(
        out, target, state, out["attention_hard"].device
    )
    truth_bool = truth.to(dtype=torch.bool)
    pred = out["attention_hard"] > 0.5

    if pred.shape != truth_bool.shape:
        raise AssertionError(
            "Attention metric invariant violated: "
            f"pred_shape={tuple(pred.shape)} "
            f"truth_shape={tuple(truth_bool.shape)}"
        )

    source_ok = (
        target["source_concept"] is None
        or (
            target["source_concept"] in names
            and out["source_logits"].argmax().item()
            == names.index(target["source_concept"])
        )
    )
    target_ok = (
        target["target_concept"] is None
        or (
            target["target_concept"] in names
            and out["target_logits"].argmax().item()
            == names.index(target["target_concept"])
        )
    )

    counter["source"] += int(source_ok)
    counter["target"] += int(target_ok)
    counter["relation"] += int(
        out["relation_logits"].argmax().item() == target["relation"]
    )

    counter["tp"] += int((pred & truth_bool).sum().item())
    counter["fp"] += int((pred & ~truth_bool).sum().item())
    counter["fn"] += int((~pred & truth_bool).sum().item())

def attention_scores(counter):
    p = counter["tp"] / max(1, counter["tp"] + counter["fp"])
    r = counter["tp"] / max(1, counter["tp"] + counter["fn"])
    f1 = 2 * p * r / max(1e-9, p + r)
    return p, r, f1


def loss_for_output(out, target, state, device):
    names = [n.concept for n in state.nodes]

    # The CURRENT state determines the attention-logit length. Never use the
    # cached oracle-sized tensor here.
    truth = aligned_attention_target(
        out, target, state, device
    )

    action_loss = F.cross_entropy(
        out["action_logits"][None],
        torch.tensor([target["action"]], device=device),
    )

    attention_loss = F.binary_cross_entropy_with_logits(
        out["attention_logits"],
        truth,
    )

    if target["source_concept"] in names:
        source_loss = F.cross_entropy(
            out["source_logits"][None],
            torch.tensor(
                [names.index(target["source_concept"])],
                device=device,
            ),
        )
    else:
        source_loss = out["source_logits"].sum() * 0.0

    if target["target_concept"] in names:
        target_loss = F.cross_entropy(
            out["target_logits"][None],
            torch.tensor(
                [names.index(target["target_concept"])],
                device=device,
            ),
        )
    else:
        target_loss = out["target_logits"].sum() * 0.0

    relation_loss = F.cross_entropy(
        out["relation_logits"][None],
        torch.tensor([target["relation"]], device=device),
    )

    return (
        action_loss
        + 0.35 * attention_loss
        + 0.35 * source_loss
        + 0.35 * target_loss
        + 0.25 * relation_loss
    )


# ---------------------------------------------------------------------------
# State transition
# ---------------------------------------------------------------------------

def predicted_transition(model, state, out):
    action_id = int(out["action_logits"].argmax().item())

    names = [n.concept for n in state.nodes]

    source_i = int(out["source_logits"].argmax().item())
    target_i = int(out["target_logits"].argmax().item())

    source = names[source_i] if 0 <= source_i < len(names) else None
    target = names[target_i] if 0 <= target_i < len(names) else None

    relation_id = int(out["relation_logits"].argmax().item())
    relation = next(
        (
            name for name, rid in RELATION_TO_ID.items()
            if rid == relation_id
        ),
        None,
    )

    return state.apply(
        action_id,
        source=source,
        target=target,
        relation=relation,
    )


def state_distance(a, b):
    a_nodes = {n.concept for n in a.nodes}
    b_nodes = {n.concept for n in b.nodes}

    a_edges = {(e.source, e.relation, e.target) for e in a.edges}
    b_edges = {(e.source, e.relation, e.target) for e in b.edges}

    node_union = len(a_nodes | b_nodes)
    edge_union = len(a_edges | b_edges)

    node_error = (
        1.0 - len(a_nodes & b_nodes) / node_union
        if node_union else 0.0
    )
    edge_error = (
        1.0 - len(a_edges & b_edges) / edge_union
        if edge_union else 0.0
    )

    return {
        "node_error": node_error,
        "edge_error": edge_error,
    }


# ---------------------------------------------------------------------------
# Training regimes
# ---------------------------------------------------------------------------

def schedule_probability(epoch, epochs, regime):
    if regime == "teacher":
        return 0.0
    if regime == "free":
        return 1.0

    # Scheduled sampling: do not jump immediately from oracle to free-run.
    if epochs <= 1:
        return 1.0

    return min(1.0, (epoch - 1) / max(1, epochs - 1))


def recurrent_epoch(
    model,
    data,
    ids,
    device,
    optimizer,
    train,
    depth,
    steps,
    regime,
    epoch,
    epochs,
    progress_every,
    rng,
):
    model.train(train)

    total_loss = 0.0
    total_decisions = 0
    c = Counter()
    generated_decisions = 0
    start = time.perf_counter()

    p_model = schedule_probability(epoch, epochs, regime)

    for pos, idx in enumerate(ids, 1):
        item = data[idx]

        current = item["initial_state"].clone()
        working = torch.zeros(
            (1, model.hidden_size),
            device=device,
        )

        sequence_loss = torch.zeros((), device=device)
        sequence_outputs = []

        with torch.set_grad_enabled(train):
            for t in range(min(steps, len(item["trajectory_states"]))):
                # At t=0 the model must always see the real initial state.
                # Afterwards scheduled/free regimes choose between the oracle
                # state and the model-generated state.
                if t > 0 and regime != "teacher":
                    use_model_state = (
                        regime == "free"
                        or rng.random() < p_model
                    )
                    if not use_model_state:
                        current = item["trajectory_states"][t].clone()
                    else:
                        generated_decisions += 1

                out = model.cognitive_step(
                    current,
                    item["goal"],
                    working,
                    depth,
                    device,
                )

                target = item["trajectory_targets"][t]

                sequence_loss = sequence_loss + (
                    (1.0 + 0.15 * t)
                    * loss_for_output(
                        out, target, current, device
                    )
                )

                metrics_update(c, out, target, current)
                sequence_outputs.append((out, target))

                # Carry the latent working state continuously.
                working = out["next_working"]

                # For the next graph state, only generated regimes use the
                # model's discrete action. The actual state transition is
                # deliberately non-differentiable.
                if t + 1 < min(
                    steps,
                    len(item["trajectory_states"]),
                ):
                    if regime == "teacher":
                        current = item["trajectory_states"][t + 1].clone()
                    elif regime == "free" or rng.random() < p_model:
                        with torch.no_grad():
                            current = predicted_transition(
                                model,
                                current,
                                out,
                            )
                        generated_decisions += 1
                    else:
                        current = item["trajectory_states"][t + 1].clone()

            loss = sequence_loss / max(1, len(sequence_outputs))

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()

        total_loss += loss.item()
        total_decisions += len(sequence_outputs)

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
                f"loss={total_loss/pos:.4f} "
                f"rate={rate:.2f}/s eta={eta:.1f}s "
                f"model_state_p={p_model:.2f}",
                flush=True,
            )

    p, r, f1 = attention_scores(c)

    return {
        "loss": total_loss / max(1, len(ids)),
        "action": c["action"] / max(1, total_decisions),
        "source": c["source"] / max(1, total_decisions),
        "target": c["target"] / max(1, total_decisions),
        "relation": c["relation"] / max(1, total_decisions),
        "hard_att_p": p,
        "hard_att_r": r,
        "hard_att_f1": f1,
        "generated_decisions": generated_decisions,
    }


# ---------------------------------------------------------------------------
# One-shot depth control: FINAL action from INITIAL state.
# This fixes the central V219 evaluation error.
# ---------------------------------------------------------------------------

def oneshot_epoch(
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

    total_loss = 0.0
    correct = 0
    start = time.perf_counter()

    for pos, idx in enumerate(ids, 1):
        item = data[idx]

        # The target is explicitly the terminal action.
        final_target = item["trajectory_targets"][-1]

        with torch.set_grad_enabled(train):
            working = torch.zeros(
                (1, model.hidden_size),
                device=device,
            )

            out = model.cognitive_step(
                item["initial_state"],
                item["goal"],
                working,
                depth,
                device,
            )

            loss = loss_for_output(
                out,
                final_target,
                item["initial_state"],
                device,
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()

        prediction = int(out["action_logits"].argmax().item())
        correct += int(prediction == item["final_action"])
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
                f"loss={total_loss/pos:.4f} "
                f"action={correct/pos:.4f} "
                f"rate={rate:.2f}/s eta={eta:.1f}s",
                flush=True,
            )

    return {
        "loss": total_loss / max(1, len(ids)),
        "action": correct / max(1, len(ids)),
    }


# ---------------------------------------------------------------------------
# Autonomous diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def autonomous_diagnostics(
    model,
    data,
    ids,
    device,
    depth,
    steps,
):
    model.eval()

    per_step = [
        {
            "cases": 0,
            "correct": 0,
            "node_error": 0.0,
            "edge_error": 0.0,
        }
        for _ in range(steps)
    ]

    final_correct = 0
    exact = 0
    cases = 0
    divergence_cases = 0

    for idx in ids:
        item = data[idx]

        rollout = model.autonomous_rollout(
            item["initial_state"],
            item["goal"],
            device,
            steps=steps,
            transformer_depth=depth,
            stop_on_terminal=False,
        )

        predictions = [
            x["action_id"]
            for x in rollout["outputs"]
        ]

        truth = [
            x["action"]
            for x in item["trajectory_targets"][:steps]
        ]

        # Compare each generated state to the corresponding oracle state
        # BEFORE the corresponding action.
        for t in range(min(steps, len(predictions))):
            if t >= len(item["trajectory_states"]):
                break

            metric = state_distance(
                rollout["states"][t],
                item["trajectory_states"][t],
            )

            per_step[t]["cases"] += 1
            per_step[t]["correct"] += int(
                predictions[t] == truth[t]
            )
            per_step[t]["node_error"] += metric["node_error"]
            per_step[t]["edge_error"] += metric["edge_error"]

        if predictions:
            final_correct += int(
                predictions[-1] == item["final_action"]
            )

        exact += int(
            predictions == truth
        )

        if any(
            state_distance(
                rollout["states"][t],
                item["trajectory_states"][t],
            )["node_error"] > 0.0
            or state_distance(
                rollout["states"][t],
                item["trajectory_states"][t],
            )["edge_error"] > 0.0
            for t in range(
                min(steps, len(item["trajectory_states"]))
            )
        ):
            divergence_cases += 1

        cases += 1

    step_metrics = []

    for t, x in enumerate(per_step):
        n = max(1, x["cases"])
        step_metrics.append({
            "step": t + 1,
            "action_accuracy": x["correct"] / n,
            "node_error": x["node_error"] / n,
            "edge_error": x["edge_error"] / n,
            "cases": x["cases"],
        })

    return {
        "autonomous_final_action": final_correct / max(1, cases),
        "autonomous_exact_trajectory": exact / max(1, cases),
        "autonomous_cases": cases,
        "state_diverged_cases": divergence_cases,
        "state_diverged_fraction": divergence_cases / max(1, cases),
        "per_step": step_metrics,
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    args,
    regime,
    steps,
    depth,
    data,
    train_ids,
    valid_ids,
):
    tag = f"v224_{regime}_d{depth}_steps{steps}"

    print(
        f"\n{'='*78}\n{tag}\n{'='*78}",
        flush=True,
    )

    model = CognitiveModel(
        hidden_size=args.hidden_size,
        heads=args.heads,
        depth=depth,
        topk=args.topk,
        mode="both" if regime != "oneshot" else "depth",
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_loss = float("inf")
    best_epoch = 0
    best_metrics = None
    checkpoint = args.output_dir / f"{tag}.pt"

    rng = random.Random(
        args.seed
        + hash(regime) % 100000
        + steps * 1000
        + depth * 1000000
    )

    for epoch in range(1, args.epochs + 1):
        print(
            f"\nEPOCH {epoch}/{args.epochs} [{tag}]",
            flush=True,
        )

        if regime == "oneshot":
            train_metrics = oneshot_epoch(
                model, data, train_ids, args.device,
                optimizer, True, depth, args.progress_every,
            )
            valid_metrics = oneshot_epoch(
                model, data, valid_ids, args.device,
                None, False, depth, args.progress_every,
            )
        else:
            train_metrics = recurrent_epoch(
                model, data, train_ids, args.device,
                optimizer, True, depth, steps, regime,
                epoch, args.epochs, args.progress_every, rng,
            )
            valid_metrics = recurrent_epoch(
                model, data, valid_ids, args.device,
                None, False, depth, steps, regime,
                epoch, args.epochs, args.progress_every, rng,
            )

        print(
            f"EPOCH {epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_action={train_metrics['action']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_action={valid_metrics['action']:.4f}",
            flush=True,
        )

        if valid_metrics["loss"] < best_loss:
            best_loss = valid_metrics["loss"]
            best_epoch = epoch
            best_metrics = valid_metrics

            torch.save(
                {
                    "version": "v224",
                    "experiment": tag,
                    "regime": regime,
                    "transformer_depth": depth,
                    "steps": steps,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": (
                        optimizer.state_dict()
                        if optimizer is not None
                        else None
                    ),
                    "valid": valid_metrics,
                },
                checkpoint,
            )

            print(
                f"checkpoint_saved: {checkpoint} epoch={epoch}",
                flush=True,
            )

    # One-shot experiments are not recurrence experiments, but we still run
    # the same final-action diagnostic from the initial state for comparability.
    if regime == "oneshot":
        auto = autonomous_diagnostics(
            model, data, valid_ids, args.device, depth, 1
        )
    else:
        state = torch.load(
            checkpoint,
            map_location=args.device,
            weights_only=False,
        )
        model.load_state_dict(state["model"])

        auto = autonomous_diagnostics(
            model, data, valid_ids, args.device, depth, steps
        )

    print(
        f"AUTONOMOUS "
        f"final_action={auto['autonomous_final_action']:.4f} "
        f"exact={auto['autonomous_exact_trajectory']:.4f}",
        flush=True,
    )

    return {
        "experiment": tag,
        "regime": regime,
        "transformer_depth": depth,
        "steps": steps,
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "best": best_metrics,
        "autonomous": auto,
        "checkpoint": str(checkpoint),
    }




# ---------------------------------------------------------------------------
# Shared dataset manifest
# ---------------------------------------------------------------------------

def state_to_json(state):
    return {
        "nodes": [
            {
                "concept": n.concept,
                "activation": n.activation,
                "role": n.role,
                "persistent": n.persistent,
            }
            for n in state.nodes
        ],
        "edges": [
            {
                "source": e.source,
                "relation": e.relation,
                "target": e.target,
                "activation": e.activation,
                "persistent": e.persistent,
            }
            for e in state.edges
        ],
    }


def write_worker_manifest(rows, train_ids, valid_ids, path):
    """
    Serialize the exact parent-owned dataset/split used by every worker.

    This intentionally avoids making workers regenerate data from a seed.
    """
    manifest_rows = []

    for row in rows:
        manifest_rows.append(row)

    payload = {
        "version": "v224",
        "rows": manifest_rows,
        "train_ids": train_ids,
        "valid_ids": valid_ids,
    }

    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def load_worker_manifest(path):
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if payload.get("version") != "v224":
        raise RuntimeError(
            f"Unexpected worker manifest version: "
            f"{payload.get('version')}"
        )

    rows = payload["rows"]
    train_ids = [int(x) for x in payload["train_ids"]]
    valid_ids = [int(x) for x in payload["valid_ids"]]

    data = prepare(rows)
    sanity_check_dataset(rows, data)

    if len(train_ids) + len(valid_ids) > len(rows):
        raise RuntimeError("Manifest split exceeds dataset size.")

    if set(train_ids) & set(valid_ids):
        raise RuntimeError(
            "Manifest contains train/validation overlap."
        )

    return rows, data, train_ids, valid_ids


# ---------------------------------------------------------------------------
# Attention alignment preflight
# ---------------------------------------------------------------------------

def attention_alignment_preflight():
    for count in (1, 2, 3, 7, 20, 21, 22, 23):
        state = State(
            [Node(f"n{i}", 1.0, 0) for i in range(count)],
            [],
        )
        out = {
            "attention_logits": torch.zeros(count),
            "attention_hard": torch.zeros(count),
        }
        target = {"attention_concepts": ("n0", "n1")}
        truth = aligned_attention_target(
            out, target, state, torch.device("cpu")
        )
        assert truth.shape == out["attention_logits"].shape

    print("attention_alignment_preflight: PASS", flush=True)


# ---------------------------------------------------------------------------
# Parallel matrix orchestration
# ---------------------------------------------------------------------------

def run_one_subprocess(args, regime, steps, output_dir, manifest):
    """
    Launch one isolated experiment process.

    CUDA models are deliberately isolated into processes instead of sharing a
    model/optimizer between Python threads. Each worker gets one CUDA context.
    The launcher limits concurrent workers to --parallelism.
    """
    import os
    import subprocess

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--samples", str(args.samples),
        "--epochs", str(args.epochs),
        "--seed", str(args.seed),
        "--lr", str(args.lr),
        "--hidden-size", str(args.hidden_size),
        "--heads", str(args.heads),
        "--topk", str(args.topk),
        "--progress-every", str(args.progress_every),
        "--depth", str(args.depth),
        "--steps", str(steps),
        "--regimes", regime,
        "--output-dir", str(output_dir),
        "--dataset-output", str(args.dataset_output),
        "--manifest", str(manifest),
    ]

    env = os.environ.copy()

    # Avoid each PyTorch process oversubscribing CPU cores while two GPU jobs
    # are already active.
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")

    return subprocess.Popen(
        cmd,
        env=env,
    )


def worker_main(args):
    """
    Execute exactly one matrix cell. The parent launcher uses this mode to keep
    GPU memory and optimizer state completely isolated between experiments.
    """
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    attention_alignment_preflight()

    if args.manifest is None:
        raise RuntimeError(
            "Worker mode requires --manifest so all workers use "
            "the exact same dataset and split."
        )

    rows, data, train_ids, valid_ids = load_worker_manifest(
        args.manifest
    )

    if len(rows) != args.samples:
        raise RuntimeError(
            f"Manifest sample count {len(rows)} != requested "
            f"sample count {args.samples}"
        )

    if len(args.regimes) != 1 or len(args.steps) != 1:
        raise ValueError(
            "Worker mode requires exactly one regime and one step count."
        )

    regime = args.regimes[0]
    steps = args.steps[0]

    result = run_experiment(
        args,
        regime,
        steps,
        args.depth,
        data,
        train_ids,
        valid_ids,
    )

    result_path = (
        args.output_dir
        / f"{result['experiment']}.json"
    )
    result_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(
        f"WORKER_COMPLETE: {result_path}",
        flush=True,
    )


def merge_worker_results(args, jobs):
    results = []

    for regime, steps in jobs:
        path = (
            args.output_dir
            / f"v224_{regime}_d{args.depth}_steps{steps}.json"
        )

        if not path.exists():
            raise RuntimeError(
                f"Missing worker result: {path}"
            )

        results.append(
            json.loads(
                path.read_text(encoding="utf-8")
            )
        )

    # Deterministic ordering independent of completion order.
    results.sort(
        key=lambda x: (
            x["regime"],
            x["steps"],
        )
    )

    summary = args.output_dir / "v224_summary.json"
    summary.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    return results, summary


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--worker", action="store_true")
    p.add_argument("--manifest", type=Path, default=None)

    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=221)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--progress-every", type=int, default=25)

    # V219/V220 showed depth was not the dominant variable.
    p.add_argument("--depth", type=int, default=8)

    # 6-step experiment deliberately removed.
    p.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[2, 4, 8],
    )

    p.add_argument(
        "--regimes",
        nargs="+",
        choices=["teacher", "scheduled", "free"],
        default=["teacher", "scheduled", "free"],
    )

    p.add_argument(
        "--parallelism",
        type=int,
        default=2,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/v224"),
    )

    p.add_argument(
        "--dataset-output",
        type=Path,
        default=Path("results/v224_parallel_dataset.jsonl"),
    )

    args = p.parse_args()

    if args.parallelism < 1:
        raise ValueError("--parallelism must be >= 1")

    if 6 in args.steps:
        raise ValueError(
            "V224 intentionally removes the 6-step experiment. "
            "Use 2, 4, and/or 8 steps."
        )

    if args.worker:
        worker_main(args)
        return

    # Parent launcher creates the deterministic dataset once for inspection.
    # Worker processes independently regenerate the same dataset from the same
    # seed, avoiding cross-process tensor serialization.
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    attention_alignment_preflight()

    rows = generate_dataset(args.samples, args.seed)
    save_dataset(rows, args.dataset_output)
    data = prepare(rows)
    sanity_check_dataset(rows, data)
    train_ids, valid_ids = stratified_split(rows, args.seed)

    print(
        "=== V224 PARALLEL RECURRENT STABILIZATION MATRIX ===",
        flush=True,
    )
    print(
        f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}",
        flush=True,
    )

    if torch.cuda.is_available():
        print(
            "gpu:",
            torch.cuda.get_device_name(0),
            flush=True,
        )

    print("dataset_size:", len(rows), flush=True)
    print(
        "action_counts:",
        dict(
            Counter(
                r["final_action"]["action"]
                for r in rows
            )
        ),
        flush=True,
    )
    print(
        "train_size:",
        len(train_ids),
        "valid_size:",
        len(valid_ids),
        flush=True,
    )
    print("dataset_sanity: PASS", flush=True)
    print("split_sanity: PASS", flush=True)

    manifest = args.output_dir / "v224_worker_manifest.json"
    write_worker_manifest(
        rows,
        train_ids,
        valid_ids,
        manifest,
    )

    # Hard verification: workers must receive the exact split that the parent
    # just validated.
    manifest_rows, _, manifest_train, manifest_valid = load_worker_manifest(
        manifest
    )
    if len(manifest_rows) != len(rows):
        raise AssertionError("Manifest dataset size mismatch.")
    if manifest_train != train_ids or manifest_valid != valid_ids:
        raise AssertionError("Manifest split differs from parent split.")

    print("shared_manifest: PASS", flush=True)
    print("manifest_samples:", len(manifest_rows), flush=True)
    print("manifest_train_size:", len(manifest_train), flush=True)
    print("manifest_valid_size:", len(manifest_valid), flush=True)
    print("depth:", args.depth, flush=True)
    print("steps:", args.steps, flush=True)
    print("regimes:", args.regimes, flush=True)
    print("parallelism:", args.parallelism, flush=True)

    jobs = [
        (regime, steps)
        for regime in args.regimes
        for steps in args.steps
    ]

    print(
        f"matrix_cells: {len(jobs)}",
        flush=True,
    )

    pending = list(jobs)
    active = []
    completed = set()

    while pending or active:
        while pending and len(active) < args.parallelism:
            regime, steps = pending.pop(0)

            print(
                f"LAUNCH {regime} steps={steps} "
                f"active={len(active)+1}/{args.parallelism}",
                flush=True,
            )

            process = run_one_subprocess(
                args,
                regime,
                steps,
                args.output_dir,
                manifest,
            )

            active.append(
                (process, regime, steps)
            )

        still_active = []

        for process, regime, steps in active:
            code = process.poll()

            if code is None:
                still_active.append(
                    (process, regime, steps)
                )
                continue

            if code != 0:
                raise RuntimeError(
                    f"Worker failed: regime={regime}, "
                    f"steps={steps}, exit_code={code}"
                )

            completed.add((regime, steps))

            print(
                f"COMPLETE {regime} steps={steps} "
                f"completed={len(completed)}/{len(jobs)}",
                flush=True,
            )

        active = still_active

        if active:
            time.sleep(0.5)

    results, summary = merge_worker_results(
        args,
        jobs,
    )

    print("\n" + "=" * 78, flush=True)
    print("V224 SUMMARY", flush=True)
    print("=" * 78, flush=True)

    for r in results:
        a = r["autonomous"]

        print(
            f"{r['regime']:10s} "
            f"steps={r['steps']} "
            f"best_ep={r['best_epoch']} "
            f"teacher_loss={r['best_valid_loss']:.4f} "
            f"auto_final={a['autonomous_final_action']:.4f} "
            f"auto_exact={a['autonomous_exact_trajectory']:.4f}",
            flush=True,
        )

    print(
        "summary_saved:",
        summary,
        flush=True,
    )


if __name__ == "__main__":
    main()
