
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dataset import generate_dataset, save_dataset
from model import StateArchitectureModel
from state import ACTIONS, ACTION_TO_ID, Edge, Node, State


ARCHITECTURES = {
    "baseline_graph": {
        "state_mode": "stateless",
        "attention_workspace": False,
        "explicit_progress": False,
        "direct_goal_to_workspace": False,
    },
    "latent_workspace": {
        "state_mode": "latent",
        "attention_workspace": False,
        "explicit_progress": False,
        "direct_goal_to_workspace": False,
    },
    "latent_action": {
        "state_mode": "latent_action",
        "attention_workspace": False,
        "explicit_progress": False,
        "direct_goal_to_workspace": False,
    },
    "workspace_attention": {
        "state_mode": "latent",
        "attention_workspace": True,
        "explicit_progress": False,
        "direct_goal_to_workspace": True,
    },
    "workspace_progress": {
        "state_mode": "latent",
        "attention_workspace": True,
        "explicit_progress": True,
        "direct_goal_to_workspace": True,
    },
    "workspace_action_progress": {
        "state_mode": "latent_action",
        "attention_workspace": True,
        "explicit_progress": True,
        "direct_goal_to_workspace": True,
    },
}


def state_from_json(p):
    return State(
        [
            Node(
                str(n["concept"]),
                float(n["activation"]),
                int(n["role"]),
                bool(n.get("persistent", False)),
            )
            for n in p["nodes"]
        ],
        [
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
    data = []

    for row in rows:
        states = [
            state_from_json(x)
            for x in row["trajectory_states"]
        ]

        targets = []

        for t, state in enumerate(states):
            action = row["trajectory_actions"][t]
            targets.append({
                "attention_concepts": tuple(
                    row["trajectory_attention"][t]
                ),
                "action": ACTION_TO_ID[
                    action["action"]
                ],
                "source_concept": action["source"],
                "target_concept": action["target"],
                "relation": 0,
            })

        data.append({
            "case_id": row["case_id"],
            "initial_state": state_from_json(
                row["initial_state"]
            ),
            "trajectory_states": states,
            "trajectory_targets": targets,
            "goal": row["goal"],
        })

    return data


def split_rows(rows, seed, valid_fraction=0.15):
    groups = {a: [] for a in ACTIONS}
    rng = random.Random(seed)

    for i, row in enumerate(rows):
        groups[
            row["final_action"]["action"]
        ].append(i)

    train_ids = []
    valid_ids = []

    for action in ACTIONS:
        ids = groups[action]

        if not ids:
            raise AssertionError(
                f"Missing action category {action}"
            )

        rng.shuffle(ids)
        n = max(
            1,
            int(len(ids) * valid_fraction),
        )

        valid_ids.extend(ids[:n])
        train_ids.extend(ids[n:])

    rng.shuffle(train_ids)
    rng.shuffle(valid_ids)

    return train_ids, valid_ids


def validate_dataset(rows, data):
    counts = {
        action: 0
        for action in ACTIONS
    }

    for row in rows:
        counts[
            row["final_action"]["action"]
        ] += 1

        if row["final_action"]["action"] not in ACTIONS:
            raise AssertionError(
                "Unknown action category"
            )

        if len(row["trajectory_states"]) != len(
            row["trajectory_actions"]
        ):
            raise AssertionError(
                f"state/action length mismatch: {row['case_id']}"
            )

        if len(row["trajectory_states"]) != len(
            row["trajectory_attention"]
        ):
            raise AssertionError(
                f"state/attention length mismatch: {row['case_id']}"
            )

    if any(v == 0 for v in counts.values()):
        raise AssertionError(
            f"action coverage failure: {counts}"
        )



def fixed_horizon_preflight(data, horizons, valid_ids):
    """
    Check that every requested horizon has enough eligible validation cases.

    This is intentionally quantitative so the survey cannot start with a
    misleading '8-step' condition supported by only a handful of trajectories.
    """
    print("", flush=True)
    print("=" * 78, flush=True)
    print("V236 FIXED-HORIZON PREFLIGHT", flush=True)
    print("=" * 78, flush=True)

    for horizon in horizons:
        eligible_train = sum(
            len(data[idx]["trajectory_targets"]) >= horizon
            and len(data[idx]["trajectory_states"]) >= horizon
            for idx in range(len(data))
        )

        eligible_valid = sum(
            len(data[idx]["trajectory_targets"]) >= horizon
            and len(data[idx]["trajectory_states"]) >= horizon
            for idx in valid_ids
        )

        print(
            f"horizon={horizon} "
            f"eligible_train={eligible_train} "
            f"eligible_valid={eligible_valid} "
            f"valid_total={len(valid_ids)}",
            flush=True,
        )

        if eligible_train == 0:
            raise AssertionError(
                f"No training trajectories support horizon={horizon}"
            )

        if eligible_valid == 0:
            raise AssertionError(
                f"No validation trajectories support horizon={horizon}"
            )

    print(
        "fixed_horizon_preflight: PASS",
        flush=True,
    )
    print("=" * 78, flush=True)
    print("", flush=True)




def dataset_horizon_preflight(rows, data, horizons, valid_ids):
    print("",flush=True)
    print("="*78,flush=True)
    print("V236 DATASET HORIZON PREFLIGHT",flush=True)
    print("="*78,flush=True)

    lengths=[len(r["trajectory_actions"]) for r in rows]
    print(
        f"trajectory_length_min={min(lengths)} "
        f"max={max(lengths)} "
        f"unique={sorted(set(lengths))}",
        flush=True,
    )

    bad_ids=[
        r["case_id"] for r in rows
        if not r["case_id"].startswith("v236_")
    ]
    if bad_ids:
        raise AssertionError(
            "legacy/non-V236 dataset case IDs: "
            + ", ".join(bad_ids[:5])
        )

    for h in horizons:
        eligible_train=sum(
            len(x["trajectory_targets"])>=h
            and len(x["trajectory_states"])>=h
            for x in data
        )
        eligible_valid=sum(
            len(data[i]["trajectory_targets"])>=h
            and len(data[i]["trajectory_states"])>=h
            for i in valid_ids
        )
        print(
            f"horizon={h} "
            f"eligible_train={eligible_train} "
            f"eligible_valid={eligible_valid} "
            f"valid_total={len(valid_ids)}",
            flush=True,
        )
        if eligible_train==0 or eligible_valid==0:
            raise AssertionError(
                f"No eligible cases for horizon={h}"
            )

    print("dataset_horizon_preflight: PASS",flush=True)
    print("="*78,flush=True)
    print("",flush=True)


def make_model(arch_name, args, device, seed):
    spec = ARCHITECTURES[arch_name]

    torch.manual_seed(seed)

    model = StateArchitectureModel(
        hidden_size=args.hidden_size,
        heads=args.heads,
        depth=args.depth,
        topk=args.topk,
        state_mode=spec["state_mode"],
        attention_workspace=spec["attention_workspace"],
        explicit_progress=spec["explicit_progress"],
        direct_goal_to_workspace=spec["direct_goal_to_workspace"],
    ).to(device)

    model.eval()
    return model


def align_attention(out, state, target, device):
    names = [
        n.concept
        for n in state.nodes
    ]
    wanted = set(
        target["attention_concepts"]
    )

    truth = torch.tensor(
        [
            float(name in wanted)
            for name in names
        ],
        dtype=torch.float32,
        device=device,
    )

    if truth.shape != out["attention_logits"].shape:
        raise AssertionError(
            "attention target/logit shape mismatch"
        )

    return truth


def loss_for(out, target, state, device):
    action_loss = F.cross_entropy(
        out["action_logits"][None],
        torch.tensor(
            [target["action"]],
            device=device,
        ),
    )

    truth = align_attention(
        out,
        state,
        target,
        device,
    )

    attention_loss = F.binary_cross_entropy_with_logits(
        out["attention_logits"],
        truth,
    )

    return action_loss + 0.35 * attention_loss


def train(model, data, ids, device, horizon, epochs, lr):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for idx in ids:
            item = data[idx]
            current = item["initial_state"].clone()
            working = torch.zeros(
                (1, model.hidden_size),
                device=device,
            )

            previous = {
                "action_id": None,
                "source": None,
                "target": None,
                "relation": None,
            }

            total_loss = torch.zeros(
                (),
                device=device,
            )

            if (
                len(item["trajectory_states"]) < horizon
                or len(item["trajectory_targets"]) < horizon
            ):
                raise AssertionError(
                    f"Training case {item['case_id']} "
                    f"does not support horizon={horizon}"
                )

            for t in range(horizon):
                target = item["trajectory_targets"][t]

                out = model.cognitive_step(
                    current,
                    item["goal"],
                    working,
                    previous["action_id"],
                    previous["source"],
                    previous["target"],
                    previous["relation"],
                    device,
                    progress=t,
                )

                total_loss = (
                    total_loss
                    + (
                        1.0 + 0.10 * t
                    ) * loss_for(
                        out,
                        target,
                        current,
                        device,
                    )
                )

                working = out["next_working"]

                if t + 1 < horizon:
                    (
                        current,
                        aid,
                        src,
                        tar,
                        rid,
                    ) = model.predicted_transition(
                        current,
                        out,
                    )

                    previous = {
                        "action_id": aid,
                        "source": src,
                        "target": tar,
                        "relation": rid,
                    }

            total_loss = total_loss / horizon

            optimizer.zero_grad(
                set_to_none=True
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )
            optimizer.step()

            running += float(
                total_loss.item()
            )

        mean_train = running / max(
            1,
            len(ids),
        )

        # Use a deterministic subset of training cases only for model selection.
        model.eval()
        probe_loss = 0.0
        probe_ids = ids[: min(32, len(ids))]

        with torch.no_grad():
            for idx in probe_ids:
                item = data[idx]
                current = item["initial_state"].clone()
                working = torch.zeros(
                    (1, model.hidden_size),
                    device=device,
                )

                for t in range(
                    min(
                        horizon,
                        len(item["trajectory_states"]),
                    )
                ):
                    target = item["trajectory_targets"][t]
                    out = model.cognitive_step(
                        current,
                        item["goal"],
                        working,
                        None,
                        None,
                        None,
                        None,
                        device,
                        progress=t,
                    )
                    probe_loss += float(
                        loss_for(
                            out,
                            target,
                            current,
                            device,
                        ).item()
                    )
                    working = out["next_working"]

        probe_loss /= max(
            1,
            len(probe_ids) * horizon,
        )

        print(
            f"    epoch={epoch}/{epochs} "
            f"train_loss={mean_train:.4f} "
            f"probe_loss={probe_loss:.4f}",
            flush=True,
        )

        if probe_loss < best_loss:
            best_loss = probe_loss
            best_state = copy.deepcopy(
                model.state_dict()
            )

    if best_state is not None:
        model.load_state_dict(
            best_state,
            strict=True,
        )

    return {
        "best_probe_loss": best_loss,
        "epochs": epochs,
    }


@torch.no_grad()
@torch.no_grad()
def autonomous_metrics(model, data, valid_ids, device, horizon):
    model.eval()
    eligible=[
        idx for idx in valid_ids
        if len(data[idx]["trajectory_targets"])>=horizon
        and len(data[idx]["trajectory_states"])>=horizon
    ]
    if not eligible:
        raise RuntimeError(
            f"No validation trajectory supports horizon={horizon}."
        )
    exact=0
    step_correct=[0]*horizon
    for idx in eligible:
        item=data[idx]
        roll=model.autonomous_rollout(
            item["initial_state"],item["goal"],device,horizon,
            stop_on_terminal=False,
        )
        predicted=[x["action_id"] for x in roll["outputs"]]
        truth=[x["action"] for x in item["trajectory_targets"][:horizon]]
        if len(predicted)!=horizon or len(truth)!=horizon:
            raise AssertionError(
                f"fixed-horizon mismatch: horizon={horizon} "
                f"predicted={len(predicted)} truth={len(truth)}"
            )
        exact += int(predicted==truth)
        for t in range(horizon):
            step_correct[t]+=int(predicted[t]==truth[t])
    return {
        "exact_trajectory":exact/len(eligible),
        "eligible_cases":len(eligible),
        "validation_cases":len(valid_ids),
        "eligibility_fraction":len(eligible)/max(1,len(valid_ids)),
        "per_step":[
            {
                "step":t+1,
                "cases":len(eligible),
                "action_accuracy":step_correct[t]/len(eligible),
            }
            for t in range(horizon)
        ],
    }

def causal_probes(model, data, valid_ids, device):
    """
    Architecture-use probes.

    These compare the SAME graph/goal under controlled interventions.
    For stateless baseline, changing the supplied working state must not affect
    the decision because that state is not in its computational path.
    """
    model.eval()

    working_delta = 0.0
    goal_delta = 0.0
    history_delta = 0.0
    n = 0

    for idx in valid_ids[: min(16, len(valid_ids))]:
        item = data[idx]
        state = item["initial_state"].clone()
        work0 = torch.zeros(
            (1, model.hidden_size),
            device=device,
        )
        work1 = torch.ones_like(work0)

        common = {
            "action_id": None,
            "source": None,
            "target": None,
            "relation": None,
        }

        a = model.cognitive_step(
            state, item["goal"], work0,
            None, None, None, None,
            device, progress=1,
        )
        b = model.cognitive_step(
            state, item["goal"], work1,
            None, None, None, None,
            device, progress=1,
        )

        working_delta += float(
            (
                a["action_logits"]
                - b["action_logits"]
            ).abs().max().item()
        )

        altered_goal = dict(item["goal"])
        altered_goal["depth"] = (
            int(altered_goal["depth"]) + 1
        ) % 6

        c = model.cognitive_step(
            state, item["goal"], work0,
            None, None, None, None,
            device, progress=1,
        )
        d = model.cognitive_step(
            state, altered_goal, work0,
            None, None, None, None,
            device, progress=1,
        )

        goal_delta += float(
            (
                c["action_logits"]
                - d["action_logits"]
            ).abs().max().item()
        )

        h0 = model.cognitive_step(
            state, item["goal"], work0,
            1, "alpha", "beta", 0,
            device, progress=1,
        )
        h1 = model.cognitive_step(
            state, item["goal"], work0,
            5, "gamma", "delta", 1,
            device, progress=1,
        )

        history_delta += float(
            (
                h0["next_working"]
                - h1["next_working"]
            ).abs().max().item()
        )

        n += 1

    return {
        "working_to_action_delta": working_delta / max(1, n),
        "goal_to_action_delta": goal_delta / max(1, n),
        "history_to_state_delta": history_delta / max(1, n),
    }


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--samples",
        type=int,
        default=500,
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=3,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=234,
    )
    p.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )
    p.add_argument(
        "--hidden-size",
        type=int,
        default=128,
    )
    p.add_argument(
        "--heads",
        type=int,
        default=4,
    )
    p.add_argument(
        "--depth",
        type=int,
        default=8,
    )
    p.add_argument(
        "--topk",
        type=int,
        default=5,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/v236"),
    )
    p.add_argument(
        "--dataset-output",
        type=Path,
        default=Path(
            "results/v236_architecture_survey_dataset.jsonl"
        ),
    )

    p.add_argument(
        "--architectures",
        nargs="+",
        choices=list(ARCHITECTURES),
        default=list(ARCHITECTURES),
    )
    p.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[2, 8],
    )

    args = p.parse_args()

    if 6 in args.horizons:
        raise ValueError(
            "6-step condition deliberately excluded."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=== V236 ARCHITECTURAL DECISION SURVEY ===",
        flush=True,
    )
    print(
        "device:",
        device,
        flush=True,
    )

    rows = generate_dataset(
        args.samples,
        args.seed,
        reasoning_steps=max(args.horizons),
    )
    save_dataset(
        rows,
        args.dataset_output,
    )

    data = prepare(rows)
    validate_dataset(
        rows,
        data,
    )

    train_ids, valid_ids = split_rows(
        rows,
        args.seed,
    )
    dataset_horizon_preflight(
        rows,
        data,
        args.horizons,
        valid_ids,
    )

    fixed_horizon_preflight(
        data,
        args.horizons,
        valid_ids,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "dataset_size:",
        len(rows),
        flush=True,
    )
    print(
        "train_size:",
        len(train_ids),
        "valid_size:",
        len(valid_ids),
        flush=True,
    )
    print(
        "architectures:",
        args.architectures,
        flush=True,
    )
    print(
        "horizons:",
        args.horizons,
        flush=True,
    )
    print(
        "matrix_cells:",
        len(args.architectures)
        * len(args.horizons),
        flush=True,
    )
    print(
        "parallelism: 1 (deliberate for survey reliability)",
        flush=True,
    )

    results = []

    for arch_index, arch in enumerate(
        args.architectures
    ):
        for horizon in args.horizons:
            tag = (
                f"v236_{arch}_"
                f"d{args.depth}_steps{horizon}"
            )

            print(
                "\n" + "=" * 78,
                flush=True,
            )
            print(tag, flush=True)
            print("=" * 78, flush=True)

            model = make_model(
                arch,
                args,
                device,
                args.seed + arch_index,
            )

            train_metrics = train(
                model,
                data,
                train_ids,
                device,
                horizon,
                args.epochs,
                args.lr,
            )

            auto = autonomous_metrics(
                model,
                data,
                valid_ids,
                device,
                horizon,
            )

            probes = causal_probes(
                model,
                data,
                valid_ids,
                device,
            )

            spec = ARCHITECTURES[arch]

            result = {
                "experiment": tag,
                "architecture": arch,
                "architecture_spec": spec,
                "horizon": horizon,
                "train": train_metrics,
                "autonomous": auto,
                "causal_probes": probes,
            }

            path = (
                args.output_dir
                / f"{tag}.json"
            )

            path.write_text(
                json.dumps(
                    result,
                    indent=2,
                ),
                encoding="utf-8",
            )

            results.append(result)

            print(
                f"RESULT "
                f"exact={auto['exact_trajectory']:.4f} "
                f"working→action={probes['working_to_action_delta']:.4e} "
                f"goal→action={probes['goal_to_action_delta']:.4e} "
                f"history→state={probes['history_to_state_delta']:.4e}",
                flush=True,
            )

    summary = (
        args.output_dir
        / "v236_summary.json"
    )

    summary.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n" + "=" * 78,
        flush=True,
    )
    print(
        "V236 SUMMARY",
        flush=True,
    )
    print(
        "=" * 78,
        flush=True,
    )

    for r in results:
        p = r["causal_probes"]
        print(
            f"{r['architecture']:25s} "
            f"steps={r['horizon']} "
            f"exact={r['autonomous']['exact_trajectory']:.4f} "
            f"working→action={p['working_to_action_delta']:.4e} "
            f"goal→action={p['goal_to_action_delta']:.4e} "
            f"history→state={p['history_to_state_delta']:.4e}",
            flush=True,
        )

    print(
        "summary_saved:",
        summary,
        flush=True,
    )


if __name__ == "__main__":
    main()
