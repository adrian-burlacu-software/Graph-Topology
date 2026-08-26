from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch
from torch import nn
from torch.optim import AdamW

# Direct execution from research/:
#   python .\\v209_teacher_distilled_cognitive\\train.py
import sys
_HERE = Path(__file__).resolve().parent
_RESEARCH_ROOT = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

from dataset import TeacherDataset
from model import TeacherDistilledController
from state import ACTIONS
from v200_graph_transformer_cognitive.long_term_memory import (
    RELATION_TO_ID,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

TEACHER_DATA = (
    RESULTS
    / "v205r_teacher_dataset.jsonl"
)

V203_CHECKPOINT = (
    RESULTS
    / "v203_multi_action_cognitive.pt"
)

MODEL_PATH = (
    RESULTS
    / "v209_teacher_distilled_cognitive.pt"
)


def loss_for_example(
    model,
    example,
    device,
    train,
    optimizer,
) -> dict[str, float]:
    current = model.encode_state(
        example["state"],
        device,
    )

    target = model.encode_state(
        example["next_state"],
        device,
    )

    goal_state = model.encode_goal(
        example["state"],
        example["goal_source_index"],
        example["goal_target_index"],
        example["goal_relation"],
        device,
    )

    graph_state = current[
        "graph_state"
    ]
    node_state = current[
        "node_state"
    ]

    action_logits = model.predict_action(
        graph_state,
        goal_state,
    )

    action_target = torch.tensor(
        [example["action_id"]],
        dtype=torch.long,
        device=device,
    )

    action_loss = nn.functional.cross_entropy(
        action_logits,
        action_target,
    )

    source_logits, target_logits = (
        model.predict_pointers(
            node_state,
            goal_state,
        )
    )

    if (
        0 <= example["source_index"]
        < len(example["state"].nodes)
    ):
        source_target = torch.tensor(
            [example["source_index"]],
            dtype=torch.long,
            device=device,
        )
        source_loss = nn.functional.cross_entropy(
            source_logits.unsqueeze(0),
            source_target,
        )
        source_correct = float(
            source_logits.argmax().item()
            == example["source_index"]
        )
    else:
        source_loss = torch.zeros(
            (),
            device=device,
        )
        source_correct = 1.0

    if (
        0 <= example["target_index"]
        < len(example["state"].nodes)
    ):
        target_target = torch.tensor(
            [example["target_index"]],
            dtype=torch.long,
            device=device,
        )
        target_loss = nn.functional.cross_entropy(
            target_logits.unsqueeze(0),
            target_target,
        )
        target_correct = float(
            target_logits.argmax().item()
            == example["target_index"]
        )
    else:
        target_loss = torch.zeros(
            (),
            device=device,
        )
        target_correct = 1.0

    # Relation classification uses the goal relation as the grounded target.
    source_index = max(
        0,
        min(
            example["source_index"],
            len(
                example["state"].nodes
            ) - 1,
        ),
    )

    target_index = max(
        0,
        min(
            example["target_index"],
            len(
                example["state"].nodes
            ) - 1,
        ),
    )

    relation_logits = model.predict_relation(
        node_state[source_index].unsqueeze(0),
        node_state[target_index].unsqueeze(0),
        goal_state,
    )

    relation_id = RELATION_TO_ID.get(
        example["relation"],
        0,
    )

    relation_target = torch.tensor(
        [relation_id],
        dtype=torch.long,
        device=device,
    )

    relation_loss = nn.functional.cross_entropy(
        relation_logits,
        relation_target,
    )

    relation_correct = float(
        relation_logits.argmax(
            dim=-1
        ).item()
        == relation_id
    )

    next_latent = model.predict_next_latent(
        graph_state,
        goal_state,
    )

    next_loss = nn.functional.mse_loss(
        next_latent,
        target[
            "graph_state"
        ].detach(),
    )

    transition_logits = model.predict_transition(
        node_state,
        graph_state,
    )

    target_activations = torch.tensor(
        [
            (
                next(
                    (
                        node.activation
                        for node
                        in example["next_state"].nodes
                        if node.concept
                        == current_node.concept
                    ),
                    0.0,
                )
            )
            for current_node
            in example["state"].nodes
        ],
        dtype=torch.float32,
        device=device,
    )

    transition_loss = nn.functional.mse_loss(
        torch.sigmoid(
            transition_logits
        ),
        target_activations,
    )

    # Give corrected teacher trajectories a small extra weight. This does not
    # pretend they are better labels; it simply makes the successful
    # "self-correction" examples harder to ignore.
    weight = (
        1.15
        if example["corrected"]
        else 1.0
    )

    total_loss = weight * (
        1.00 * action_loss
        + 0.35 * source_loss
        + 0.35 * target_loss
        + 0.65 * relation_loss
        + 0.70 * next_loss
        + 0.20 * transition_loss
    )

    if train:
        optimizer.zero_grad(
            set_to_none=True
        )
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

    return {
        "loss": float(
            total_loss.detach().item()
        ),
        "action_correct": float(
            action_logits.argmax(
                dim=-1
            ).item()
            == example["action_id"]
        ),
        "source_correct": source_correct,
        "target_correct": target_correct,
        "relation_correct": relation_correct,
        "next_loss": float(
            next_loss.detach().item()
        ),
        "transition_loss": float(
            transition_loss.detach().item()
        ),
        "corrected_weight": weight,
    }


def run_epoch(
    model,
    dataset,
    indices,
    device,
    optimizer,
    train,
) -> dict[str, float]:
    model.train(train)

    totals: dict[str, float] = {}
    count = 0

    iterable = indices[:]
    if train:
        random.shuffle(iterable)

    for index in iterable:
        with torch.set_grad_enabled(
            train
        ):
            stats = loss_for_example(
                model,
                dataset[index],
                device,
                train,
                optimizer,
            )

        for key, value in stats.items():
            totals[key] = (
                totals.get(
                    key,
                    0.0,
                )
                + value
            )

        count += 1

    return {
        key: value / max(
            1,
            count,
        )
        for key, value
        in totals.items()
    }


def load_v203(
    model,
    checkpoint_path,
    device,
) -> int:
    if not checkpoint_path.exists():
        return 0

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    source = checkpoint.get(
        "model",
        checkpoint,
    )

    current = model.state_dict()

    compatible = {
        key: value
        for key, value in source.items()
        if key in current
        and current[key].shape == value.shape
    }

    model.load_state_dict(
        compatible,
        strict=False,
    )

    return len(compatible)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )
    parser.add_argument(
        "--no-v203-init",
        action="store_true",
    )
    args = parser.parse_args()

    random.seed(209)
    torch.manual_seed(209)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=== V209 TEACHER-DISTILLED COGNITIVE TRAINING ===",
        flush=True,
    )
    print(
        "device:",
        device,
        flush=True,
    )

    if device.type == "cuda":
        print(
            "gpu:",
            torch.cuda.get_device_name(0),
            flush=True,
        )

    print(
        "teacher_dataset:",
        TEACHER_DATA.resolve(),
        "exists=",
        TEACHER_DATA.exists(),
        flush=True,
    )

    if not TEACHER_DATA.exists():
        raise FileNotFoundError(
            TEACHER_DATA.resolve()
        )

    dataset = TeacherDataset(
        TEACHER_DATA
    )

    train_indices, valid_indices = (
        dataset.split(
            valid_fraction=0.15,
            seed=209,
        )
    )

    print(
        "dataset_size:",
        len(dataset),
        flush=True,
    )
    print(
        "train_size:",
        len(train_indices),
        flush=True,
    )
    print(
        "valid_size:",
        len(valid_indices),
        flush=True,
    )

    model = TeacherDistilledController(
        vocab_size=50000,
        relation_count=len(
            RELATION_TO_ID
        ),
        hidden_size=args.hidden_size,
        heads=4,
        layers=3,
    ).to(device)

    if not args.no_v203_init:
        loaded = load_v203(
            model,
            V203_CHECKPOINT,
            device,
        )
        print(
            "v203_compatible_tensors:",
            loaded,
            flush=True,
        )
    else:
        print(
            "v203_init: disabled",
            flush=True,
        )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-2,
    )

    best_score = float("inf")

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_stats = run_epoch(
            model,
            dataset,
            train_indices,
            device,
            optimizer,
            True,
        )

        valid_stats = run_epoch(
            model,
            dataset,
            valid_indices,
            device,
            optimizer,
            False,
        )

        score = (
            valid_stats["loss"]
            + 0.50 * valid_stats["next_loss"]
        )

        print(
            f"EPOCH {epoch} "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_action={train_stats['action_correct']:.4f} "
            f"train_src={train_stats['source_correct']:.4f} "
            f"train_tgt={train_stats['target_correct']:.4f} "
            f"train_rel={train_stats['relation_correct']:.4f} "
            f"valid_loss={valid_stats['loss']:.4f} "
            f"valid_action={valid_stats['action_correct']:.4f} "
            f"valid_src={valid_stats['source_correct']:.4f} "
            f"valid_tgt={valid_stats['target_correct']:.4f} "
            f"valid_rel={valid_stats['relation_correct']:.4f} "
            f"valid_next={valid_stats['next_loss']:.4f}",
            flush=True,
        )

        if score < best_score:
            best_score = score

            RESULTS.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "vocab_size": 50000,
                        "relation_count": len(
                            RELATION_TO_ID
                        ),
                        "hidden_size": args.hidden_size,
                        "heads": 4,
                        "layers": 3,
                        "actions": ACTIONS,
                    },
                    "best_score": best_score,
                    "teacher_dataset": str(
                        TEACHER_DATA
                    ),
                    "initialized_from": (
                        str(
                            V203_CHECKPOINT
                        )
                        if (
                            not args.no_v203_init
                            and V203_CHECKPOINT.exists()
                        )
                        else None
                    ),
                },
                MODEL_PATH,
            )

            print(
                "checkpoint_saved:",
                MODEL_PATH.resolve(),
                "exists=",
                MODEL_PATH.exists(),
                flush=True,
            )

    print(
        "FINAL_CHECKPOINT:",
        MODEL_PATH.resolve(),
        "exists=",
        MODEL_PATH.exists(),
        flush=True,
    )

    if MODEL_PATH.exists():
        print(
            "bytes:",
            MODEL_PATH.stat().st_size,
            flush=True,
        )


if __name__ == "__main__":
    main()
