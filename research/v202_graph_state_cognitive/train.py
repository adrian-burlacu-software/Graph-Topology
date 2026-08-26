from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

try:
    from .dataset import GraphTrajectoryDataset, TrajectoryExample
    from .graph_state import ACTIONS
    from .model import CognitiveLoopModel
    from v200_graph_transformer_cognitive.long_term_memory import RELATION_TO_ID
except ImportError:
    import sys
    from pathlib import Path as _Path

    _RESEARCH_ROOT = _Path(__file__).resolve().parents[1]
    if str(_RESEARCH_ROOT) not in sys.path:
        sys.path.insert(0, str(_RESEARCH_ROOT))

    from v202_graph_state_cognitive.dataset import (
        GraphTrajectoryDataset,
        TrajectoryExample,
    )
    from v202_graph_state_cognitive.graph_state import ACTIONS
    from v202_graph_state_cognitive.model import CognitiveLoopModel
    from v200_graph_transformer_cognitive.long_term_memory import RELATION_TO_ID


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DB_PATH = DATA / "conceptnet_compact.db"
MODEL_PATH = (
    RESULTS
    / "v202_graph_state_cognitive.pt"
)


def state_distance_target(
    current,
    target,
) -> list[float]:
    """
    Node-level transition target.

    A node should become highly active when it is newly activated in the target
    state, otherwise its target activation is retained.
    """
    current_by_concept = {
        node.concept: node.activation
        for node in current.nodes
    }

    return [
        node.activation
        for node in target.nodes
    ]


def run_example(
    model: CognitiveLoopModel,
    example: TrajectoryExample,
    device: torch.device,
    train: bool,
    optimizer,
) -> dict[str, float]:
    current = model.forward_state(
        example.current,
        device,
    )

    target = model.forward_state(
        example.target_next,
        device,
    )

    graph_state = current[
        "graph_state"
    ]

    source_state = current[
        "node_state"
    ][
        example.source_index
    ].unsqueeze(0)

    target_state = current[
        "node_state"
    ][
        example.target_index
    ].unsqueeze(0)

    action_logits = model.predict_action(
        graph_state
    )

    action_target = torch.tensor(
        [example.action_id],
        dtype=torch.long,
        device=device,
    )

    action_loss = nn.functional.cross_entropy(
        action_logits,
        action_target,
    )

    action_prediction = int(
        action_logits.argmax(
            dim=-1
        ).item()
    )

    relation_logits = model.predict_relation(
        source_state,
        target_state,
    )

    relation_target = torch.tensor(
        [example.relation_id],
        dtype=torch.long,
        device=device,
    )

    relation_loss = nn.functional.cross_entropy(
        relation_logits,
        relation_target,
    )

    relation_prediction = int(
        relation_logits.argmax(
            dim=-1
        ).item()
    )

    predicted_next_latent = model.predict_next_latent(
        graph_state
    )

    next_latent_loss = nn.functional.mse_loss(
        predicted_next_latent,
        target[
            "graph_state"
        ].detach(),
    )

    current_node = current[
        "node_state"
    ]

    transition_logits = model.predict_node_transition(
        current_node,
        graph_state,
    )

    target_activations = torch.tensor(
        state_distance_target(
            example.current,
            example.target_next,
        ),
        dtype=torch.float32,
        device=device,
    )

    transition_loss = nn.functional.mse_loss(
        torch.sigmoid(
            transition_logits
        ),
        target_activations,
    )

    # Graph edit target: the target state must contain one more semantically
    # active edge than the current state in the synthetic trajectory.
    current_edge_keys = {
        (
            edge.source,
            edge.relation_id,
            edge.target,
        )
        for edge in example.current.edges
    }

    target_edge_keys = {
        (
            edge.source,
            edge.relation_id,
            edge.target,
        )
        for edge in example.target_next.edges
    }

    edge_gain = float(
        len(
            target_edge_keys
            - current_edge_keys
        )
    )

    edge_edit_loss = nn.functional.mse_loss(
        predicted_next_latent.norm(
            dim=-1
        ),
        target[
            "graph_state"
        ].detach().norm(
            dim=-1
        ),
    )

    loss = (
        1.00 * action_loss
        + 0.80 * relation_loss
        + 0.80 * next_latent_loss
        + 0.30 * transition_loss
        + 0.20 * edge_edit_loss
    )

    if train:
        optimizer.zero_grad(
            set_to_none=True
        )
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

    return {
        "loss": float(
            loss.detach().item()
        ),
        "action_loss": float(
            action_loss.detach().item()
        ),
        "relation_loss": float(
            relation_loss.detach().item()
        ),
        "next_latent_loss": float(
            next_latent_loss.detach().item()
        ),
        "transition_loss": float(
            transition_loss.detach().item()
        ),
        "edge_edit_loss": float(
            edge_edit_loss.detach().item()
        ),
        "action_correct": float(
            action_prediction
            == example.action_id
        ),
        "relation_correct": float(
            relation_prediction
            == example.relation_id
        ),
        "edge_gain": edge_gain,
    }


def run_epoch(
    model,
    loader,
    device,
    optimizer,
    train,
) -> dict[str, float]:
    model.train(train)

    totals: dict[str, float] = {}
    count = 0

    for batch in loader:
        for example in batch:
            with torch.set_grad_enabled(train):
                stats = run_example(
                    model,
                    example,
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
        for key, value in totals.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=12000,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
    )
    args = parser.parse_args()

    random.seed(202)
    torch.manual_seed(202)

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(
        "=== V202 GRAPH-STATE COGNITIVE LOOP ===",
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
        "conceptnet_db:",
        DB_PATH.resolve(),
        "exists=",
        DB_PATH.exists(),
        flush=True,
    )

    if not DB_PATH.exists():
        raise FileNotFoundError(
            DB_PATH
        )

    dataset = GraphTrajectoryDataset(
        DB_PATH,
        samples=args.samples,
    )

    train_size = int(
        len(dataset)
        * 0.85
    )

    valid_size = (
        len(dataset)
        - train_size
    )

    train_set, valid_set = random_split(
        dataset,
        [
            train_size,
            valid_size,
        ],
        generator=torch.Generator().manual_seed(
            202
        ),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=list,
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=list,
    )

    model = CognitiveLoopModel(
        vocab_size=50000,
        relation_count=None,
        hidden_size=args.hidden_size,
        heads=4,
        layers=3,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-2,
    )

    best_score = float("inf")

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_stats = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            train=True,
        )

        valid_stats = run_epoch(
            model,
            valid_loader,
            device,
            optimizer,
            train=False,
        )

        composite = (
            valid_stats["loss"]
            + 0.50
            * valid_stats[
                "next_latent_loss"
            ]
        )

        print(
            f"EPOCH {epoch} "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_action={train_stats['action_correct']:.4f} "
            f"train_rel={train_stats['relation_correct']:.4f} "
            f"valid_loss={valid_stats['loss']:.4f} "
            f"valid_action={valid_stats['action_correct']:.4f} "
            f"valid_rel={valid_stats['relation_correct']:.4f} "
            f"valid_next={valid_stats['next_latent_loss']:.4f} "
            f"valid_transition={valid_stats['transition_loss']:.4f}",
            flush=True,
        )

        if composite < best_score:
            best_score = composite
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
