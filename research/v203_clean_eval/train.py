from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

try:
    from .dataset import (
        MultiActionGraphDataset,
        ControllerExample,
    )
    from .graph_state import ACTIONS
    from .model import MultiActionController
    from v200_graph_transformer_cognitive.long_term_memory import (
        RELATION_TO_ID,
    )
except ImportError:
    import sys
    from pathlib import Path as _Path

    _RESEARCH_ROOT = _Path(
        __file__
    ).resolve().parents[1]

    if str(_RESEARCH_ROOT) not in sys.path:
        sys.path.insert(
            0,
            str(_RESEARCH_ROOT),
        )

    from v203_multi_action_cognitive.dataset import (
        MultiActionGraphDataset,
        ControllerExample,
    )
    from v203_multi_action_cognitive.graph_state import ACTIONS
    from v203_multi_action_cognitive.model import MultiActionController
    from v200_graph_transformer_cognitive.long_term_memory import (
        RELATION_TO_ID,
    )


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DB_PATH = DATA / "conceptnet_compact.db"
INIT_CHECKPOINT = (
    RESULTS
    / "v202_graph_state_cognitive.pt"
)
MODEL_PATH = (
    RESULTS
    / "v203_multi_action_cognitive.pt"
)


def load_v202(
    model: MultiActionController,
    path: Path,
    device: torch.device,
) -> bool:
    if not path.exists():
        print(
            "v202_checkpoint: not found; training from scratch",
            flush=True,
        )
        return False

    checkpoint = torch.load(
        path,
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

    print(
        "v202_checkpoint: loaded compatible tensors",
        len(compatible),
        flush=True,
    )
    return True


def episode_loss(
    model: MultiActionController,
    example: ControllerExample,
    device: torch.device,
    train: bool,
    optimizer,
) -> dict[str, float]:
    current = model.encode(
        example.state,
        device,
    )

    target = model.encode(
        example.target_state,
        device,
    )

    graph = current[
        "graph_state"
    ]
    nodes = current[
        "node_state"
    ]

    action_logits = model.predict_action(
        graph
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

    source_logits, target_logits = (
        model.predict_pointers(nodes)
    )

    source_target = torch.tensor(
        [example.source_index],
        dtype=torch.long,
        device=device,
    )

    target_target = torch.tensor(
        [example.target_index],
        dtype=torch.long,
        device=device,
    )

    source_loss = nn.functional.cross_entropy(
        source_logits.unsqueeze(0),
        source_target,
    )

    target_loss = nn.functional.cross_entropy(
        target_logits.unsqueeze(0),
        target_target,
    )

    source_state = nodes[
        example.source_index
    ].unsqueeze(0)

    target_state = nodes[
        example.target_index
    ].unsqueeze(0)

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

    next_latent = model.predict_next_latent(
        graph
    )

    next_loss = nn.functional.mse_loss(
        next_latent,
        target[
            "graph_state"
        ].detach(),
    )

    transition_logits = model.predict_transition(
        nodes,
        graph,
    )

    target_activations = torch.tensor(
        [
            node.activation
            for node in example.target_state.nodes
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

    # State value is trained against the inverse number of remaining edits.
    # This gives the controller a general notion of "closer to goal" without
    # semantic task labels.
    goal_complete = (
        example.state.signature()
        == example.target_state.signature()
    )

    value_target = torch.tensor(
        [1.0 if goal_complete else 0.0],
        dtype=torch.float32,
        device=device,
    )

    value = model.predict_value(
        graph
    ).reshape(1)

    value_loss = nn.functional.mse_loss(
        torch.sigmoid(value),
        value_target,
    )

    loss = (
        1.00 * action_loss
        + 0.45 * source_loss
        + 0.45 * target_loss
        + 0.70 * relation_loss
        + 0.65 * next_loss
        + 0.20 * transition_loss
        + 0.10 * value_loss
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

    action_prediction = int(
        action_logits.argmax(
            dim=-1
        ).item()
    )

    source_prediction = int(
        source_logits.argmax().item()
    )

    target_prediction = int(
        target_logits.argmax().item()
    )

    relation_prediction = int(
        relation_logits.argmax(
            dim=-1
        ).item()
    )

    return {
        "loss": float(
            loss.detach().item()
        ),
        "action_correct": float(
            action_prediction
            == example.action_id
        ),
        "source_correct": float(
            source_prediction
            == example.source_index
        ),
        "target_correct": float(
            target_prediction
            == example.target_index
        ),
        "relation_correct": float(
            relation_prediction
            == example.relation_id
        ),
        "next_loss": float(
            next_loss.detach().item()
        ),
        "transition_loss": float(
            transition_loss.detach().item()
        ),
        "value_loss": float(
            value_loss.detach().item()
        ),
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
            with torch.set_grad_enabled(
                train
            ):
                stats = episode_loss(
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
        for key, value
        in totals.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
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
    parser.add_argument(
        "--no-v202-init",
        action="store_true",
    )
    args = parser.parse_args()

    random.seed(203)
    torch.manual_seed(203)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=== V203 MULTI-ACTION COGNITIVE CONTROLLER ===",
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

    dataset = MultiActionGraphDataset(
        DB_PATH,
        samples=args.samples,
    )

    try:
        train_size = int(
            len(dataset)
            * 0.85
        )

        valid_size = (
            len(dataset)
            - train_size
        )

        train_set, valid_set = (
            random_split(
                dataset,
                [
                    train_size,
                    valid_size,
                ],
                generator=torch.Generator().manual_seed(
                    203
                ),
            )
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

        model = MultiActionController(
            vocab_size=50000,
            relation_count=len(
                RELATION_TO_ID
            ),
            hidden_size=args.hidden_size,
            heads=4,
            layers=3,
        ).to(device)

        if not args.no_v202_init:
            load_v202(
                model,
                INIT_CHECKPOINT,
                device,
            )

        optimizer = AdamW(
            model.parameters(),
            lr=2e-4,
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
                True,
            )

            valid_stats = run_epoch(
                model,
                valid_loader,
                device,
                optimizer,
                False,
            )

            composite = (
                valid_stats["loss"]
                + 0.50
                * valid_stats["next_loss"]
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
                        "initialized_from": (
                            str(
                                INIT_CHECKPOINT
                            )
                            if (
                                not args.no_v202_init
                                and INIT_CHECKPOINT.exists()
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

    finally:
        dataset.close()


if __name__ == "__main__":
    main()
