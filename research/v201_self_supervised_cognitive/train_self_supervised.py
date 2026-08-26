from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

try:
    from .cognitive_graph_encoder import CognitiveGraphEncoder
    from .self_supervised_dataset import (
        CognitiveExample,
        SelfSupervisedConceptNetDataset,
    )
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path

    _RESEARCH_ROOT = _Path(__file__).resolve().parents[1]
    if str(_RESEARCH_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_RESEARCH_ROOT))

    from v201_self_supervised_cognitive.cognitive_graph_encoder import (
        CognitiveGraphEncoder,
    )
    from v201_self_supervised_cognitive.self_supervised_dataset import (
        CognitiveExample,
        SelfSupervisedConceptNetDataset,
    )


# train_self_supervised.py lives at:
#   <repo>/research/v201_self_supervised_cognitive/train_self_supervised.py
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DB_PATH = DATA / "conceptnet_compact.db"
MODEL_PATH = RESULTS / "v201_self_supervised.pt"


def state_device_move(
    _state,
    _device,
):
    return None


def run_one(
    model: CognitiveGraphEncoder,
    example: CognitiveExample,
    device: torch.device,
    train: bool,
    optimizer,
) -> dict[str, float]:
    current = model.encode_state(
        example.current,
        device,
    )

    next_state = model.encode_state(
        example.next_state,
        device,
    )

    masked = model.encode_state(
        example.masked_state,
        device,
    )

    permuted = model.encode_state(
        example.permuted_view,
        device,
    )

    source_index = example.source_index
    target_index = example.target_index

    source_state = current[
        "node_state"
    ][
        source_index
    ].unsqueeze(0)

    target_state = current[
        "node_state"
    ][
        target_index
    ].unsqueeze(0)

    relation_logits = model.predict_relation(
        source_state,
        target_state,
    )

    relation_target = torch.tensor(
        [example.target_relation],
        dtype=torch.long,
        device=device,
    )

    relation_loss = nn.functional.cross_entropy(
        relation_logits,
        relation_target,
    )

    binding_logits = model.predict_binding(
        source_state,
        target_state,
    )

    binding_target = torch.tensor(
        [example.positive_binding],
        dtype=torch.float32,
        device=device,
    )

    binding_loss = nn.functional.binary_cross_entropy_with_logits(
        binding_logits,
        binding_target,
    )

    predicted_next = model.predict_next_state(
        current["graph_state"],
        target_state,
    )

    target_next = next_state[
        "graph_state"
    ].detach()

    next_state_loss = nn.functional.mse_loss(
        predicted_next,
        target_next,
    )

    # Masked identity objective. The target node is reconstructed from the
    # graph representation in the current state.
    masked_node = masked[
        "node_state"
    ][
        example.masked_target_index
    ]

    node_logits = model.reconstruct_node(
        masked_node.unsqueeze(0)
    )

    node_id_target = model.concept_ids(
        [example.masked_target_concept],
        device,
    )

    masked_loss = nn.functional.cross_entropy(
        node_logits,
        node_id_target,
    )

    # Order-invariance consistency.
    permuted_graph = permuted[
        "graph_state"
    ]

    current_graph = current[
        "graph_state"
    ]

    consistency_loss = nn.functional.mse_loss(
        current_graph,
        permuted_graph.detach(),
    )

    loss = (
        1.00 * relation_loss
        + 0.40 * binding_loss
        + 0.60 * next_state_loss
        + 0.30 * masked_loss
        + 0.15 * consistency_loss
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
        relation_prediction = int(
            relation_logits.argmax(
                dim=-1
            ).item()
        )

        binding_prediction = float(
            torch.sigmoid(
                binding_logits
            ).item()
        )

        consistency = float(
            consistency_loss.item()
        )

    return {
        "loss": float(
            loss.detach().item()
        ),
        "relation_loss": float(
            relation_loss.detach().item()
        ),
        "binding_loss": float(
            binding_loss.detach().item()
        ),
        "next_state_loss": float(
            next_state_loss.detach().item()
        ),
        "masked_loss": float(
            masked_loss.detach().item()
        ),
        "consistency_loss": float(
            consistency_loss.detach().item()
        ),
        "relation_correct": float(
            relation_prediction
            == example.target_relation
        ),
        "binding_correct": float(
            binding_prediction
            >= 0.5
        ),
        "consistency": consistency,
    }


def run_epoch(
    model: CognitiveGraphEncoder,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    train: bool,
) -> dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    totals: dict[str, float] = {}
    count = 0

    for batch in loader:
        for example in batch:
            with torch.set_grad_enabled(train):
                stats = run_one(
                    model,
                    example,
                    device,
                    train,
                    optimizer,
                )

            for key, value in stats.items():
                totals[key] = (
                    totals.get(key, 0.0)
                    + value
                )

            count += 1

    return {
        key: value / max(1, count)
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
        default=32,
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
    )
    args = parser.parse_args()

    random.seed(201)
    torch.manual_seed(201)

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(
        "=== V201 SELF-SUPERVISED COGNITIVE TRAINING ==="
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

    dataset = SelfSupervisedConceptNetDataset(
        DB_PATH,
        samples=args.samples,
    )

    train_size = int(
        len(dataset)
        * 0.85
    )

    valid_size = len(dataset) - train_size

    train_set, valid_set = random_split(
        dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(201),
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

    vocab_size = 50000

    model = CognitiveGraphEncoder(
        vocab_size=vocab_size,
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
            optimizer,
            device,
            train=True,
        )

        valid_stats = run_epoch(
            model,
            valid_loader,
            optimizer,
            device,
            train=False,
        )

        composite = (
            valid_stats["loss"]
            + valid_stats["next_state_loss"]
        )

        print(
            f"EPOCH {epoch} "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_rel={train_stats['relation_correct']:.4f} "
            f"train_bind={train_stats['binding_correct']:.4f} "
            f"valid_loss={valid_stats['loss']:.4f} "
            f"valid_rel={valid_stats['relation_correct']:.4f} "
            f"valid_bind={valid_stats['binding_correct']:.4f} "
            f"valid_next={valid_stats['next_state_loss']:.4f} "
            f"valid_consistency={valid_stats['consistency']:.4f}",
            flush=True,
        )

        if composite < best_score:
            best_score = composite
            RESULTS.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "hidden_size": args.hidden_size,
                        "heads": 4,
                        "layers": 3,
                        "vocab_size": vocab_size,
                        "relation_count": len(__import__("v200_graph_transformer_cognitive.long_term_memory", fromlist=["RELATION_TO_ID"]).RELATION_TO_ID),
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
