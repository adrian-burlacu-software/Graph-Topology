from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

try:
    from .dataset import ConceptNetEdgeDataset, TrainingExample
    from .graph_transformer import GraphBatch, GraphTransformer, DesignerHead
    from .long_term_memory import RELATIONS
except ImportError:
    import sys

    RESEARCH_ROOT = Path(__file__).resolve().parents[1]
    if str(RESEARCH_ROOT) not in sys.path:
        sys.path.insert(0, str(RESEARCH_ROOT))

    from v200_graph_transformer_cognitive.dataset import ConceptNetEdgeDataset, TrainingExample
    from v200_graph_transformer_cognitive.graph_transformer import GraphBatch, GraphTransformer, DesignerHead
    from v200_graph_transformer_cognitive.long_term_memory import RELATIONS


# train.py lives at:
#   <repo>/research/v200_graph_transformer_cognitive/train.py
# data/ and results/ live directly under <repo>.
ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "conceptnet_compact.db"
RESULTS = ROOT / "results"
MODEL_PATH = RESULTS / "v200_graph_transformer.pt"


def make_batch(
    examples: list[TrainingExample],
    device: torch.device,
) -> list[GraphBatch]:
    batches = []

    for example in examples:
        node_ids = torch.tensor(
            [
                abs(hash(concept)) % 50000
                for concept in example.node_concepts
            ],
            dtype=torch.long,
            device=device,
        )

        node_roles = torch.tensor(
            example.node_roles,
            dtype=torch.long,
            device=device,
        )

        activations = torch.tensor(
            example.node_activations,
            dtype=torch.float32,
            device=device,
        )

        if example.edges:
            edge_index = torch.tensor(
                [
                    [edge[0] for edge in example.edges],
                    [edge[1] for edge in example.edges],
                ],
                dtype=torch.long,
                device=device,
            )

            edge_relation_ids = torch.tensor(
                [edge[2] for edge in example.edges],
                dtype=torch.long,
                device=device,
            )

            edge_weights = torch.tensor(
                [
                    edge[3]
                    for edge in example.edges
                ],
                dtype=torch.float32,
                device=device,
            ).log1p()

        else:
            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
                device=device,
            )
            edge_relation_ids = torch.empty(
                (0,),
                dtype=torch.long,
                device=device,
            )
            edge_weights = torch.empty(
                (0,),
                dtype=torch.float32,
                device=device,
            )

        batches.append(
            GraphBatch(
                node_ids=node_ids,
                node_roles=node_roles,
                node_activations=activations,
                edge_index=edge_index,
                edge_relation_ids=edge_relation_ids,
                edge_weights=edge_weights,
                edge_target_relation=torch.tensor(
                    example.target_relation,
                    dtype=torch.long,
                    device=device,
                ),
            )
        )

    return batches


def run_epoch(
    model: GraphTransformer,
    designer: DesignerHead,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    train: bool,
) -> dict[str, float]:
    if train:
        model.train()
        designer.train()
    else:
        model.eval()
        designer.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    designer_loss_total = 0.0

    for raw_batch in loader:
        examples = list(raw_batch)
        batches = make_batch(
            examples,
            device,
        )

        for example, batch in zip(
            examples,
            batches,
        ):
            with torch.set_grad_enabled(train):
                out = model(
                    batch
                )

                node_state = out[
                    "node_state"
                ]

                source_state = node_state[
                    example.target_source
                ]
                target_state = node_state[
                    example.target_target
                ]

                relation_logits = model.predict_relation(
                    source_state.unsqueeze(0),
                    target_state.unsqueeze(0),
                )

                relation_loss = nn.functional.cross_entropy(
                    relation_logits,
                    torch.tensor(
                        [example.target_relation],
                        dtype=torch.long,
                        device=device,
                    ),
                )

                designer_logits = designer(
                    out[
                        "graph_state"
                    ]
                )

                # Initial action supervision is intentionally simple:
                # if the known edge exists in the graph, BIND is the useful
                # generic action. This is a scaffold for future learned
                # designer training, not a semantic relation hand-rule.
                action_target = torch.tensor(
                    [4],  # BIND
                    dtype=torch.long,
                    device=device,
                )

                designer_loss = nn.functional.cross_entropy(
                    designer_logits,
                    action_target,
                )

                loss = (
                    relation_loss
                    + 0.10
                    * designer_loss
                )

                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters())
                        + list(designer.parameters()),
                        1.0,
                    )
                    optimizer.step()

            total_loss += float(
                loss.detach().item()
            )
            designer_loss_total += float(
                designer_loss.detach().item()
            )

            predicted = int(
                relation_logits.argmax(
                    dim=-1
                ).item()
            )

            correct += int(
                predicted
                == example.target_relation
            )
            total += 1

    return {
        "loss": (
            total_loss
            / max(1, total)
        ),
        "relation_accuracy": (
            correct
            / max(1, total)
        ),
        "designer_loss": (
            designer_loss_total
            / max(1, total)
        ),
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
    parser.add_argument(
        "--heads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=3,
    )
    args = parser.parse_args()

    random.seed(
        200
    )
    torch.manual_seed(
        200
    )

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(
        "=== V200 GRAPH TRANSFORMER TRAINING ==="
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

    dataset = ConceptNetEdgeDataset(
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
        [
            train_size,
            valid_size,
        ],
        generator=torch.Generator().manual_seed(200),
    )

    loader = DataLoader(
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

    model = GraphTransformer(
        vocab_size=50000,
        relation_count=len(RELATIONS),
        hidden_size=args.hidden_size,
        heads=args.heads,
        layers=args.layers,
    ).to(device)

    designer = DesignerHead(
        hidden_size=args.hidden_size
    ).to(device)

    optimizer = AdamW(
        list(model.parameters())
        + list(designer.parameters()),
        lr=3e-4,
        weight_decay=1e-2,
    )

    best_valid = -1.0

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_stats = run_epoch(
            model,
            designer,
            loader,
            optimizer,
            device,
            train=True,
        )

        valid_stats = run_epoch(
            model,
            designer,
            valid_loader,
            optimizer,
            device,
            train=False,
        )

        print(
            f"EPOCH {epoch} "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_acc={train_stats['relation_accuracy']:.4f} "
            f"valid_loss={valid_stats['loss']:.4f} "
            f"valid_acc={valid_stats['relation_accuracy']:.4f}",
            flush=True,
        )

        if (
            valid_stats[
                "relation_accuracy"
            ]
            > best_valid
        ):
            best_valid = (
                valid_stats[
                    "relation_accuracy"
                ]
            )

            torch.save(
                {
                    "model": model.state_dict(),
                    "designer": designer.state_dict(),
                    "config": {
                        "hidden_size": args.hidden_size,
                        "heads": args.heads,
                        "layers": args.layers,
                        "relations": RELATIONS,
                    },
                    "best_valid_accuracy": best_valid,
                },
                MODEL_PATH,
            )

    print(
        "saved:",
        MODEL_PATH,
        flush=True,
    )
    print(
        "best_valid_accuracy:",
        best_valid,
    )


if __name__ == "__main__":
    main()
