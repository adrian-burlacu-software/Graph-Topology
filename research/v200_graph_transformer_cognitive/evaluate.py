from __future__ import annotations

import argparse
from pathlib import Path
import json
import random
import sqlite3

import torch

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


# evaluate.py lives at:
#   <repo>/research/v200_graph_transformer_cognitive/evaluate.py
#
# data/ and results/ live directly under <repo>.
ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "conceptnet_compact.db"
MODEL_PATH = ROOT / "results" / "v200_graph_transformer.pt"
OUTPUT_PATH = ROOT / "results" / "v200_graph_transformer_eval.json"


def example_to_batch(
    example: TrainingExample,
    device: torch.device,
) -> GraphBatch:
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

    node_activations = torch.tensor(
        example.node_activations,
        dtype=torch.float32,
        device=device,
    )

    edge_index = torch.tensor(
        [
            [edge[0] for edge in example.edges],
            [edge[1] for edge in example.edges],
        ],
        dtype=torch.long,
        device=device,
    )

    relation_ids = torch.tensor(
        [edge[2] for edge in example.edges],
        dtype=torch.long,
        device=device,
    )

    edge_weights = torch.tensor(
        [edge[3] for edge in example.edges],
        dtype=torch.float32,
        device=device,
    ).log1p()

    return GraphBatch(
        node_ids=node_ids,
        node_roles=node_roles,
        node_activations=node_activations,
        edge_index=edge_index,
        edge_relation_ids=relation_ids,
        edge_weights=edge_weights,
        edge_target_relation=torch.tensor(
            example.target_relation,
            dtype=torch.long,
            device=device,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
    )
    args = parser.parse_args()

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    config = checkpoint[
        "config"
    ]

    model = GraphTransformer(
        vocab_size=50000,
        relation_count=len(RELATIONS),
        hidden_size=config[
            "hidden_size"
        ],
        heads=config[
            "heads"
        ],
        layers=config[
            "layers"
        ],
    ).to(device)

    designer = DesignerHead(
        hidden_size=config[
            "hidden_size"
        ]
    ).to(device)

    model.load_state_dict(
        checkpoint["model"]
    )
    designer.load_state_dict(
        checkpoint["designer"]
    )

    model.eval()
    designer.eval()

    dataset = ConceptNetEdgeDataset(
        DB_PATH,
        samples=args.samples,
        seed=999,
    )

    correct = 0
    total = 0
    top5 = 0

    action_counts = {}

    traces = []

    with torch.inference_mode():
        for index in range(
            len(dataset)
        ):
            example = dataset[index]
            batch = example_to_batch(
                example,
                device,
            )

            out = model(
                batch
            )

            source = out[
                "node_state"
            ][
                example.target_source
            ].unsqueeze(0)

            target = out[
                "node_state"
            ][
                example.target_target
            ].unsqueeze(0)

            logits = model.predict_relation(
                source,
                target,
            )

            prediction = int(
                logits.argmax(
                    dim=-1
                ).item()
            )

            topk = logits.topk(
                k=min(
                    5,
                    logits.shape[-1],
                ),
                dim=-1,
            ).indices[
                0
            ].tolist()

            correct += int(
                prediction
                == example.target_relation
            )
            top5 += int(
                example.target_relation
                in topk
            )
            total += 1

            action_logits = designer(
                out[
                    "graph_state"
                ]
            )

            action_id = int(
                action_logits.argmax(
                    dim=-1
                ).item()
            )

            action_name = (
                DesignerHead.action_names()[
                    action_id
                ]
            )

            action_counts[
                action_name
            ] = (
                action_counts.get(
                    action_name,
                    0,
                )
                + 1
            )

            if len(
                traces
            ) < 20:
                traces.append(
                    {
                        "concepts": example.node_concepts,
                        "target_relation": RELATIONS[
                            example.target_relation
                        ],
                        "predicted_relation": RELATIONS[
                            prediction
                        ],
                        "top5": [
                            RELATIONS[
                                relation_id
                            ]
                            for relation_id
                            in topk
                        ],
                        "designer_action": action_name,
                    }
                )

    report = {
        "experiment": "V200 graph transformer evaluation",
        "samples": total,
        "relation_accuracy": (
            correct
            / max(
                1,
                total,
            )
        ),
        "top5_accuracy": (
            top5
            / max(
                1,
                total,
            )
        ),
        "designer_action_distribution": action_counts,
        "traces": traces,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "relation_accuracy:",
        report[
            "relation_accuracy"
        ],
    )
    print(
        "top5_accuracy:",
        report[
            "top5_accuracy"
        ],
    )
    print(
        "designer_actions:",
        action_counts,
    )
    print(
        "saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
