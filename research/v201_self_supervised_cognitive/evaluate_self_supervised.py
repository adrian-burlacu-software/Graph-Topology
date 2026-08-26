from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from .cognitive_graph_encoder import CognitiveGraphEncoder
    from .self_supervised_dataset import SelfSupervisedConceptNetDataset
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
        SelfSupervisedConceptNetDataset,
    )

from v200_graph_transformer_cognitive.long_term_memory import RELATIONS


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

DB_PATH = DATA / "conceptnet_compact.db"
MODEL_PATH = RESULTS / "v201_self_supervised.pt"
OUTPUT_PATH = RESULTS / "v201_self_supervised_eval.json"


def main() -> None:
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(
        "conceptnet_db:",
        DB_PATH.resolve(),
        "exists=",
        DB_PATH.exists(),
        flush=True,
    )
    print(
        "checkpoint:",
        MODEL_PATH.resolve(),
        "exists=",
        MODEL_PATH.exists(),
        flush=True,
    )

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"ConceptNet database not found: {DB_PATH.resolve()}"
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"V201 checkpoint not found: {MODEL_PATH.resolve()}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    config = checkpoint["config"]

    model = CognitiveGraphEncoder(
        vocab_size=config["vocab_size"],
        relation_count=config.get("relation_count"),
        hidden_size=config["hidden_size"],
        heads=config["heads"],
        layers=config["layers"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model"]
    )
    model.eval()

    dataset = SelfSupervisedConceptNetDataset(
        DB_PATH,
        samples=2000,
        seed=202,
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=list,
    )

    totals = {
        "relation": 0,
        "binding": 0,
        "count": 0,
        "next_state_mse": 0.0,
        "consistency_mse": 0.0,
        "masked_correct": 0,
    }

    traces = []

    with torch.inference_mode():
        for batch in loader:
            for example in batch:
                current = model.encode_state(
                    example.current,
                    device,
                )

                nxt = model.encode_state(
                    example.next_state,
                    device,
                )

                perm = model.encode_state(
                    example.permuted_view,
                    device,
                )

                source = current[
                    "node_state"
                ][
                    example.source_index
                ].unsqueeze(0)

                target = current[
                    "node_state"
                ][
                    example.target_index
                ].unsqueeze(0)

                relation_logits = model.predict_relation(
                    source,
                    target,
                )

                relation_prediction = int(
                    relation_logits.argmax(
                        dim=-1
                    ).item()
                )

                relation_correct = (
                    relation_prediction
                    == example.target_relation
                )

                binding_logits = model.predict_binding(
                    source,
                    target,
                )

                binding_correct = (
                    float(
                        torch.sigmoid(
                            binding_logits
                        ).item()
                    )
                    >= 0.5
                )

                predicted_next = model.predict_next_state(
                    current[
                        "graph_state"
                    ],
                    target,
                )

                next_state_mse = float(
                    torch.mean(
                        (
                            predicted_next
                            - nxt["graph_state"]
                        ) ** 2
                    ).item()
                )

                consistency_mse = float(
                    torch.mean(
                        (
                            current[
                                "graph_state"
                            ]
                            - perm[
                                "graph_state"
                            ]
                        ) ** 2
                    ).item()
                )

                masked_node = current[
                    "node_state"
                ][
                    example.masked_target_index
                ].unsqueeze(0)

                masked_logits = model.reconstruct_node(
                    masked_node
                )

                target_id = int(
                    model.concept_ids(
                        [example.masked_target_concept],
                        device,
                    )[0].item()
                )

                masked_prediction = int(
                    masked_logits.argmax(
                        dim=-1
                    ).item()
                )

                totals[
                    "relation"
                ] += int(
                    relation_correct
                )
                totals[
                    "binding"
                ] += int(
                    binding_correct
                )
                totals[
                    "next_state_mse"
                ] += next_state_mse
                totals[
                    "consistency_mse"
                ] += consistency_mse
                totals[
                    "masked_correct"
                ] += int(
                    masked_prediction
                    == target_id
                )
                totals[
                    "count"
                ] += 1

                if len(traces) < 30:
                    traces.append(
                        {
                            "source": example.current.node_concepts[
                                example.source_index
                            ],
                            "target": example.masked_target_concept,
                            "relation": RELATIONS[
                                example.target_relation
                            ],
                            "predicted_relation": RELATIONS[
                                relation_prediction
                            ],
                            "relation_correct": relation_correct,
                            "binding_correct": binding_correct,
                            "next_state_mse": next_state_mse,
                            "consistency_mse": consistency_mse,
                        }
                    )

    count = max(
        1,
        totals["count"],
    )

    report = {
        "experiment": "V201 self-supervised cognitive evaluation",
        "samples": totals["count"],
        "relation_accuracy": totals["relation"] / count,
        "binding_accuracy": totals["binding"] / count,
        "masked_node_accuracy": totals["masked_correct"] / count,
        "mean_next_state_mse": totals["next_state_mse"] / count,
        "mean_consistency_mse": totals["consistency_mse"] / count,
        "checkpoint_best_score": checkpoint.get(
            "best_score"
        ),
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
        report["relation_accuracy"],
    )
    print(
        "binding_accuracy:",
        report["binding_accuracy"],
    )
    print(
        "masked_node_accuracy:",
        report["masked_node_accuracy"],
    )
    print(
        "mean_next_state_mse:",
        report["mean_next_state_mse"],
    )
    print(
        "mean_consistency_mse:",
        report["mean_consistency_mse"],
    )
    print(
        "saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
