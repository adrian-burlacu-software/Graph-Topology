from __future__ import annotations

import json
from pathlib import Path

import torch

# Direct execution from research/:
#   python .\\v209_teacher_distilled_cognitive\\evaluate.py
import sys
_HERE = Path(__file__).resolve().parent
_RESEARCH_ROOT = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

from dataset import TeacherDataset
from model import TeacherDistilledController
from state import State


from v200_graph_transformer_cognitive.long_term_memory import (
    RELATION_TO_ID,
    RELATIONS,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

TEACHER_DATA = (
    RESULTS
    / "v205r_teacher_dataset.jsonl"
)
MODEL_PATH = (
    RESULTS
    / "v209_teacher_distilled_cognitive.pt"
)
OUTPUT_PATH = (
    RESULTS
    / "v209_teacher_distilled_eval.json"
)

ACTIONS = (
    "NOOP",
    "REUSE",
    "CREATE",
    "BRANCH",
    "INHIBIT",
    "BIND",
    "COMMIT",
)


def goal_reached(
    state: State,
    example: dict,
) -> bool:
    source = example["state"].nodes[
        example["goal_source_index"]
    ].concept

    target = example["state"].nodes[
        example["goal_target_index"]
    ].concept

    relation = example[
        "goal_relation"
    ]

    target_node = state.node(
        target
    )

    target_active = (
        target_node is not None
        and target_node.activation > 0.5
    )

    return (
        target_active
        and state.has_edge(
            source,
            relation,
            target,
            active_only=True,
        )
    )


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=== V209 EVALUATION ===",
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

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            MODEL_PATH.resolve()
        )

    dataset = TeacherDataset(
        TEACHER_DATA
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    config = checkpoint[
        "config"
    ]

    model = TeacherDistilledController(
        vocab_size=config["vocab_size"],
        relation_count=config["relation_count"],
        hidden_size=config["hidden_size"],
        heads=config["heads"],
        layers=config["layers"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model"]
    )
    model.eval()

    _train, valid_indices = dataset.split(
        valid_fraction=0.15,
        seed=209,
    )

    totals = {
        "count": 0,
        "action_correct": 0,
        "source_correct": 0,
        "target_correct": 0,
        "relation_correct": 0,
        "goal_success": 0,
        "next_state_mse": 0.0,
    }

    per_action = {
        action: {
            "count": 0,
            "action_correct": 0,
            "goal_success": 0,
        }
        for action in ACTIONS
    }

    traces = []

    with torch.inference_mode():
        for index in valid_indices:
            example = dataset[index]

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

            action = int(
                model.predict_action(
                    current["graph_state"],
                    goal_state,
                ).argmax().item()
            )

            source_logits, target_logits = (
                model.predict_pointers(
                    current["node_state"],
                    goal_state,
                )
            )

            source = int(
                source_logits.argmax().item()
            )
            target_index = int(
                target_logits.argmax().item()
            )

            relation = int(
                model.predict_relation(
                    current["node_state"][
                        min(
                            source,
                            len(
                                example["state"].nodes
                            ) - 1,
                        )
                    ].unsqueeze(0),
                    current["node_state"][
                        min(
                            target_index,
                            len(
                                example["state"].nodes
                            ) - 1,
                        )
                    ].unsqueeze(0),
                    goal_state,
                ).argmax(
                    dim=-1
                ).item()
            )

            source_concept = None
            if (
                0 <= source
                < len(
                    example["state"].nodes
                )
            ):
                source_concept = (
                    example["state"].nodes[
                        source
                    ].concept
                )

            target_concept = None
            if (
                0 <= target_index
                < len(
                    example["state"].nodes
                )
            ):
                target_concept = (
                    example["state"].nodes[
                        target_index
                    ].concept
                )

            relation_name = RELATIONS[
                relation
            ]

            predicted_state = example[
                "state"
            ].apply(
                action,
                source=source_concept,
                target=target_concept,
                relation=relation_name,
            )

            action_ok = (
                action
                == example["action_id"]
            )

            source_ok = (
                source
                == example["source_index"]
                if example["source_index"] >= 0
                else True
            )

            target_ok = (
                target_index
                == example["target_index"]
                if example["target_index"] >= 0
                else True
            )

            relation_id = RELATION_TO_ID.get(
                example["relation"],
                0,
            )

            relation_ok = (
                relation
                == relation_id
            )

            success = goal_reached(
                predicted_state,
                example,
            )

            next_latent = model.predict_next_latent(
                current["graph_state"],
                goal_state,
            )

            next_mse = float(
                torch.mean(
                    (
                        next_latent
                        - target[
                            "graph_state"
                        ]
                    ) ** 2
                ).item()
            )

            totals["count"] += 1
            totals["action_correct"] += int(action_ok)
            totals["source_correct"] += int(source_ok)
            totals["target_correct"] += int(target_ok)
            totals["relation_correct"] += int(relation_ok)
            totals["goal_success"] += int(success)
            totals["next_state_mse"] += next_mse

            gold_action = ACTIONS[
                example["action_id"]
            ]

            per_action[
                gold_action
            ]["count"] += 1
            per_action[
                gold_action
            ]["action_correct"] += int(
                action_ok
            )
            per_action[
                gold_action
            ]["goal_success"] += int(
                success
            )

            if len(traces) < 100:
                traces.append(
                    {
                        "case_id": example[
                            "case_id"
                        ],
                        "gold_action": gold_action,
                        "predicted_action": ACTIONS[
                            action
                        ],
                        "gold_source": example[
                            "state"
                        ].nodes[
                            example[
                                "source_index"
                            ]
                        ].concept
                        if example[
                            "source_index"
                        ] >= 0
                        else None,
                        "predicted_source": source_concept,
                        "gold_target": example[
                            "state"
                        ].nodes[
                            example[
                                "target_index"
                            ]
                        ].concept
                        if example[
                            "target_index"
                        ] >= 0
                        else None,
                        "predicted_target": target_concept,
                        "gold_relation": example[
                            "relation"
                        ],
                        "predicted_relation": relation_name,
                        "action_correct": action_ok,
                        "source_correct": source_ok,
                        "target_correct": target_ok,
                        "relation_correct": relation_ok,
                        "goal_success": success,
                        "next_state_mse": next_mse,
                    }
                )

    count = max(
        1,
        totals["count"],
    )

    report = {
        "experiment": (
            "V209 teacher-distilled graph cognitive controller"
        ),
        "validation_samples": totals["count"],
        "action_accuracy": (
            totals["action_correct"] / count
        ),
        "source_accuracy": (
            totals["source_correct"] / count
        ),
        "target_accuracy": (
            totals["target_correct"] / count
        ),
        "relation_accuracy": (
            totals["relation_correct"] / count
        ),
        "goal_one_step_success": (
            totals["goal_success"] / count
        ),
        "mean_next_state_mse": (
            totals["next_state_mse"] / count
        ),
        "per_action": {
            action: {
                "count": metrics["count"],
                "action_accuracy": (
                    metrics["action_correct"]
                    / max(
                        1,
                        metrics["count"],
                    )
                ),
                "goal_success": (
                    metrics["goal_success"]
                    / max(
                        1,
                        metrics["count"],
                    )
                ),
            }
            for action, metrics
            in per_action.items()
            if metrics["count"]
        },
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
        "action_accuracy:",
        report["action_accuracy"],
    )
    print(
        "source_accuracy:",
        report["source_accuracy"],
    )
    print(
        "target_accuracy:",
        report["target_accuracy"],
    )
    print(
        "relation_accuracy:",
        report["relation_accuracy"],
    )
    print(
        "goal_one_step_success:",
        report[
            "goal_one_step_success"
        ],
    )
    print(
        "mean_next_state_mse:",
        report[
            "mean_next_state_mse"
        ],
    )
    print(
        "saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
