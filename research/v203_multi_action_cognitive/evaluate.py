from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader

try:
    from .dataset import MultiActionGraphDataset
    from .graph_state import ACTIONS
    from .model import MultiActionController
    from v200_graph_transformer_cognitive.long_term_memory import RELATIONS
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

    from v203_multi_action_cognitive.dataset import MultiActionGraphDataset
    from v203_multi_action_cognitive.graph_state import ACTIONS
    from v203_multi_action_cognitive.model import MultiActionController
    from v200_graph_transformer_cognitive.long_term_memory import RELATIONS


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

DB_PATH = DATA / "conceptnet_compact.db"
MODEL_PATH = (
    RESULTS
    / "v203_multi_action_cognitive.pt"
)
OUTPUT_PATH = (
    RESULTS
    / "v203_multi_action_cognitive_eval.json"
)


def choose_action(
    model,
    state,
    device,
):
    encoded = model.encode(
        state,
        device,
    )

    graph = encoded[
        "graph_state"
    ]
    nodes = encoded[
        "node_state"
    ]

    action_logits = model.predict_action(
        graph
    )
    action = int(
        action_logits.argmax().item()
    )

    source_logits, target_logits = (
        model.predict_pointers(nodes)
    )

    source = int(
        source_logits.argmax().item()
    )
    target = int(
        target_logits.argmax().item()
    )

    source_state = nodes[
        source
    ].unsqueeze(0)

    target_state = nodes[
        target
    ].unsqueeze(0)

    relation_logits = model.predict_relation(
        source_state,
        target_state,
    )

    relation = int(
        relation_logits.argmax(
            dim=-1
        ).item()
    )

    return (
        action,
        source,
        target,
        relation,
    )


def rollout(
    model,
    example,
    device,
    steps: int = 3,
) -> dict:
    state = example.state.clone()

    trace = []

    for step in range(
        steps
    ):
        action, source, target, relation = (
            choose_action(
                model,
                state,
                device,
            )
        )

        trace.append(
            {
                "step": step,
                "action": ACTIONS[action],
                "source": source,
                "target": target,
                "relation": RELATIONS[relation],
            }
        )

        state = state.apply(
            action,
            source=source,
            target=target,
            relation_id=relation,
        )

        # Reached graph goal.
        if (
            example.scenario
            in {
                "bind",
                "commit",
                "noop",
            }
            and state.has_edge(
                example.goal_source,
                example.goal_relation,
                example.goal_target,
                active_only=True,
            )
        ):
            break

    return {
        "final_state": state,
        "trace": trace,
        "goal_edge_present": state.has_edge(
            example.goal_source,
            example.goal_relation,
            example.goal_target,
            active_only=True,
        ),
        "steps": len(trace),
    }


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=== V203 EVALUATION ===",
        flush=True,
    )
    print(
        "device:",
        device,
        flush=True,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            MODEL_PATH.resolve()
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    config = checkpoint[
        "config"
    ]

    model = MultiActionController(
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

    dataset = MultiActionGraphDataset(
        DB_PATH,
        samples=2800,
        seed=2203,
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=list,
    )

    action_correct = 0
    source_correct = 0
    target_correct = 0
    relation_correct = 0
    one_step_success = 0
    rollout_success = 0
    commit_success = 0
    noop_success = 0
    inhibit_success = 0
    branch_success = 0
    create_success = 0

    total = 0

    per_action_total = Counter()
    per_action_correct = Counter()

    traces = []

    with torch.inference_mode():
        for batch in loader:
            for example in batch:
                encoded = model.encode(
                    example.state,
                    device,
                )

                graph = encoded[
                    "graph_state"
                ]
                nodes = encoded[
                    "node_state"
                ]

                action_logits = model.predict_action(
                    graph
                )
                action = int(
                    action_logits.argmax().item()
                )

                source_logits, target_logits = (
                    model.predict_pointers(nodes)
                )

                source = int(
                    source_logits.argmax().item()
                )

                target = int(
                    target_logits.argmax().item()
                )

                relation = int(
                    model.predict_relation(
                        nodes[
                            source
                        ].unsqueeze(0),
                        nodes[
                            target
                        ].unsqueeze(0),
                    ).argmax(
                        dim=-1
                    ).item()
                )

                action_name = ACTIONS[
                    example.action_id
                ]

                action_ok = (
                    action
                    == example.action_id
                )
                source_ok = (
                    source
                    == example.source_index
                )
                target_ok = (
                    target
                    == example.target_index
                )
                relation_ok = (
                    relation
                    == example.relation_id
                )

                predicted_state = example.state.apply(
                    action,
                    source=source,
                    target=target,
                    relation_id=relation,
                )

                edge_ok = predicted_state.has_edge(
                    example.goal_source,
                    example.goal_relation,
                    example.goal_target,
                    active_only=True,
                )

                trajectory = rollout(
                    model,
                    example,
                    device,
                    steps=3,
                )

                rollout_ok = trajectory[
                    "goal_edge_present"
                ]

                action_correct += int(action_ok)
                source_correct += int(source_ok)
                target_correct += int(target_ok)
                relation_correct += int(relation_ok)
                one_step_success += int(edge_ok)
                rollout_success += int(rollout_ok)

                per_action_total[
                    action_name
                ] += 1
                per_action_correct[
                    action_name
                ] += int(action_ok)

                if (
                    example.scenario
                    == "commit"
                ):
                    commit_success += int(
                        any(
                            e.persistent
                            for e in predicted_state.edges
                        )
                    )

                if example.scenario == "noop":
                    noop_success += int(
                        predicted_state.signature()
                        == example.state.signature()
                    )

                if example.scenario == "inhibit":
                    distractor_index = (
                        example.target_index
                    )
                    if (
                        0 <= distractor_index
                        < len(
                            predicted_state.nodes
                        )
                    ):
                        inhibit_success += int(
                            predicted_state.nodes[
                                distractor_index
                            ].activation
                            < 0.20
                        )

                if example.scenario == "branch":
                    branch_success += int(
                        len(
                            predicted_state.nodes
                        )
                        > len(
                            example.state.nodes
                        )
                    )

                if example.scenario == "create":
                    create_success += int(
                        len(
                            predicted_state.nodes
                        )
                        > len(
                            example.state.nodes
                        )
                    )

                if len(traces) < 80:
                    traces.append(
                        {
                            "scenario": example.scenario,
                            "gold_action": action_name,
                            "predicted_action": ACTIONS[action],
                            "gold_relation": RELATIONS[
                                example.relation_id
                            ],
                            "predicted_relation": RELATIONS[
                                relation
                            ],
                            "action_correct": action_ok,
                            "source_correct": source_ok,
                            "target_correct": target_ok,
                            "relation_correct": relation_ok,
                            "one_step_success": edge_ok,
                            "rollout": trajectory[
                                "trace"
                            ],
                        }
                    )

                total += 1

    denominator = max(
        1,
        total,
    )

    report = {
        "experiment": (
            "V203 multi-action cognitive controller"
        ),
        "samples": total,
        "action_accuracy": (
            action_correct
            / denominator
        ),
        "source_accuracy": (
            source_correct
            / denominator
        ),
        "target_accuracy": (
            target_correct
            / denominator
        ),
        "relation_accuracy": (
            relation_correct
            / denominator
        ),
        "one_step_success": (
            one_step_success
            / denominator
        ),
        "three_step_rollout_success": (
            rollout_success
            / denominator
        ),
        "commit_success": (
            commit_success
            / max(
                1,
                per_action_total["COMMIT"],
            )
        ),
        "noop_success": (
            noop_success
            / max(
                1,
                per_action_total["NOOP"],
            )
        ),
        "inhibit_success": (
            inhibit_success
            / max(
                1,
                per_action_total["INHIBIT"],
            )
        ),
        "branch_success": (
            branch_success
            / max(
                1,
                per_action_total["BRANCH"],
            )
        ),
        "create_success": (
            create_success
            / max(
                1,
                per_action_total["CREATE"],
            )
        ),
        "per_action_accuracy": {
            action: (
                per_action_correct[action]
                / max(
                    1,
                    per_action_total[action],
                )
            )
            for action in ACTIONS
            if per_action_total[action]
        },
        "per_action_counts": dict(
            per_action_total
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
        "one_step_success:",
        report["one_step_success"],
    )
    print(
        "three_step_rollout_success:",
        report[
            "three_step_rollout_success"
        ],
    )
    print(
        "per_action_accuracy:",
        report[
            "per_action_accuracy"
        ],
    )
    print(
        "saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
