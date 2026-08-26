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

    _RESEARCH_ROOT = _Path(__file__).resolve().parents[1]
    if str(_RESEARCH_ROOT) not in sys.path:
        sys.path.insert(0, str(_RESEARCH_ROOT))

    from v203_multi_action_cognitive.dataset import MultiActionGraphDataset
    from v203_multi_action_cognitive.graph_state import ACTIONS
    from v203_multi_action_cognitive.model import MultiActionController
    from v200_graph_transformer_cognitive.long_term_memory import RELATIONS


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

DB_PATH = DATA / "conceptnet_compact.db"
MODEL_PATH = RESULTS / "v203_multi_action_cognitive.pt"
OUTPUT_PATH = RESULTS / "v203_clean_eval.json"

# Environment-level tolerance for floating point activation comparisons.
ACTIVATION_EPS = 1e-4


def graph_transition_metrics(
    before,
    after,
    expected,
    scenario: str,
) -> dict[str, bool]:
    """
    Score the actual semantic effect of an action.

    The old evaluator used the same "goal edge exists" condition for every
    action. That made REUSE/CREATE/NOOP/INHIBIT/COMMIT look wrong or right for
    the wrong reasons.

    This evaluator scores each action according to its intended state change.
    """

    def node_activation(
        state,
        concept: str,
    ) -> float | None:
        for node in state.nodes:
            if node.concept == concept:
                return node.activation
        return None

    def persistent_edge(
        state,
        source: int,
        relation_id: int,
        target: int,
    ) -> bool:
        return any(
            edge.source == source
            and edge.relation_id == relation_id
            and edge.target == target
            and edge.persistent
            for edge in state.edges
        )

    if scenario == "NOOP" or scenario == "noop":
        return {
            "transition_success": (
                before.signature()
                == after.signature()
            )
        }

    if scenario == "REUSE" or scenario == "reuse":
        before_activation = node_activation(
            before,
            expected["target_concept"],
        )
        after_activation = node_activation(
            after,
            expected["target_concept"],
        )

        return {
            "transition_success": (
                before_activation is not None
                and after_activation is not None
                and after_activation
                > before_activation + ACTIVATION_EPS
            )
        }

    if scenario == "INHIBIT" or scenario == "inhibit":
        before_activation = node_activation(
            before,
            expected["target_concept"],
        )
        after_activation = node_activation(
            after,
            expected["target_concept"],
        )

        return {
            "transition_success": (
                before_activation is not None
                and after_activation is not None
                and after_activation
                < before_activation - ACTIVATION_EPS
            )
        }

    if scenario == "CREATE" or scenario == "create":
        return {
            "transition_success": (
                len(after.nodes)
                > len(before.nodes)
            )
        }

    if scenario == "BRANCH" or scenario == "branch":
        has_new_node = (
            len(after.nodes)
            > len(before.nodes)
        )
        has_new_edge = (
            len(after.edges)
            > len(before.edges)
        )
        return {
            "transition_success": (
                has_new_node
                and has_new_edge
            )
        }

    if scenario == "COMMIT" or scenario == "commit":
        before_persistent_nodes = sum(
            node.persistent
            for node in before.nodes
        )
        after_persistent_nodes = sum(
            node.persistent
            for node in after.nodes
        )

        before_persistent_edges = sum(
            edge.persistent
            for edge in before.edges
        )
        after_persistent_edges = sum(
            edge.persistent
            for edge in after.edges
        )

        return {
            "transition_success": (
                after_persistent_nodes
                > before_persistent_nodes
                or after_persistent_edges
                > before_persistent_edges
            )
        }

    if scenario == "BIND" or scenario == "bind":
        source = expected["source"]
        target = expected["target"]
        relation = expected["relation"]

        return {
            "transition_success": after.has_edge(
                source,
                relation,
                target,
                active_only=True,
            )
        }

    raise ValueError(
        f"Unknown scenario: {scenario}"
    )


def choose_action(
    model,
    state,
    device,
) -> tuple[int, int, int, int]:
    encoded = model.encode(
        state,
        device,
    )

    graph = encoded["graph_state"]
    nodes = encoded["node_state"]

    action = int(
        model.predict_action(
            graph
        ).argmax().item()
    )

    source_logits, target_logits = (
        model.predict_pointers(
            nodes
        )
    )

    source = int(
        source_logits.argmax().item()
    )
    target = int(
        target_logits.argmax().item()
    )

    relation = int(
        model.predict_relation(
            nodes[source].unsqueeze(0),
            nodes[target].unsqueeze(0),
        ).argmax(
            dim=-1
        ).item()
    )

    return (
        action,
        source,
        target,
        relation,
    )


def run_closed_loop(
    model,
    example,
    device,
    steps: int = 3,
) -> dict:
    state = example.state.clone()
    trace = []
    all_transition_success = True

    for step in range(steps):
        action, source, target, relation = (
            choose_action(
                model,
                state,
                device,
            )
        )

        predicted_state = state.apply(
            action,
            source=source,
            target=target,
            relation_id=relation,
        )

        expected = {
            "source": example.goal_source,
            "target": example.goal_target,
            "relation": example.goal_relation,
            "source_concept": example.state.nodes[
                example.goal_source
            ].concept,
            "target_concept": example.state.nodes[
                example.goal_target
            ].concept,
        }

        step_metrics = graph_transition_metrics(
            state,
            predicted_state,
            {
                **expected,
                "target_concept": example.state.nodes[
                    example.goal_target
                ].concept,
            },
            example.scenario,
        )

        trace.append(
            {
                "step": step,
                "action": ACTIONS[action],
                "source": source,
                "target": target,
                "relation": RELATIONS[relation],
                "transition_success": step_metrics[
                    "transition_success"
                ],
            }
        )

        all_transition_success &= step_metrics[
            "transition_success"
        ]

        state = predicted_state

        # Stop once the action has achieved the scenario's immediate goal.
        if step_metrics["transition_success"]:
            break

    return {
        "success": all_transition_success,
        "steps": len(trace),
        "trace": trace,
        "final_state": state,
    }


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=== V203 CLEAN COMPREHENSIVE EVALUATION ===",
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
        "checkpoint:",
        MODEL_PATH.resolve(),
        "exists=",
        MODEL_PATH.exists(),
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

    config = checkpoint["config"]

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
        seed=3203,
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=list,
    )

    total = 0

    action_correct = 0
    source_correct = 0
    target_correct = 0
    relation_correct = 0

    immediate_transition_success = 0
    rollout_transition_success = 0

    per_action_total = Counter()
    per_action_action_correct = Counter()
    per_action_transition = Counter()

    traces = []

    with torch.inference_mode():
        for batch in loader:
            for example in batch:
                encoded = model.encode(
                    example.state,
                    device,
                )
                graph = encoded["graph_state"]
                nodes = encoded["node_state"]

                action = int(
                    model.predict_action(
                        graph
                    ).argmax().item()
                )

                source_logits, target_logits = (
                    model.predict_pointers(
                        nodes
                    )
                )

                source = int(
                    source_logits.argmax().item()
                )
                target = int(
                    target_logits.argmax().item()
                )

                relation = int(
                    model.predict_relation(
                        nodes[source].unsqueeze(0),
                        nodes[target].unsqueeze(0),
                    ).argmax(
                        dim=-1
                    ).item()
                )

                expected = {
                    "source": example.goal_source,
                    "target": example.goal_target,
                    "relation": example.goal_relation,
                    "target_concept": example.state.nodes[
                        example.goal_target
                    ].concept,
                }

                predicted_state = example.state.apply(
                    action,
                    source=source,
                    target=target,
                    relation_id=relation,
                )

                one_step = graph_transition_metrics(
                    example.state,
                    predicted_state,
                    expected,
                    example.scenario,
                )["transition_success"]

                action_ok = (
                    action == example.action_id
                )
                source_ok = (
                    source == example.source_index
                )
                target_ok = (
                    target == example.target_index
                )
                relation_ok = (
                    relation == example.relation_id
                )

                rollout = run_closed_loop(
                    model,
                    example,
                    device,
                    steps=3,
                )

                action_name = ACTIONS[
                    example.action_id
                ]

                total += 1
                action_correct += int(
                    action_ok
                )
                source_correct += int(
                    source_ok
                )
                target_correct += int(
                    target_ok
                )
                relation_correct += int(
                    relation_ok
                )
                immediate_transition_success += int(
                    one_step
                )
                rollout_transition_success += int(
                    rollout["success"]
                )

                per_action_total[
                    action_name
                ] += 1
                per_action_action_correct[
                    action_name
                ] += int(
                    action_ok
                )
                per_action_transition[
                    action_name
                ] += int(
                    one_step
                )

                if len(traces) < 100:
                    traces.append(
                        {
                            "scenario": example.scenario,
                            "gold_action": action_name,
                            "predicted_action": ACTIONS[
                                action
                            ],
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
                            "one_step_transition_success": one_step,
                            "rollout": rollout["trace"],
                        }
                    )

    denominator = max(
        1,
        total,
    )

    per_action = {}
    for action in ACTIONS:
        if not per_action_total[action]:
            continue

        per_action[action] = {
            "count": per_action_total[action],
            "action_accuracy": (
                per_action_action_correct[action]
                / per_action_total[action]
            ),
            "transition_success": (
                per_action_transition[action]
                / per_action_total[action]
            ),
        }

    report = {
        "experiment": (
            "V203 clean action-specific transition evaluation"
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
        "immediate_transition_success": (
            immediate_transition_success
            / denominator
        ),
        "three_step_rollout_success": (
            rollout_transition_success
            / denominator
        ),
        "per_action": per_action,
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
        "immediate_transition_success:",
        report[
            "immediate_transition_success"
        ],
    )
    print(
        "three_step_rollout_success:",
        report[
            "three_step_rollout_success"
        ],
    )

    print(
        "\nper_action:"
    )
    for action, metrics in per_action.items():
        print(
            f"  {action:8s} "
            f"action={metrics['action_accuracy']:.4f} "
            f"transition={metrics['transition_success']:.4f} "
            f"n={metrics['count']}"
        )

    print(
        "\nsaved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
