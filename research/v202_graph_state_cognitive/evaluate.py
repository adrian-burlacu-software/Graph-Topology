from __future__ import annotations

from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader

try:
    from .dataset import GraphTrajectoryDataset
    from .model import CognitiveLoopModel
    from .graph_state import ACTIONS
except ImportError:
    import sys
    from pathlib import Path as _Path

    _RESEARCH_ROOT = _Path(__file__).resolve().parents[1]
    if str(_RESEARCH_ROOT) not in sys.path:
        sys.path.insert(0, str(_RESEARCH_ROOT))

    from v202_graph_state_cognitive.dataset import GraphTrajectoryDataset
    from v202_graph_state_cognitive.model import CognitiveLoopModel
    from v202_graph_state_cognitive.graph_state import ACTIONS

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DB_PATH = DATA / "conceptnet_compact.db"
MODEL_PATH = RESULTS / "v202_graph_state_cognitive.pt"
OUTPUT_PATH = RESULTS / "v202_graph_state_cognitive_eval.json"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0), flush=True)

    print(
        "checkpoint:",
        MODEL_PATH.resolve(),
        "exists=",
        MODEL_PATH.exists(),
        flush=True,
    )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    config = checkpoint["config"]
    model = CognitiveLoopModel(
        vocab_size=config["vocab_size"],
        relation_count=config["relation_count"],
        hidden_size=config["hidden_size"],
        heads=config["heads"],
        layers=config["layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = GraphTrajectoryDataset(
        DB_PATH,
        samples=1500,
        seed=1202,
    )
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=list,
    )

    totals = {
        "count": 0,
        "action_correct": 0,
        "relation_correct": 0,
        "next_latent_mse": 0.0,
        "transition_mse": 0.0,
        "rollout_edge_success": 0,
    }
    traces = []

    with torch.inference_mode():
        for batch in loader:
            for example in batch:
                current = model.forward_state(
                    example.current, device
                )
                target = model.forward_state(
                    example.target_next, device
                )

                action_logits = model.predict_action(
                    current["graph_state"]
                )
                action = int(
                    action_logits.argmax(dim=-1).item()
                )

                source = current["node_state"][
                    example.source_index
                ].unsqueeze(0)
                target_node = current["node_state"][
                    example.target_index
                ].unsqueeze(0)

                relation_logits = model.predict_relation(
                    source,
                    target_node,
                )
                relation = int(
                    relation_logits.argmax(dim=-1).item()
                )

                predicted_next = model.predict_next_latent(
                    current["graph_state"]
                )
                next_latent_mse = float(
                    torch.mean(
                        (predicted_next - target["graph_state"]) ** 2
                    ).item()
                )

                transition_logits = model.predict_node_transition(
                    current["node_state"],
                    current["graph_state"],
                )
                target_activation = torch.tensor(
                    [
                        node.activation
                        for node in example.target_next.nodes
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                transition_mse = float(
                    torch.mean(
                        (
                            torch.sigmoid(transition_logits)
                            - target_activation
                        ) ** 2
                    ).item()
                )

                predicted_state = example.current.apply(
                    action,
                    source=example.source_index,
                    target=example.target_index,
                    relation_id=relation,
                )

                expected_edge = (
                    example.source_index,
                    example.relation_id,
                    example.target_index,
                )
                rollout_edges = {
                    (
                        edge.source,
                        edge.relation_id,
                        edge.target,
                    )
                    for edge in predicted_state.edges
                    if edge.activation > 0.5
                }
                rollout_success = expected_edge in rollout_edges

                action_correct = (
                    action == example.action_id
                )
                relation_correct = (
                    relation == example.relation_id
                )

                totals["count"] += 1
                totals["action_correct"] += int(action_correct)
                totals["relation_correct"] += int(relation_correct)
                totals["next_latent_mse"] += next_latent_mse
                totals["transition_mse"] += transition_mse
                totals["rollout_edge_success"] += int(
                    rollout_success
                )

                if len(traces) < 40:
                    traces.append(
                        {
                            "source": example.current.nodes[
                                example.source_index
                            ].concept,
                            "target": example.current.nodes[
                                example.target_index
                            ].concept,
                            "gold_action": ACTIONS[
                                example.action_id
                            ],
                            "predicted_action": ACTIONS[action],
                            "gold_relation_id": example.relation_id,
                            "predicted_relation_id": relation,
                            "action_correct": action_correct,
                            "relation_correct": relation_correct,
                            "next_latent_mse": next_latent_mse,
                            "transition_mse": transition_mse,
                            "rollout_edge_success": rollout_success,
                        }
                    )

    count = max(1, totals["count"])
    report = {
        "experiment": "V202 graph-state cognitive loop",
        "samples": totals["count"],
        "action_accuracy": totals["action_correct"] / count,
        "relation_accuracy": totals["relation_correct"] / count,
        "mean_next_latent_mse": totals["next_latent_mse"] / count,
        "mean_transition_mse": totals["transition_mse"] / count,
        "rollout_edge_success": totals["rollout_edge_success"] / count,
        "traces": traces,
    }

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("action_accuracy:", report["action_accuracy"])
    print("relation_accuracy:", report["relation_accuracy"])
    print("mean_next_latent_mse:", report["mean_next_latent_mse"])
    print("mean_transition_mse:", report["mean_transition_mse"])
    print("rollout_edge_success:", report["rollout_edge_success"])
    print("saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
