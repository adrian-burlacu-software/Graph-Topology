"""Observation-level ablations that remove information from every model input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from attention_dataset import read_jsonl


def ablate(records, kind):
    output = json.loads(json.dumps(records))
    for episode in output:
        for step in episode["trajectory"]:
            state = step["state"]
            if kind == "no_relation_activation":
                state["relation_activation"] = {}
                for candidate in state["candidate_features"]:
                    candidate["relation_activation"] = 0.0
            elif kind == "no_candidate_activation":
                state["candidate_activation"] = {}
                for candidate in state["candidate_features"]:
                    candidate["candidate_activation"] = 0.0
            elif kind == "no_history":
                state["visited_nodes"] = []; state["visited_relations"] = []; state["attention_history"] = []
                state["relation_activation"] = {}; state["candidate_activation"] = {}
                for candidate in state["candidate_features"]:
                    candidate["already_visited"] = 0.0
                    candidate["relation_activation"] = 0.0
                    candidate["candidate_activation"] = 0.0
            elif kind == "no_recurrent_state":
                state["attention_history"] = []
            elif kind != "full":
                raise ValueError(f"unknown ablation {kind}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="./results/v680/ablation.json")
    args = parser.parse_args()
    from attention_evaluate import evaluate, load_student
    records = read_jsonl(args.dataset); model = load_student(args.checkpoint)
    report = {}
    for kind in ("full", "no_relation_activation", "no_candidate_activation", "no_history", "no_recurrent_state"):
        model.use_recurrent = kind != "no_recurrent_state"
        report[kind] = evaluate(ablate(records, kind), model, recurrent=model.use_recurrent)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
