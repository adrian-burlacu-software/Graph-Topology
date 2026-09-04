"""Evaluate which symbolic observation fields the student depends on."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from attention_dataset import read_jsonl
from attention_evaluate import evaluate, load_student


def ablate(records, kind):
    copied = []
    for record in records:
        record = json.loads(json.dumps(record))
        for step in record["trajectory"]:
            state = step["state"]
            if kind == "no_relation_activation":
                state["relation_activation"] = {}
            elif kind == "no_candidate_activation":
                state["candidate_activation"] = {}
            elif kind == "no_history":
                state["visited_nodes"] = []; state["visited_relations"] = []
        copied.append(record)
    return copied


def main():
    parser = argparse.ArgumentParser(description="Run V680 symbolic-state ablations.")
    parser.add_argument("--dataset", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="./results/v680/ablation.json")
    args = parser.parse_args()
    records = read_jsonl(args.dataset); model = load_student(args.checkpoint)
    report = {"full_state": evaluate(records, model)}
    for kind in ("no_relation_activation", "no_candidate_activation", "no_history"):
        report[kind] = evaluate(ablate(records, kind), model)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
