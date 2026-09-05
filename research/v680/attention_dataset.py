"""Validated sequential teacher/student trajectory records."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

from attention_env import AttentionEnv, benchmark_episodes, episodes_from_database
from attention_teacher import V679AttentionTeacher
from attention_types import DATASET_VERSION, JEPA_VERSION, STUDENT_VERSION, TEACHER_VERSION, validate_step_record


def action_candidates(state):
    return [
        {"action": {"kind": "traverse", "candidate_id": index}, "features": candidate.__dict__}
        for index, candidate in enumerate(state.candidate_features)
    ] + [{"action": {"kind": "stop", "candidate_id": None}, "features": {}},
         {"action": {"kind": "abstain", "candidate_id": None}, "features": {}}]


def step_record(episode_id, split, step, state, teacher, action, next_state, reward, terminal_outcome, oracle,
                provenance, partition="", category="", no_proof=False):
    record = {
        "episode_id": episode_id, "split": split, "step": step,
        "state": state.as_dict(), "candidates": action_candidates(state),
        "teacher": {
            "logits": teacher["logits"], "probabilities": teacher["probabilities"],
            "selected_action": teacher["selected_action"], "outcome": teacher["outcome"],
        },
        "action": action.as_dict(), "next_state": next_state.as_dict(),
        "reward": float(reward), "terminal_outcome": terminal_outcome,
        "oracle": dict(oracle), "provenance": dict(provenance),
        "partition": partition, "category": category, "no_proof": bool(no_proof),
        "teacher_version": TEACHER_VERSION, "dataset_version": DATASET_VERSION,
        "student_version": STUDENT_VERSION, "jepa_version": JEPA_VERSION,
    }
    return validate_step_record(record)


def collect_teacher_episodes(episodes=None, temperature=2.0, database=""):
    teacher = V679AttentionTeacher(temperature)
    source = episodes or (episodes_from_database(database) + [benchmark_episodes()[-1]]
                          if database else benchmark_episodes())
    records = []
    for spec in source:
        env = AttentionEnv(spec)
        state = env.reset()
        trajectory = []
        while not env.done:
            decision = teacher.select_action(state, deterministic=True)
            next_state, reward, _, oracle = env.step(decision["action"])
            trajectory.append(step_record(
                spec["episode_id"], spec["split"], len(trajectory), state, decision,
                decision["action"], next_state, reward, oracle["terminal_outcome"], oracle,
                {"generator": "frozen_v679_teacher", "round": 0},
                spec.get("partition", ""), spec.get("category", ""), spec.get("no_proof", False),
            ))
            state = next_state
        records.append({
            "episode_id": spec["episode_id"], "split": spec["split"],
            "partition": spec.get("partition", ""), "category": spec.get("category", ""),
            "no_proof": spec.get("no_proof", False),
            "trajectory": trajectory, "terminal_outcome": trajectory[-1]["terminal_outcome"],
            "provenance": {"generator": "frozen_v679_teacher", "round": 0},
            "teacher_version": teacher.version, "dataset_version": DATASET_VERSION,
            "student_version": STUDENT_VERSION, "jepa_version": JEPA_VERSION,
        })
    return records


def collect_jepa_transition_episodes(episodes=None):
    """Probe every bounded action once to train dynamics without policy labels leaking into it."""
    teacher = V679AttentionTeacher()
    records = []
    for spec in episodes or benchmark_episodes():
        env = AttentionEnv(spec)
        initial = env.reset()
        for action in env.available_actions():
            env.reset()
            state = env.state
            decision = teacher.select_action(state, deterministic=True)
            next_state, reward, _, oracle = env.step(action)
            record = step_record(
                f"{spec['episode_id']}_transition_{action.index(len(state.candidate_features))}",
                spec["split"], 0, state, decision, action, next_state, reward,
                oracle["terminal_outcome"], oracle,
                {"generator": "observable_transition_probe", "round": 0},
                spec.get("partition", ""), spec.get("category", ""), spec.get("no_proof", False),
            )
            records.append({"episode_id": record["episode_id"], "split": spec["split"],
                            "partition": spec.get("partition", ""), "category": spec.get("category", ""),
                            "no_proof": spec.get("no_proof", False), "trajectory": [record],
                            "terminal_outcome": record["terminal_outcome"],
                            "provenance": {"generator": "observable_transition_probe", "round": 0}})
    return records


def write_jsonl(path, records):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    for episode in records:
        if not isinstance(episode.get("trajectory"), list):
            raise ValueError(f"episode {episode.get('episode_id')} has no trajectory")
        for record in episode["trajectory"]:
            validate_step_record(record)
    return records


def dataset_stats(records):
    steps = [step for episode in records for step in episode["trajectory"]]
    return {
        "episodes": len(records), "states": len(steps),
        "unique_states": len({json.dumps(step["state"], sort_keys=True) for step in steps}),
        "teacher_labels": len(steps),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate serialized frozen-V679 teacher trajectories.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--database", default="")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--jepa-transitions", action="store_true")
    parser.add_argument("--decision-boundary", action="store_true")
    parser.add_argument("--samples-per-category", type=int, default=100)
    args = parser.parse_args()
    if args.decision_boundary:
        from attention_benchmark import decision_boundary_episodes
        source = decision_boundary_episodes(args.samples_per_category)
        records = (collect_jepa_transition_episodes(source) if args.jepa_transitions
                   else collect_teacher_episodes(source, temperature=args.temperature))
    else:
        records = (collect_jepa_transition_episodes() if args.jepa_transitions else
                   collect_teacher_episodes(temperature=args.temperature, database=args.database))
    write_jsonl(args.output, records)


if __name__ == "__main__":
    main()
