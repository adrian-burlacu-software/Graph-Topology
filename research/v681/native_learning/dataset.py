# V681-owned learning implementation; derived from V680.
"""Validated sequential teacher/student trajectory records."""
from __future__ import annotations

import json
import argparse
from copy import deepcopy
from pathlib import Path

from .environment import AttentionEnv, benchmark_episodes, episodes_from_database
from .teacher import V679AttentionTeacher
from .types import (DATASET_VERSION, JEPA_VERSION, STUDENT_VERSION, TEACHER_VERSION,
                    validate_jepa_transition_record, validate_step_record)


def action_candidates(state):
    return [
        {"action": {"kind": "traverse", "candidate_id": index}, "features": candidate.__dict__}
        for index, candidate in enumerate(state.candidate_features)
    ] + [{"action": {"kind": "stop", "candidate_id": None}, "features": {}},
         {"action": {"kind": "abstain", "candidate_id": None}, "features": {}}]


def step_record(episode_id, split, step, state, teacher, action, next_state, reward, terminal_outcome, oracle,
                provenance, partition="", category="", no_proof=False, source="native_teacher"):
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
        "source": source,
        "teacher_version": TEACHER_VERSION, "dataset_version": DATASET_VERSION,
        "student_version": STUDENT_VERSION, "jepa_version": JEPA_VERSION,
    }
    return validate_step_record(record)


def jepa_transition_record(episode_id, step, state, action, next_state, provenance):
    return validate_jepa_transition_record({
        "episode_id": episode_id, "step": step, "state": state.as_dict(),
        "action": action.as_dict(), "next_state": next_state.as_dict(), "provenance": dict(provenance),
    })


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
            "source": spec.get("source", "native_teacher"),
            "no_proof": spec.get("no_proof", False),
            "trajectory": trajectory, "terminal_outcome": trajectory[-1]["terminal_outcome"],
            "provenance": {"generator": "frozen_v679_teacher", "round": 0},
            "teacher_version": teacher.version, "dataset_version": DATASET_VERSION,
            "student_version": STUDENT_VERSION, "jepa_version": JEPA_VERSION,
        })
    return records


def collect_jepa_transition_episodes(episodes=None):
    """Probe every bounded action once to train dynamics without policy labels leaking into it."""
    records = []
    for spec in episodes or benchmark_episodes():
        env = AttentionEnv(spec)
        env.reset()
        for action in env.available_actions():
            env.reset()
            state = env.state
            next_state, _, _, _ = env.step(action)
            records.append(jepa_transition_record(
                f"{spec['episode_id']}_transition_{action.index(len(state.candidate_features))}",
                0, state, action, next_state,
                {"generator": "observable_transition_probe", "round": 0},
            ))
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


def read_jepa_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return [validate_jepa_transition_record(record) for record in records]


def training_records(records):
    """Keep validation and every held-out form out of student fitting."""
    return [record for record in records if not str(record["split"]).startswith("held_out")
            and record["split"] != "heldout" and record.get("partition") != "validation"]


def augment_training_candidate_order(records):
    """Cycle candidate order in train copies while retaining STOP/ABSTAIN terminal slots."""
    augmented = list(records)
    for record in records:
        candidate_count = max((len(step["state"]["candidate_features"]) for step in record["trajectory"]), default=0)
        for shift in range(1, candidate_count):
            variant = deepcopy(record)
            variant["episode_id"] = f"{record['episode_id']}_candidate_rotation_{shift}"
            variant["candidate_order_augmentation"] = f"rotate:{shift}"
            for step in variant["trajectory"]:
                step["episode_id"] = variant["episode_id"]
                count = len(step["state"]["candidate_features"])
                if count < 2:
                    continue
                order = list(range(shift % count, count)) + list(range(shift % count))
                inverse = {old: new for new, old in enumerate(order)}
                step["state"]["candidate_features"] = [step["state"]["candidate_features"][old] for old in order]
                step["candidates"] = [step["candidates"][old] for old in order] + step["candidates"][count:]
                for candidate_id, candidate in enumerate(step["candidates"][:count]):
                    candidate["action"]["candidate_id"] = candidate_id
                teacher = step["teacher"]
                teacher["logits"] = [teacher["logits"][old] for old in order] + teacher["logits"][count:]
                teacher["probabilities"] = [teacher["probabilities"][old] for old in order] + teacher["probabilities"][count:]
                if teacher["selected_action"] < count:
                    teacher["selected_action"] = inverse[teacher["selected_action"]]
                if step["action"]["kind"] == "traverse":
                    step["action"]["candidate_id"] = inverse[step["action"]["candidate_id"]]
                step["provenance"] = {**step["provenance"], "candidate_order_augmentation": f"rotate:{shift}"}
            augmented.append(variant)
    return augmented


def dataset_stats(records):
    steps = [step for episode in records for step in episode["trajectory"]]
    distributions = {}
    for episode in records:
        for step in episode["trajectory"]:
            source = step.get("source", episode.get("source", "unknown"))
            source_counts = distributions.setdefault(source, {"traverse": 0, "stop": 0, "abstain": 0})
            source_counts[teacher_action_kind(step)] += 1
    return {
        "episodes": len(records), "states": len(steps),
        "unique_states": len({json.dumps(step["state"], sort_keys=True) for step in steps}),
        "teacher_labels": len(steps),
        "teacher_action_distribution_by_source": distributions,
    }


def teacher_action_kind(step):
    count = len(step["state"]["candidate_features"])
    selected = step["teacher"]["selected_action"]
    return "traverse" if selected < count else ("stop" if selected == count else "abstain")


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
        from .benchmark import decision_boundary_episodes
        from .environment import stop_boundary_training_episodes
        source = (decision_boundary_episodes(args.samples_per_category)
                  + stop_boundary_training_episodes(max(1, int(args.samples_per_category) // 4)))
        records = (collect_jepa_transition_episodes(source) if args.jepa_transitions
                   else collect_teacher_episodes(source, temperature=args.temperature))
    else:
        records = (collect_jepa_transition_episodes() if args.jepa_transitions else
                   collect_teacher_episodes(temperature=args.temperature, database=args.database))
    write_jsonl(args.output, records)


if __name__ == "__main__":
    main()
