"""Teacher trajectory export and JSONL dataset I/O."""
from __future__ import annotations

import json
from pathlib import Path

from attention_env import AttentionEnv, benchmark_episodes, episodes_from_database
from attention_teacher import V679AttentionTeacher


def collect_teacher_episodes(episodes=None, temperature=2.0, database=""):
    teacher = V679AttentionTeacher(temperature)
    output = []
    source = episodes or (
        episodes_from_database(database) + [benchmark_episodes()[-1]]
        if database else benchmark_episodes()
    )
    for spec in source:
        env = AttentionEnv(spec)
        state = env.reset()
        observations = []
        actions = []
        rewards = []
        while not env.done:
            decision = teacher.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(decision["action"])
            observations.append({
                "state": state.as_dict(),
                "candidates": [
                    {"action": "traverse", "candidate_id": index, "features": candidate.__dict__}
                    for index, candidate in enumerate(state.candidate_features)
                ] + [{"action": "stop", "features": {}}, {"action": "abstain", "features": {}}],
                "teacher": {
                    "logits": decision["logits"], "probabilities": decision["probabilities"],
                    "selected_action": decision["selected_action"], "outcome": decision["outcome"],
                },
                "action": {"kind": decision["action"].kind.value, "candidate_id": decision["action"].candidate_id},
                "reward": reward, "resulting_state": next_state.as_dict(), "terminal_outcome": info["terminal_outcome"],
            })
            actions.append(decision["action"])
            rewards.append(reward)
            state = next_state
        output.append({
            "episode_id": spec["episode_id"], "split": spec["split"], "trajectory": observations,
            "terminal_outcome": observations[-1]["terminal_outcome"], "rewards": rewards,
        })
    return output


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
