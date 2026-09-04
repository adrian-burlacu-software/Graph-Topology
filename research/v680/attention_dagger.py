"""Dataset aggregation: student-induced states receive frozen-teacher labels."""
from __future__ import annotations

import argparse
from pathlib import Path

from attention_dataset import collect_teacher_episodes, write_jsonl
from attention_distill import train
from attention_env import AttentionEnv, benchmark_episodes
from attention_evaluate import load_student
from attention_teacher import V679AttentionTeacher


def collect_dagger(model, rounds=4):
    teacher = V679AttentionTeacher()
    records = []
    for round_index in range(rounds):
        teacher_fraction = max(.25, 1.0 - round_index * .25)
        for spec in benchmark_episodes():
            env = AttentionEnv(spec); state = env.reset(); trajectory = []
            while not env.done:
                label = teacher.select_action(state, deterministic=True)
                student_action = model.select_action(state, deterministic=True)["action"]
                use_teacher = (round_index == 0 or teacher_fraction >= .5)
                action = label["action"] if use_teacher else student_action
                next_state, reward, _, info = env.step(action)
                trajectory.append({
                    "state": state.as_dict(), "teacher": {
                        "logits": label["logits"], "probabilities": label["probabilities"],
                        "selected_action": label["selected_action"], "outcome": label["outcome"],
                    }, "student_action": {"kind": student_action.kind.value, "candidate_id": student_action.candidate_id},
                    "executed_action": {"kind": action.kind.value, "candidate_id": action.candidate_id},
                    "reward": reward, "resulting_state": next_state.as_dict(), "terminal_outcome": info["terminal_outcome"],
                })
                state = next_state
            records.append({"episode_id": f"dagger_{round_index}_{spec['episode_id']}",
                            "split": spec["split"], "round": round_index, "trajectory": trajectory})
    return records


def main():
    parser = argparse.ArgumentParser(description="Collect V680 DAgger attention states.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="./results/v680/dagger_dataset.jsonl")
    parser.add_argument("--retrained-checkpoint", default="./results/v680/student_dagger_checkpoint.pt")
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()
    records = collect_teacher_episodes() + collect_dagger(
        load_student(args.checkpoint), args.rounds
    )
    write_jsonl(args.output, records)
    model = train(records, epochs=80)
    Path(args.retrained_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save({"state_dict": model.state_dict(), "hidden_size": model.hidden_size},
               args.retrained_checkpoint)


if __name__ == "__main__":
    main()
