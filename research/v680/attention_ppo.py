"""Minimal recurrent PPO fine-tuning scaffold initialized from the student."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from attention_env import AttentionEnv, benchmark_episodes
from attention_evaluate import load_student
from attention_student import tensors_from_observation
from attention_teacher import V679AttentionTeacher


def train_ppo(model, episodes=20, beta=.5, learning_rate=.001):
    teacher = V679AttentionTeacher()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(episodes):
        for spec in benchmark_episodes():
            env = AttentionEnv(spec); state = env.reset()
            while not env.done:
                logits, _, _ = model(*tensors_from_observation(state))
                action_index = int(logits.argmax())
                action = model.select_action(state)["action"]
                _, reward, _, _ = env.step(action)
                teacher_probs = F.softmax(torch.tensor(teacher.score_candidates(state)), dim=-1)
                old_log_probability = F.log_softmax(logits.detach(), dim=-1)[action_index]
                new_log_probability = F.log_softmax(logits, dim=-1)[action_index]
                ratio = torch.exp(new_log_probability - old_log_probability)
                _, value, _ = model(*tensors_from_observation(state))
                advantage = torch.tensor(float(reward)) - value.detach()
                clipped_ratio = torch.clamp(ratio, .8, 1.2)
                policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
                value_loss = F.mse_loss(value, torch.tensor(float(reward)))
                loss = policy_loss + .5 * value_loss
                loss += beta * F.kl_div(F.log_softmax(logits, dim=-1), teacher_probs, reduction="batchmean")
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                state = env.state
    return model


def main():
    parser = argparse.ArgumentParser(description="Teacher-regularized V680 PPO experiment.")
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--checkpoint", default="./results/v680/ppo_checkpoint.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--teacher-kl-beta", type=float, default=.5)
    args = parser.parse_args()
    model = train_ppo(load_student(args.student_checkpoint), args.episodes, args.teacher_kl_beta)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hidden_size": model.hidden_size}, args.checkpoint)


if __name__ == "__main__":
    main()
