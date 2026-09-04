"""Batched PPO for bounded sequential attention actions."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from attention_env import AttentionEnv, benchmark_episodes
from attention_student import NeuralAttentionPolicy, tensors_from_observation
from attention_teacher import V679AttentionTeacher


def collect_rollouts(model, episodes, episode_count=8, seed=0):
    random.seed(seed); torch.manual_seed(seed)
    transitions = []
    for episode_number in range(episode_count):
        env = AttentionEnv(episodes[episode_number % len(episodes)])
        state, hidden = env.reset(), None
        while not env.done:
            chosen = model.select_action(state, deterministic=False, hidden=hidden)
            next_state, reward, done, oracle = env.step(chosen["action"])
            transitions.append({"state": state.as_dict(), "action": chosen["selected_action"],
                                "reward": reward, "old_log_probability": chosen["log_probability"],
                                "value": chosen["value"], "done": done,
                                "teacher_logits": V679AttentionTeacher().select_action(state)["logits"]})
            state, hidden = next_state, chosen["hidden"]
    return transitions


def gae(transitions, gamma=.99, gae_lambda=.95):
    advantages, returns, next_value, accumulator = [], [], 0.0, 0.0
    for item in reversed(transitions):
        next_value = 0.0 if item["done"] else next_value
        delta = item["reward"] + gamma * next_value - item["value"]
        accumulator = delta + gamma * gae_lambda * (0.0 if item["done"] else accumulator)
        advantages.append(accumulator); returns.append(accumulator + item["value"])
        next_value = item["value"]
    advantages.reverse(); returns.reverse()
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


def ppo_update(model, optimizer, transitions, ppo_epochs=4, minibatch_size=16, clip_epsilon=.2,
               value_coef=.5, entropy_coef=.01, teacher_kl_coef=.05, gamma=.99, gae_lambda=.95):
    advantages, returns = gae(transitions, gamma, gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    total_loss = 0.0
    for _ in range(ppo_epochs):
        order = torch.randperm(len(transitions))
        for batch in order.split(minibatch_size):
            losses = []
            for index in batch.tolist():
                item = transitions[index]
                from attention_types import AttentionObservation
                state = AttentionObservation.from_dict(item["state"])
                vector, candidates = tensors_from_observation(state)
                logits, value, _ = model(vector, candidates, action_mask=model.action_mask(state))
                distribution = torch.distributions.Categorical(logits=logits)
                action = torch.tensor(item["action"])
                ratio = torch.exp(distribution.log_prob(action) - torch.tensor(item["old_log_probability"]))
                unclipped = ratio * advantages[index]
                clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages[index]
                policy_loss = -torch.minimum(unclipped, clipped)
                value_loss = F.mse_loss(value, returns[index])
                entropy = distribution.entropy()
                teacher = F.softmax(torch.tensor(item["teacher_logits"]), -1)
                teacher_kl = F.kl_div(F.log_softmax(logits, -1), teacher, reduction="sum")
                losses.append(policy_loss + value_coef * value_loss - entropy_coef * entropy +
                              teacher_kl_coef * teacher_kl)
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total_loss += float(loss.detach())
    return {"loss": total_loss / max(1, ppo_epochs)}


def run_ppo(episodes=None, episode_count=8, seed=0, checkpoint=None, resume=None,
            initial_checkpoint=None, **hyperparameters):
    torch.manual_seed(seed); random.seed(seed)
    model = NeuralAttentionPolicy(); optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters.pop("learning_rate", 3e-4))
    prior_step = prior_episode = 0
    if resume or initial_checkpoint:
        payload = torch.load(resume or initial_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
        prior_step, prior_episode = payload.get("step", 0), payload.get("episode", 0)
    trajectories = collect_rollouts(model, episodes or benchmark_episodes(), episode_count, seed)
    metrics = ppo_update(model, optimizer, trajectories, **hyperparameters)
    if checkpoint:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": prior_step + len(trajectories), "episode": prior_episode + episode_count, "seed": seed,
                    "hyperparameters": hyperparameters}, checkpoint)
    return model, trajectories, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--resume"); parser.add_argument("--student-checkpoint")
    parser.add_argument("--teacher-kl-coef", type=float, default=.05); parser.add_argument("--entropy-coef", type=float, default=.01)
    args = parser.parse_args()
    _, transitions, metrics = run_ppo(episode_count=args.episodes, seed=args.seed, checkpoint=args.checkpoint, resume=args.resume,
                                      initial_checkpoint=args.student_checkpoint,
                                      ppo_epochs=args.ppo_epochs, teacher_kl_coef=args.teacher_kl_coef,
                                      entropy_coef=args.entropy_coef)
    print({"transitions": len(transitions), **metrics})


if __name__ == "__main__":
    main()
