"""Action-conditioned JEPA for observable future attention-state representations."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from attention_dataset import read_jsonl
from attention_distill import training_records
from attention_student import CANDIDATE_DIM, STATE_DIM
from attention_types import AttentionAction, AttentionObservation, audit_model_input, validate_step_record


OBSERVATION_DIM = STATE_DIM + CANDIDATE_DIM


def observation_vector(state):
    """Observable state summary only; it deliberately excludes the transition oracle."""
    audit_model_input(state.as_dict())
    candidates = [candidate.vector() for candidate in state.candidate_features]
    mean_candidate = (torch.tensor(candidates, dtype=torch.float32).mean(0)
                      if candidates else torch.zeros(CANDIDATE_DIM))
    return torch.cat([torch.tensor(state.state_vector(), dtype=torch.float32), mean_candidate])


def action_vector(state, action_index):
    candidate_count = len(state.candidate_features)
    if action_index < candidate_count:
        base = torch.tensor(state.candidate_features[action_index].vector(), dtype=torch.float32)
        kind = torch.tensor([1.0, 0.0])
    elif action_index == candidate_count:
        base = torch.zeros(CANDIDATE_DIM); kind = torch.tensor([0.0, 1.0])
    elif action_index == candidate_count + 1:
        base = torch.zeros(CANDIDATE_DIM); kind = torch.tensor([-1.0, 1.0])
    else:
        raise ValueError("action index does not belong to observation action space")
    return torch.cat([base, kind])


class AttentionJEPA(nn.Module):
    """Predict target-encoder Z(t+1) from context Z(t) and an action embedding."""
    def __init__(self, representation_dim=24, hidden_size=48, target_momentum=.99):
        super().__init__()
        self.representation_dim = int(representation_dim)
        self.hidden_size = int(hidden_size)
        self.target_momentum = float(target_momentum)
        self.context_encoder = nn.Sequential(
            nn.Linear(OBSERVATION_DIM, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, representation_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(CANDIDATE_DIM + 2, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, representation_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(representation_dim * 2, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, representation_dim),
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False

    def encode_context(self, state):
        return self.context_encoder(observation_vector(state))

    @torch.no_grad()
    def encode_target(self, state):
        return self.target_encoder(observation_vector(state))

    def predict_actions(self, state):
        context = self.encode_context(state)
        actions = torch.stack([action_vector(state, index)
                               for index in range(len(state.candidate_features) + 2)])
        encoded_actions = self.action_encoder(actions)
        return self.predictor(torch.cat([context.expand_as(encoded_actions), encoded_actions], dim=-1))

    def predict_transition(self, state, action_index):
        return self.predict_actions(state)[action_index]

    @torch.no_grad()
    def update_target_encoder(self):
        for target, source in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            target.mul_(self.target_momentum).add_(source, alpha=1 - self.target_momentum)


class JEPAFeatureControl:
    """Frozen, shape-preserving causal feature control with no oracle inputs."""
    def __init__(self, model, mode="real", seed=0, mean=0.0, std=1.0, state_permutation=None):
        self.model = model
        self.mode = mode
        self.seed = int(seed)
        self.representation_dim = model.representation_dim
        self.mean = float(mean)
        self.std = float(std)
        self._sample_generator = torch.Generator().manual_seed(self.seed)
        self.state_permutation = state_permutation or {}

    def _generator(self, state, per_state):
        source = str(state.as_dict()).encode() if per_state else b"fixed"
        digest = hashlib.sha256(source).digest()
        value = int.from_bytes(digest[:8], "big") if per_state else 0
        return torch.Generator().manual_seed(self.seed + value % (2**31 - 1))

    @torch.no_grad()
    def predict_actions(self, state):
        prediction = self.model.predict_actions(state)
        if self.mode == "real":
            return prediction
        if self.mode == "action_shuffled":
            return prediction.roll(1, 0)
        if self.mode == "dimension_permuted":
            order = torch.randperm(prediction.shape[-1], generator=self._generator(state, False))
            return prediction[:, order]
        if self.mode == "state_permuted":
            try:
                return self.state_permutation[str(state.as_dict())].clone()
            except KeyError as error:
                raise ValueError("state permutation lacks this observable state") from error
        if self.mode == "zero":
            return torch.zeros_like(prediction)
        if self.mode in {"fixed_random", "per_state_random", "per_sample_random"}:
            generator = (self._sample_generator if self.mode == "per_sample_random"
                         else self._generator(state, self.mode == "per_state_random"))
            return torch.randn(prediction.shape, generator=generator, dtype=prediction.dtype) * self.std + self.mean
        raise ValueError(f"unknown JEPA control mode {self.mode!r}")


@torch.no_grad()
def representation_statistics(records, model):
    """Control calibration data from observable, frozen JEPA outputs only."""
    vectors = []
    for state, _, _, _ in transition_records(records):
        vectors.append(model.predict_actions(state))
    values = torch.cat(vectors, dim=0)
    return {
        "mean": float(values.mean()), "std": float(values.std(unbiased=False)),
        "min": float(values.min()), "max": float(values.max()),
        "mean_l2_norm": float(values.norm(dim=1).mean()),
        "per_dimension_variance": values.var(dim=0, unbiased=False).tolist(),
    }


def transition_records(records):
    output = []
    for episode in records:
        for step in episode["trajectory"]:
            step = validate_step_record(step)
            state = AttentionObservation.from_dict(step["state"])
            action = AttentionAction.from_dict(step["action"], len(state.candidate_features))
            output.append((state, action.index(len(state.candidate_features)),
                           AttentionObservation.from_dict(step["next_state"]), step))
    if not output:
        raise ValueError("JEPA requires sequential transition records")
    return output


def transition_category(state, action_index, record):
    if record["oracle"].get("valid_proof_edge"):
        return "useful"
    if record["oracle"].get("already_visited"):
        return "invalid"
    if action_index < len(state.candidate_features):
        candidate = state.candidate_features[action_index]
        if candidate.lexical_score or candidate.relation == "related_to":
            return "misleading"
    return "irrelevant"


def train_jepa(records, epochs=8, learning_rate=1e-3, seed=0, model=None):
    torch.manual_seed(seed); random.seed(seed)
    model = model or AttentionJEPA()
    optimizer = torch.optim.Adam(
        list(model.context_encoder.parameters()) + list(model.action_encoder.parameters())
        + list(model.predictor.parameters()), lr=learning_rate)
    transitions = transition_records(training_records(records))
    for _ in range(int(epochs)):
        order = torch.randperm(len(transitions)).tolist()
        for index in order:
            state, action, next_state, _ = transitions[index]
            prediction = model.predict_transition(state, action)
            with torch.no_grad():
                target = model.encode_target(next_state)
            loss = F.mse_loss(prediction, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            model.update_target_encoder()
    return model, optimizer


@torch.no_grad()
def evaluate_jepa(records, model):
    transitions = transition_records(records)
    targets = torch.stack([model.encode_target(next_state) for _, _, next_state, _ in transitions])
    mean_target = targets.mean(0)
    totals = {"jepa": 0.0, "zero": 0.0, "mean": 0.0, "persistence": 0.0, "random": 0.0,
              "shuffled_action": 0.0}
    categories, action_distances = {}, []
    for (state, action, next_state, record), target in zip(transitions, targets):
        prediction = model.predict_transition(state, action)
        error = float(F.mse_loss(prediction, target))
        totals["jepa"] += error
        totals["zero"] += float(F.mse_loss(torch.zeros_like(target), target))
        totals["mean"] += float(F.mse_loss(mean_target, target))
        totals["persistence"] += float(F.mse_loss(model.encode_context(state), target))
        generator = torch.Generator().manual_seed(action * 7919 + len(state.candidate_features))
        totals["random"] += float(F.mse_loss(torch.randn(target.shape, generator=generator), target))
        wrong_action = (action + 1) % (len(state.candidate_features) + 2)
        totals["shuffled_action"] += float(F.mse_loss(model.predict_transition(state, wrong_action), target))
        bucket = transition_category(state, action, record)
        categories.setdefault(bucket, []).append(error)
        if len(state.candidate_features) > 1:
            predictions = model.predict_actions(state)
            action_distances.append(float(torch.pdist(predictions).mean()))
    count = len(transitions)
    prediction_error = {name: value / count for name, value in totals.items()}
    return {
        "prediction_error": prediction_error,
        "relative_improvement_vs_baseline": {
            name: 1 - prediction_error["jepa"] / max(value, 1e-12)
            for name, value in prediction_error.items() if name != "jepa"},
        "by_transition_category": {name: sum(values) / len(values) for name, values in categories.items()},
        "mean_action_conditioned_prediction_distance": sum(action_distances) / max(1, len(action_distances)),
        "action_conditioned": bool(action_distances and min(action_distances) > 1e-6),
        "representation_std": float(targets.std(unbiased=False)),
        "transitions": count,
    }


def load_jepa(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = payload["configuration"]
    model = AttentionJEPA(**config)
    model.load_state_dict(payload["model"]); model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Train action-conditioned attention JEPA.")
    parser.add_argument("--dataset", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0); parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--representation-dim", type=int, default=24); parser.add_argument("--target-momentum", type=float, default=.99)
    args = parser.parse_args()
    records = read_jsonl(args.dataset)
    model, optimizer = train_jepa(records, args.epochs, args.learning_rate, args.seed,
                                  AttentionJEPA(args.representation_dim, target_momentum=args.target_momentum))
    result = evaluate_jepa(records, model)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "seed": args.seed,
                "configuration": {"representation_dim": args.representation_dim, "hidden_size": 48,
                                  "target_momentum": args.target_momentum}}, args.checkpoint)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
