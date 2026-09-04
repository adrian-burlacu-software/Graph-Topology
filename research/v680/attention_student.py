"""Small recurrent masked policy over only the candidates currently available."""
from __future__ import annotations

import torch
from torch import nn

from attention_types import AttentionAction, AttentionActionKind

STATE_DIM = 6
CANDIDATE_DIM = 12


def tensors_from_observation(state, recurrent=True):
    return (
        torch.tensor([state.state_vector(recurrent=recurrent)], dtype=torch.float32),
        torch.tensor([candidate.vector() for candidate in state.candidate_features], dtype=torch.float32)
        .reshape(len(state.candidate_features), CANDIDATE_DIM),
    )


class NeuralAttentionPolicy(nn.Module):
    def __init__(self, hidden_size=32, recurrent=True, jepa_dim=0):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.recurrent = bool(recurrent)
        self.use_recurrent = bool(recurrent)
        self.jepa_dim = int(jepa_dim)
        self.state_encoder = nn.Sequential(nn.Linear(STATE_DIM, hidden_size), nn.Tanh())
        self.candidate_encoder = nn.Sequential(nn.Linear(CANDIDATE_DIM, hidden_size), nn.Tanh())
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.candidate_head = nn.Linear(hidden_size * 2 + self.jepa_dim, 1)
        self.terminal_head = nn.Linear(hidden_size + self.jepa_dim + 2, 1)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, state_vector, candidate_vectors, hidden=None, action_mask=None, future_representations=None):
        if candidate_vectors.ndim != 2:
            raise ValueError("candidate vectors must be [candidate, feature]")
        state_embedding = self.state_encoder(state_vector)
        if self.use_recurrent:
            prior = hidden if hidden is not None else torch.zeros_like(state_embedding)
            hidden = self.gru(state_embedding, prior)
        else:
            hidden = state_embedding
        candidate_embeddings = self.candidate_encoder(candidate_vectors)
        action_count = candidate_embeddings.shape[0] + 2
        if self.jepa_dim:
            if future_representations is None or tuple(future_representations.shape) != (action_count, self.jepa_dim):
                raise ValueError("JEPA predictions must cover every bounded action")
        elif future_representations is not None:
            raise ValueError("policy was not configured for JEPA features")
        candidate_future = (future_representations[:candidate_embeddings.shape[0]]
                            if self.jepa_dim else candidate_embeddings.new_zeros((candidate_embeddings.shape[0], 0)))
        candidate_logits = self.candidate_head(torch.cat([
            hidden.expand(candidate_embeddings.shape[0], -1), candidate_embeddings, candidate_future
        ], dim=-1)).squeeze(-1)
        terminal_future = future_representations[-2:].reshape(-1, self.jepa_dim) if self.jepa_dim else hidden.new_zeros((2, 0))
        terminal_kind = torch.eye(2, dtype=hidden.dtype, device=hidden.device)
        terminal_logits = self.terminal_head(torch.cat([
            hidden.expand(2, -1), terminal_future, terminal_kind
        ], dim=-1)).squeeze(-1)
        logits = torch.cat([candidate_logits, terminal_logits])
        if action_mask is not None:
            if len(action_mask) != len(logits):
                raise ValueError("action mask must match candidate actions plus STOP and ABSTAIN")
            logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
        return logits, self.value_head(hidden).squeeze(), hidden

    @staticmethod
    def action_mask(state):
        return torch.ones(len(state.candidate_features) + 2, dtype=torch.bool)

    def distribution(self, state, hidden=None, deterministic=False, recurrent=None, jepa=None, shuffled_jepa=False):
        recurrent = self.use_recurrent if recurrent is None else bool(recurrent)
        state_vector, candidate_vectors = tensors_from_observation(state, recurrent)
        future = None
        if jepa is not None:
            with torch.no_grad():
                future = jepa.predict_actions(state)
                if shuffled_jepa and len(future) > 1:
                    future = future.roll(1, 0)
        logits, value, hidden = self(state_vector, candidate_vectors, hidden, self.action_mask(state), future)
        distribution = torch.distributions.Categorical(logits=logits)
        index = torch.argmax(logits) if deterministic else distribution.sample()
        return logits, distribution, int(index), value, hidden

    def select_action(self, state, candidates=None, deterministic=False, hidden=None, jepa=None, shuffled_jepa=False):
        if candidates is not None and candidates is not state.candidate_features:
            from dataclasses import replace
            state = replace(state, candidate_features=candidates)
        logits, distribution, index, value, hidden = self.distribution(
            state, hidden=hidden, deterministic=deterministic, jepa=jepa, shuffled_jepa=shuffled_jepa
        )
        count = len(state.candidate_features)
        action = AttentionAction(
            AttentionActionKind.TRAVERSE if index < count else (
                AttentionActionKind.STOP if index == count else AttentionActionKind.ABSTAIN
            ),
            index if index < count else None,
        )
        return {
            "logits": logits.detach().tolist(), "selected_action": index, "action": action,
            "log_probability": float(distribution.log_prob(torch.tensor(index)).detach()),
            "value": float(value.detach()), "hidden": hidden.detach(),
        }

    def score_candidates(self, state, candidates=None):
        return self.select_action(state, candidates, deterministic=True)["logits"]

    def value(self, state):
        return self.select_action(state, deterministic=True)["value"]
