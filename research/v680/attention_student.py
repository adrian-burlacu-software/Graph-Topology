"""Small recurrent bounded-action attention student."""
from __future__ import annotations

import torch
from torch import nn


STATE_DIM = 6
CANDIDATE_DIM = 12


def tensors_from_observation(state):
    return (
        torch.tensor([state.state_vector()], dtype=torch.float32),
        torch.tensor(
            [candidate.vector() for candidate in state.candidate_features],
            dtype=torch.float32,
        ),
    )


class NeuralAttentionPolicy(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.state_encoder = nn.Sequential(nn.Linear(STATE_DIM, hidden_size), nn.Tanh())
        self.candidate_encoder = nn.Sequential(nn.Linear(CANDIDATE_DIM, hidden_size), nn.Tanh())
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.candidate_head = nn.Linear(hidden_size * 2, 1)
        self.terminal_head = nn.Linear(hidden_size, 2)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, state_vector, candidate_vectors, hidden=None):
        state_embedding = self.state_encoder(state_vector)
        hidden = self.gru(state_embedding, hidden if hidden is not None else torch.zeros_like(state_embedding))
        candidate_embeddings = self.candidate_encoder(candidate_vectors)
        shared = hidden.expand(candidate_embeddings.shape[0], -1)
        candidate_logits = self.candidate_head(torch.cat([shared, candidate_embeddings], dim=-1)).squeeze(-1)
        logits = torch.cat([candidate_logits, self.terminal_head(hidden).squeeze(0)])
        return logits, self.value_head(hidden).squeeze(), hidden

    def score_candidates(self, state, candidates=None):
        candidates = candidates if candidates is not None else state.candidate_features
        with torch.no_grad():
            observation = state
            if candidates is not state.candidate_features:
                from dataclasses import replace
                observation = replace(state, candidate_features=candidates)
            logits, _, _ = self(*tensors_from_observation(observation))
        return logits.tolist()

    def select_action(self, state, candidates=None, deterministic=False):
        from attention_types import AttentionAction, AttentionActionKind
        candidates = candidates if candidates is not None else state.candidate_features
        logits = self.score_candidates(state, candidates)
        selected = int(torch.tensor(logits).argmax())
        if selected < len(candidates):
            action = AttentionAction(AttentionActionKind.TRAVERSE, selected)
        else:
            action = AttentionAction(
                AttentionActionKind.STOP if selected == len(candidates) else AttentionActionKind.ABSTAIN
            )
        return {"logits": logits, "selected_action": selected, "action": action}

    def value(self, state):
        with torch.no_grad():
            _, value, _ = self(
                torch.tensor([state.state_vector()], dtype=torch.float32),
                torch.tensor([candidate.vector() for candidate in state.candidate_features], dtype=torch.float32),
            )
        return float(value)
