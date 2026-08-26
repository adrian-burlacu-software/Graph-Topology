from __future__ import annotations

from pathlib import Path
import hashlib

import torch
from torch import nn

from v200_graph_transformer_cognitive.graph_transformer import (
    GraphAttentionBlock,
)
from v200_graph_transformer_cognitive.long_term_memory import (
    RELATION_TO_ID,
    RELATIONS,
)

from .graph_state import ACTIONS


def concept_id(
    concept: str,
    vocab_size: int,
) -> int:
    value = int.from_bytes(
        hashlib.blake2b(
            concept.encode("utf-8"),
            digest_size=8,
        ).digest(),
        "little",
        signed=False,
    )
    return value % vocab_size


class CognitiveLoopModel(nn.Module):
    """
    A small graph-transformer state-transition model.

    Inputs:
        current working graph

    Outputs:
        latent state
        designer action
        relation choice
        predicted next graph latent state
        node transition scores

    The model does not know the semantic meaning of a relation ID.
    """

    def __init__(
        self,
        *,
        vocab_size: int = 50000,
        relation_count: int | None = None,
        role_count: int = 8,
        hidden_size: int = 128,
        heads: int = 4,
        layers: int = 3,
    ) -> None:
        super().__init__()

        if relation_count is None:
            relation_count = len(
                RELATION_TO_ID
            )

        self.vocab_size = vocab_size
        self.relation_count = relation_count
        self.hidden_size = hidden_size

        self.node_embedding = nn.Embedding(
            vocab_size,
            hidden_size,
        )
        self.role_embedding = nn.Embedding(
            role_count,
            hidden_size,
        )
        self.activation_proj = nn.Linear(
            1,
            hidden_size,
        )
        self.relation_embedding = nn.Embedding(
            relation_count,
            hidden_size,
        )

        self.blocks = nn.ModuleList(
            [
                GraphAttentionBlock(
                    hidden_size,
                    heads,
                    dropout=0.05,
                )
                for _ in range(layers)
            ]
        )

        self.norm = nn.LayerNorm(
            hidden_size
        )

        self.action_head = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_size,
                len(ACTIONS),
            ),
        )

        self.relation_head = nn.Sequential(
            nn.Linear(
                hidden_size * 2,
                hidden_size,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_size,
                relation_count,
            ),
        )

        self.next_state_head = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size * 2,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_size * 2,
                hidden_size,
            ),
        )

        self.node_transition_head = nn.Sequential(
            nn.Linear(
                hidden_size * 2,
                hidden_size,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_size,
                1,
            ),
        )

    def _encode(
        self,
        state,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_ids = torch.tensor(
            [
                concept_id(
                    node.concept,
                    self.vocab_size,
                )
                for node in state.nodes
            ],
            dtype=torch.long,
            device=device,
        )

        roles = torch.tensor(
            [
                node.role
                for node in state.nodes
            ],
            dtype=torch.long,
            device=device,
        )

        activations = torch.tensor(
            [
                node.activation
                for node in state.nodes
            ],
            dtype=torch.float32,
            device=device,
        )

        x = (
            self.node_embedding(
                node_ids
            )
            + self.role_embedding(
                roles
            )
            + self.activation_proj(
                activations.unsqueeze(-1)
            )
        )

        if state.edges:
            edge_index = torch.tensor(
                [
                    [edge.source for edge in state.edges],
                    [edge.target for edge in state.edges],
                ],
                dtype=torch.long,
                device=device,
            )

            relation_ids = torch.tensor(
                [
                    edge.relation_id
                    for edge in state.edges
                ],
                dtype=torch.long,
                device=device,
            )

            edge_weights = torch.tensor(
                [
                    edge.activation
                    for edge in state.edges
                ],
                dtype=torch.float32,
                device=device,
            )

            edge_state = (
                self.relation_embedding(
                    relation_ids
                )
                + edge_weights.unsqueeze(-1)
            )
        else:
            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
                device=device,
            )
            edge_state = torch.empty(
                (0, self.hidden_size),
                dtype=x.dtype,
                device=device,
            )

        for block in self.blocks:
            x = block(
                x,
                edge_index,
                edge_state,
            )

        x = self.norm(x)
        graph_state = x.mean(
            dim=0,
            keepdim=True,
        )

        return x, graph_state

    def forward_state(
        self,
        state,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        node_state, graph_state = self._encode(
            state,
            device,
        )

        return {
            "node_state": node_state,
            "graph_state": graph_state,
        }

    def predict_action(
        self,
        graph_state: torch.Tensor,
    ) -> torch.Tensor:
        return self.action_head(
            graph_state
        )

    def predict_relation(
        self,
        source_state: torch.Tensor,
        target_state: torch.Tensor,
    ) -> torch.Tensor:
        return self.relation_head(
            torch.cat(
                [
                    source_state,
                    target_state,
                ],
                dim=-1,
            )
        )

    def predict_next_latent(
        self,
        graph_state: torch.Tensor,
    ) -> torch.Tensor:
        return self.next_state_head(
            graph_state
        )

    def predict_node_transition(
        self,
        current_node_state: torch.Tensor,
        graph_state: torch.Tensor,
    ) -> torch.Tensor:
        graph_repeat = graph_state.expand(
            current_node_state.shape[0],
            -1,
        )

        return self.node_transition_head(
            torch.cat(
                [
                    current_node_state,
                    graph_repeat,
                ],
                dim=-1,
            )
        ).squeeze(-1)
