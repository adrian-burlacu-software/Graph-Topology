from __future__ import annotations

import hashlib

import torch
from torch import nn

from v200_graph_transformer_cognitive.graph_transformer import (
    GraphAttentionBlock,
)
from v200_graph_transformer_cognitive.long_term_memory import (
    RELATION_TO_ID,
)
from .graph_state import ACTIONS


def concept_id(
    concept: str,
    vocab_size: int,
) -> int:
    digest = hashlib.blake2b(
        concept.encode("utf-8"),
        digest_size=8,
    ).digest()

    return int.from_bytes(
        digest,
        "little",
        signed=False,
    ) % vocab_size


class MultiActionController(nn.Module):
    """
    Graph transformer + real multi-action controller.

    Outputs:
        action
        source pointer
        target pointer
        relation module
        next-state latent
        node activation transition
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

        self.source_head = nn.Linear(
            hidden_size,
            1,
        )

        self.target_head = nn.Linear(
            hidden_size,
            1,
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

        self.transition_head = nn.Sequential(
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

        self.value_head = nn.Sequential(
            nn.Linear(
                hidden_size,
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
                    [e.source for e in state.edges],
                    [e.target for e in state.edges],
                ],
                dtype=torch.long,
                device=device,
            )

            relation_ids = torch.tensor(
                [
                    e.relation_id
                    for e in state.edges
                ],
                dtype=torch.long,
                device=device,
            )

            edge_weights = torch.tensor(
                [
                    e.activation
                    for e in state.edges
                ],
                dtype=torch.float32,
                device=device,
            )

            if (
                edge_index.numel()
                and int(
                    edge_index.max().item()
                )
                >= len(state.nodes)
            ):
                raise RuntimeError(
                    "Invalid graph edge index."
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

    def encode(
        self,
        state,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        nodes, graph = self._encode(
            state,
            device,
        )
        return {
            "node_state": nodes,
            "graph_state": graph,
        }

    def predict_action(
        self,
        graph_state: torch.Tensor,
    ) -> torch.Tensor:
        return self.action_head(
            graph_state
        )

    def predict_pointers(
        self,
        node_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.source_head(
                node_state
            ).squeeze(-1),
            self.target_head(
                node_state
            ).squeeze(-1),
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

    def predict_transition(
        self,
        node_state: torch.Tensor,
        graph_state: torch.Tensor,
    ) -> torch.Tensor:
        graph = graph_state.expand(
            node_state.shape[0],
            -1,
        )
        return self.transition_head(
            torch.cat(
                [
                    node_state,
                    graph,
                ],
                dim=-1,
            )
        ).squeeze(-1)

    def predict_value(
        self,
        graph_state: torch.Tensor,
    ) -> torch.Tensor:
        return self.value_head(
            graph_state
        ).squeeze(-1)
