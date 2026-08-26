from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class GraphBatch:
    node_ids: torch.Tensor
    node_roles: torch.Tensor
    node_activations: torch.Tensor
    edge_index: torch.Tensor
    edge_relation_ids: torch.Tensor
    edge_weights: torch.Tensor
    edge_target_relation: torch.Tensor


class GraphAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads: int,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        if hidden_size % heads != 0:
            raise ValueError(
                "hidden_size must be divisible by heads"
            )

        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = hidden_size // heads

        self.q = nn.Linear(
            hidden_size,
            hidden_size,
        )
        self.k = nn.Linear(
            hidden_size,
            hidden_size,
        )
        self.v = nn.Linear(
            hidden_size,
            hidden_size,
        )
        self.out = nn.Linear(
            hidden_size,
            hidden_size,
        )

        self.edge_bias = nn.Linear(
            hidden_size,
            heads,
            bias=False,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.norm1 = nn.LayerNorm(
            hidden_size
        )
        self.norm2 = nn.LayerNorm(
            hidden_size
        )

        self.ff = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size * 4,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_size * 4,
                hidden_size,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: [N, H]
        edge_index: [2, E]
        edge_state: [E, H]
        """
        residual = x
        x_norm = self.norm1(x)

        q = self.q(
            x_norm
        ).view(
            -1,
            self.heads,
            self.head_dim,
        )
        k = self.k(
            x_norm
        ).view(
            -1,
            self.heads,
            self.head_dim,
        )
        v = self.v(
            x_norm
        ).view(
            -1,
            self.heads,
            self.head_dim,
        )

        if edge_index.numel() == 0:
            return (
                residual
                + self.dropout(
                    self.ff(
                        self.norm2(
                            residual
                        )
                    )
                )
            )

        source = edge_index[0]
        target = edge_index[1]

        scores = (
            (
                q[source]
                * k[target]
            ).sum(
                dim=-1
            )
            / (
                self.head_dim
                ** 0.5
            )
        )

        scores = (
            scores
            + self.edge_bias(
                edge_state
            )
        )

        # Edge-list softmax: normalize over outgoing edges per source node.
        attention = torch.zeros_like(
            scores
        )

        for node_id in torch.unique(
            source
        ):
            mask = (
                source == node_id
            )
            attention[mask] = torch.softmax(
                scores[mask],
                dim=0,
            )

        messages = (
            attention.unsqueeze(-1)
            * v[target]
        )

        aggregated = torch.zeros_like(
            x
        )

        aggregated.index_add_(
            0,
            source,
            messages.reshape(
                -1,
                self.hidden_size,
            ),
        )

        x = residual + self.dropout(
            self.out(
                aggregated
            )
        )

        x = x + self.dropout(
            self.ff(
                self.norm2(x)
            )
        )

        return x


class GraphTransformer(nn.Module):
    """
    Small graph-transformer encoder.

    Nothing in this module knows that IsA means category, that RelatedTo means
    association, or that a node is a "dog". Those semantics are learned from
    the graph-derived training signal.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        relation_count: int,
        role_count: int = 16,
        hidden_size: int = 128,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.relation_count = relation_count

        self.node_embedding = nn.Embedding(
            vocab_size,
            hidden_size,
        )
        self.role_embedding = nn.Embedding(
            role_count,
            hidden_size,
        )
        self.activation_projection = nn.Linear(
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
                    dropout,
                )
                for _ in range(layers)
            ]
        )

        self.pool = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.GELU(),
        )

        self.relation_head = nn.Linear(
            hidden_size * 2,
            relation_count,
        )

        self.edge_score = nn.Sequential(
            nn.Linear(
                hidden_size * 3,
                hidden_size,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_size,
                1,
            ),
        )

    def encode_nodes(
        self,
        node_ids: torch.Tensor,
        node_roles: torch.Tensor,
        node_activations: torch.Tensor,
    ) -> torch.Tensor:
        x = (
            self.node_embedding(
                node_ids
            )
            + self.role_embedding(
                node_roles
            )
            + self.activation_projection(
                node_activations.unsqueeze(-1)
            )
        )
        return x

    def encode_edges(
        self,
        relation_ids: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> torch.Tensor:
        relation = self.relation_embedding(
            relation_ids
        )
        return (
            relation
            + edge_weights.unsqueeze(-1)
        )

    def forward(
        self,
        batch: GraphBatch,
    ) -> dict[str, torch.Tensor]:
        x = self.encode_nodes(
            batch.node_ids,
            batch.node_roles,
            batch.node_activations,
        )

        edge_state = self.encode_edges(
            batch.edge_relation_ids,
            batch.edge_weights,
        )

        for block in self.blocks:
            x = block(
                x,
                batch.edge_index,
                edge_state,
            )

        # Mean pooling is deliberately simple. Working-memory structure is
        # represented by the graph attention updates, not by a hand-designed
        # semantic summary.
        graph_state = self.pool(
            x.mean(
                dim=0,
                keepdim=True,
            )
        )

        return {
            "node_state": x,
            "graph_state": graph_state,
        }

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

    def score_edges(
        self,
        source_state: torch.Tensor,
        target_state: torch.Tensor,
        edge_state: torch.Tensor,
    ) -> torch.Tensor:
        graph_state = torch.cat(
            [
                source_state,
                target_state,
                edge_state,
            ],
            dim=-1,
        )
        return self.edge_score(
            graph_state
        ).squeeze(-1)


class DesignerHead(nn.Module):
    ACTIONS = (
        "REUSE",
        "CREATE",
        "BRANCH",
        "INHIBIT",
        "BIND",
        "COMMIT",
    )

    def __init__(
        self,
        hidden_size: int,
        action_count: int | None = None,
    ) -> None:
        super().__init__()

        if action_count is None:
            action_count = len(
                self.ACTIONS
            )

        self.net = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_size,
                action_count,
            ),
        )

    def forward(
        self,
        graph_state: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(
            graph_state
        )

    @classmethod
    def action_names(cls) -> tuple[str, ...]:
        return cls.ACTIONS
