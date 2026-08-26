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


def stable_vocab_id(
    concept: str,
    vocab_size: int,
) -> int:
    digest = hashlib.blake2b(
        concept.encode("utf-8"),
        digest_size=8,
    ).digest()
    value = int.from_bytes(
        digest,
        "little",
        signed=False,
    )
    return value % vocab_size


class CognitiveGraphEncoder(nn.Module):
    """
    CUDA-safe graph-transformer encoder.

    Compared with the first V201 version:
      - uses a deterministic vocabulary hash
      - allocates relation embeddings for the full relation vocabulary
      - validates every index before CUDA indexing
      - correctly accepts the masked graph state
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        relation_count: int | None = None,
        hidden_size: int = 128,
        heads: int = 4,
        layers: int = 3,
        roles: int = 8,
    ) -> None:
        super().__init__()

        if relation_count is None:
            relation_count = len(RELATION_TO_ID)

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.relation_count = relation_count

        self.node_embedding = nn.Embedding(
            vocab_size,
            hidden_size,
        )
        self.role_embedding = nn.Embedding(
            roles,
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
                    dropout=0.05,
                )
                for _ in range(layers)
            ]
        )

        self.graph_norm = nn.LayerNorm(
            hidden_size
        )

        self.next_state_head = nn.Sequential(
            nn.Linear(
                hidden_size * 2,
                hidden_size,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
        )

        self.relation_head = nn.Linear(
            hidden_size * 2,
            relation_count,
        )

        self.binding_head = nn.Linear(
            hidden_size * 2,
            1,
        )

        self.node_reconstruction_head = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_size,
                vocab_size,
            ),
        )

    def concept_ids(
        self,
        concepts: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        ids = [
            stable_vocab_id(
                concept,
                self.vocab_size,
            )
            for concept in concepts
        ]

        result = torch.tensor(
            ids,
            dtype=torch.long,
            device=device,
        )

        self._validate_node_ids(
            result
        )

        return result

    def _validate_node_ids(
        self,
        node_ids: torch.Tensor,
    ) -> None:
        if node_ids.numel() == 0:
            return

        min_id = int(
            node_ids.min().item()
        )
        max_id = int(
            node_ids.max().item()
        )

        if min_id < 0 or max_id >= self.vocab_size:
            raise RuntimeError(
                "Invalid node embedding index: "
                f"min={min_id} max={max_id} "
                f"vocab_size={self.vocab_size}"
            )

    def _validate_roles(
        self,
        roles: torch.Tensor,
    ) -> None:
        if roles.numel() == 0:
            return

        min_role = int(
            roles.min().item()
        )
        max_role = int(
            roles.max().item()
        )
        limit = self.role_embedding.num_embeddings

        if min_role < 0 or max_role >= limit:
            raise RuntimeError(
                "Invalid node role index: "
                f"min={min_role} max={max_role} "
                f"role_count={limit}"
            )

    def _validate_edges(
        self,
        edge_index: torch.Tensor,
        edge_relation_ids: torch.Tensor,
        node_count: int,
    ) -> None:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise RuntimeError(
                "edge_index must have shape [2, E], "
                f"got {tuple(edge_index.shape)}"
            )

        edge_count = edge_index.shape[1]

        if edge_relation_ids.numel() != edge_count:
            raise RuntimeError(
                "Relation/index length mismatch: "
                f"edges={edge_count} "
                f"relations={edge_relation_ids.numel()}"
            )

        if edge_count:
            min_index = int(
                edge_index.min().item()
            )
            max_index = int(
                edge_index.max().item()
            )
            if (
                min_index < 0
                or max_index >= node_count
            ):
                raise RuntimeError(
                    "Invalid graph edge index: "
                    f"min={min_index} max={max_index} "
                    f"node_count={node_count}"
                )

        if edge_relation_ids.numel():
            min_relation = int(
                edge_relation_ids.min().item()
            )
            max_relation = int(
                edge_relation_ids.max().item()
            )
            if (
                min_relation < 0
                or max_relation >= self.relation_count
            ):
                raise RuntimeError(
                    "Invalid relation embedding index: "
                    f"min={min_relation} max={max_relation} "
                    f"relation_count={self.relation_count}"
                )

    def encode_state(
        self,
        state,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        node_ids = self.concept_ids(
            state.node_concepts,
            device,
        )

        node_roles = torch.tensor(
            state.node_roles,
            dtype=torch.long,
            device=device,
        )

        node_activations = torch.tensor(
            state.node_activations,
            dtype=torch.float32,
            device=device,
        )

        self._validate_roles(
            node_roles
        )

        if node_ids.numel() != node_roles.numel():
            raise RuntimeError(
                "Node id / role length mismatch: "
                f"ids={node_ids.numel()} "
                f"roles={node_roles.numel()}"
            )

        if node_ids.numel() != node_activations.numel():
            raise RuntimeError(
                "Node id / activation length mismatch: "
                f"ids={node_ids.numel()} "
                f"activations={node_activations.numel()}"
            )

        if state.edges:
            edge_index = torch.tensor(
                [
                    [edge[0] for edge in state.edges],
                    [edge[1] for edge in state.edges],
                ],
                dtype=torch.long,
                device=device,
            )

            edge_relation_ids = torch.tensor(
                [
                    edge[2]
                    for edge in state.edges
                ],
                dtype=torch.long,
                device=device,
            )

            edge_weights = torch.tensor(
                [
                    edge[3]
                    for edge in state.edges
                ],
                dtype=torch.float32,
                device=device,
            ).log1p()
        else:
            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
                device=device,
            )
            edge_relation_ids = torch.empty(
                (0,),
                dtype=torch.long,
                device=device,
            )
            edge_weights = torch.empty(
                (0,),
                dtype=torch.float32,
                device=device,
            )

        # Validate BEFORE any downstream CUDA indexing.
        self._validate_edges(
            edge_index,
            edge_relation_ids,
            node_ids.numel(),
        )

        x = (
            self.node_embedding(node_ids)
            + self.role_embedding(node_roles)
            + self.activation_projection(
                node_activations.unsqueeze(-1)
            )
        )

        if edge_relation_ids.numel():
            edge_state = (
                self.relation_embedding(
                    edge_relation_ids
                )
                + edge_weights.unsqueeze(-1)
            )
        else:
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

        x = self.graph_norm(x)

        graph_state = x.mean(
            dim=0,
            keepdim=True,
        )

        return {
            "node_state": x,
            "graph_state": graph_state,
            "node_ids": node_ids,
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

    def predict_binding(
        self,
        source_state: torch.Tensor,
        target_state: torch.Tensor,
    ) -> torch.Tensor:
        return self.binding_head(
            torch.cat(
                [
                    source_state,
                    target_state,
                ],
                dim=-1,
            )
        ).squeeze(-1)

    def predict_next_state(
        self,
        current_graph_state: torch.Tensor,
        target_graph_state: torch.Tensor,
    ) -> torch.Tensor:
        return self.next_state_head(
            torch.cat(
                [
                    current_graph_state,
                    target_graph_state,
                ],
                dim=-1,
            )
        )

    def reconstruct_node(
        self,
        node_state: torch.Tensor,
    ) -> torch.Tensor:
        return self.node_reconstruction_head(
            node_state
        )
