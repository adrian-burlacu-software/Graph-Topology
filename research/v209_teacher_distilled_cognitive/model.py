from __future__ import annotations

import hashlib

import torch
from torch import nn

# Allow direct execution from research/ without package installation.
import sys
_RESEARCH_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

from v200_graph_transformer_cognitive.graph_transformer import (
    GraphAttentionBlock,
)
from v200_graph_transformer_cognitive.long_term_memory import (
    RELATION_TO_ID,
)

try:
    from .state import ACTIONS
except ImportError:
    import sys
    _HERE = __import__("pathlib").Path(__file__).resolve().parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    from state import ACTIONS



def stable_id(
    text: str,
    size: int,
) -> int:
    digest = hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=8,
    ).digest()

    return int.from_bytes(
        digest,
        "little",
        signed=False,
    ) % size


class TeacherDistilledController(nn.Module):
    """
    Goal-conditioned graph controller.

    The V203 graph encoder is retained, but V209 adds explicit goal
    conditioning so the controller is not forced to infer the task solely from
    the current working graph.
    """

    def __init__(
        self,
        *,
        vocab_size: int = 50000,
        relation_count: int | None = None,
        hidden_size: int = 128,
        heads: int = 4,
        layers: int = 3,
        role_count: int = 8,
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

        self.goal_relation_embedding = nn.Embedding(
            relation_count,
            hidden_size,
        )
        self.goal_role_embedding = nn.Embedding(
            4,
            hidden_size,
        )

        self.goal_encoder = nn.Sequential(
            nn.Linear(
                hidden_size * 3,
                hidden_size,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
        )

        self.action_head = nn.Sequential(
            nn.Linear(
                hidden_size * 2,
                hidden_size,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_size,
                len(ACTIONS),
            ),
        )

        self.source_head = nn.Linear(
            hidden_size * 2,
            1,
        )

        self.target_head = nn.Linear(
            hidden_size * 2,
            1,
        )

        self.relation_head = nn.Sequential(
            nn.Linear(
                hidden_size * 3,
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
                hidden_size * 2,
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

    def concept_embedding(
        self,
        concept: str,
        device: torch.device,
    ) -> torch.Tensor:
        index = torch.tensor(
            [
                stable_id(
                    concept,
                    self.vocab_size,
                )
            ],
            dtype=torch.long,
            device=device,
        )

        return self.node_embedding(
            index
        )

    def encode_state(
        self,
        state,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        node_ids = torch.tensor(
            [
                stable_id(
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
                min(
                    max(
                        node.role,
                        0,
                    ),
                    self.role_embedding.num_embeddings - 1,
                )
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
            self.node_embedding(node_ids)
            + self.role_embedding(roles)
            + self.activation_proj(
                activations.unsqueeze(-1)
            )
        )

        if state.edges:
            edge_index = torch.tensor(
                [
                    [
                        next(
                            i
                            for i, n in enumerate(
                                state.nodes
                            )
                            if n.concept == edge.source
                        )
                        for edge in state.edges
                    ],
                    [
                        next(
                            i
                            for i, n in enumerate(
                                state.nodes
                            )
                            if n.concept == edge.target
                        )
                        for edge in state.edges
                    ],
                ],
                dtype=torch.long,
                device=device,
            )

            relation_ids = torch.tensor(
                [
                    RELATION_TO_ID.get(
                        edge.relation,
                        0,
                    )
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

        return {
            "node_state": x,
            "graph_state": x.mean(
                dim=0,
                keepdim=True,
            ),
        }

    def encode_goal(
        self,
        state,
        goal_source_index: int,
        goal_target_index: int,
        goal_relation: str,
        device: torch.device,
    ) -> torch.Tensor:
        if (
            0 <= goal_source_index
            < len(state.nodes)
        ):
            source = self.concept_embedding(
                state.nodes[
                    goal_source_index
                ].concept,
                device,
            )
        else:
            source = torch.zeros(
                (1, self.hidden_size),
                device=device,
            )

        if (
            0 <= goal_target_index
            < len(state.nodes)
        ):
            target = self.concept_embedding(
                state.nodes[
                    goal_target_index
                ].concept,
                device,
            )
        else:
            target = torch.zeros(
                (1, self.hidden_size),
                device=device,
            )

        relation_id = RELATION_TO_ID.get(
            goal_relation,
            0,
        )

        relation = self.goal_relation_embedding(
            torch.tensor(
                [relation_id],
                dtype=torch.long,
                device=device,
            )
        )

        return self.goal_encoder(
            torch.cat(
                [
                    source,
                    target,
                    relation,
                ],
                dim=-1,
            )
        )

    def combined_graph_goal(
        self,
        graph_state: torch.Tensor,
        goal_state: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                graph_state,
                goal_state,
            ],
            dim=-1,
        )

    def predict_action(
        self,
        graph_state: torch.Tensor,
        goal_state: torch.Tensor,
    ) -> torch.Tensor:
        return self.action_head(
            self.combined_graph_goal(
                graph_state,
                goal_state,
            )
        )

    def predict_pointers(
        self,
        node_state: torch.Tensor,
        goal_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        goal = goal_state.expand(
            node_state.shape[0],
            -1,
        )

        combined = torch.cat(
            [
                node_state,
                goal,
            ],
            dim=-1,
        )

        return (
            self.source_head(
                combined
            ).squeeze(-1),
            self.target_head(
                combined
            ).squeeze(-1),
        )

    def predict_relation(
        self,
        source_state: torch.Tensor,
        target_state: torch.Tensor,
        goal_state: torch.Tensor,
    ) -> torch.Tensor:
        return self.relation_head(
            torch.cat(
                [
                    source_state,
                    target_state,
                    goal_state,
                ],
                dim=-1,
            )
        )

    def predict_next_latent(
        self,
        graph_state: torch.Tensor,
        goal_state: torch.Tensor,
    ) -> torch.Tensor:
        return self.next_state_head(
            torch.cat(
                [
                    graph_state,
                    goal_state,
                ],
                dim=-1,
            )
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
