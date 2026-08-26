from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
from torch import nn

_HERE = Path(__file__).resolve().parent
_RESEARCH_ROOT = _HERE.parent
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

from v200_graph_transformer_cognitive.graph_transformer import GraphAttentionBlock
from v200_graph_transformer_cognitive.long_term_memory import RELATION_TO_ID
from state import ACTIONS

def stable_id(text: str, size: int) -> int:
    return int.from_bytes(
        hashlib.blake2b(text.encode(), digest_size=8).digest(), "little"
    ) % size

class HardAttentionController(nn.Module):
    """
    V214:
      graph -> attention logits -> hard/straight-through mask
            -> ONLY selected node states -> action/pointers/relation

    The full graph is deliberately NOT available to downstream heads.
    """

    def __init__(
        self,
        *,
        vocab_size=50000,
        relation_count=None,
        hidden_size=128,
        heads=4,
        layers=3,
        role_count=8,
        attention_temperature=0.75,
        topk=5,
    ):
        super().__init__()
        relation_count = relation_count or len(RELATION_TO_ID)
        self.vocab_size = vocab_size
        self.relation_count = relation_count
        self.hidden_size = hidden_size
        self.attention_temperature = attention_temperature
        self.topk = topk

        self.node_embedding = nn.Embedding(vocab_size, hidden_size)
        self.role_embedding = nn.Embedding(role_count, hidden_size)
        self.activation_proj = nn.Linear(1, hidden_size)
        self.relation_embedding = nn.Embedding(relation_count, hidden_size)

        self.blocks = nn.ModuleList([
            GraphAttentionBlock(hidden_size, heads, dropout=0.05)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)

        self.goal_relation_embedding = nn.Embedding(relation_count, hidden_size)
        self.goal_depth_embedding = nn.Embedding(8, hidden_size)
        self.goal_encoder = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.attention_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

        # Downstream heads ONLY see attended_graph/node_state.
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, len(ACTIONS)),
        )
        self.source_head = nn.Linear(hidden_size * 2, 1)
        self.target_head = nn.Linear(hidden_size * 2, 1)

        self.relation_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, relation_count),
        )

    def encode_state(self, state, device):
        ids = torch.tensor(
            [stable_id(n.concept, self.vocab_size) for n in state.nodes],
            dtype=torch.long,
            device=device,
        )
        roles = torch.tensor(
            [min(max(n.role, 0), self.role_embedding.num_embeddings - 1) for n in state.nodes],
            dtype=torch.long,
            device=device,
        )
        acts = torch.tensor(
            [n.activation for n in state.nodes],
            dtype=torch.float32,
            device=device,
        )

        x = (
            self.node_embedding(ids)
            + self.role_embedding(roles)
            + self.activation_proj(acts.unsqueeze(-1))
        )

        if state.edges:
            index = {n.concept: i for i, n in enumerate(state.nodes)}
            edge_index = torch.tensor(
                [
                    [index[e.source] for e in state.edges],
                    [index[e.target] for e in state.edges],
                ],
                dtype=torch.long,
                device=device,
            )
            rel_ids = torch.tensor(
                [RELATION_TO_ID.get(e.relation, 0) for e in state.edges],
                dtype=torch.long,
                device=device,
            )
            edge_state = self.relation_embedding(rel_ids)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_state = torch.empty(
                (0, self.hidden_size),
                dtype=x.dtype,
                device=device,
            )

        for block in self.blocks:
            x = block(x, edge_index, edge_state)

        return self.norm(x)

    def encode_goal(self, goal, device):
        def node_emb(name):
            if not name:
                return torch.zeros((1, self.hidden_size), device=device)
            return self.node_embedding(
                torch.tensor([stable_id(name, self.vocab_size)], device=device)
            )

        source = node_emb(goal.get("source"))
        target = node_emb(goal.get("target"))
        relation = self.goal_relation_embedding(
            torch.tensor(
                [RELATION_TO_ID.get(goal.get("relation"), 0)],
                device=device,
            )
        )
        depth = self.goal_depth_embedding(
            torch.tensor(
                [min(int(goal.get("depth", 1)), 7)],
                device=device,
            )
        )

        return self.goal_encoder(
            torch.cat([source, target, relation, depth], dim=-1)
        )

    def hard_mask(self, logits):
        # Straight-through top-k selection.
        k = min(self.topk, logits.numel())
        soft = torch.sigmoid(logits / self.attention_temperature)

        _, top_idx = torch.topk(logits, k=k)
        hard = torch.zeros_like(logits)
        hard[top_idx] = 1.0

        # Forward pass is hard; backward pass uses soft attention.
        mask = hard + soft - soft.detach()
        return mask, soft, hard

    def forward(self, state, goal, device=None):
        device = device or next(self.parameters()).device

        node_state = self.encode_state(state, device)
        goal_state = self.encode_goal(goal, device)
        goal_expanded = goal_state.expand(node_state.shape[0], -1)

        attention_logits = self.attention_head(
            torch.cat([node_state, goal_expanded], dim=-1)
        ).squeeze(-1)

        attention_mask, attention_soft, attention_hard = self.hard_mask(
            attention_logits
        )

        # THIS is the critical V214 change.
        # Everything downstream receives masked node representations.
        masked_nodes = node_state * attention_mask.unsqueeze(-1)

        denom = attention_mask.sum().clamp_min(1e-5)
        attended_graph = (
            masked_nodes.sum(dim=0, keepdim=True) / denom
        )

        combined = torch.cat(
            [attended_graph, goal_state],
            dim=-1,
        )

        action_logits = self.action_head(combined)

        pointer_input = torch.cat(
            [masked_nodes, goal_expanded],
            dim=-1,
        )
        source_logits = self.source_head(pointer_input).squeeze(-1)
        target_logits = self.target_head(pointer_input).squeeze(-1)

        src = source_logits.argmax()
        tgt = target_logits.argmax()

        relation_logits = self.relation_head(
            torch.cat(
                [
                    masked_nodes[src:src + 1],
                    masked_nodes[tgt:tgt + 1],
                    goal_state,
                ],
                dim=-1,
            )
        ).squeeze(0)

        return {
            "node_state": node_state,
            "masked_nodes": masked_nodes,
            "goal_state": goal_state,
            "attention_logits": attention_logits,
            "attention_soft": attention_soft,
            "attention_hard": attention_hard,
            "attention_mask": attention_mask,
            "attended_graph": attended_graph,
            "action_logits": action_logits.squeeze(0),
            "source_logits": source_logits,
            "target_logits": target_logits,
            "relation_logits": relation_logits,
        }
