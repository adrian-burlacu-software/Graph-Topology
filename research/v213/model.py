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
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little") % size

class TeacherDistilledController(nn.Module):
    def __init__(self, *, vocab_size=50000, relation_count=None, hidden_size=128, heads=4, layers=3, role_count=8):
        super().__init__()
        relation_count = relation_count or len(RELATION_TO_ID)
        self.vocab_size = vocab_size
        self.relation_count = relation_count
        self.hidden_size = hidden_size

        self.node_embedding = nn.Embedding(vocab_size, hidden_size)
        self.role_embedding = nn.Embedding(role_count, hidden_size)
        self.activation_proj = nn.Linear(1, hidden_size)
        self.relation_embedding = nn.Embedding(relation_count, hidden_size)

        self.blocks = nn.ModuleList([GraphAttentionBlock(hidden_size, heads, dropout=0.05) for _ in range(layers)])
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
        self.next_state_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.transition_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def encode_state(self, state, device):
        ids = torch.tensor([stable_id(n.concept, self.vocab_size) for n in state.nodes], device=device)
        roles = torch.tensor([min(max(n.role, 0), self.role_embedding.num_embeddings-1) for n in state.nodes], device=device)
        acts = torch.tensor([n.activation for n in state.nodes], dtype=torch.float32, device=device)
        x = self.node_embedding(ids) + self.role_embedding(roles) + self.activation_proj(acts.unsqueeze(-1))

        if state.edges:
            index_by_name = {n.concept: i for i, n in enumerate(state.nodes)}
            edge_index = torch.tensor([
                [index_by_name[e.source] for e in state.edges],
                [index_by_name[e.target] for e in state.edges],
            ], dtype=torch.long, device=device)
            rel_ids = torch.tensor([RELATION_TO_ID.get(e.relation, 0) for e in state.edges], dtype=torch.long, device=device)
            edge_state = self.relation_embedding(rel_ids) + torch.tensor([e.activation for e in state.edges], dtype=torch.float32, device=device).unsqueeze(-1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_state = torch.empty((0, self.hidden_size), dtype=x.dtype, device=device)

        for block in self.blocks:
            x = block(x, edge_index, edge_state)
        x = self.norm(x)
        return {"node_state": x, "graph_state": x.mean(0, keepdim=True)}

    def encode_goal(self, state, goal, device):
        def emb(name):
            if not name:
                return torch.zeros((1, self.hidden_size), device=device)
            return self.node_embedding(torch.tensor([stable_id(name, self.vocab_size)], device=device))
        source = emb(goal.get("source"))
        target = emb(goal.get("target"))
        rel = self.goal_relation_embedding(torch.tensor([RELATION_TO_ID.get(goal.get("relation"), 0)], device=device))
        depth = self.goal_depth_embedding(torch.tensor([min(int(goal.get("depth", 1)), 7)], device=device))
        return self.goal_encoder(torch.cat([source, target, rel, depth], dim=-1))

    def forward(self, state, goal, device=None):
        device = device or next(self.parameters()).device
        enc = self.encode_state(state, device)
        goal_state = self.encode_goal(state, goal, device)
        node_state = enc["node_state"]
        goal_expanded = goal_state.expand(node_state.shape[0], -1)

        att_logits = self.attention_head(torch.cat([node_state, goal_expanded], dim=-1)).squeeze(-1)
        att_weights = torch.sigmoid(att_logits)
        denom = att_weights.sum().clamp_min(1e-5)
        attended = (node_state * att_weights.unsqueeze(-1)).sum(0, keepdim=True) / denom

        combined = torch.cat([attended, goal_state], dim=-1)
        action_logits = self.action_head(combined)

        pointer_input = torch.cat([node_state, goal_expanded], dim=-1)
        source_logits = self.source_head(pointer_input).squeeze(-1)
        target_logits = self.target_head(pointer_input).squeeze(-1)

        src_idx = source_logits.argmax()
        tgt_idx = target_logits.argmax()
        relation_logits = self.relation_head(torch.cat([node_state[src_idx:src_idx+1], node_state[tgt_idx:tgt_idx+1], goal_state], dim=-1)).squeeze(0)
        next_latent = self.next_state_head(combined)
        transition = self.transition_head(torch.cat([node_state, enc["graph_state"].expand(node_state.shape[0], -1)], dim=-1)).squeeze(-1)

        return {
            "node_state": node_state,
            "graph_state": enc["graph_state"],
            "goal_state": goal_state,
            "attention_logits": att_logits,
            "attention_weights": att_weights,
            "attended_graph": attended,
            "action_logits": action_logits.squeeze(0),
            "source_logits": source_logits,
            "target_logits": target_logits,
            "relation_logits": relation_logits,
            "next_state": next_latent,
            "transition_logits": transition,
        }

    # Compatibility helpers retained for V209-style callers.
    def predict_action(self, graph_state, goal_state):
        return self.action_head(torch.cat([graph_state, goal_state], dim=-1))

    def predict_pointers(self, node_state, goal_state):
        g = goal_state.expand(node_state.shape[0], -1)
        x = torch.cat([node_state, g], dim=-1)
        return self.source_head(x).squeeze(-1), self.target_head(x).squeeze(-1)

    def predict_relation(self, source_state, target_state, goal_state):
        return self.relation_head(torch.cat([source_state, target_state, goal_state], dim=-1))

    def predict_next_latent(self, graph_state, goal_state):
        return self.next_state_head(torch.cat([graph_state, goal_state], dim=-1))

    def predict_transition(self, node_state, graph_state):
        g = graph_state.expand(node_state.shape[0], -1)
        return self.transition_head(torch.cat([node_state, g], dim=-1)).squeeze(-1)
