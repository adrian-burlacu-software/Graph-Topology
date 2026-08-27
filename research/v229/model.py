
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from v200_graph_transformer_cognitive.graph_transformer import GraphAttentionBlock
from v200_graph_transformer_cognitive.long_term_memory import RELATION_TO_ID
from state import ACTIONS


def stable_id(text: str, size: int) -> int:
    return int.from_bytes(
        hashlib.blake2b(text.encode(), digest_size=8).digest(), "little"
    ) % size


class StateArchitectureModel(nn.Module):
    def __init__(
        self,
        vocab_size=50000,
        relation_count=None,
        hidden_size=128,
        heads=4,
        depth=8,
        topk=5,
        state_mode="latent",
    ):
        super().__init__()
        relation_count = relation_count or len(RELATION_TO_ID)
        self.vocab_size = vocab_size
        self.relation_count = relation_count
        self.hidden_size = hidden_size
        self.depth = depth
        self.topk = topk
        self.state_mode = state_mode

        self.node_embedding = nn.Embedding(vocab_size, hidden_size)
        self.role_embedding = nn.Embedding(8, hidden_size)
        self.activation_projection = nn.Linear(1, hidden_size)
        self.relation_embedding = nn.Embedding(relation_count, hidden_size)

        self.blocks = nn.ModuleList([
            GraphAttentionBlock(hidden_size, heads, dropout=0.05)
            for _ in range(depth)
        ])

        self.goal_relation = nn.Embedding(relation_count, hidden_size)
        self.goal_depth = nn.Embedding(8, hidden_size)
        self.goal_encoder = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.attention_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, len(ACTIONS)),
        )
        self.source_head = nn.Linear(hidden_size * 3, 1)
        self.target_head = nn.Linear(hidden_size * 3, 1)
        self.relation_head = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, relation_count),
        )

        self.state_update = nn.GRUCell(hidden_size * 2, hidden_size)
        self.action_embedding = nn.Embedding(len(ACTIONS), hidden_size)
        self.pointer_embedding = nn.Embedding(vocab_size, hidden_size)
        self.history_update = nn.GRUCell(hidden_size * 4, hidden_size)

    def encode_graph(self, state, device):
        ids = torch.tensor(
            [stable_id(n.concept, self.vocab_size) for n in state.nodes],
            dtype=torch.long, device=device)
        roles = torch.tensor(
            [min(max(n.role, 0), 7) for n in state.nodes],
            dtype=torch.long, device=device)
        activations = torch.tensor(
            [n.activation for n in state.nodes],
            dtype=torch.float32, device=device)

        x = (
            self.node_embedding(ids)
            + self.role_embedding(roles)
            + self.activation_projection(activations[:, None])
        )

        if state.edges:
            index = {n.concept: i for i, n in enumerate(state.nodes)}
            edge_index = torch.tensor(
                [
                    [index[e.source] for e in state.edges],
                    [index[e.target] for e in state.edges],
                ],
                dtype=torch.long, device=device)
            relation_ids = torch.tensor(
                [RELATION_TO_ID.get(e.relation, 0) for e in state.edges],
                dtype=torch.long, device=device)
            edge_state = self.relation_embedding(relation_ids)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_state = torch.empty((0, self.hidden_size),
                                     dtype=x.dtype, device=device)

        for block in self.blocks:
            x = block(x, edge_index, edge_state)

        return nn.functional.layer_norm(x, (self.hidden_size,))

    def encode_goal(self, goal, device):
        def node(name):
            if not name:
                return torch.zeros((1, self.hidden_size), device=device)
            return self.node_embedding(
                torch.tensor([stable_id(name, self.vocab_size)],
                             dtype=torch.long, device=device)
            )

        relation = self.goal_relation(
            torch.tensor([RELATION_TO_ID.get(goal["relation"], 0)],
                         dtype=torch.long, device=device)
        )
        depth = self.goal_depth(
            torch.tensor([min(int(goal["depth"]), 7)],
                         dtype=torch.long, device=device)
        )
        return self.goal_encoder(
            torch.cat(
                [node(goal["source"]), node(goal["target"]), relation, depth],
                dim=-1)
        )

    def cognitive_step(
        self, state, goal, working,
        previous_action_id, previous_source, previous_target,
        previous_relation, device
    ):
        x = self.encode_graph(state, device)
        g = self.encode_goal(goal, device)
        gx = g.expand(x.size(0), -1)
        wx = working.expand(x.size(0), -1)

        attention_logits = self.attention_head(
            torch.cat([x, gx, wx], dim=-1)
        ).squeeze(-1)

        k = min(self.topk, x.size(0))
        _, idx = torch.topk(attention_logits, k=k)
        hard = torch.zeros_like(attention_logits)
        hard[idx] = 1.0
        soft = torch.sigmoid(attention_logits)
        mask = hard + soft - soft.detach()

        masked = x * mask[:, None]
        attended = masked.sum(0, keepdim=True) / mask.sum().clamp_min(1e-5)

        rep = torch.cat([attended, g, working], dim=-1)
        action_logits = self.action_head(rep).squeeze(0)

        pointer_input = torch.cat([masked, gx, wx], dim=-1)
        source_logits = self.source_head(pointer_input).squeeze(-1)
        target_logits = self.target_head(pointer_input).squeeze(-1)

        si = source_logits.argmax()
        ti = target_logits.argmax()

        relation_logits = self.relation_head(
            torch.cat([
                masked[si:si + 1],
                masked[ti:ti + 1],
                g,
                working,
            ], dim=-1)
        ).squeeze(0)

        if self.state_mode == "stateless":
            next_working = torch.zeros_like(working)
        elif self.state_mode == "latent":
            next_working = self.state_update(
                torch.cat([attended, g], dim=-1), working)
        elif self.state_mode == "latent_action":
            aid = 0 if previous_action_id is None else int(previous_action_id)
            action_vec = self.action_embedding(
                torch.tensor([aid], dtype=torch.long, device=device))
            source_vec = (
                self.pointer_embedding(
                    torch.tensor([stable_id(previous_source, self.vocab_size)],
                                 dtype=torch.long, device=device))
                if previous_source else torch.zeros_like(action_vec)
            )
            target_vec = (
                self.pointer_embedding(
                    torch.tensor([stable_id(previous_target, self.vocab_size)],
                                 dtype=torch.long, device=device))
                if previous_target else torch.zeros_like(action_vec)
            )
            relation_vec = self.relation_embedding(
                torch.tensor([0 if previous_relation is None else previous_relation],
                             dtype=torch.long, device=device))
            history = torch.cat([
                attended, g, action_vec,
                source_vec + target_vec + relation_vec
            ], dim=-1)
            next_working = self.history_update(history, working)
        else:
            raise ValueError(self.state_mode)

        return {
            "node_state": x,
            "attention_logits": attention_logits,
            "attention_soft": soft,
            "attention_hard": hard,
            "masked_nodes": masked,
            "attended": attended,
            "action_logits": action_logits,
            "source_logits": source_logits,
            "target_logits": target_logits,
            "relation_logits": relation_logits,
            "next_working": next_working,
        }

    @torch.no_grad()
    def predicted_transition(self, state, out):
        """
        Apply the discrete action represented by a model output to a State.

        Kept on the model API because the training loop uses the exact same
        transition logic for free/scheduled state generation and autonomous
        rollout.
        """
        from state import ACTION_TO_ID

        names = [n.concept for n in state.nodes]

        source_index = int(
            out["source_logits"].argmax().item()
        )
        target_index = int(
            out["target_logits"].argmax().item()
        )

        source = (
            names[source_index]
            if 0 <= source_index < len(names)
            else None
        )
        target = (
            names[target_index]
            if 0 <= target_index < len(names)
            else None
        )

        relation_id = int(
            out["relation_logits"].argmax().item()
        )

        relation = next(
            (
                name
                for name, rid in RELATION_TO_ID.items()
                if rid == relation_id
            ),
            None,
        )

        action_id = int(
            out["action_logits"].argmax().item()
        )

        next_state = state.apply(
            action_id,
            source=source,
            target=target,
            relation=relation,
        )

        return (
            next_state,
            action_id,
            source,
            target,
            relation_id,
        )

    @torch.no_grad()
    def autonomous_rollout(
        self, initial_state, goal, device, steps, stop_on_terminal=False
    ):
        """Closed-loop inference; no teacher state is injected after step 0."""
        from state import ACTION_TO_ID

        current = initial_state.clone()
        working = torch.zeros((1, self.hidden_size), device=device)

        prev_action = None
        prev_source = None
        prev_target = None
        prev_relation = None

        outputs = []
        states = [current.clone()]

        for _ in range(int(steps)):
            out = self.cognitive_step(
                current, goal, working,
                prev_action, prev_source, prev_target, prev_relation,
                device
            )

            action_id = int(out["action_logits"].argmax().item())
            names = [n.concept for n in current.nodes]
            si = int(out["source_logits"].argmax().item())
            ti = int(out["target_logits"].argmax().item())

            source = names[si] if 0 <= si < len(names) else None
            target = names[ti] if 0 <= ti < len(names) else None

            rid = int(out["relation_logits"].argmax().item())
            relation = next(
                (name for name, value in RELATION_TO_ID.items() if value == rid),
                None
            )

            current = current.apply(
                action_id,
                source=source,
                target=target,
                relation=relation,
            )

            outputs.append({
                "out": out,
                "action_id": action_id,
                "source": source,
                "target": target,
                "relation": relation,
            })
            states.append(current.clone())

            working = out["next_working"]
            prev_action = action_id
            prev_source = source
            prev_target = target
            prev_relation = rid

            if stop_on_terminal and action_id in (
                ACTION_TO_ID["NOOP"],
                ACTION_TO_ID["COMMIT"],
            ):
                break

        return {
            "outputs": outputs,
            "states": states,
            "final_state": current,
        }

    def forward_static(self, state, goal, device, transformer_depth=None):
        depth = transformer_depth or self.depth
        working = torch.zeros((1, self.hidden_size), device=device)
        return self.cognitive_step(
            state, goal, working,
            None, None, None, None, device
        )

    def forward_iterative(
        self, states, goal, device, transformer_depth=None, max_steps=None
    ):
        # Kept for compatibility with the previous trainer API.
        # V229 matrix uses run-time cognitive steps directly.
        selected = states if max_steps is None else states[:max_steps]
        working = torch.zeros((1, self.hidden_size), device=device)
        outputs = []
        prev_action = prev_source = prev_target = prev_relation = None

        for state in selected:
            out = self.cognitive_step(
                state, goal, working,
                prev_action, prev_source, prev_target, prev_relation,
                device
            )
            outputs.append(out)
            working = out["next_working"]
            prev_action = int(out["action_logits"].argmax().item())
            names = [n.concept for n in state.nodes]
            si = int(out["source_logits"].argmax().item())
            ti = int(out["target_logits"].argmax().item())
            prev_source = names[si] if 0 <= si < len(names) else None
            prev_target = names[ti] if 0 <= ti < len(names) else None
            prev_relation = int(out["relation_logits"].argmax().item())

        return outputs
