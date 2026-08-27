
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


def stable_id(text, size):
    return int.from_bytes(
        hashlib.blake2b(text.encode(), digest_size=8).digest(),
        "little",
    ) % size


class CognitiveModel(nn.Module):
    """
    Three modes:

    depth:
        one static cognitive decision after N distinct Transformer layers.

    iterative:
        repeated cognitive decisions over teacher states, carrying a learned
        working state from one step to the next. One Transformer layer per step.

    both:
        repeated cognitive decisions, with N distinct Transformer layers at
        every cognitive step.

    IMPORTANT: iterative state is not "reusing a Transformer block". The
    recurrent object is the learned cognitive/working state between states.
    """

    def __init__(
        self,
        vocab_size=50000,
        relation_count=None,
        hidden_size=128,
        heads=4,
        depth=6,
        topk=5,
        mode="depth",
    ):
        super().__init__()

        relation_count = relation_count or len(RELATION_TO_ID)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.topk = topk
        self.mode = mode

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

        # Attention sees the current working state.
        self.attention_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

        # Downstream heads only see attended information.
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

        # Actual iterative cognitive state transition.
        self.state_update = nn.GRUCell(hidden_size * 2, hidden_size)

    def encode_graph(self, state, device, depth):
        ids = torch.tensor(
            [stable_id(n.concept, self.vocab_size) for n in state.nodes],
            dtype=torch.long,
            device=device,
        )
        roles = torch.tensor(
            [min(max(n.role, 0), 7) for n in state.nodes],
            dtype=torch.long,
            device=device,
        )
        activations = torch.tensor(
            [n.activation for n in state.nodes],
            dtype=torch.float32,
            device=device,
        )

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
                dtype=torch.long,
                device=device,
            )

            relation_ids = torch.tensor(
                [RELATION_TO_ID.get(e.relation, 0) for e in state.edges],
                dtype=torch.long,
                device=device,
            )

            edge_state = self.relation_embedding(relation_ids)
        else:
            edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=device
            )
            edge_state = torch.empty(
                (0, self.hidden_size), dtype=x.dtype, device=device
            )

        for block in self.blocks[:depth]:
            x = block(x, edge_index, edge_state)

        return nn.functional.layer_norm(x, (self.hidden_size,))

    def encode_goal(self, goal, device):
        def node(name):
            if not name:
                return torch.zeros(
                    (1, self.hidden_size), device=device
                )
            return self.node_embedding(
                torch.tensor(
                    [stable_id(name, self.vocab_size)],
                    dtype=torch.long,
                    device=device,
                )
            )

        relation = self.goal_relation(
            torch.tensor(
                [RELATION_TO_ID.get(goal["relation"], 0)],
                dtype=torch.long,
                device=device,
            )
        )

        depth = self.goal_depth(
            torch.tensor(
                [min(int(goal["depth"]), 7)],
                dtype=torch.long,
                device=device,
            )
        )

        return self.goal_encoder(
            torch.cat(
                [
                    node(goal["source"]),
                    node(goal["target"]),
                    relation,
                    depth,
                ],
                dim=-1,
            )
        )

    def cognitive_step(self, graph_state, goal, working, transformer_depth, device):
        x = self.encode_graph(graph_state, device, transformer_depth)
        g = self.encode_goal(goal, device)

        gx = g.expand(x.size(0), -1)
        wx = working.expand(x.size(0), -1)

        attention_logits = self.attention_head(
            torch.cat([x, gx, wx], dim=-1)
        ).squeeze(-1)

        k = min(self.topk, x.size(0))
        _, top_indices = torch.topk(attention_logits, k=k)

        hard = torch.zeros_like(attention_logits)
        hard[top_indices] = 1.0

        soft = torch.sigmoid(attention_logits)

        # Straight-through hard attention.
        mask = hard + soft - soft.detach()

        masked = x * mask[:, None]
        attended = masked.sum(0, keepdim=True) / mask.sum().clamp_min(1e-5)

        representation = torch.cat(
            [attended, g, working],
            dim=-1,
        )

        action_logits = self.action_head(representation).squeeze(0)

        pointer_input = torch.cat(
            [masked, gx.expand_as(masked), wx],
            dim=-1,
        )

        source_logits = self.source_head(pointer_input).squeeze(-1)
        target_logits = self.target_head(pointer_input).squeeze(-1)

        source_index = source_logits.argmax()
        target_index = target_logits.argmax()

        relation_logits = self.relation_head(
            torch.cat(
                [
                    masked[source_index:source_index + 1],
                    masked[target_index:target_index + 1],
                    g,
                    working,
                ],
                dim=-1,
            )
        ).squeeze(0)

        # THIS is the iterative state:
        # the result of the current cognitive computation becomes working
        # memory for the next state.
        next_working = self.state_update(
            torch.cat([attended, g], dim=-1),
            working,
        )

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
            "working": working,
        }

    @torch.no_grad()
    def autonomous_rollout(
        self,
        initial_state,
        goal,
        device,
        steps,
        transformer_depth=None,
        stop_on_terminal=False,
    ):
        """
        Closed-loop rollout.

        The model receives the initial graph once. Every later graph is created
        from the model's own previous action. No oracle state is injected.

        stop_on_terminal defaults to False for experiments: otherwise a model
        predicting COMMIT/NOOP could hide the quality of later transitions.
        """
        from state import ACTION_TO_ID
        from v200_graph_transformer_cognitive.long_term_memory import RELATION_TO_ID

        depth = transformer_depth or self.depth
        current = initial_state.clone()
        working = torch.zeros((1, self.hidden_size), device=device)

        outputs = []
        states = [current.clone()]

        for _ in range(steps):
            out = self.cognitive_step(
                current, goal, working, depth, device
            )

            action_id = int(out["action_logits"].argmax().item())
            source_i = int(out["source_logits"].argmax().item())
            target_i = int(out["target_logits"].argmax().item())

            source = (
                current.nodes[source_i].concept
                if 0 <= source_i < len(current.nodes)
                else None
            )
            target = (
                current.nodes[target_i].concept
                if 0 <= target_i < len(current.nodes)
                else None
            )

            relation_id = int(out["relation_logits"].argmax().item())
            relation = next(
                (name for name, rid in RELATION_TO_ID.items()
                 if rid == relation_id),
                None,
            )

            next_state = current.apply(
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

            current = next_state
            states.append(current.clone())
            working = out["next_working"]

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
        working = torch.zeros(
            (1, self.hidden_size),
            device=device,
        )
        return self.cognitive_step(
            state,
            goal,
            working,
            depth,
            device,
        )

    def forward_iterative(
        self,
        states,
        goal,
        device,
        transformer_depth=None,
        max_steps=None,
    ):
        depth = transformer_depth or self.depth

        # One persistent learned working state across cognitive steps.
        working = torch.zeros(
            (1, self.hidden_size),
            device=device,
        )

        outputs = []

        # The teacher trajectory is authoritative. max_steps is only a cap;
        # it never fabricates states that do not exist in the deterministic
        # teacher trajectory.
        if max_steps is None:
            selected_states = states
        else:
            selected_states = states[:max_steps]

        for state in selected_states:
            out = self.cognitive_step(
                state,
                goal,
                working,
                depth,
                device,
            )
            outputs.append(out)
            working = out["next_working"]

        return outputs
