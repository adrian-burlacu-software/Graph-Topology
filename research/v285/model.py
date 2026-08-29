
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
RESEARCH = HERE.parent
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from v200_graph_transformer_cognitive.graph_transformer import GraphAttentionBlock
from v200_graph_transformer_cognitive.long_term_memory import RELATION_TO_ID
from state import ACTIONS


def stable_id(text: str, size: int) -> int:
    return int.from_bytes(
        hashlib.blake2b(text.encode(), digest_size=8).digest(),
        "little",
    ) % size


class StateArchitectureModel(nn.Module):
    """
    V235 architecture model.

    state_mode:
      stateless:
        working state is structurally absent from the decision pathway.

      latent:
        persistent workspace updates from attended graph + goal and is fed
        back into attention and the controller.

      latent_action:
        persistent workspace additionally receives previous operation history.

    survey flags:
      attention_workspace:
        workspace participates directly in the attention scoring pathway.

      explicit_progress:
        a scalar progress/cursor embedding participates in the controller and
        workspace update.

      direct_goal_to_workspace:
        goal is explicitly injected into workspace update.

    These are architectural switches, not post-hoc score biases.
    """

    def __init__(
        self,
        vocab_size=50000,
        relation_count=None,
        hidden_size=128,
        heads=4,
        depth=8,
        topk=5,
        state_mode="latent",
        attention_workspace=False,
        explicit_progress=False,
        direct_goal_to_workspace=False,
        read_mode="standard",
        progress_read_gain=False,
        slow_memory=False,
        persistent_progress=False,
        terminal_query=False,
        action_memory_binding=False,
        terminal_memory_bridge=False,
    ):
        super().__init__()

        relation_count = relation_count or len(RELATION_TO_ID)

        self.vocab_size = vocab_size
        self.relation_count = relation_count
        self.hidden_size = hidden_size
        self.depth = depth
        self.topk = topk
        self.state_mode = state_mode
        self.read_mode = read_mode
        self.progress_read_gain = bool(progress_read_gain)
        self.slow_memory = bool(slow_memory)
        self.persistent_progress = bool(persistent_progress)
        self.terminal_query = bool(terminal_query)
        self.action_memory_binding = bool(action_memory_binding)
        self.terminal_memory_bridge = bool(terminal_memory_bridge)

        self.attention_workspace = bool(attention_workspace)
        self.explicit_progress = bool(explicit_progress)
        self.direct_goal_to_workspace = bool(direct_goal_to_workspace)

        self.node_embedding = nn.Embedding(vocab_size, hidden_size)
        self.role_embedding = nn.Embedding(8, hidden_size)
        self.activation_projection = nn.Linear(1, hidden_size)
        self.relation_embedding = nn.Embedding(relation_count, hidden_size)

        self.blocks = nn.ModuleList([
            GraphAttentionBlock(
                hidden_size,
                heads,
                dropout=0.05,
            )
            for _ in range(depth)
        ])

        self.goal_relation = nn.Embedding(
            relation_count,
            hidden_size,
        )
        self.goal_depth = nn.Embedding(
            8,
            hidden_size,
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Explicit query/focus tokens are optional.  When present they are
        # projected into the goal representation and therefore really reach
        # the decision pathway; when absent this contributes exactly zero.
        self.goal_focus_embedding = nn.Embedding(
            vocab_size,
            hidden_size,
        )

        # Previous action must reach the decision BEFORE action logits for the
        # latent_action family.  The older architecture only used previous
        # action when constructing next_working, which made a "previous action
        # changes current decision" benchmark impossible.
        self.action_decision_gate = nn.Linear(
            hidden_size,
            hidden_size,
        )

        # V285: protected persistent workspace.
        # retention_gate ~= 0 means "keep old memory";
        # retention_gate ~= 1 means "replace with candidate".
        self.retention_gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.retention_bias = nn.Parameter(
            torch.full(
                (hidden_size,),
                -2.0,
            )
        )

        # V285: explicit persistent-memory read path.
        self.workspace_read = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.workspace_read_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.workspace_read_bias = nn.Parameter(
            torch.full(
                (hidden_size,),
                -1.0,
            )
        )

        self.workspace_decision_norm = nn.LayerNorm(
            hidden_size
        )

        self.action_workspace_candidate = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.action_retention_gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.action_retention_bias = nn.Parameter(
            torch.full((hidden_size,), -2.0)
        )

        self.progress_embedding = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.slow_memory_update = nn.GRUCell(
            hidden_size,
            hidden_size,
        )
        self.slow_memory_mix = nn.Parameter(
            torch.tensor(0.12)
        )

        self.progress_memory_update = nn.GRUCell(
            hidden_size,
            hidden_size,
        )
        self.progress_memory_mix = nn.Parameter(
            torch.tensor(0.15)
        )

        self.progress_read_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.attention_graph = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

        self.attention_workspace_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

        self.controller = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, len(ACTIONS)),
        )

        self.terminal_memory_query = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, len(ACTIONS)),
        )

        # V285: action-specific memory binding. The remembered workspace is
        # mapped through action-conditioned gates so the terminal decision
        # can use a compact action-relevant subspace.
        self.action_memory_bind = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )

        self.terminal_memory_bridge_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )

        self.terminal_memory_bridge_head = nn.Sequential(
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

        self.state_update = nn.GRUCell(
            hidden_size * 2,
            hidden_size,
        )

        self.action_embedding = nn.Embedding(
            len(ACTIONS),
            hidden_size,
        )
        self.pointer_embedding = nn.Embedding(
            vocab_size,
            hidden_size,
        )
        self.history_update = nn.GRUCell(
            hidden_size * 4,
            hidden_size,
        )

        # Optional progress signal is deliberately projected into the same
        # dimensionality as the workspace.
        self.progress_gate = nn.Linear(
            hidden_size,
            hidden_size,
        )

    def encode_graph(self, state, device):
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
            + self.activation_projection(
                activations[:, None]
            )
        )

        if state.edges:
            index = {
                n.concept: i
                for i, n in enumerate(state.nodes)
            }
            edge_index = torch.tensor(
                [
                    [index[e.source] for e in state.edges],
                    [index[e.target] for e in state.edges],
                ],
                dtype=torch.long,
                device=device,
            )
            relation_ids = torch.tensor(
                [
                    RELATION_TO_ID.get(e.relation, 0)
                    for e in state.edges
                ],
                dtype=torch.long,
                device=device,
            )
            edge_state = self.relation_embedding(
                relation_ids
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

        return nn.functional.layer_norm(
            x,
            (self.hidden_size,),
        )

    def encode_goal(self, goal, device):
        def node(name):
            if not name:
                return torch.zeros(
                    (1, self.hidden_size),
                    device=device,
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

        base = self.goal_encoder(
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

        focus = goal.get("focus")
        if focus:
            base = base + self.goal_focus_embedding(
                torch.tensor(
                    [stable_id(str(focus), self.vocab_size)],
                    dtype=torch.long,
                    device=device,
                )
            )

        return base

    def encode_progress(self, progress, device):
        p = float(progress)
        x = torch.tensor(
            [[
                max(0.0, min(1.0, p / 8.0)),
                1.0 if p > 0 else 0.0,
            ]],
            dtype=torch.float32,
            device=device,
        )
        return self.progress_embedding(x)

    def _history_vector(
        self,
        previous_action_id,
        previous_source,
        previous_target,
        previous_relation,
        device,
    ):
        aid = 0 if previous_action_id is None else int(previous_action_id)

        action_vec = self.action_embedding(
            torch.tensor(
                [aid],
                dtype=torch.long,
                device=device,
            )
        )

        source_vec = (
            self.pointer_embedding(
                torch.tensor(
                    [stable_id(previous_source, self.vocab_size)],
                    dtype=torch.long,
                    device=device,
                )
            )
            if previous_source
            else torch.zeros_like(action_vec)
        )

        target_vec = (
            self.pointer_embedding(
                torch.tensor(
                    [stable_id(previous_target, self.vocab_size)],
                    dtype=torch.long,
                    device=device,
                )
            )
            if previous_target
            else torch.zeros_like(action_vec)
        )

        relation_vec = self.relation_embedding(
            torch.tensor(
                [
                    0
                    if previous_relation is None
                    else int(previous_relation)
                ],
                dtype=torch.long,
                device=device,
            )
        )

        return action_vec, source_vec, target_vec, relation_vec

    def cognitive_step(
        self,
        state,
        goal,
        working,
        previous_action_id,
        previous_source,
        previous_target,
        previous_relation,
        device,
        progress=0,
        slow_memory_state=None,
        progress_memory_state=None,
    ):
        x = self.encode_graph(state, device)
        g = self.encode_goal(goal, device)

        # Previous action is a true decision-time input only for latent_action.
        # This makes P2/P5 test the intended architectural distinction:
        # "does the current decision condition on the previous operation?"
        decision_history = torch.zeros_like(g)
        if self.state_mode == "latent_action":
            action_vec, _, _, _ = self._history_vector(
                previous_action_id,
                previous_source,
                previous_target,
                previous_relation,
                device,
            )
            decision_history = self.action_decision_gate(
                action_vec
            )

        # CRITICAL: the stateless control has no working-state pathway at all.
        if self.state_mode == "stateless":
            decision_workspace = torch.zeros_like(
                g
            )
        else:
            decision_workspace = working

        progress_vec = (
            self.encode_progress(progress, device)
            if self.explicit_progress
            else torch.zeros_like(g)
        )

        # Workspace can explicitly participate in attention, but only for
        # architectures that claim that decision.
        if self.attention_workspace:
            wx = decision_workspace.expand(
                x.size(0),
                -1,
            )
            gx = g.expand(
                x.size(0),
                -1,
            )
            attention_logits = self.attention_workspace_head(
                torch.cat([x, gx, wx], dim=-1)
            ).squeeze(-1)
        else:
            gx = g.expand(
                x.size(0),
                -1,
            )
            attention_logits = self.attention_graph(
                torch.cat([x, gx], dim=-1)
            ).squeeze(-1)

        k = min(self.topk, x.size(0))
        _, indices = torch.topk(
            attention_logits,
            k=k,
        )

        hard = torch.zeros_like(
            attention_logits
        )
        hard[indices] = 1.0

        soft = torch.sigmoid(
            attention_logits
        )

        mask = (
            hard
            + soft
            - soft.detach()
        )

        masked = x * mask[:, None]
        attended = (
            masked.sum(
                dim=0,
                keepdim=True,
            )
            / mask.sum().clamp_min(1e-5)
        )

        if slow_memory_state is None:
            slow_memory_state = torch.zeros_like(working)

        if progress_memory_state is None:
            progress_memory_state = torch.zeros_like(working)

        if self.slow_memory:
            slow_candidate = self.slow_memory_update(
                g + attended,
                slow_memory_state,
            )
            slow_alpha = torch.sigmoid(
                self.slow_memory_mix
            )
            slow_memory_for_read = (
                (1.0 - slow_alpha) * slow_memory_state
                + slow_alpha * slow_candidate
            )
        else:
            slow_candidate = slow_memory_state
            slow_memory_for_read = torch.zeros_like(working)

        if self.persistent_progress and self.explicit_progress:
            progress_candidate = self.progress_memory_update(
                progress_vec,
                progress_memory_state,
            )
            progress_alpha = torch.sigmoid(
                self.progress_memory_mix
            )
            progress_memory_for_read = (
                (1.0 - progress_alpha) * progress_memory_state
                + progress_alpha * progress_candidate
            )
        else:
            progress_candidate = progress_memory_state
            progress_memory_for_read = torch.zeros_like(working)

        if self.state_mode != "stateless":
            controller_workspace = decision_workspace
        else:
            controller_workspace = torch.zeros_like(g)

        if self.explicit_progress:
            controller_workspace = (
                controller_workspace
                + self.progress_gate(progress_vec)
            )

        controller_workspace = (
            controller_workspace + decision_history
        )

        # V285 explicit memory read path.
        if self.read_mode != "standard":
            memory_for_read = working

            if self.slow_memory:
                memory_for_read = (
                    memory_for_read + slow_memory_for_read
                )

            if self.persistent_progress:
                memory_for_read = (
                    memory_for_read + progress_memory_for_read
                )

            if self.read_mode in (
                "normalized",
                "gated_read",
                "protected_read",
            ):
                memory_for_read = F.normalize(
                    memory_for_read,
                    p=2,
                    dim=-1,
                    eps=1e-8,
                )

            memory_read = self.workspace_read(
                memory_for_read
            )

            if self.progress_read_gain and self.explicit_progress:
                read_gain = torch.sigmoid(
                    self.progress_read_gate(
                        torch.cat(
                            [
                                memory_for_read,
                                progress_vec,
                            ],
                            dim=-1,
                        )
                    )
                )
                memory_read = memory_read * (
                    0.5 + read_gain
                )

            if self.read_mode in (
                "gated_read",
                "protected_read",
            ):
                read_gate = torch.sigmoid(
                    self.workspace_read_gate(
                        torch.cat(
                            [
                                working,
                                g,
                            ],
                            dim=-1,
                        )
                    )
                    + self.workspace_read_bias
                )
                memory_read = read_gate * memory_read

            controller_workspace = (
                controller_workspace + memory_read
            )

            controller_workspace = (
                self.workspace_decision_norm(
                    controller_workspace
                )
            )

        representation = torch.cat(
            [
                attended,
                g,
                controller_workspace,
            ],
            dim=-1,
        )

        action_logits = self.controller(
            representation
        ).squeeze(0)

        if self.action_memory_binding:
            bind_gate = self.action_memory_bind(
                torch.cat(
                    [
                        working,
                        g,
                    ],
                    dim=-1,
                )
            )
            bound_memory = working * bind_gate
            bound_representation = torch.cat(
                [
                    bound_memory,
                    g,
                    progress_vec,
                ],
                dim=-1,
            )
            action_logits = (
                action_logits
                + self.controller(
                    bound_representation
                ).squeeze(0)
            )

        if self.terminal_query:
            query_input = torch.cat(
                [
                    working,
                    g,
                    progress_vec,
                ],
                dim=-1,
            )
            query_logits = self.terminal_memory_query(
                query_input
            ).squeeze(0)

            action_logits = (
                action_logits + query_logits
            )

        if self.terminal_memory_bridge:
            bridge_gate = self.terminal_memory_bridge_gate(
                torch.cat(
                    [
                        working,
                        g,
                    ],
                    dim=-1,
                )
            )
            bridge_memory = working * bridge_gate

            bridge_input = torch.cat(
                [
                    bridge_memory,
                    g,
                    progress_vec,
                ],
                dim=-1,
            )

            bridge_logits = self.terminal_memory_bridge_head(
                bridge_input
            ).squeeze(0)

            action_logits = (
                action_logits + bridge_logits
            )

        pointer_input = torch.cat(
            [
                masked,
                gx,
                controller_workspace.expand_as(masked),
            ],
            dim=-1,
        )

        source_logits = self.source_head(
            pointer_input
        ).squeeze(-1)

        target_logits = self.target_head(
            pointer_input
        ).squeeze(-1)

        source_index = source_logits.argmax()
        target_index = target_logits.argmax()

        relation_logits = self.relation_head(
            torch.cat(
                [
                    masked[
                        source_index:source_index + 1
                    ],
                    masked[
                        target_index:target_index + 1
                    ],
                    g,
                    controller_workspace,
                ],
                dim=-1,
            )
        ).squeeze(0)

        if self.state_mode == "stateless":
            next_working = torch.zeros_like(
                working
            )

        elif self.state_mode in ("latent", "gated_latent"):
            update_input = torch.cat(
                [attended, g],
                dim=-1,
            )

            if self.direct_goal_to_workspace:
                update_input = torch.cat(
                    [
                        attended + g,
                        g,
                    ],
                    dim=-1,
                )

            if self.explicit_progress:
                update_input = torch.cat(
                    [
                        attended + progress_vec,
                        g,
                    ],
                    dim=-1,
                )

            candidate = self.state_update(
                update_input,
                working,
            )

            if self.state_mode == "gated_latent":
                gate_input = torch.cat(
                    [
                        working,
                        candidate,
                        g,
                    ],
                    dim=-1,
                )

                retain = torch.sigmoid(
                    self.retention_gate(
                        gate_input
                    )
                    + self.retention_bias
                )

                # retain=1 -> keep old workspace.
                # retain=0 -> accept new candidate.
                next_working = (
                    retain * working
                    + (1.0 - retain) * candidate
                )

            else:
                next_working = candidate

        elif self.state_mode in (
            "latent_action",
            "latent_action_protected",
        ):
            action_vec, source_vec, target_vec, relation_vec = (
                self._history_vector(
                    previous_action_id,
                    previous_source,
                    previous_target,
                    previous_relation,
                    device,
                )
            )

            history = torch.cat(
                [
                    attended,
                    g,
                    action_vec,
                    source_vec
                    + target_vec
                    + relation_vec,
                ],
                dim=-1,
            )

            if self.state_mode == "latent_action_protected":
                candidate = self.action_workspace_candidate(
                    history
                )

                retain = torch.sigmoid(
                    self.action_retention_gate(
                        torch.cat(
                            [
                                working,
                                candidate,
                                g,
                            ],
                            dim=-1,
                        )
                    )
                    + self.action_retention_bias
                )

                next_working = (
                    retain * working
                    + (1.0 - retain) * candidate
                )
            else:
                next_working = self.history_update(
                    history,
                    working,
                )

            if self.direct_goal_to_workspace:
                next_working = next_working + 0.10 * g

            if self.explicit_progress:
                next_working = next_working + 0.10 * progress_vec

        else:
            raise ValueError(
                f"Unknown state_mode={self.state_mode}"
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
            "next_slow_memory": (
                slow_candidate if self.slow_memory
                else slow_memory_state
            ),
            "next_progress_memory": (
                progress_candidate
                if (
                    self.persistent_progress
                    and self.explicit_progress
                )
                else progress_memory_state
            ),
        }

    @torch.no_grad()
    def predicted_transition(self, state, out):
        names = [
            n.concept
            for n in state.nodes
        ]

        si = int(
            out["source_logits"].argmax().item()
        )
        ti = int(
            out["target_logits"].argmax().item()
        )

        source = (
            names[si]
            if 0 <= si < len(names)
            else None
        )
        target = (
            names[ti]
            if 0 <= ti < len(names)
            else None
        )

        rid = int(
            out["relation_logits"].argmax().item()
        )

        relation = next(
            (
                name
                for name, value
                in RELATION_TO_ID.items()
                if value == rid
            ),
            None,
        )

        aid = int(
            out["action_logits"].argmax().item()
        )

        return (
            state.apply(
                aid,
                source=source,
                target=target,
                relation=relation,
            ),
            aid,
            source,
            target,
            rid,
        )

    @torch.no_grad()
    def autonomous_rollout(
        self,
        initial_state,
        goal,
        device,
        steps,
        stop_on_terminal=False,
    ):
        current = initial_state.clone()
        working = torch.zeros(
            (1, self.hidden_size),
            device=device,
        )

        previous_action_id = None
        previous_source = None
        previous_target = None
        previous_relation = None

        outputs = []
        states = [current.clone()]

        for t in range(int(steps)):
            out = self.cognitive_step(
                current,
                goal,
                working,
                previous_action_id,
                previous_source,
                previous_target,
                previous_relation,
                device,
                progress=t,
            )

            current, action_id, source, target, relation_id = (
                self.predicted_transition(
                    current,
                    out,
                )
            )

            outputs.append({
                "out": out,
                "action_id": action_id,
                "source": source,
                "target": target,
                "relation": relation_id,
            })
            states.append(
                current.clone()
            )

            working = out["next_working"]
            previous_action_id = action_id
            previous_source = source
            previous_target = target
            previous_relation = relation_id

            if stop_on_terminal and action_id in (0, 6):
                break

        return {
            "outputs": outputs,
            "states": states,
            "final_state": current,
        }
