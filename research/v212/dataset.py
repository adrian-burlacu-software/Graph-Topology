from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_RESEARCH_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

try:
    from .state import ACTIONS, State, Node, Edge
except ImportError:
    from state import ACTIONS, State, Node, Edge

from v200_graph_transformer_cognitive.long_term_memory import RELATION_TO_ID


RELATIONS = tuple(
    r for r in (
        "IsA", "RelatedTo", "CapableOf", "HasProperty", "UsedFor",
        "HasA", "PartOf", "SimilarTo", "Antonym", "Causes", "AtLocation",
    ) if r in RELATION_TO_ID
)


def _action_quota(total: int) -> dict[str, int]:
    q, rem = divmod(total, len(ACTIONS))
    return {a: q + (i < rem) for i, a in enumerate(ACTIONS)}


def _new_state() -> State:
    return State(nodes=[], edges=[])


def _add_distractors(state: State, rng: random.Random, prefix: str, count: int) -> None:
    for i in range(count):
        state.add_node(
            f"{prefix}_d_{i}",
            activation=rng.choice((0.0, 0.15, 0.25, 0.35)),
            role=3,
        )


def _make_case(rng: random.Random, ordinal: int, action: str) -> dict[str, Any]:
    # Larger working memory than V211: 18-24 nodes, 12-22 edges.
    n = rng.randint(18, 24)
    prefix = f"v212_{action.lower()}_{ordinal:04d}"
    concepts = [f"{prefix}_n_{i}" for i in range(n)]
    state = _new_state()

    for i, concept in enumerate(concepts):
        if i < 4:
            activation = rng.choice((0.65, 0.8, 1.0))
            role = 1 if i == 0 else 2
        else:
            activation = rng.choice((0.0, 0.1, 0.2, 0.3, 0.4))
            role = 3
        state.add_node(concept, activation=activation, role=role)

    # Build a connected relevant chain. The chain is deliberately longer than
    # the immediate goal so attention must learn a subgraph rather than only
    # matching the two goal nodes.
    chain_len = rng.randint(3, 5)
    chain_nodes = list(range(chain_len))
    chain_relations = [rng.choice(RELATIONS) for _ in range(chain_len - 1)]
    for a, rel, b in zip(chain_nodes[:-1], chain_relations, chain_nodes[1:]):
        state.add_edge(concepts[a], rel, concepts[b], activation=rng.choice((0.55, 0.7, 0.85)))

    # Add decoy components, including edges that share relations and even one
    # of the goal endpoints. This prevents simple relation-only shortcuts.
    for _ in range(rng.randint(10, 18)):
        a, b = rng.sample(range(n), 2)
        rel = rng.choice(RELATIONS)
        if not state.has_edge(concepts[a], rel, concepts[b], active_only=False):
            state.add_edge(concepts[a], rel, concepts[b], activation=rng.choice((0.1, 0.2, 0.3, 0.45)))

    source_i = 0
    target_i = min(2, chain_len - 1)
    relation = chain_relations[min(1, len(chain_relations) - 1)]

    # Relevant nodes form the minimal reasoning subgraph plus the operation's
    # direct target. Decoys are deliberately abundant.
    relevant = set(chain_nodes)

    if action == "NOOP":
        # Goal already satisfied by an existing multi-hop edge.
        state.add_edge(concepts[source_i], relation, concepts[target_i], activation=1.0)
        state.node(concepts[source_i]).activation = 1.0
        state.node(concepts[target_i]).activation = 1.0
        source_idx, target_idx = source_i, target_i

    elif action == "REUSE":
        target_idx = n - 1
        state.node(concepts[target_idx]).activation = 0.0
        source_idx = -1
        relevant.add(target_idx)

    elif action == "CREATE":
        source_idx = -1
        target_idx = -1
        # CREATE has no existing target; relevant context is the chain that
        # motivates adding a fresh concept.
        new_concept = f"{prefix}_new"

    elif action == "BRANCH":
        source_idx = source_i
        target_idx = -1
        relevant.add(source_i)
        new_concept = f"{concepts[source_i]}#branch{n}"

    elif action == "INHIBIT":
        target_idx = target_i
        source_idx = -1
        state.node(concepts[target_idx]).activation = 1.0
        relevant.add(target_idx)

    elif action == "BIND":
        # The target edge is absent/inactive. The two endpoints are embedded
        # inside a longer relevant chain to force graph discrimination.
        source_idx, target_idx = source_i, target_i
        state.edges = [
            e for e in state.edges
            if not (e.source == concepts[source_i] and e.relation == relation and e.target == concepts[target_i])
        ]
        state.node(concepts[source_i]).activation = 0.65
        state.node(concepts[target_i]).activation = 0.35
        relevant.update((source_i, target_i))

    elif action == "COMMIT":
        source_idx, target_idx = source_i, target_i
        relevant = {i for i, node in enumerate(state.nodes) if node.activation > 0.5}

    else:
        raise ValueError(action)

    # goal_mode is intentionally the semantic objective, not the action id.
    # It tells the model what kind of state change is requested, while the
    # state determines whether that objective is already satisfied and which
    # operation is the minimal valid solution.
    goal_mode = {
        "NOOP": 0,
        "REUSE": 1,
        "CREATE": 2,
        "BRANCH": 3,
        "INHIBIT": 4,
        "BIND": 5,
        "COMMIT": 6,
    }[action]

    if action == "CREATE":
        goal_target = f"{prefix}_new"
        goal_source_index = source_idx
        goal_target_index = -1
    elif action == "BRANCH":
        goal_target = f"{concepts[source_i]}#branch{n}"
        goal_source_index = source_i
        goal_target_index = -1
    else:
        goal_target = concepts[target_idx] if target_idx >= 0 else concepts[target_i]
        goal_source_index = source_idx if source_idx >= 0 else source_i
        goal_target_index = target_idx if target_idx >= 0 else target_i

    # For CREATE/BRANCH State.apply creates deterministic node names; use the
    # actual source/relation arguments only when the operation requires them.
    next_state = state.apply(
        ACTIONS.index(action),
        source=concepts[source_idx] if source_idx >= 0 else None,
        target=concepts[target_idx] if target_idx >= 0 else None,
        relation=relation,
    )

    # Add the operation's newly-created node to the relevance target only if it
    # exists in the input state (it does not for CREATE/BRANCH).
    attention_target = [1.0 if i in relevant else 0.0 for i in range(len(state.nodes))]

    return {
        "case_id": f"v212_{action.lower()}_{ordinal:04d}",
        "state": state,
        "next_state": next_state,
        "action_id": ACTIONS.index(action),
        "source_index": source_idx,
        "target_index": target_idx,
        "relation": relation,
        "goal_source_index": goal_source_index,
        "goal_target_index": goal_target_index,
        "goal_target_concept": goal_target,
        "goal_relation": relation,
        "goal_mode": goal_mode,
        "attention_target": attention_target,
        "teacher_confidence": 1.0,
        "corrected": False,
        "chain_length": chain_len,
        "node_count": len(state.nodes),
        "edge_count": len(state.edges),
    }


class TeacherDataset:
    """V212 deterministic oracle dataset with hard distractors and graph chains."""

    def __init__(self, path: Path | None = None, *, size: int = 500, seed: int = 212, regenerate: bool = True):
        self.seed = seed
        self.rows: list[dict[str, Any]] = []
        if path is not None and Path(path).exists() and not regenerate:
            self._load(Path(path))
        else:
            self._generate(size)

    def _generate(self, size: int) -> None:
        rng = random.Random(self.seed)
        quotas = _action_quota(size)
        for action in ACTIONS:
            for ordinal in range(quotas[action]):
                self.rows.append(_make_case(rng, ordinal, action))
        rng.shuffle(self.rows)

    def _load(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    row["state"] = state_from_json(row["state"])
                    row["next_state"] = state_from_json(row["next_state"])
                    self.rows.append(row)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in self.rows:
                payload = dict(row)
                payload["state"] = row["state"].signature()
                payload["next_state"] = row["next_state"].signature()
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]

    def action_counts(self) -> dict[str, int]:
        counts = {a: 0 for a in ACTIONS}
        for row in self.rows:
            counts[ACTIONS[row["action_id"]]] += 1
        return counts

    def split(self, valid_fraction: float = 0.15, seed: int = 212):
        rng = random.Random(seed)
        by_action = {i: [] for i in range(len(ACTIONS))}
        for i, row in enumerate(self.rows):
            by_action[row["action_id"]].append(i)
        train, valid = [], []
        for indices in by_action.values():
            rng.shuffle(indices)
            n_valid = max(1, int(len(indices) * valid_fraction))
            valid.extend(indices[:n_valid])
            train.extend(indices[n_valid:])
        rng.shuffle(train)
        rng.shuffle(valid)
        return train, valid


def state_from_json(payload: dict) -> State:
    return State(
        nodes=[Node(str(n["concept"]), float(n.get("activation", 0.0)), int(n.get("role", 0)), bool(n.get("persistent", False))) for n in payload.get("nodes", [])],
        edges=[Edge(str(e["source"]), str(e["relation"]), str(e["target"]), float(e.get("activation", 0.0)), bool(e.get("persistent", False))) for e in payload.get("edges", [])],
    )
