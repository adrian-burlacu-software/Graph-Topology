from __future__ import annotations

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


def state_from_json(payload: dict) -> State:
    return State(
        nodes=[
            Node(
                concept=str(n["concept"]),
                activation=float(n.get("activation", 0.0)),
                role=int(n.get("role", 0)),
                persistent=bool(n.get("persistent", False)),
            )
            for n in payload.get("nodes", [])
        ],
        edges=[
            Edge(
                source=str(e["source"]),
                relation=str(e["relation"]),
                target=str(e["target"]),
                activation=float(e.get("activation", 0.0)),
                persistent=bool(e.get("persistent", False)),
            )
            for e in payload.get("edges", [])
        ],
    )


def _relation_pool() -> list[str]:
    preferred = (
        "IsA", "RelatedTo", "CapableOf", "HasProperty",
        "UsedFor", "HasA", "PartOf", "SimilarTo",
        "Antonym", "Causes", "AtLocation",
    )
    return [r for r in preferred if r in RELATION_TO_ID]


def _action_quota(total: int) -> dict[str, int]:
    q, rem = divmod(total, len(ACTIONS))
    return {a: q + (i < rem) for i, a in enumerate(ACTIONS)}


def _make_state(rng: random.Random, index: int, action: str) -> tuple[State, dict]:
    rels = _relation_pool()
    r = rng.choice(rels)

    concepts = [f"concept_{index}_{i}" for i in range(8)]
    state = State(nodes=[], edges=[])

    # Two active nodes, several inactive distractors.
    for i, concept in enumerate(concepts):
        activation = 1.0 if i < 2 else (0.35 if i < 6 else 0.0)
        role = 1 if i == 0 else (2 if i == 1 else 3)
        state.add_node(concept, activation=activation, role=role)

    # Active semantic structure plus distractor structure.
    state.add_edge(concepts[0], r, concepts[1], activation=0.65)
    state.add_edge(concepts[2], rng.choice(rels), concepts[3], activation=0.25)
    state.add_edge(concepts[4], rng.choice(rels), concepts[5], activation=0.20)

    source = concepts[0]
    target = concepts[1]
    focus = concepts[6]

    # Construct a task-specific state where the deterministic answer is known.
    if action == "NOOP":
        # Goal is already satisfied.
        state.add_edge(source, r, target, activation=1.0)
        state.node(target).activation = 1.0
        source_idx, target_idx = 0, 1
        attention = [0, 1]

    elif action == "REUSE":
        target = concepts[6]
        state.node(target).activation = 0.0
        source_idx, target_idx = -1, 6
        attention = [6]

    elif action == "CREATE":
        source = concepts[0]
        target = concepts[6]
        source_idx, target_idx = 0, 6
        attention = [0, 6]

    elif action == "BRANCH":
        source = concepts[0]
        target = concepts[6]
        source_idx, target_idx = 0, -1
        attention = [0]

    elif action == "INHIBIT":
        target = concepts[1]
        state.node(target).activation = 1.0
        source_idx, target_idx = -1, 1
        attention = [1]

    elif action == "BIND":
        # Goal edge exists in long-term semantics but is inactive in working memory.
        target = concepts[1]
        state.edges = [
            e for e in state.edges
            if not (e.source == source and e.relation == r and e.target == target)
        ]
        state.node(source).activation = 0.7
        state.node(target).activation = 0.35
        source_idx, target_idx = 0, 1
        attention = [0, 1]

    elif action == "COMMIT":
        # Active working-memory material should become persistent.
        source_idx, target_idx = 0, 1
        attention = [
            i for i, n in enumerate(state.nodes)
            if n.activation > 0.5
        ]

    else:
        raise ValueError(action)

    # State-level objective. This is not the teacher/LLM label; it is the
    # deterministic description of what the resulting working memory must do.
    goal_mode = {
        "NOOP": 0,
        "REUSE": 1,    # make an existing node active
        "CREATE": 2,  # add a new node
        "BRANCH": 3,  # extend the graph from a source
        "INHIBIT": 4, # suppress an active node
        "BIND": 5,    # activate a semantic edge
        "COMMIT": 6,  # make active memory persistent
    }[action]

    goal = {
        "source": source,
        "target": target,
        "relation": r,
        "goal_mode": goal_mode,
    }

    next_state = state.apply(
        ACTIONS.index(action),
        source=source if source_idx >= 0 else None,
        target=target if target_idx >= 0 else None,
        relation=r,
    )

    attention_target = [1.0 if i in attention else 0.0 for i in range(len(state.nodes))]

    return state, {
        "next_state": next_state,
        "goal": goal,
        "source_index": source_idx,
        "target_index": target_idx,
        "goal_source_index": source_idx if source_idx >= 0 else 0,
        "goal_target_index": target_idx if target_idx >= 0 else 1,
        "relation": r,
        "attention_target": attention_target,
    }


class TeacherDataset:
    """
    Deterministic oracle dataset for V209.

    No LLM is used. Every target is generated directly from State.apply().
    The default dataset contains exactly 500 samples, balanced across all
    seven action types.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        size: int = 500,
        seed: int = 209,
        regenerate: bool = True,
    ) -> None:
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
                index = len(self.rows)
                state, info = _make_state(
                    rng,
                    index,
                    action,
                )
                self.rows.append({
                    "case_id": f"oracle_{action.lower()}_{ordinal:04d}",
                    "state": state,
                    "next_state": info["next_state"],
                    "action_id": ACTIONS.index(action),
                    "source_index": info["source_index"],
                    "target_index": info["target_index"],
                    "relation": info["relation"],
                    "goal_source_index": info["goal_source_index"],
                    "goal_target_index": info["goal_target_index"],
                    "goal_relation": info["relation"],
                    "goal_mode": info["goal"]["goal_mode"],
                    "goal": info["goal"],
                    "attention_target": info["attention_target"],
                    "teacher_confidence": 1.0,
                    "corrected": False,
                })

        # Shuffle after balancing so action is not encoded by dataset order.
        rng.shuffle(self.rows)

    def _load(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = __import__("json").loads(line)
                    self.rows.append(record)

    def save(self, path: Path) -> None:
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in self.rows:
                state = row["state"]
                next_state = row["next_state"]
                payload = {
                    **{k: v for k, v in row.items()
                       if k not in ("state", "next_state")},
                    "state": state.signature(),
                    "next_state": next_state.signature(),
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        if isinstance(row["state"], dict):
            row = dict(row)
            row["state"] = state_from_json(row["state"])
            row["next_state"] = state_from_json(row["next_state"])
            self.rows[index] = row
        return row

    def action_counts(self) -> dict[str, int]:
        counts = {a: 0 for a in ACTIONS}
        for row in self.rows:
            counts[ACTIONS[row["action_id"]]] += 1
        return counts

    def split(self, valid_fraction: float = 0.15, seed: int = 209):
        # Stratified split keeps every action represented in validation.
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
