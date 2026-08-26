from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from collections import Counter

_HERE = Path(__file__).resolve().parent
_RESEARCH_ROOT = _HERE.parent
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

from state import ACTIONS, ACTION_TO_ID, Edge, Node, State

RELATIONS = (
    "IsA", "PartOf", "Causes", "UsedFor", "RelatedTo",
    "AtLocation", "HasProperty", "CapableOf",
)

BASE_CONCEPTS = (
    "dog", "animal", "living_thing", "entity", "pet",
    "wolf", "mammal", "organism", "creature", "thing",
    "cat", "vehicle", "car", "travel", "tree", "forest",
    "food", "water", "home", "person", "tool", "place",
)

def _balanced_counts(total: int) -> dict[str, int]:
    q, r = divmod(total, len(ACTIONS))
    return {a: q + (i < r) for i, a in enumerate(ACTIONS)}

def _add_edge(s: State, a: str, rel: str, b: str, activation=0.9):
    s.add_node(a, activation, 1)
    s.add_node(b, activation, 1)
    s.add_edge(a, rel, b, activation)

def _make_chain(rng: random.Random, depth: int) -> tuple[list[str], list[str]]:
    pool = list(BASE_CONCEPTS)
    rng.shuffle(pool)
    concepts = pool[:depth + 1]
    relations = [rng.choice(("IsA", "PartOf", "RelatedTo")) for _ in range(depth)]
    return concepts, relations

def make_example(action: str, rng: random.Random, index: int) -> dict:
    # Multi-hop core. The final goal deliberately names only the endpoints.
    depth = rng.randint(2, 5)
    chain_nodes, chain_rels = _make_chain(rng, depth)
    source = chain_nodes[0]
    goal_target = chain_nodes[-1]

    s = State(nodes=[], edges=[])

    # Core chain.
    for i, rel in enumerate(chain_rels):
        _add_edge(s, chain_nodes[i], rel, chain_nodes[i + 1], 0.95)

    # Distractor graph: overlapping relations and plausible alternative paths.
    distractor_count = rng.randint(10, 18)
    used = set(chain_nodes)
    available = [x for x in BASE_CONCEPTS if x not in used]
    while len(s.nodes) < depth + 1 + distractor_count:
        concept = rng.choice(available) if available else f"distractor_{len(s.nodes)}"
        if s.node(concept) is not None:
            concept = f"{concept}_{len(s.nodes)}"
        s.add_node(concept, rng.choice((0.15, 0.3, 0.5, 0.7)), rng.randint(0, 5))

    # Add distractor edges, including decoy paths using the same relations.
    node_names = [n.concept for n in s.nodes]
    for _ in range(rng.randint(16, 28)):
        a, b = rng.sample(node_names, 2)
        rel = rng.choice(RELATIONS)
        if not s.has_edge(a, rel, b, active_only=False):
            s.add_edge(a, rel, b, rng.choice((0.15, 0.35, 0.55, 0.75)))

    # The action target is deterministic. For operations whose semantics do not
    # require a multi-hop transition, the chain remains the attention problem.
    if action == "BIND":
        target = goal_target
        relation = chain_rels[-1]
        action_source = source
        action_target = target
        # The oracle operation is the final hop; relevance is the whole chain.
        next_state = s.apply(ACTION_TO_ID[action], source=action_source, target=action_target, relation=relation)
    elif action == "REUSE":
        action_source = None
        action_target = source
        relation = None
        next_state = s.apply(ACTION_TO_ID[action], target=source)
    elif action == "INHIBIT":
        action_source = None
        action_target = source
        relation = None
        next_state = s.apply(ACTION_TO_ID[action], target=source)
    elif action == "BRANCH":
        action_source = source
        action_target = None
        relation = chain_rels[0]
        next_state = s.apply(ACTION_TO_ID[action], source=source, relation=relation)
    elif action in ("CREATE", "COMMIT", "NOOP"):
        action_source = None
        action_target = None
        relation = None
        next_state = s.apply(ACTION_TO_ID[action])
    else:
        raise ValueError(action)

    relevant = set(chain_nodes)
    # For action semantics, include the actual operands as well.
    if action in ("REUSE", "INHIBIT"):
        relevant.add(source)
    if action == "BRANCH":
        relevant.update((source, chain_nodes[1]))
    if action == "BIND":
        relevant.update((source, goal_target))

    goal = {
        "source": source,
        "target": goal_target,
        "relation": chain_rels[-1],
        "depth": depth,
        "mode": "multi_hop_endpoint_goal",
    }

    return {
        "version": "v213",
        "case_id": f"v213_{index:05d}_{action.lower()}",
        "initial_state": s.signature(),
        "final_state": next_state.signature(),
        "goal": goal,
        "action": {
            "action": action,
            "source": action_source,
            "target": action_target,
            "relation": relation,
        },
        "attention_target": [n.concept for n in s.nodes if n.concept in relevant],
        "reasoning_chain": chain_nodes,
        "reasoning_relations": chain_rels,
        "chain_depth": depth,
        "final_action": action,
    }

class TeacherDataset:
    def __init__(
        self,
        path: Path | None = None,
        *,
        samples: int = 500,
        seed: int = 213,
        records: list[dict] | None = None,
    ):
        self.path = Path(path) if path else None
        if records is not None:
            self.rows = records
        elif self.path is not None and self.path.exists():
            self.rows = [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x.strip()]
        else:
            self.rows = generate_dataset(samples=samples, seed=seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]

    def split(self, valid_fraction=0.15, seed=213):
        groups = {a: [] for a in ACTIONS}
        for i, r in enumerate(self.rows):
            groups[r["final_action"]].append(i)
        rng = random.Random(seed)
        train, valid = [], []
        for ids in groups.values():
            rng.shuffle(ids)
            n = max(1, int(len(ids) * valid_fraction))
            valid.extend(ids[:n])
            train.extend(ids[n:])
        rng.shuffle(train); rng.shuffle(valid)
        return train, valid

def generate_dataset(samples=500, seed=213):
    rng = random.Random(seed)
    counts = _balanced_counts(samples)
    rows = []
    for action, count in counts.items():
        for _ in range(count):
            rows.append(make_example(action, rng, len(rows)))
    rng.shuffle(rows)
    return rows

def save_dataset(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=213)
    p.add_argument("--output", type=Path, default=Path("results/v213_multihop_oracle_dataset.jsonl"))
    args = p.parse_args()
    rows = generate_dataset(args.samples, args.seed)
    save_dataset(rows, args.output)
    print("V213 DATASET")
    print("samples:", len(rows))
    print("actions:", dict(Counter(r["final_action"] for r in rows)))
    print("chain_depth:", min(r["chain_depth"] for r in rows), "-", max(r["chain_depth"] for r in rows))
    print("saved:", args.output)
