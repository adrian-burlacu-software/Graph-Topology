from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RESEARCH_ROOT = _HERE.parent
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

from state import ACTIONS, ACTION_TO_ID, State

RELATIONS = (
    "IsA", "PartOf", "Causes", "UsedFor", "RelatedTo",
    "AtLocation", "HasProperty", "CapableOf",
)

CONCEPTS = (
    "dog", "animal", "living_thing", "entity", "pet", "wolf",
    "mammal", "organism", "creature", "thing", "cat", "vehicle",
    "car", "travel", "tree", "forest", "food", "water", "home",
    "person", "tool", "place", "machine", "object", "plant",
    "house", "road", "city", "bird", "fish", "metal", "material",
    "energy", "motion", "weather", "space", "land", "river",
)

def balanced_counts(total: int) -> dict[str, int]:
    q, r = divmod(total, len(ACTIONS))
    return {a: q + (i < r) for i, a in enumerate(ACTIONS)}

def add_edge(state, source, relation, target, activation=0.95):
    state.add_node(source, activation, 1)
    state.add_node(target, activation, 1)
    state.add_edge(source, relation, target, activation)

def make_chain(rng: random.Random, depth: int, offset: int):
    # Use unique concepts for the reasoning chain so the oracle chain is unambiguous.
    pool = list(CONCEPTS)
    rng.shuffle(pool)
    nodes = pool[:depth + 1]
    rels = [rng.choice(("IsA", "PartOf", "RelatedTo")) for _ in range(depth)]
    return nodes, rels

def make_example(action: str, rng: random.Random, index: int) -> dict:
    depth = rng.randint(2, 5)
    chain, chain_rels = make_chain(rng, depth, index)

    state = State(nodes=[], edges=[])

    # The only high-confidence connected path from source to goal.
    for i, rel in enumerate(chain_rels):
        add_edge(state, chain[i], rel, chain[i + 1], 0.98)

    # Many distractor nodes.
    used = set(chain)
    distractors = [c for c in CONCEPTS if c not in used]
    rng.shuffle(distractors)
    target_nodes = rng.randint(18, 24)

    for c in distractors:
        if len(state.nodes) >= target_nodes:
            break
        state.add_node(c, rng.choice((0.10, 0.20, 0.35, 0.55, 0.70)), rng.randint(0, 5))

    while len(state.nodes) < target_nodes:
        state.add_node(f"distractor_{len(state.nodes)}", rng.choice((0.1, 0.25, 0.5)), rng.randint(0, 5))

    names = [n.concept for n in state.nodes]

    # Decoys use the same relation vocabulary and occasionally connect to chain nodes,
    # but do not form a second valid source->goal path.
    for _ in range(rng.randint(20, 32)):
        a, b = rng.sample(names, 2)
        rel = rng.choice(RELATIONS)
        if not state.has_edge(a, rel, b, active_only=False):
            state.add_edge(a, rel, b, rng.choice((0.08, 0.15, 0.25, 0.40, 0.60)))

    source = chain[0]
    goal_target = chain[-1]

    # The task asks for the endpoint relationship. The architecture must inspect
    # the chain to know which local edge is the final operation.
    if action == "BIND":
        action_source = chain[-2]
        action_target = chain[-1]
        action_relation = chain_rels[-1]
    elif action == "BRANCH":
        action_source = chain[0]
        action_target = None
        action_relation = chain_rels[0]
    elif action in ("REUSE", "INHIBIT"):
        action_source = None
        action_target = chain[0]
        action_relation = None
    else:
        action_source = None
        action_target = None
        action_relation = None

    next_state = state.apply(
        ACTION_TO_ID[action],
        source=action_source,
        target=action_target,
        relation=action_relation,
    )

    # Attention MUST identify the complete reasoning path. The downstream model
    # will receive only this selected subgraph.
    attention_target = list(chain)

    return {
        "version": "v214",
        "case_id": f"v214_{index:05d}_{action.lower()}",
        "initial_state": state.signature(),
        "final_state": next_state.signature(),
        "goal": {
            "source": source,
            "target": goal_target,
            "relation": chain_rels[-1],
            "depth": depth,
            "mode": "endpoint_requires_path",
        },
        "action": {
            "action": action,
            "source": action_source,
            "target": action_target,
            "relation": action_relation,
        },
        "attention_target": attention_target,
        "reasoning_chain": chain,
        "reasoning_relations": chain_rels,
        "chain_depth": depth,
        "final_action": action,
    }

def generate_dataset(samples=500, seed=214):
    rng = random.Random(seed)
    counts = balanced_counts(samples)
    rows = []
    for action in ACTIONS:
        for _ in range(counts[action]):
            rows.append(make_example(action, rng, len(rows)))
    rng.shuffle(rows)
    return rows

def save_dataset(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

class TeacherDataset:
    def __init__(self, records):
        self.rows = records
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        return self.rows[i]

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=214)
    p.add_argument("--output", type=Path, default=Path("results/v214_hard_attention_dataset.jsonl"))
    args = p.parse_args()
    rows = generate_dataset(args.samples, args.seed)
    save_dataset(rows, args.output)
    print("V214 DATASET")
    print("samples:", len(rows))
    print("actions:", dict(Counter(r["final_action"] for r in rows)))
    print("depth:", min(r["chain_depth"] for r in rows), "-", max(r["chain_depth"] for r in rows))
    print("saved:", args.output)
