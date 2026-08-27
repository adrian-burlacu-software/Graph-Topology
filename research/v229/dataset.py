
from __future__ import annotations

import json
import random
from pathlib import Path

from state import ACTIONS, ACTION_TO_ID, State

RELATIONS = (
    "IsA", "PartOf", "Causes", "UsedFor",
    "RelatedTo", "AtLocation", "HasProperty", "CapableOf",
)

CONCEPTS = (
    "dog", "animal", "living_thing", "entity", "pet", "wolf", "mammal",
    "organism", "creature", "thing", "cat", "vehicle", "car", "travel",
    "tree", "forest", "food", "water", "home", "person", "tool", "place",
    "machine", "object", "plant", "house", "road", "city", "bird", "fish",
    "metal", "material", "energy", "motion", "weather", "space", "land", "river",
)


def add_edge(s, source, relation, target):
    s.add_node(source, 0.98, 1)
    s.add_node(target, 0.98, 1)
    s.add_edge(source, relation, target, 0.98)


def make_example(action, rng, index):
    depth = rng.randint(2, 5)

    pool = list(CONCEPTS)
    rng.shuffle(pool)

    chain = pool[: depth + 1]
    relations = [rng.choice(("IsA", "PartOf", "RelatedTo")) for _ in range(depth)]

    initial = State([], [])

    for j, relation in enumerate(relations):
        add_edge(initial, chain[j], relation, chain[j + 1])

    target_nodes = rng.randint(18, 24)

    for concept in [x for x in pool if x not in chain]:
        if len(initial.nodes) >= target_nodes:
            break
        initial.add_node(
            concept,
            rng.choice((0.10, 0.20, 0.35, 0.55, 0.70)),
            rng.randint(0, 5),
        )

    while len(initial.nodes) < target_nodes:
        initial.add_node(
            f"distractor_{len(initial.nodes)}",
            rng.choice((0.10, 0.25, 0.50)),
            rng.randint(0, 5),
        )

    names = [n.concept for n in initial.nodes]

    for _ in range(rng.randint(20, 32)):
        a, b = rng.sample(names, 2)
        relation = rng.choice(RELATIONS)
        if not initial.has_edge(a, relation, b, active_only=False):
            initial.add_edge(a, relation, b, rng.choice((0.08, 0.15, 0.25, 0.40, 0.60)))

    # Oracle trajectory.
    if action == "BIND":
        trajectory_actions = [
            {"action": "REUSE", "source": None, "target": chain[j], "relation": None}
            for j in range(depth - 1)
        ]
        trajectory_actions.append({
            "action": "BIND",
            "source": chain[-2],
            "target": chain[-1],
            "relation": relations[-1],
        })
    elif action == "BRANCH":
        trajectory_actions = [
            {"action": "REUSE", "source": None, "target": chain[j], "relation": None}
            for j in range(depth)
        ]
        trajectory_actions.append({
            "action": "BRANCH",
            "source": chain[0],
            "target": None,
            "relation": relations[0],
        })
    elif action == "INHIBIT":
        trajectory_actions = [
            {"action": "REUSE", "source": None, "target": chain[j], "relation": None}
            for j in range(depth - 1)
        ]
        trajectory_actions.append({
            "action": "INHIBIT",
            "source": None,
            "target": chain[-1],
            "relation": None,
        })
    elif action == "REUSE":
        trajectory_actions = [
            {"action": "REUSE", "source": None, "target": chain[j], "relation": None}
            for j in range(depth)
        ]
    elif action == "CREATE":
        trajectory_actions = [
            {"action": "REUSE", "source": None, "target": chain[j], "relation": None}
            for j in range(depth - 1)
        ]
        trajectory_actions.append({
            "action": "CREATE", "source": None, "target": None, "relation": None
        })
    elif action == "COMMIT":
        trajectory_actions = [
            {"action": "REUSE", "source": None, "target": chain[j], "relation": None}
            for j in range(depth - 1)
        ]
        trajectory_actions.append({
            "action": "COMMIT", "source": None, "target": None, "relation": None
        })
    else:
        trajectory_actions = [{
            "action": "NOOP", "source": None, "target": None, "relation": None
        }]

    trajectory_states = []
    trajectory_attention = []

    current = initial
    for t, _ in enumerate(trajectory_actions):
        # Increasing prefix makes the attention target change as cognition advances.
        upto = min(t + 2, len(chain))
        trajectory_states.append(current.signature())
        trajectory_attention.append(chain[:upto])

        step = trajectory_actions[t]
        current = current.apply(
            ACTION_TO_ID[step["action"]],
            source=step["source"],
            target=step["target"],
            relation=step["relation"],
        )

    # Final decision task used by the depth-only family.
    final_action = trajectory_actions[-1]

    return {
        "version": "v224",
        "case_id": f"v224_{index:05d}_{action.lower()}",
        "initial_state": initial.signature(),
        "goal": {
            "source": chain[0],
            "target": chain[-1],
            "relation": relations[-1],
            "depth": depth,
        },
        "trajectory_states": trajectory_states,
        "trajectory_attention": trajectory_attention,
        "trajectory_actions": trajectory_actions,
        "final_action": final_action,
        "reasoning_chain": chain,
        "reasoning_relations": relations,
        "chain_depth": depth,
    }


def generate_dataset(samples=500, seed=216):
    rng = random.Random(seed)
    q, r = divmod(samples, len(ACTIONS))

    rows = []
    for i, action in enumerate(ACTIONS):
        for _ in range(q + int(i < r)):
            rows.append(make_example(action, rng, len(rows)))

    rng.shuffle(rows)
    return rows


def save_dataset(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
