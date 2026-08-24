from __future__ import annotations

from copy import deepcopy
import random


GENOME = {
    "growth": {
        "max_children_per_cell": 2,
        "growth_threshold": 0.55,
        "growth_cost": 1.0,
    },

    "connection": {
        "initial_strength": 0.50,
        "learning_rate": 0.08,
        "max_connections_per_cell": 8,
    },

    "ordering": {
        "decay": 0.90,
        "sequence_strength": 1.0,
    },

    "inhibition": {
        "strength": 0.35,
        "radius": 1,
    },

    "reuse": {
        "match_threshold": 0.70,
        "reuse_reward": 1.0,
    },

    "pruning": {
        "minimum_strength": 0.05,
        "minimum_activity": 0.01,
    },

    "designer": {
        "input_gain": 0.80,
        "match_gain": 1.35,
        "context_gain": 0.20,
        "branch_bias": 0.45,
        "reuse_bias": 0.00,
        "decision_margin": 0.05,
        "leak": 0.90,
        "threshold": 1.0,
    },

    "plasticity": {
        "reward_learning_rate": 0.08,
        "weight_learning_rate": 0.05,
        "eligibility_decay": 0.90,
    },

    "evolution": {
        "mutation_rate": 0.10,
        "mutation_scale": 0.10,
        "elite_fraction": 0.20,
        "population_size": 16,
        "generations": 20,
    },
}


def clone_genome(genome: dict | None = None) -> dict:
    return deepcopy(GENOME if genome is None else genome)


def _mutate_value(value: float, scale: float) -> float:
    return value + random.uniform(-scale, scale)


def mutate_genome(
    genome: dict | None = None,
    mutation_rate: float | None = None,
    mutation_scale: float | None = None,
) -> dict:
    result = clone_genome(genome)

    evolution = result["evolution"]

    rate = (
        evolution["mutation_rate"]
        if mutation_rate is None
        else mutation_rate
    )

    scale = (
        evolution["mutation_scale"]
        if mutation_scale is None
        else mutation_scale
    )

    def mutate_tree(node):
        for key, value in list(node.items()):
            if isinstance(value, dict):
                mutate_tree(value)
                continue

            if not isinstance(value, (int, float)):
                continue

            if random.random() > rate:
                continue

            mutated = _mutate_value(float(value), scale)

            if isinstance(value, int):
                node[key] = int(round(mutated))
            else:
                node[key] = mutated

    mutate_tree(result)

    # Hard validity limits.
    result["growth"]["max_children_per_cell"] = max(
        1,
        int(result["growth"]["max_children_per_cell"]),
    )

    result["connection"]["max_connections_per_cell"] = max(
        1,
        int(result["connection"]["max_connections_per_cell"]),
    )

    result["inhibition"]["radius"] = max(
        0,
        int(result["inhibition"]["radius"]),
    )

    result["designer"]["threshold"] = max(
        0.1,
        result["designer"]["threshold"],
    )

    result["designer"]["leak"] = min(
        0.999,
        max(0.0, result["designer"]["leak"]),
    )

    result["designer"]["decision_margin"] = max(
        0.0,
        result["designer"]["decision_margin"],
    )

    result["reuse"]["match_threshold"] = min(
        1.0,
        max(0.0, result["reuse"]["match_threshold"]),
    )

    result["plasticity"]["reward_learning_rate"] = max(
        0.0,
        result["plasticity"]["reward_learning_rate"],
    )

    result["plasticity"]["weight_learning_rate"] = max(
        0.0,
        result["plasticity"]["weight_learning_rate"],
    )

    return result


def genome_distance(a: dict, b: dict) -> float:
    total = 0.0
    count = 0

    def walk(x, y):
        nonlocal total, count

        for key in x:
            if key not in y:
                continue

            xv = x[key]
            yv = y[key]

            if isinstance(xv, dict) and isinstance(yv, dict):
                walk(xv, yv)

            elif isinstance(xv, (int, float)) and isinstance(yv, (int, float)):
                total += abs(float(xv) - float(yv))
                count += 1

    walk(a, b)

    return total / count if count else 0.0


def genome_summary(genome: dict) -> str:
    d = genome["designer"]
    p = genome["plasticity"]
    r = genome["reuse"]

    return (
        f"match={r['match_threshold']:.3f} "
        f"input={d['input_gain']:.3f} "
        f"match_gain={d['match_gain']:.3f} "
        f"branch={d['branch_bias']:.3f} "
        f"reuse={d['reuse_bias']:.3f} "
        f"margin={d['decision_margin']:.3f} "
        f"leak={d['leak']:.3f} "
        f"lr={p['reward_learning_rate']:.3f}"
    )