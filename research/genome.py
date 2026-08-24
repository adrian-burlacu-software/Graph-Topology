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

    # Experimental genome parameters.
    #
    # These affect the actual simulator dynamics rather than merely
    # describing an external evolutionary process.
    "designer": {
        "input_gain": 0.80,
        "match_gain": 1.45,
        "context_gain": 0.20,
        "branch_bias": 0.45,
        "reuse_bias": 0.00,
        "decision_margin": 0.10,
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
    """
    Produce a mutated genome.

    This is deliberately generic: every numeric leaf has a chance to
    mutate, while integer/bounded parameters are kept valid.
    """
    result = clone_genome(genome)

    evolution = result["evolution"]
    rate = evolution["mutation_rate"] if mutation_rate is None else mutation_rate
    scale = evolution["mutation_scale"] if mutation_scale is None else mutation_scale

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
                node[key] = max(1, int(round(mutated)))
            else:
                node[key] = mutated

    mutate_tree(result)

    # Keep the genome sane.
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

    return result


def genome_distance(a: dict, b: dict) -> float:
    """
    Simple numeric genome distance.
    Useful for later evolutionary experiments.
    """
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