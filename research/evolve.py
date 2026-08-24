from __future__ import annotations

import copy
import random
from dataclasses import dataclass

from simulator import Config, Network
from genome import GENOME


TRAINING = [
    "CAT",
    "CAR",
    "CAN",
    "CARD",
    "CART",
    "DOG",
    "DOT",
    "BAT",
]


BASELINE = {
    "accuracy": 1.0,
    "cells": 18,
    "synapses": 31,
    "reward": 126.5,
}


@dataclass
class Result:
    genome: dict
    score: float
    accuracy: float
    reward: float
    cells: int
    synapses: int
    correct_reuse: int
    correct_branch: int
    wrong_reuse: int
    wrong_branch: int


def config_from_genome(genome: dict) -> Config:
    """
    Convert genome parameters into the EXISTING simulator's
    parameters.

    The simulator itself remains untouched.
    """

    connection = genome["connection"]
    inhibition = genome["inhibition"]
    reuse = genome["reuse"]

    return Config(
        designer_learning_rate=connection["learning_rate"],

        # Existing simulator parameters.
        spike_threshold=1.0,
        leak=genome["ordering"]["decay"],

        excite_weight=connection["initial_strength"],
        inhibit_weight=inhibition["strength"],

        reward_correct_reuse=reuse["reuse_reward"],
        reward_correct_branch=1.0,

        reward_wrong_reuse=-1.0,
        reward_wrong_branch=-0.25,

        branch_cost=-0.25,

        max_designer_cells=32,
    )


def evaluate_genome(genome: dict) -> Result:
    config = config_from_genome(genome)

    network = Network(config)

    network.train(
        TRAINING,
        epochs=5,
    )

    total_decisions = (
        network.correct_reuse
        + network.correct_branch
        + network.wrong_reuse
        + network.wrong_branch
    )

    correct = (
        network.correct_reuse
        + network.correct_branch
    )

    accuracy = (
        correct / total_decisions
        if total_decisions
        else 0.0
    )

    cells = len(network.cells)
    synapses = len(network.synapses)

    # ------------------------------------------------------------
    # Fitness hierarchy
    #
    # Correctness dominates EVERYTHING.
    #
    # Then reward.
    #
    # Then compactness.
    # ------------------------------------------------------------

    score = (
        accuracy * 1_000_000.0
        + network.total_reward * 100.0
        - cells * 10.0
        - synapses * 2.5
    )

    return Result(
        genome=copy.deepcopy(genome),
        score=score,
        accuracy=accuracy,
        reward=network.total_reward,
        cells=cells,
        synapses=synapses,
        correct_reuse=network.correct_reuse,
        correct_branch=network.correct_branch,
        wrong_reuse=network.wrong_reuse,
        wrong_branch=network.wrong_branch,
    )


def mutate(
    genome: dict,
    rng: random.Random,
) -> dict:
    """
    Mutate DEVELOPMENTAL PARAMETERS only.

    Never mutate topology.
    """

    child = copy.deepcopy(genome)

    mutation_rate = child["evolution"]["mutation_rate"]
    mutation_scale = child["evolution"]["mutation_scale"]

    def mutate_float(
        section: str,
        key: str,
        minimum: float,
        maximum: float,
    ):
        if rng.random() > mutation_rate:
            return

        value = child[section][key]

        delta = rng.uniform(
            -mutation_scale,
            mutation_scale,
        )

        value += delta

        child[section][key] = max(
            minimum,
            min(maximum, value),
        )

    mutate_float(
        "connection",
        "initial_strength",
        0.05,
        2.0,
    )

    mutate_float(
        "connection",
        "learning_rate",
        0.001,
        0.5,
    )

    mutate_float(
        "ordering",
        "decay",
        0.50,
        0.999,
    )

    mutate_float(
        "ordering",
        "sequence_strength",
        0.05,
        2.0,
    )

    mutate_float(
        "inhibition",
        "strength",
        0.05,
        1.5,
    )

    mutate_float(
        "reuse",
        "match_threshold",
        0.05,
        1.0,
    )

    mutate_float(
        "reuse",
        "reuse_reward",
        0.05,
        3.0,
    )

    mutate_float(
        "growth",
        "growth_threshold",
        0.05,
        1.5,
    )

    mutate_float(
        "growth",
        "growth_cost",
        0.0,
        3.0,
    )

    return child


def genome_signature(genome: dict) -> str:
    return repr(genome)


def run():
    seed = 42

    rng = random.Random(seed)

    population_size = (
        GENOME
        .get("evolution", {})
        .get("population_size", 16)
    )

    generations = (
        GENOME
        .get("evolution", {})
        .get("generations", 50)
    )

    elite_fraction = (
        GENOME
        .get("evolution", {})
        .get("elite_fraction", 0.25)
    )

    mutation_rate = (
        GENOME
        .get("evolution", {})
        .get("mutation_rate", 0.15)
    )

    mutation_scale = (
        GENOME
        .get("evolution", {})
        .get("mutation_scale", 0.10)
    )

    print()
    print("=== GENOME EVOLUTION EXPERIMENT ===")
    print(f"population : {population_size}")
    print(f"generations: {generations}")
    print(f"seed       : {seed}")
    print()
    print("=== BASELINE ===")
    print(f"accuracy   : {BASELINE['accuracy']:.2%}")
    print(f"cells      : {BASELINE['cells']}")
    print(f"synapses   : {BASELINE['synapses']}")
    print(f"reward     : {BASELINE['reward']:.2f}")
    print()

    # ------------------------------------------------------------
    # Initial population.
    #
    # Start around the current known-good genome.
    # ------------------------------------------------------------

    population = []

    for _ in range(population_size):
        population.append(
            mutate(
                GENOME,
                rng,
            )
        )

    best: Result | None = None

    # ------------------------------------------------------------
    # Evolution.
    # ------------------------------------------------------------

    for generation in range(1, generations + 1):

        results = [
            evaluate_genome(genome)
            for genome in population
        ]

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        generation_best = results[0]

        if (
            best is None
            or generation_best.score > best.score
        ):
            best = generation_best

        print(
            f"generation={generation:3d} "
            f"score={generation_best.score:12.2f} "
            f"accuracy={generation_best.accuracy:7.2%} "
            f"reward={generation_best.reward:8.2f} "
            f"cells={generation_best.cells:3d} "
            f"synapses={generation_best.synapses:3d} "
            f"reuse={generation_best.correct_reuse:3d} "
            f"branch={generation_best.correct_branch:3d} "
            f"wrong={generation_best.wrong_reuse + generation_best.wrong_branch:3d}"
        )

        elite_count = max(
            1,
            int(population_size * elite_fraction),
        )

        elites = [
            copy.deepcopy(result.genome)
            for result in results[:elite_count]
        ]

        next_population = []

        # Preserve elites exactly.
        next_population.extend(elites)

        # Fill population through mutation.
        while len(next_population) < population_size:

            parent = rng.choice(elites)

            child = mutate(
                parent,
                rng,
            )

            next_population.append(child)

        population = next_population

    assert best is not None

    print()
    print("=== BEST GENOME ===")
    print(f"score       : {best.score:.2f}")
    print(f"accuracy    : {best.accuracy:.2%}")
    print(f"reward      : {best.reward:.2f}")
    print(f"cells       : {best.cells}")
    print(f"synapses    : {best.synapses}")
    print(f"correct reuse : {best.correct_reuse}")
    print(f"correct branch: {best.correct_branch}")
    print(f"wrong reuse   : {best.wrong_reuse}")
    print(f"wrong branch  : {best.wrong_branch}")
    print()

    print("=== BEST GENOME PARAMETERS ===")
    print()

    for section, values in best.genome.items():
        print(f"[{section}]")

        if isinstance(values, dict):
            for key, value in values.items():
                print(f"  {key} = {value}")

        else:
            print(f"  {values}")

        print()

    print("=== BASELINE COMPARISON ===")

    print(
        f"accuracy: "
        f"{BASELINE['accuracy']:.2%}"
        f" -> "
        f"{best.accuracy:.2%}"
    )

    print(
        f"cells: "
        f"{BASELINE['cells']}"
        f" -> "
        f"{best.cells}"
    )

    print(
        f"synapses: "
        f"{BASELINE['synapses']}"
        f" -> "
        f"{best.synapses}"
    )

    print(
        f"reward: "
        f"{BASELINE['reward']:.2f}"
        f" -> "
        f"{best.reward:.2f}"
    )

    print()

    if (
        best.accuracy >= 1.0
        and best.cells < BASELINE["cells"]
    ):
        print("🏆 NEW CHAMPION: smaller perfect topology")

    elif (
        best.accuracy >= 1.0
        and best.synapses < BASELINE["synapses"]
    ):
        print("🏆 NEW CHAMPION: fewer synapses")

    elif best.accuracy >= 1.0:
        print("✓ PERFECT: baseline accuracy preserved")

    else:
        print("✗ EVOLUTION LOST PERFECT ACCURACY")


if __name__ == "__main__":
    run()