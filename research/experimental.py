from __future__ import annotations

import copy
import random
from dataclasses import dataclass

from genome import GENOME
from simulator import Config, Network


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


@dataclass
class Result:
    score: float
    accuracy: float
    reuse: int
    branch: int
    cells: int
    genome: dict


def make_config(genome: dict) -> Config:
    return Config(
        designer_learning_rate=0.05,
        vocabulary_learning_rate=0.05,

        spike_threshold=1.0,
        leak=0.90,

        excite_weight=1.0,
        inhibit_weight=0.6,

        reward_correct_reuse=1.0,
        reward_correct_branch=1.0,
        reward_wrong_reuse=-1.0,
        reward_wrong_branch=-0.25,

        branch_cost=-0.25,

        max_designer_cells=32,
    )


def evaluate_genome(genome: dict) -> Result:
    network = Network(make_config(genome))

    network.train(
        TRAINING,
        epochs=5,
    )

    correct = (
        network.correct_reuse
        + network.correct_branch
    )

    wrong = (
        network.wrong_reuse
        + network.wrong_branch
    )

    total = correct + wrong

    accuracy = (
        correct / total
        if total
        else 0.0
    )

    # We want:
    #
    #   high correctness
    #   high reward
    #   compact topology
    #
    # Correctness dominates.
    score = (
        network.total_reward
        + accuracy * 100.0
        - max(0, len(network.cells) - 32) * 0.25
    )

    return Result(
        score=score,
        accuracy=accuracy,
        reuse=network.correct_reuse,
        branch=network.correct_branch,
        cells=len(network.cells),
        genome=copy.deepcopy(genome),
    )


def mutate(
    genome: dict,
    rng: random.Random,
) -> dict:
    child = copy.deepcopy(genome)

    rate = child["evolution"]["mutation_rate"]
    scale = child["evolution"]["mutation_scale"]

    def mutate_number(
        section: str,
        key: str,
        minimum: float,
        maximum: float,
    ):
        if rng.random() >= rate:
            return

        value = child[section][key]

        value += rng.uniform(
            -scale,
            scale,
        )

        child[section][key] = max(
            minimum,
            min(maximum, value),
        )

    mutate_number(
        "growth",
        "growth_threshold",
        0.05,
        1.5,
    )

    mutate_number(
        "connection",
        "initial_strength",
        0.05,
        2.0,
    )

    mutate_number(
        "connection",
        "learning_rate",
        0.001,
        0.5,
    )

    mutate_number(
        "ordering",
        "decay",
        0.5,
        1.0,
    )

    mutate_number(
        "ordering",
        "sequence_strength",
        0.05,
        2.0,
    )

    mutate_number(
        "inhibition",
        "strength",
        0.05,
        1.5,
    )

    mutate_number(
        "reuse",
        "match_threshold",
        0.1,
        1.0,
    )

    mutate_number(
        "reuse",
        "reuse_reward",
        0.1,
        3.0,
    )

    return child


def run():
    seed = GENOME["evolution"].get(
        "seed",
        42,
    )

    rng = random.Random(seed)

    population_size = GENOME["evolution"][
        "population_size"
    ]

    generations = GENOME["evolution"][
        "generations"
    ]

    elite_fraction = GENOME["evolution"][
        "elite_fraction"
    ]

    population = [
        copy.deepcopy(GENOME)
        for _ in range(population_size)
    ]

    print()
    print("=== GENOME EVOLUTION EXPERIMENT ===")
    print(f"population : {population_size}")
    print(f"generations: {generations}")
    print()

    best = None

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
            f"score={generation_best.score:8.2f} "
            f"accuracy={generation_best.accuracy:6.2%} "
            f"reuse={generation_best.reuse:3d} "
            f"branch={generation_best.branch:3d} "
            f"cells={generation_best.cells:3d}"
        )

        elite_count = max(
            1,
            int(population_size * elite_fraction),
        )

        elites = [
            copy.deepcopy(result.genome)
            for result in results[:elite_count]
        ]

        next_population = elites.copy()

        while len(next_population) < population_size:
            parent = rng.choice(elites)

            child = mutate(
                parent,
                rng,
            )

            next_population.append(child)

        population = next_population

    print()
    print("=== BEST GENOME ===")
    print(f"score    : {best.score:.4f}")
    print(f"accuracy : {best.accuracy:.4%}")
    print(f"reuse    : {best.reuse}")
    print(f"branch   : {best.branch}")
    print(f"cells    : {best.cells}")
    print()

    print(best.genome)


if __name__ == "__main__":
    run()