from __future__ import annotations

import random

from simulator import Network, Config
from genome import GENOME, clone_genome, mutate_genome, genome_summary


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

TEST = [
    "BAT",
    "BAR",
    "BOAT",
    "BOAR",
    "BOATD",
    "CAB",
    "COAT",
    "COAR",
    "CART",
    "CARTD",
    "BART",
    "BARD",
    "BOARD"
]

EXPECTED_REUSE = 50
EXPECTED_CREATE = 17


def config_from_genome(genome: dict) -> Config:
    """
    The simulator is already genome-driven.

    Do NOT flatten the genome into Config fields here.
    Network expects the complete nested genome:
        genome["designer"]
        genome["growth"]
        genome["connection"]
        ...
    """

    return Config(
        genome=genome,
    )


def expected_for_word(
    network: Network,
    word: str,
):
    """
    Determine the structural expectation before mutation.
    """

    reuse = 0
    create = 0

    current = None

    for symbol in word:
        existing = network.find_child(
            current,
            symbol,
        )

        if existing is None:
            create += 1
        else:
            reuse += 1
            current = existing

    return reuse, create


def run_once(
    genome: dict,
    verbose: bool = True,
):
    config = config_from_genome(
        genome
    )

    network = Network(
        config
    )

    if verbose:
        print(
            "genome:",
            genome_summary(genome),
        )

    print()
    print("=== TRAINING ===")

    network.train(
        TRAINING,
        epochs=5,
    )

    print()
    print("=== FREEZE ===")
    print("No learning after training.")

    network.config.designer_learning_rate = 0.0
    network.config.vocabulary_learning_rate = 0.0

    before_cells = len(
        network.cells
    )

    total_reuse = 0
    total_create = 0
    exact_words = 0

    print()
    print("=== NOVEL TEST ===")

    for word in TEST:

        expected_reuse, expected_create = (
            expected_for_word(
                network,
                word,
            )
        )

        reuse_before = (
            network.total_reuse
        )

        create_before = (
            network.total_create
        )

        result = network.process_word(
            word,
            learn=False,
        )

        word_reuse = (
            network.total_reuse
            - reuse_before
        )

        word_create = (
            network.total_create
            - create_before
        )

        total_reuse += word_reuse
        total_create += word_create

        exact = (
            word_reuse == expected_reuse
            and word_create == expected_create
        )

        if exact:
            exact_words += 1

        print(
            f"{word:5s} "
            f"reuse={word_reuse:2d} "
            f"create={word_create:2d} "
            f"expected_reuse={expected_reuse:2d} "
            f"expected_create={expected_create:2d} "
            f"exact={exact}"
        )

    after_cells = len(
        network.cells
    )

    reuse_error = abs(
        total_reuse
        - EXPECTED_REUSE
    )

    create_error = abs(
        total_create
        - EXPECTED_CREATE
    )

    score = (
        exact_words * 100.0
        - reuse_error * 10.0
        - create_error * 10.0
    )

    print()
    print("=== GENERALIZATION ===")

    print(
        f"training_words       : "
        f"{len(TRAINING)}"
    )

    print(
        f"test_words           : "
        f"{len(TEST)}"
    )

    print(
        f"cells_before_test    : "
        f"{before_cells}"
    )

    print(
        f"cells_after_test     : "
        f"{after_cells}"
    )

    print(
        f"new_cells            : "
        f"{after_cells - before_cells}"
    )

    print(
        f"test_reuse           : "
        f"{total_reuse}"
    )

    print(
        f"test_create          : "
        f"{total_create}"
    )

    print(
        f"expected_reuse       : "
        f"{EXPECTED_REUSE}"
    )

    print(
        f"expected_create      : "
        f"{EXPECTED_CREATE}"
    )

    print(
        f"exact_words          : "
        f"{exact_words}/{len(TEST)}"
    )

    return {
        "score": score,
        "exact_words": exact_words,
        "reuse": total_reuse,
        "create": total_create,
        "reuse_error": reuse_error,
        "create_error": create_error,
        "before_cells": before_cells,
        "after_cells": after_cells,
        "network": network,
    }


def evolve(
    generations: int = 10,
    population_size: int = 8,
    seed: int = 7,
):
    random.seed(seed)

    base = clone_genome()

    population = [
        clone_genome(base)
    ]

    for _ in range(
        population_size - 1
    ):
        population.append(
            mutate_genome(
                base,
                mutation_rate=0.30,
                mutation_scale=0.08,
            )
        )

    best_genome = None
    best_score = float("-inf")

    print()
    print("=== GENOME EXPERIMENT ===")
    print(
        f"generations : {generations}"
    )
    print(
        f"population  : {population_size}"
    )
    print(
        f"seed        : {seed}"
    )
    print()

    for generation in range(
        1,
        generations + 1,
    ):
        scored = []

        for genome in population:

            result = run_once(
                genome,
                verbose=False,
            )

            scored.append(
                (
                    result["score"],
                    genome,
                    result,
                )
            )

        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        score, champion, result = (
            scored[0]
        )

        if score > best_score:
            best_score = score
            best_genome = clone_genome(
                champion
            )

        print(
            f"generation={generation:2d} "
            f"score={score:7.1f} "
            f"exact={result['exact_words']:2d}/{len(TEST)} "
            f"reuse={result['reuse']:2d} "
            f"create={result['create']:2d}"
        )

        print(
            "  ",
            genome_summary(
                champion
            ),
        )

        elite_count = max(
            1,
            int(
                population_size
                * GENOME["evolution"]["elite_fraction"]
            ),
        )

        elites = [
            clone_genome(
                item[1]
            )
            for item in scored[
                :elite_count
            ]
        ]

        next_population = list(
            elites
        )

        while len(
            next_population
        ) < population_size:

            parent = random.choice(
                elites
            )

            child = mutate_genome(
                parent
            )

            next_population.append(
                child
            )

        population = (
            next_population
        )

    print()
    print("=== BEST GENOME ===")

    print(
        genome_summary(
            best_genome
        )
    )

    print(
        f"score : {best_score:.1f}"
    )

    return best_genome


def evaluate():

    print("=== BASELINE ===")

    baseline = run_once(
        clone_genome(),
        verbose=True,
    )

    print()
    print()
    print("=== EVOLUTION ===")

    best = evolve(
        generations=10,
        population_size=8,
        seed=7,
    )

    print()
    print()
    print("=== BEST GENOME REPLAY ===")

    final = run_once(
        best,
        verbose=True,
    )

    print()
    print("=== LEARNED NETWORK ===")

    final["network"].print_summary()

    print()
    final["network"].print_vocabulary_tree()


if __name__ == "__main__":
    evaluate()