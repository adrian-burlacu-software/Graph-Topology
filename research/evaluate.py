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
    "CAT",
    "CAR",
    "CAN",
    "CARD",
    "CART",
    "DOG",
    "DOT",
    "BAT",

    "CAD",
    "COD",
    "COT",
    "BAD",
    "BAR",
    "BARD",
    "BAN",
    "DART",
    "DAT",
    "BOT",
    "BOAT",
    "CARTD",
]


EXPECTED_REUSE = 50
EXPECTED_CREATE = 17


def config_from_genome(genome: dict) -> Config:
    d = genome["designer"]
    p = genome["plasticity"]
    i = genome["inhibition"]

    return Config(
        designer_learning_rate=p["reward_learning_rate"],
        vocabulary_learning_rate=p["weight_learning_rate"],

        spike_threshold=d["threshold"],
        leak=d["leak"],

        excite_weight=genome["connection"]["initial_strength"] * 2.0,
        inhibit_weight=max(
            0.05,
            i["strength"] + 0.25,
        ),

        reward_correct_reuse=genome["reuse"]["reuse_reward"],
        reward_correct_branch=1.0,

        reward_wrong_reuse=-1.0,
        reward_wrong_branch=-0.25,

        branch_cost=-0.25 * genome["growth"]["growth_cost"],

        feedback_weight=max(
            0.01,
            min(
                0.5,
                genome["ordering"]["sequence_strength"] * 0.10,
            ),
        ),
    )


def expected_for_word(network: Network, word: str):
    """
    Calculate the structural expectation BEFORE mutating the graph.
    """
    expected_reuse = 0
    expected_create = 0

    current = None

    for symbol in word:
        existing = network.find_child(current, symbol)

        if existing is None:
            expected_create += 1
        else:
            expected_reuse += 1
            current = existing

    return expected_reuse, expected_create


def run_once(genome: dict, verbose: bool = True):
    config = config_from_genome(genome)
    network = Network(config)

    if verbose:
        print("=== TRAINING ===")
        print()
        print(
            "genome:",
            genome_summary(genome),
        )

    network.train(TRAINING, epochs=5)

    if verbose:
        print()
        print("=== FREEZE ===")
        print("No learning after training.")

    network.config.designer_learning_rate = 0.0
    network.config.vocabulary_learning_rate = 0.0

    before_cells = len(network.cells)

    total_reuse = 0
    total_create = 0
    exact_words = 0

    if verbose:
        print()
        print("=== NOVEL TEST ===")

    for word in TEST:
        expected_reuse, expected_create = expected_for_word(
            network,
            word,
        )

        before_reuse = network.total_reuse
        before_create = network.total_create

        result = network.process_word(word)

        word_reuse = network.total_reuse - before_reuse
        word_create = network.total_create - before_create

        total_reuse += word_reuse
        total_create += word_create

        exact = (
            word_reuse == expected_reuse
            and word_create == expected_create
        )

        if exact:
            exact_words += 1

        if verbose:
            print(
                f"{word:5s} "
                f"reuse={word_reuse:2d} "
                f"create={word_create:2d} "
                f"expected_reuse={expected_reuse:2d} "
                f"expected_create={expected_create:2d} "
                f"exact={exact}"
            )

    after_cells = len(network.cells)

    reuse_error = abs(
        total_reuse - EXPECTED_REUSE
    )

    create_error = abs(
        total_create - EXPECTED_CREATE
    )

    # Exact structural generalization is the primary fitness.
    score = (
        exact_words * 100.0
        - reuse_error * 10.0
        - create_error * 10.0
    )

    result = {
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

    if verbose:
        print()
        print("=== GENERALIZATION ===")
        print(f"training_words       : {len(TRAINING)}")
        print(f"test_words           : {len(TEST)}")
        print(f"cells_before_test    : {before_cells}")
        print(f"cells_after_test     : {after_cells}")
        print(f"new_cells            : {after_cells - before_cells}")
        print(f"test_reuse           : {total_reuse}")
        print(f"test_create          : {total_create}")
        print(f"expected_reuse       : {EXPECTED_REUSE}")
        print(f"expected_create      : {EXPECTED_CREATE}")
        print(f"exact_words          : {exact_words}/{len(TEST)}")

    return result


def evolve(
    generations: int = 10,
    population_size: int = 8,
    seed: int = 7,
):
    random.seed(seed)

    base = clone_genome()

    population = [clone_genome(base)]

    for _ in range(population_size - 1):
        population.append(
            mutate_genome(
                base,
                mutation_rate=0.30,
                mutation_scale=0.08,
            )
        )

    best_genome = None
    best_score = float("-inf")

    print("=== GENOME EXPERIMENT ===")
    print(f"generations : {generations}")
    print(f"population  : {population_size}")
    print(f"seed        : {seed}")
    print()

    for generation in range(1, generations + 1):
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
            key=lambda item: item[0],
            reverse=True,
        )

        score, champion, result = scored[0]

        if score > best_score:
            best_score = score
            best_genome = clone_genome(champion)

        print(
            f"generation={generation:2d} "
            f"score={score:7.1f} "
            f"exact={result['exact_words']:2d}/{len(TEST)} "
            f"reuse={result['reuse']:2d} "
            f"create={result['create']:2d}"
        )

        print(
            "  ",
            genome_summary(champion),
        )

        elite_count = max(
            1,
            int(population_size * 0.25),
        )

        elites = [
            clone_genome(item[1])
            for item in scored[:elite_count]
        ]

        next_population = list(elites)

        while len(next_population) < population_size:
            parent = random.choice(elites)

            child = mutate_genome(
                parent,
                mutation_rate=0.25,
                mutation_scale=0.05,
            )

            next_population.append(child)

        population = next_population

    print()
    print("=== BEST GENOME ===")
    print(genome_summary(best_genome))
    print(f"score : {best_score:.1f}")

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