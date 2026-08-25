from __future__ import annotations

import random
from copy import deepcopy

from genome import GENOME, clone_genome, mutate_genome, genome_summary
from evaluate_dual_vocabulary_v6 import DualVocabularyV6


TRAINING = [
    "CAT", "CAR", "CAN", "CARD", "CART", "DOG", "DOT", "BAT",
]

# Evolution sees this validation set.
# The final TEST set is never used to select genomes.
VALIDATION = [
    "CAD", "COD", "COT", "BAD", "BAR",
    "BARD", "BAN", "DART", "DAT", "BOT",
    "BOAT", "CARTD",
]

FINAL_TEST = [
    "CAT", "CAR", "CAN", "CARD", "CART",
    "CAD", "COD", "COT", "BAD", "BAR", "BARD", "BAN",
    "DART", "DAT", "BOT", "BOAT", "CARTD",
    "COARD", "BAND", "BOARD",
]

POPULATION_SIZE = 16
GENERATIONS = 20
ELITE_FRACTION = 0.20
SEED = 23


def train_genome(genome):
    net = DualVocabularyV6(genome)
    net.train(TRAINING, epochs=5)
    return net


def score_genome(genome):
    """
    Fitness is based ONLY on the held-out validation set.

    The structural graph is learned from TRAINING first. Validation is
    frozen: no learning and no structural mutation.

    Fitness strongly rewards exact boundary decisions and lightly rewards
    compact, stable behavior. It does not use FINAL_TEST.
    """
    net = train_genome(genome)

    correct = 0
    total = 0
    exact_words = 0

    for word in VALIDATION:
        word_correct = True

        for pos in range(len(word)):
            available = net.available(word, pos)
            action = net.designer_action(available, learn=False)

            expected = "reuse" if available else "branch"
            actual = "reuse" if action == "REUSE" else "branch"

            if actual == expected:
                correct += 1
            else:
                word_correct = False

            total += 1

        if word_correct:
            exact_words += 1

    # Primary objective: position accuracy.
    # Secondary objective: whole-word exactness.
    position_accuracy = correct / total if total else 0.0
    word_accuracy = exact_words / len(VALIDATION)

    fitness = (
        position_accuracy * 1000.0
        + word_accuracy * 100.0
    )

    return {
        "fitness": fitness,
        "position_accuracy": position_accuracy,
        "word_accuracy": word_accuracy,
        "correct": correct,
        "total": total,
        "exact_words": exact_words,
        "net": net,
    }


def make_population(seed_genome):
    population = [clone_genome(seed_genome)]

    while len(population) < POPULATION_SIZE:
        population.append(
            mutate_genome(
                seed_genome,
                mutation_rate=seed_genome["evolution"]["mutation_rate"],
                mutation_scale=seed_genome["evolution"]["mutation_scale"],
            )
        )

    return population


def evolve():
    random.seed(SEED)

    # Start from the checked-in genome, but do NOT modify it.
    seed = clone_genome(GENOME)

    population = make_population(seed)
    elite_count = max(1, int(POPULATION_SIZE * ELITE_FRACTION))

    best_genome = None
    best_result = None

    print("=== EVOLUTION: DUAL VOCABULARY V6 ===")
    print()
    print(f"population_size : {POPULATION_SIZE}")
    print(f"generations     : {GENERATIONS}")
    print(f"elite_fraction  : {ELITE_FRACTION:.2f}")
    print(f"seed            : {SEED}")
    print()
    print("Fitness uses VALIDATION only.")
    print("FINAL_TEST is completely withheld until evolution finishes.")
    print()

    for generation in range(1, GENERATIONS + 1):
        scored = []

        for genome in population:
            result = score_genome(genome)
            scored.append((result["fitness"], genome, result))

        scored.sort(key=lambda item: item[0], reverse=True)

        generation_best = scored[0]
        fitness, genome, result = generation_best

        if best_result is None or fitness > best_result["fitness"]:
            best_genome = clone_genome(genome)
            best_result = {
                key: value
                for key, value in result.items()
                if key != "net"
            }

        print(
            f"generation={generation:3d} "
            f"fitness={fitness:9.2f} "
            f"positions={result['correct']:2d}/{result['total']:2d} "
            f"words={result['exact_words']:2d}/{len(VALIDATION):2d} "
            f"genome={genome_summary(genome)}"
        )

        elites = [
            clone_genome(item[1])
            for item in scored[:elite_count]
        ]

        next_population = elites[:]

        while len(next_population) < POPULATION_SIZE:
            parent = random.choice(elites)

            child = mutate_genome(
                parent,
                mutation_rate=parent["evolution"]["mutation_rate"],
                mutation_scale=parent["evolution"]["mutation_scale"],
            )

            # Keep the evolutionary controller itself stable enough that
            # one mutation cannot collapse the population into zero/huge
            # generations or population sizes.
            child["evolution"]["population_size"] = POPULATION_SIZE
            child["evolution"]["generations"] = GENERATIONS
            child["evolution"]["elite_fraction"] = ELITE_FRACTION

            next_population.append(child)

        population = next_population

    print()
    print("=== BEST EVOLVED GENOME ===")
    print(genome_summary(best_genome))

    print()
    print("=== BEST GENOME PARAMETERS ===")
    print(best_genome)

    print()
    print("=== VALIDATION SCORE ===")
    print(
        f"fitness           : {best_result['fitness']:.2f}"
    )
    print(
        f"positions         : "
        f"{best_result['correct']}/{best_result['total']}"
    )
    print(
        f"exact_words       : "
        f"{best_result['exact_words']}/{len(VALIDATION)}"
    )

    print()
    print("=== FINAL TEST WITH EVOLVED GENOME ===")

    final_net = train_genome(best_genome)

    links_before = len(final_net.boundaries.links)
    prefix_before = final_net.prefix.next_id
    suffix_before = final_net.suffix.next_id

    exact_words = 0
    total_positions = 0
    correct_positions = 0

    for word in FINAL_TEST:
        expected_reuse = 0
        expected_branch = 0
        actual_reuse = 0
        actual_branch = 0

        for pos in range(len(word)):
            available = final_net.available(word, pos)
            action = final_net.designer_action(available, learn=False)

            if available:
                expected_reuse += 1
            else:
                expected_branch += 1

            if action == "REUSE":
                actual_reuse += 1
            else:
                actual_branch += 1

            total_positions += 1

            expected = "REUSE" if available else "BRANCH"
            if action == expected:
                correct_positions += 1

        exact = (
            actual_reuse == expected_reuse
            and actual_branch == expected_branch
        )

        if exact:
            exact_words += 1

        print(
            f"{word:6s} "
            f"available_reuse={expected_reuse:2d} "
            f"available_branch={expected_branch:2d} "
            f"designer_reuse={actual_reuse:2d} "
            f"designer_branch={actual_branch:2d} "
            f"exact={exact}"
        )

    print()
    print("=== FINAL GENERALIZATION ===")
    print(f"test_words               : {len(FINAL_TEST)}")
    print(f"exact_words              : {exact_words}/{len(FINAL_TEST)}")
    print(f"correct_positions        : {correct_positions}/{total_positions}")
    print(
        f"position_accuracy        : "
        f"{correct_positions / total_positions:.4f}"
    )

    print()
    print("=== FROZEN INVARIANTS ===")
    print(f"boundary_links_before    : {links_before}")
    print(f"boundary_links_after     : {len(final_net.boundaries.links)}")
    print(f"prefix_nodes_before      : {prefix_before}")
    print(f"prefix_nodes_after       : {final_net.prefix.next_id}")
    print(f"suffix_nodes_before      : {suffix_before}")
    print(f"suffix_nodes_after       : {final_net.suffix.next_id}")

    print()
    print("=== FINAL LEARNED NETWORK ===")
    print(f"designer_spikes          : {final_net.net.designer_spikes}")
    print(f"total_reward             : {final_net.total_reward:.2f}")
    print(f"action_reuse             : {final_net.action_reuse}")
    print(f"action_branch            : {final_net.action_branch}")
    print(f"correct_reuse            : {final_net.correct_reuse}")
    print(f"correct_branch           : {final_net.correct_branch}")
    print(f"wrong_reuse              : {final_net.wrong_reuse}")
    print(f"wrong_branch             : {final_net.wrong_branch}")

    return best_genome


if __name__ == "__main__":
    evolve()
