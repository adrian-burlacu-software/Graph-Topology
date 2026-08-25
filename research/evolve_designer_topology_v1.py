from __future__ import annotations

import random
from copy import deepcopy

from genome import GENOME, clone_genome, genome_summary
from evaluate_dual_vocabulary_v6 import DualVocabularyV6


TRAINING = [
    "CAT", "CAR", "CAN", "CARD", "CART", "DOG", "DOT", "BAT",
]

VALIDATION = [
    "CAD", "COD", "COT", "BAD", "BAR", "BARD",
    "BAN", "DART", "DAT", "BOT", "BOAT", "CARTD",
]

FINAL_TEST = [
    "CAT", "CAR", "CAN", "CARD", "CART",
    "CAD", "COD", "COT", "BAD", "BAR", "BARD", "BAN",
    "DART", "DAT", "BOT", "BOAT", "CARTD",
    "COARD", "BAND", "BOARD",
]

POPULATION = 24
GENERATIONS = 30
ELITE = 6
SEED = 24


# The topology being evolved is the small designer circuit itself.
#
#       0 ROOT
#       /    \
#      1      2
#    REUSE  BRANCH
#      \      /
#       \    /
#       inhibitory competition
#
# The structural dual vocabulary stays fixed. Evolution is now allowed
# to discover how the designer circuit couples those signals.
DEFAULT_TOPOLOGY = {
    "root_to_reuse": 1.0,
    "root_to_branch": 1.0,
    "reuse_to_branch_inhibit": 0.60,
    "branch_to_reuse_inhibit": 0.60,
}


def topology_of(genome):
    g = clone_genome(genome)

    if "designer_topology" not in g:
        g["designer_topology"] = deepcopy(DEFAULT_TOPOLOGY)

    for key, value in DEFAULT_TOPOLOGY.items():
        g["designer_topology"].setdefault(key, value)

    return g


def mutate_topology(genome):
    child = topology_of(genome)

    for key in DEFAULT_TOPOLOGY:
        if random.random() < 0.35:
            child["designer_topology"][key] += random.gauss(0.0, 0.12)

    # Keep the circuit physically meaningful.
    for key in DEFAULT_TOPOLOGY:
        child["designer_topology"][key] = max(
            0.0,
            min(2.5, child["designer_topology"][key]),
        )

    return child


def apply_topology(net, genome):
    t = topology_of(genome)["designer_topology"]

    # Current simulator topology:
    # 0 -> 1 : EXCITE
    # 0 -> 2 : EXCITE
    # 1 -> 2 : INHIBIT
    # 2 -> 1 : INHIBIT
    #
    # We evolve these actual designer synaptic strengths.
    n = net.net

    n.synapses[(n.designer_root, n.reuse_cell)].weight = t["root_to_reuse"]
    n.synapses[(n.designer_root, n.branch_cell)].weight = t["root_to_branch"]

    n.synapses[(n.reuse_cell, n.branch_cell)].weight = (
        t["reuse_to_branch_inhibit"]
    )
    n.synapses[(n.branch_cell, n.reuse_cell)].weight = (
        t["branch_to_reuse_inhibit"]
    )


def build(genome):
    net = DualVocabularyV6(topology_of(genome))
    apply_topology(net, genome)
    net.train(TRAINING, epochs=5)
    return net


def evaluate_words(net, words):
    correct = 0
    total = 0
    exact_words = 0

    for word in words:
        exact = True

        for pos in range(len(word)):
            available = net.available(word, pos)
            action = net.designer_action(available, learn=False)

            expected = "REUSE" if available else "BRANCH"

            if action == expected:
                correct += 1
            else:
                exact = False

            total += 1

        if exact:
            exact_words += 1

    return correct, total, exact_words


def score(genome):
    net = build(genome)

    correct, total, exact_words = evaluate_words(
        net,
        VALIDATION,
    )

    accuracy = correct / total

    return {
        "fitness": accuracy * 1000.0 + exact_words * 100.0,
        "correct": correct,
        "total": total,
        "exact_words": exact_words,
        "net": net,
    }


def print_topology(genome):
    t = topology_of(genome)["designer_topology"]

    print(
        "root->reuse={:.4f} "
        "root->branch={:.4f} "
        "reuse->branch_inh={:.4f} "
        "branch->reuse_inh={:.4f}".format(
            t["root_to_reuse"],
            t["root_to_branch"],
            t["reuse_to_branch_inhibit"],
            t["branch_to_reuse_inhibit"],
        )
    )


def evolve():
    random.seed(SEED)

    seed = topology_of(GENOME)
    population = [clone_genome(seed)]

    while len(population) < POPULATION:
        population.append(mutate_topology(seed))

    best_genome = None
    best_result = None

    print("=== EVOLVE DESIGNER TOPOLOGY ===")
    print()
    print("Structural representation: FIXED")
    print("Designer topology: EVOLVED")
    print("Validation: held out")
    print("Final test: withheld")
    print()
    print(
        "population={} generations={} elite={} seed={}".format(
            POPULATION,
            GENERATIONS,
            ELITE,
            SEED,
        )
    )
    print()

    for generation in range(1, GENERATIONS + 1):
        scored = []

        for genome in population:
            result = score(genome)
            scored.append((result["fitness"], genome, result))

        scored.sort(key=lambda x: x[0], reverse=True)

        fitness, genome, result = scored[0]

        if best_result is None or fitness > best_result["fitness"]:
            best_genome = clone_genome(genome)
            best_result = {
                k: v
                for k, v in result.items()
                if k != "net"
            }

        print(
            "generation={:3d} "
            "fitness={:8.2f} "
            "validation={:2d}/{:2d} "
            "words={:2d}/{:2d} "
            .format(
                generation,
                fitness,
                result["correct"],
                result["total"],
                result["exact_words"],
                len(VALIDATION),
            ),
            end="",
        )
        print_topology(genome)

        elites = [
            clone_genome(item[1])
            for item in scored[:ELITE]
        ]

        population = elites[:]

        while len(population) < POPULATION:
            parent = random.choice(elites)
            population.append(mutate_topology(parent))

    print()
    print("=== BEST EVOLVED TOPOLOGY ===")
    print_topology(best_genome)

    print()
    print("=== VALIDATION ===")
    print(
        "fitness={:.2f} positions={}/{} words={}/{}".format(
            best_result["fitness"],
            best_result["correct"],
            best_result["total"],
            best_result["exact_words"],
            len(VALIDATION),
        )
    )

    print()
    print("=== BEST GENOME ===")
    print(genome_summary(best_genome))
    print(best_genome)

    print()
    print("=== FINAL HELD-OUT TEST ===")

    final_net = build(best_genome)

    links_before = len(final_net.boundaries.links)
    prefix_before = final_net.prefix.next_id
    suffix_before = final_net.suffix.next_id

    correct = 0
    total = 0
    exact_words = 0

    for word in FINAL_TEST:
        expected_reuse = 0
        expected_branch = 0
        actual_reuse = 0
        actual_branch = 0
        exact = True

        for pos in range(len(word)):
            available = final_net.available(word, pos)

            if available:
                expected_reuse += 1
            else:
                expected_branch += 1

            action = final_net.designer_action(
                available,
                learn=False,
            )

            if action == "REUSE":
                actual_reuse += 1
            else:
                actual_branch += 1

            expected = "REUSE" if available else "BRANCH"

            if action == expected:
                correct += 1
            else:
                exact = False

            total += 1

        if exact:
            exact_words += 1

        print(
            "{:6s} available_reuse={:2d} "
            "available_branch={:2d} "
            "designer_reuse={:2d} "
            "designer_branch={:2d} exact={}".format(
                word,
                expected_reuse,
                expected_branch,
                actual_reuse,
                actual_branch,
                exact,
            )
        )

    print()
    print("=== FINAL GENERALIZATION ===")
    print("test_words        :", len(FINAL_TEST))
    print("exact_words       :", "{}/{}".format(exact_words, len(FINAL_TEST)))
    print("correct_positions :", "{}/{}".format(correct, total))
    print("accuracy          :", "{:.4f}".format(correct / total))

    print()
    print("=== FROZEN INVARIANTS ===")
    print("boundary_links_before:", links_before)
    print("boundary_links_after :", len(final_net.boundaries.links))
    print("prefix_nodes_before  :", prefix_before)
    print("prefix_nodes_after   :", final_net.prefix.next_id)
    print("suffix_nodes_before  :", suffix_before)
    print("suffix_nodes_after   :", final_net.suffix.next_id)

    print()
    print("=== EVOLVED DESIGNER SYNAPSES ===")
    n = final_net.net

    for edge in [
        (n.designer_root, n.reuse_cell),
        (n.designer_root, n.branch_cell),
        (n.reuse_cell, n.branch_cell),
        (n.branch_cell, n.reuse_cell),
    ]:
        syn = n.synapses[edge]
        print(
            "{} -> {} {} weight={:.4f}".format(
                edge[0],
                edge[1],
                syn.kind,
                syn.weight,
            )
        )


if __name__ == "__main__":
    evolve()
