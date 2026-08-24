from __future__ import annotations

from simulator import Network, Config


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

    # Re-combinations.
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


def expected_counts_before_processing(
    network: Network,
    word: str,
) -> tuple[int, int]:
    """
    Calculate the ground-truth structural actions against a snapshot
    of the graph BEFORE process_word mutates it.

    Missing paths are counted locally without mutating the network.
    """

    expected_reuse = 0
    expected_create = 0

    current = None

    for symbol in word:
        existing = network.find_child(
            current,
            symbol,
        )

        if existing is not None:
            expected_reuse += 1
            current = existing
            continue

        expected_create += 1

        # We need to follow the hypothetical new path.
        #
        # Do NOT mutate the real network. Instead, once a missing edge
        # occurs, the rest of this word is necessarily new unless the
        # same hypothetical path can reconnect to an existing node.
        #
        # For this tree topology, a newly created node has no existing
        # children, so every remaining character is also a creation.
        break

    remaining = len(word) - (
        expected_reuse + expected_create
    )

    expected_create += remaining

    return expected_reuse, expected_create


def evaluate():
    config = Config(
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
        feedback_weight=0.10,
    )

    network = Network(config)

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

    before_cells = len(network.cells)

    total_expected_reuse = 0
    total_expected_create = 0

    total_actual_reuse = 0
    total_actual_create = 0

    correct_reuse = 0
    correct_create = 0

    print()
    print("=== NOVEL TEST ===")

    for word in TEST:
        # ----------------------------------------------------------
        # IMPORTANT:
        # Ground truth is captured BEFORE process_word.
        # ----------------------------------------------------------

        expected_reuse, expected_create = (
            expected_counts_before_processing(
                network,
                word,
            )
        )

        before_reuse = network.total_reuse
        before_create = network.total_create

        result = network.process_word(
            word,
            learn=False,
        )

        actual_reuse = (
            network.total_reuse - before_reuse
        )

        actual_create = (
            network.total_create - before_create
        )

        total_actual_reuse += actual_reuse
        total_actual_create += actual_create

        total_expected_reuse += expected_reuse
        total_expected_create += expected_create

        if actual_reuse == expected_reuse:
            correct_reuse += 1

        if actual_create == expected_create:
            correct_create += 1

        print(
            f"{word:5s} "
            f"reuse={actual_reuse:2d} "
            f"create={actual_create:2d} "
            f"expected_reuse={expected_reuse:2d} "
            f"expected_create={expected_create:2d} "
            f"result={result}"
        )

    after_cells = len(network.cells)

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
        f"{total_actual_reuse}"
    )

    print(
        f"test_create          : "
        f"{total_actual_create}"
    )

    print(
        f"expected_reuse       : "
        f"{total_expected_reuse}"
    )

    print(
        f"expected_create      : "
        f"{total_expected_create}"
    )

    print(
        f"exact_reuse_words    : "
        f"{correct_reuse}/{len(TEST)}"
    )

    print(
        f"exact_create_words   : "
        f"{correct_create}/{len(TEST)}"
    )

    print()
    print("=== LEARNED NETWORK ===")

    network.print_summary()
    network.print_vocabulary_tree()


if __name__ == "__main__":
    evaluate()