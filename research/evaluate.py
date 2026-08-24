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

    # recombinations
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
    network.train(TRAINING, epochs=5)

    print()
    print("=== FREEZE ===")
    print("No learning after training.")

    # Disable learning.
    network.config.designer_learning_rate = 0.0
    network.config.vocabulary_learning_rate = 0.0

    before_cells = len(network.cells)

    correct = 0
    total = 0
    created = 0
    reused = 0

    print()
    print("=== NOVEL TEST ===")

    for word in TEST:
        before_reuse = network.total_reuse
        before_create = network.total_create

        result = network.process_word(word)

        word_reuse = network.total_reuse - before_reuse
        word_create = network.total_create - before_create

        reused += word_reuse
        created += word_create

        print(
            f"{word:5s} "
            f"reuse={word_reuse:2d} "
            f"create={word_create:2d} "
            f"result={result}"
        )

        # A test is structurally correct if:
        # existing paths were reused and missing paths were created.
        expected_reuse = 0
        expected_create = 0

        current = None

        for order, symbol in enumerate(word):
            existing = network.find_child(current, symbol)

            if existing is not None:
                expected_reuse += 1
                current = existing
            else:
                expected_create += 1
                # Do not mutate here; this is only a reference
                # calculation for reporting.

        # We don't score against the post-process graph above,
        # because process_word may have created the missing path.
        total += len(word)

    after_cells = len(network.cells)

    print()
    print("=== GENERALIZATION ===")
    print(f"training_words       : {len(TRAINING)}")
    print(f"test_words           : {len(TEST)}")
    print(f"cells_before_test    : {before_cells}")
    print(f"cells_after_test     : {after_cells}")
    print(f"new_cells            : {after_cells - before_cells}")
    print(f"test_reuse           : {reused}")
    print(f"test_create          : {created}")

    print()
    print("=== LEARNED NETWORK ===")
    network.print_summary()
    network.print_vocabulary_tree()


if __name__ == "__main__":
    evaluate()