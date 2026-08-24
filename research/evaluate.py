from simulator import Network, Config


def evaluate():
    training = [
        "CAT", "CAR", "CAN", "CARD", "CART", "DOG", "DOT", "BAT",
    ]

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
    network.train(training, epochs=5)
    network.print_summary()
    network.print_topology()
    network.print_vocabulary_tree()


if __name__ == "__main__":
    evaluate()
