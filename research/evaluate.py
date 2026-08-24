import json

from genome import GENOME
from simulator import DevelopmentalNetwork


def load_config():

    with open("config.json") as f:
        return json.load(f)


def evaluate():

    config = load_config()

    network = DevelopmentalNetwork(
        GENOME,
        seed=config["experiment"]["seed"]
    )

    training = config["vocabulary"]["training"]

    network.train(training)

    stats = network.stats()

    print()
    print("=== EXPERIMENT ===")

    for key, value in stats.items():
        print(f"{key:18}: {value}")

    print()
    print("=== TOPOLOGY ===")

    for cell in network.topology():

        print(
            f"{cell['id']:>3} "
            f"{cell['role']:>10} "
            f"{str(cell['symbol']):>2} "
            f"parent={str(cell['parent']):>3} "
            f"in={cell['incoming']} "
            f"out={cell['outgoing']} "
            f"order={cell['order']}"
        )


if __name__ == "__main__":
    evaluate()