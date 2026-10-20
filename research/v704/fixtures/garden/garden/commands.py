"""The nursery from a terminal: `python -m garden.commands plants`."""
from __future__ import annotations

import sys

from garden import plants as P
from garden.web import serve


def run(args: list) -> int:
    if not args:
        print("commands: plants | worth | about")
        return 1
    command = args[0]
    if command == "plants":
        for plant in P.read_plants():
            print(f"{plant.name}: {plant.stock} in stock")
    elif command == "worth":
        print(round(P.stock_value(P.read_plants()), 2))
    elif command == "about":
        print(serve("/about"))
    else:
        print(f"no command {command}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
