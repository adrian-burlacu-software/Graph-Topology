"""The nursery's plants, read from data/plants.csv."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


@dataclass
class Plant:
    code: str
    name: str
    kind: str
    price: float
    stock: int


def read_plants(path: Path = DATA / "plants.csv") -> list:
    """Every plant the nursery lists, in the file's order."""
    with open(path, encoding="utf-8") as stream:
        return [Plant(row["code"], row["name"], row["kind"],
                      float(row["price"]), int(row["stock"]))
                for row in csv.DictReader(stream)]


def of_kind(plants: list, kind: str) -> list:
    """The plants of one kind (shrub, herb, tree, ...)."""
    return [plant for plant in plants if plant.kind == kind]


def stock_value(plants: list) -> float:
    """What all the plants in stock are worth together."""
    return sum(plant.price * plant.stock for plant in plants)


def cheapest(plants: list):
    """The plant with the lowest price, or None for no plants."""
    return min(plants, key=lambda plant: plant.price) if plants else None


def tidy_code(code: str) -> str:
    """A plant code in capitals, its spaces taken out."""
    return code.replace(" ", "").upper()
