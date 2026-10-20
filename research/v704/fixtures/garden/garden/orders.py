"""Orders customers place, and when they must be shipped."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from garden.plants import Plant

#: days the nursery has to ship an order
SHIP_DAYS = 3
#: the most plants one order may hold
ORDER_LIMIT = 12


@dataclass
class Order:
    customer: str
    plants: list = field(default_factory=list)
    placed: date | None = None

    def total(self) -> float:
        """What the order costs."""
        return sum(plant.price for plant in self.plants)


def ship_by(placed: date) -> date:
    """The last day an order placed on `placed` may be shipped."""
    return placed + timedelta(days=SHIP_DAYS)


def place_order(customer: str, plants: list, orders: list,
                placed: date) -> Order:
    """A new order, refused when it holds more than ORDER_LIMIT plants."""
    if len(plants) > ORDER_LIMIT:
        raise ValueError(f"an order holds at most {ORDER_LIMIT} plants")
    order = Order(customer, list(plants), placed)
    orders.append(order)
    return order


def late(orders: list, today: date) -> list:
    """The orders whose shipping day has passed."""
    return [order for order in orders if ship_by(order.placed) < today]
