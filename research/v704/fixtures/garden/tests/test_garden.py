import unittest
from datetime import date

from garden import orders as O
from garden import plants as P


class PlantsTests(unittest.TestCase):
    def test_stock_value(self):
        plants = [P.Plant("A", "a", "herb", 2.0, 3),
                  P.Plant("B", "b", "tree", 10.0, 1)]
        self.assertEqual(P.stock_value(plants), 16.0)

    def test_tidy_code(self):
        self.assertEqual(P.tidy_code("ab 12"), "AB12")


class OrdersTests(unittest.TestCase):
    def test_ship_by(self):
        self.assertEqual(O.ship_by(date(2026, 3, 1)), date(2026, 3, 4))


if __name__ == "__main__":
    unittest.main()
