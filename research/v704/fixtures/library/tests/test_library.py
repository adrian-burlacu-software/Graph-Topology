import unittest
from datetime import date

from library import books as B
from library import loans as L


class BooksTests(unittest.TestCase):
    def test_total_pages(self):
        books = [B.Book(1, "a", "x", 10, 2000), B.Book(2, "b", "y", 5, 1990)]
        self.assertEqual(B.total_pages(books), 15)

    def test_isbn(self):
        self.assertEqual(B.parse_isbn("978-0 14"), "978014")


class LoansTests(unittest.TestCase):
    def test_due(self):
        self.assertEqual(L.due_date(date(2026, 1, 1)), date(2026, 1, 15))


if __name__ == "__main__":
    unittest.main()
