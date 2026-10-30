import unittest

from clinic import patients as P
from clinic import visits as V


class PatientTests(unittest.TestCase):
    def test_tidy_number(self):
        self.assertEqual(P.tidy_number("P-1"), "P1")


class VisitTests(unittest.TestCase):
    def test_cancel_fee(self):
        self.assertEqual(V.cancel_fee(2), 15)


if __name__ == "__main__":
    unittest.main()
