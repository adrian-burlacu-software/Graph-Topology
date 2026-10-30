import unittest

from rental import hires as H
from rental import tools as T


class ToolTests(unittest.TestCase):
    def test_clean_code(self):
        self.assertEqual(T.clean_code("d 1"), "D1")


class HireTests(unittest.TestCase):
    def test_late_charge(self):
        self.assertEqual(H.late_charge(2), 14)


if __name__ == "__main__":
    unittest.main()
