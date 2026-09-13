"""`within.py` over the real store: which kinds of a class do something."""
from __future__ import annotations

import unittest
from pathlib import Path

from research.v687 import within

STORE = Path(__file__).resolve().parents[2] / "data" / "v684_reasoning.sqlite"


class WithinTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from research.v687.reasoning import ReasoningEngine

        cls.engine = ReasoningEngine(STORE)

    def found(self, question):
        payload = within.answer(self.engine, question)
        self.assertIsNotNone(payload, question)
        return payload

    def test_the_reading(self):
        self.assertEqual(within.read("which birds cannot fly"),
                         ("birds", "can fly", True))
        self.assertEqual(within.read("what mammals lay eggs?"),
                         ("mammals", "lay eggs", False))
        for question in ("what animal has a trunk", "what kinds of dogs are there",
                         "which is heavier, a feather or a brick",
                         "which one is black"):
            self.assertIsNone(within.read(question), question)

    def test_which_birds_cannot_fly_names_the_exceptions(self):
        payload = self.found("which birds cannot fly")
        found = payload["within"]
        self.assertEqual(payload["verdict"], "LISTING")
        self.assertTrue({"emu", "penguin"} <= set(found["norms"]), found)
        self.assertNotIn("eagle", found["norms"])
        self.assertEqual(found["store"], [])

    def test_which_dogs_bark_and_what_mammals_lay_eggs(self):
        self.assertIn("collie", self.found("which dogs bark")["within"]["norms"])
        self.assertIn("platypus",
                      self.found("what mammals lay eggs")["within"]["norms"])

    def test_the_engine_routes_them_here_and_leaves_identification_alone(self):
        self.assertEqual(self.engine.ask("which birds cannot fly")["parse"]
                         ["relation"], "R20")
        self.assertEqual(self.engine.ask("what animal has a trunk")["verdict"],
                         "IDENTIFIED")


if __name__ == "__main__":
    unittest.main()
