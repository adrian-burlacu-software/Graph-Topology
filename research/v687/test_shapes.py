"""`attributes.py` and `shapes.py` over the real store."""
from __future__ import annotations

import unittest
from pathlib import Path

from research.v687 import attributes, shapes

STORE = Path(__file__).resolve().parents[2] / "data" / "v684_reasoning.sqlite"


class ShapeTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from research.v687.reasoning import ReasoningEngine

        cls.engine = ReasoningEngine(STORE)

    def test_reading_a_value_question(self):
        self.assertEqual(attributes.read("what color is a banana"),
                         ("color", "banana"))
        self.assertEqual(attributes.read("how fast can a cheetah run?"),
                         ("speed", "cheetah"))
        self.assertEqual(attributes.read("how much does an elephant weigh"),
                         ("weight", "elephant"))
        for question in ("what is a dog made of", "how long do dogs live",
                         "how many legs does a spider have"):
            self.assertIsNone(attributes.read(question), question)

    def test_a_value_is_what_the_thing_carries(self):
        banana = attributes.answer(self.engine, "what color is a banana")
        self.assertIn("yellow", banana["attribute"]["values"])
        elephant = attributes.answer(self.engine, "how big is an elephant")
        self.assertTrue({"big", "large", "huge"}
                        & set(elephant["attribute"]["values"]), elephant["note"])
        self.assertIn("no numbers", elephant["note"])

    def test_the_kind_above_and_the_parts(self):
        above = shapes.answer(self.engine, "what is a dog a kind of")
        self.assertIn("canine", above["above"]["chain"])
        parts = shapes.answer(self.engine, "what are the parts of a car")
        self.assertEqual(parts["verdict"], "LISTING")
        self.assertTrue(parts["parts"]["parts"])

    def test_a_choice_and_likeness(self):
        tomato = shapes.answer(self.engine,
                               "is a tomato a fruit or a vegetable")
        self.assertIn("vegetable", tomato["choice"]["holds"])
        # Likeness is R21's to answer or to decline by name -- never a yes on
        # the word `fish` in a dolphin's norms.
        dolphin = shapes.answer(self.engine, "is a dolphin like a fish")
        self.assertIsNotNone(dolphin)
        self.assertEqual(dolphin["parse"]["relation"], "R21")
        self.assertTrue("contrast" in dolphin
                        or "feature norms" in dolphin["note"], dolphin["note"])

    def test_an_analogy_in_another_shape_is_declined_by_name(self):
        payload = shapes.answer(self.engine, "wing is to bird as fin is to what")
        self.assertEqual((payload["verdict"], payload["parse"]["relation"]),
                         ("UNKNOWN", "R24"))

    def test_a_value_comes_from_the_best_source_that_has_one(self):
        banana = attributes.answer(self.engine, "what color is a banana")
        self.assertIn("yellow", banana["attribute"]["values"])
        self.assertLessEqual(len(banana["attribute"]["values"]), 3,
                             banana["note"])

    def test_a_verb_sense_is_not_a_kind(self):
        payload = self.engine.ask("is water wet")
        self.assertNotIn("water.v.01", payload.get("note") or "")
        self.assertNotEqual(payload.get("concept"), "water.v.01")

    def test_the_engine_routes_them(self):
        self.assertEqual(self.engine.ask("what color is a banana")["parse"]
                         ["relation"], "attribute")
        self.assertEqual(self.engine.ask("what is a dog a kind of")["parse"]
                         ["relation"], "R1")


if __name__ == "__main__":
    unittest.main()
