"""S1 to S4: where things are against each other, and how big.

Run: python -m unittest research.v689.test_relations -v

`relations.py` on its own first, over told events as the stream would carry
them; then in conversation, on v687's own `Reasoner` and `Parser` over a small
store built here.
"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from research.v687.language import Parser
from research.v687.reason import Reasoner
from research.v689 import relations
from research.v689.asker import Asker
from research.v689.events import Event
from research.v689.session import Session
from research.v689.test_v689 import SCHEMA

CONCEPTS = (
    ("entity.n.01", ("entity",), None),
    ("person.n.01", ("person",), "entity.n.01"),
    ("room.n.01", ("room",), "entity.n.01"),
    ("kitchen.n.01", ("kitchen",), "room.n.01"),
    ("office.n.01", ("office",), "room.n.01"),
    ("hallway.n.01", ("hallway",), "entity.n.01"),
    ("garden.n.01", ("garden",), "entity.n.01"),
    ("bedroom.n.01", ("bedroom",), "room.n.01"),
    ("shape.n.01", ("shape",), "entity.n.01"),
    ("triangle.n.01", ("triangle",), "shape.n.01"),
    ("rectangle.n.01", ("rectangle",), "shape.n.01"),
    ("square.n.01", ("square",), "shape.n.01"),
    ("sphere.n.01", ("sphere",), "shape.n.01"),
    ("container.n.01", ("container",), "entity.n.01"),
    ("box.n.01", ("box",), "container.n.01"),
    ("chest.n.02", ("chest",), "container.n.01"),
    ("suitcase.n.01", ("suitcase",), "container.n.01"),
    ("chocolate.n.02", ("chocolate",), "entity.n.01"))

STORE: dict = {}


def setUpModule() -> None:                      # noqa: N802
    folder = tempfile.TemporaryDirectory()
    path = Path(folder.name) / "relations.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    connection.executemany(
        "INSERT INTO concepts VALUES (?, ?, ?, 1, ?, 1)",
        [(concept, lemmas[0], concept.split(".")[-2], f"a {lemmas[0]}")
         for concept, lemmas, _ in CONCEPTS])
    connection.executemany(
        "INSERT INTO taxonomy VALUES (?, ?)",
        [(concept, parent) for concept, _, parent in CONCEPTS if parent])
    connection.executemany(
        "INSERT INTO lemmas VALUES (?, ?, 1)",
        [(lemma, concept) for concept, lemmas, _ in CONCEPTS
         for lemma in lemmas])
    connection.commit()
    connection.close()
    reasoner = Reasoner(path)
    parser = Parser(vocabulary=reasoner.vocabulary(),
                    nouns=reasoner.noun_vocabulary())
    STORE.update(folder=folder, reasoner=reasoner, parser=parser)


def tearDownModule() -> None:                   # noqa: N802
    if STORE:
        STORE["reasoner"].connection.close()
        try:
            STORE["folder"].cleanup()
        except OSError:
            pass


class Log:
    """Just enough of `events.Log` for a projection to be fed by hand."""

    def __init__(self) -> None:
        self.handlers: dict = {}
        self.seq = 0

    def on(self, kind: str, handler) -> None:
        self.handlers.setdefault(kind, []).append(handler)

    def told(self, node: str, obj: str, bound: str) -> None:
        self.seq += 1
        event = Event("conversation", self.seq, "told",
                      {"node": node, "relation": "has_property",
                       "object": obj, "bound": bound, "said": obj})
        for handler in self.handlers.get("told", ()):
            handler(event)


class PhraseTests(unittest.TestCase):
    def test_the_longest_phrase_with_articles_left_out(self):
        found = relations.phrase(["to", "the", "left", "of", "the",
                                  "triangle"])
        self.assertEqual((found.dimension, found.side, found.length),
                         ("lateral", -1, 4))
        self.assertEqual(relations.phrase(["north", "of", "an", "office"])
                         .dimension, "north-south")
        self.assertEqual(relations.phrase(["fits", "inside", "the", "box"])
                         .side, -1)
        self.assertIsNone(relations.phrase(["in", "the", "kitchen"]))


class RuleTests(unittest.TestCase):
    """S1 to S4 over told relations, without anything read."""

    def setUp(self):
        self.log = Log()
        self.relations = relations.Relations(self.log)

    def test_s1_a_relation_and_its_converse(self):
        self.log.told("kitchen", "north of an office", "office")
        self.assertTrue(self.relations.compare(
            "office", "kitchen", "north-south", -1).value)
        self.assertEqual([one for one, _ in self.relations.beside(
            "office", "north-south", 1)], ["kitchen"])
        self.assertEqual([one for one, _ in self.relations.beside(
            "kitchen", "north-south", -1)], ["office"])

    def test_s2_an_order_walked_and_absent_not_false(self):
        self.log.told("box", "bigger than a chest", "chest")
        self.log.told("chocolate", "fits inside a chest", "chest")
        found = self.relations.compare("box", "chocolate", "size", 1)
        self.assertTrue(found.value)
        self.assertEqual(len(found.path), 2)
        self.assertFalse(self.relations.compare("chocolate", "box", "size",
                                                1).value)
        self.log.told("suitcase", "bigger than a chest", "chest")
        self.assertIsNone(self.relations.compare("box", "suitcase", "size",
                                                 1).value)

    def test_s3_a_direction_puts_the_two_in_line(self):
        self.log.told("triangle", "above a rectangle", "rectangle")
        self.log.told("square", "to the left of a triangle", "triangle")
        self.assertTrue(self.relations.compare(
            "rectangle", "square", "lateral", 1).value)
        self.assertFalse(self.relations.compare(
            "square", "rectangle", "vertical", -1).value)
        level = self.relations.compare("triangle", "rectangle", "lateral", 1)
        self.assertFalse(level.value)
        self.assertTrue(level.level)

    def test_s4_the_shortest_walk_on_the_compass(self):
        self.log.told("kitchen", "north of an office", "office")
        self.log.told("office", "west of a garden", "garden")
        steps = self.relations.route("kitchen", "garden")
        self.assertEqual([direction for direction, _, _ in steps],
                         ["south", "east"])
        self.assertIsNone(self.relations.route("kitchen", "bedroom"))


class RelationAsker(Asker):
    def __init__(self) -> None:
        super().__init__(STORE["reasoner"], STORE["parser"])

    def run(self, question: str) -> dict:
        return {"summary": {"outcome": "unknown", "trust": "",
                            "lines": [f"unknown — {question}"]}}


def talk(*lines: str):
    session = Session(RelationAsker())
    return session, [session.say(line) for line in lines]


def said(turn) -> str:
    return (turn.answer.get("text") or "").split(" — ", 1)[0]


class ConversationTests(unittest.TestCase):
    """The same rules, from what is said."""

    def test_what_is_north_and_what_it_is_north_of(self):
        _, turns = talk("The kitchen is north of the office.",
                        "What is north of the office?",
                        "What is the kitchen north of?",
                        "What is south of the kitchen?")
        self.assertEqual(said(turns[1]), "the kitchen")
        self.assertEqual(said(turns[2]), "the office")
        self.assertEqual(said(turns[3]), "the office")

    def test_how_to_go(self):
        _, turns = talk("The office is east of the hallway.",
                        "The kitchen is north of the office.",
                        "How do you go from the kitchen to the hallway?")
        self.assertTrue(said(turns[2]).startswith("south, then west"),
                        said(turns[2]))

    def test_in_line(self):
        _, turns = talk("The triangle is above the pink rectangle.",
                        "The blue square is to the left of the triangle.",
                        "Is the pink rectangle to the right of the blue "
                        "square?",
                        "Is the blue square below the pink rectangle?")
        self.assertEqual(turns[2].answer["outcome"], "verified")
        self.assertEqual(turns[3].answer["outcome"], "denied")

    def test_what_fits_inside_what(self):
        _, turns = talk("The box of chocolates fits inside the chest.",
                        "The box is bigger than the chest.",
                        "Does the box fit in the box of chocolates?",
                        "Is the box bigger than the box of chocolates?")
        self.assertEqual(turns[2].answer["outcome"], "denied")
        self.assertEqual(turns[3].answer["outcome"], "verified")

    def test_not_told_either_way(self):
        _, turns = talk("The box is bigger than the chest.",
                        "The suitcase is bigger than the chest.",
                        "Is the box bigger than the suitcase?")
        self.assertEqual(turns[2].answer["outcome"], "unknown")
        self.assertTrue(said(turns[2]).startswith("not told"),
                        said(turns[2]))

    def test_replayed_the_order_is_the_same(self):
        session, _ = talk("The box is bigger than the chest.",
                          "The chest is bigger than the chocolate.")
        again = Session.rebuild(RelationAsker(), session.conversation,
                                session.memory.log.conversation.events)
        turn = again.say("Is the chocolate bigger than the box?")
        self.assertEqual(turn.answer["outcome"], "denied")


if __name__ == "__main__":
    unittest.main()
