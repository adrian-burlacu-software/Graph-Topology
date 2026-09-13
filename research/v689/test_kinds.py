"""What one of a kind is like, is toward, and is doing it for.

Run: python -m unittest research.v689.test_kinds -v

Deduction (a quality toward something descends: E1 is about what a thing is
like), induction (I1: nothing told of Greg's colour, and the swans told of
here are grey), and motives (the store's motivations: tired moves one to
sleep, and a bedroom is for sleeping). On v687's own `Reasoner` and `Parser`
over a small store built here, with the colours and ConceptNet rows those
need.
"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from research.v687.language import Parser
from research.v687.reason import Reasoner
from research.v689 import change
from research.v689.asker import Asker
from research.v689.episodic import toward
from research.v689.session import Session
from research.v689.test_v689 import SCHEMA

CONCEPTS = (
    ("entity.n.01", ("entity",), None),
    ("organism.n.01", ("organism",), "entity.n.01"),
    ("person.n.01", ("person",), "organism.n.01"),
    ("animal.n.01", ("animal",), "organism.n.01"),
    ("mouse.n.01", ("mouse",), "animal.n.01"),
    ("wolf.n.01", ("wolf",), "animal.n.01"),
    ("cat.n.01", ("cat",), "animal.n.01"),
    ("frog.n.01", ("frog",), "animal.n.01"),
    ("swan.n.01", ("swan",), "animal.n.01"),
    ("color.n.01", ("color", "colour"), "entity.n.01"),
    ("chromatic color.n.01", ("chromatic color",), "color.n.01"),
    ("green.n.01", ("green",), "chromatic color.n.01"),
    ("white.n.02", ("white",), "color.n.01"),
    ("gray.n.01", ("gray", "grey"), "color.n.01"),
    ("yellow.n.01", ("yellow",), "chromatic color.n.01"),
    ("room.n.01", ("room",), "entity.n.01"),
    ("bedroom.n.01", ("bedroom",), "room.n.01"),
    ("kitchen.n.01", ("kitchen",), "room.n.01"),
    ("apple.n.01", ("apple",), "entity.n.01"),
    ("sleeping.n.01", ("sleeping",), "entity.n.01"),
    ("eat.v.01", ("eat",), None),
    ("travel.v.01", ("travel", "go", "move"), None),
    ("get.v.01", ("get",), None))

FACTS = (("sleeping.n.01", "has_prerequisite", "tired", "conceptnet", 0.9, 0),
         ("bedroom.n.01", "used_for", "sleeping", "conceptnet", 0.9, 0),
         ("eat.v.01", "motivated_by_goal", "hungry", "conceptnet", 0.9, 0),
         ("kitchen.n.01", "used_for", "eating meals", "conceptnet", 0.9, 0))

STORE: dict = {}


def setUpModule() -> None:                      # noqa: N802
    folder = tempfile.TemporaryDirectory()
    path = Path(folder.name) / "kinds.sqlite"
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
    connection.executemany("INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?)",
                           FACTS)
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


class KindAsker(Asker):
    def __init__(self) -> None:
        super().__init__(STORE["reasoner"], STORE["parser"])

    def run(self, question: str) -> dict:
        return {"summary": {"outcome": "unknown", "trust": "",
                            "lines": [f"unknown — {question}"]}}


def talk(*lines: str):
    session = Session(KindAsker())
    return session, [session.say(line) for line in lines]


def said(turn) -> str:
    return (turn.answer.get("text") or "").split(" — ", 1)[0]


class DeductionTests(unittest.TestCase):
    """A quality toward something descends; what a thing is like does not."""

    def test_toward_is_a_quality_with_a_complement(self):
        self.assertTrue(toward("afraid of wolves"))
        self.assertFalse(toward("black"))
        self.assertFalse(toward("very black"))

    def test_what_one_of_them_is_afraid_of(self):
        _, turns = talk("Mice are afraid of wolves.", "Gertrude is a mouse.",
                        "What is gertrude afraid of?",
                        "Is Gertrude afraid of wolves?")
        self.assertEqual(said(turns[2]), "wolves")
        self.assertEqual(turns[3].answer["outcome"], "verified")

    def test_what_was_told_of_this_one_comes_first(self):
        _, turns = talk("Mice are afraid of wolves.", "Gertrude is a mouse.",
                        "Gertrude is afraid of cats.",
                        "What is Gertrude afraid of?")
        self.assertEqual(said(turns[3]), "cats")

    def test_what_a_kind_is_like_still_does_not_descend(self):
        _, turns = talk("Swans are white.", "Greg is a swan.",
                        "Is Greg white?")
        self.assertEqual(turns[2].answer["outcome"], "unknown")


class InductionTests(unittest.TestCase):
    """I1: nothing told of this one, and something told of others of its
    kind."""

    def test_its_own_value_first(self):
        _, turns = talk("Lily is a frog.", "Lily is yellow.",
                        "Bernhard is a frog.", "Bernhard is green.",
                        "What color is Lily?")
        self.assertEqual(said(turns[4]), "yellow")

    def test_from_the_others_of_its_kind(self):
        _, turns = talk("Lily is a frog.", "Bernhard is a frog.",
                        "Bernhard is green.", "What color is Lily?")
        self.assertEqual(said(turns[3]), "probably green")
        self.assertEqual(turns[3].answer["source"], "induced")

    def test_where_they_differ_the_last_told(self):
        _, turns = talk("Julius is a swan.", "Julius is white.",
                        "Brian is a swan.", "Brian is grey.",
                        "Greg is a swan.", "What colour is Greg?")
        self.assertEqual(said(turns[5]), "probably grey")
        self.assertIn("differ", turns[5].answer["text"])

    def test_a_different_kind_is_no_evidence(self):
        _, turns = talk("Julius is a swan.", "Julius is white.",
                        "Lily is a frog.", "What color is Lily?")
        self.assertFalse(said(turns[3]).startswith("probably"))


@unittest.skipUnless(change.frames(), "needs data/verbnet3.3")
class MotiveTests(unittest.TestCase):
    """The store's motivations: why someone did something, and where they
    will go."""

    def test_where_she_will_go(self):
        _, turns = talk("John went to the kitchen.",
                        "John went to the bedroom.", "Sumit is tired.",
                        "Where will Sumit go?")
        self.assertEqual(said(turns[3]), "probably the bedroom")

    def test_nowhere_here_is_for_it(self):
        _, turns = talk("Sumit is tired.", "Where will Sumit go?")
        self.assertTrue(said(turns[1]).startswith("not told"))

    def test_why_she_went_there(self):
        _, turns = talk("Sumit is tired.", "Sumit went to the bedroom.",
                        "Why did Sumit go to the bedroom?")
        self.assertEqual(said(turns[2]), "because Sumit is tired")

    def test_only_the_last_state_is_perhaps(self):
        _, turns = talk("Sumit is hungry.", "Sumit got the apple.",
                        "Why did Sumit get the apple?")
        self.assertEqual(said(turns[2]), "perhaps because Sumit is hungry")


if __name__ == "__main__":
    unittest.main()
