"""Goals: questions nobody wrote a pattern for, answered by composing what
memory already does -- where each one is, what is with whom, where an
occurrence went -- and what a pattern answered before, answered the same.

Run: python -m unittest research.v689.test_goals -v

On `test_people`'s store. It has no sense of `have`, so VerbNet reads `has`
through any of its classes, as `change.py` backs off to when the store
offers none.
"""
from __future__ import annotations

import unittest

from research.v689 import test_people as people
from research.v689.goals import read_goal
from research.v689.test_people import said, talk


def setUpModule() -> None:                      # noqa: N802
    people.setUpModule()


def tearDownModule() -> None:                   # noqa: N802
    people.tearDownModule()


class ReadingGoalsTests(unittest.TestCase):
    def lexicon(self):
        from research.v689.session import Session, Taught
        session = Session(people.PeopleAsker())
        return Taught(session.asker, session.memory.kinds)

    def test_the_slots(self):
        lexicon = self.lexicon()
        names = frozenset({"mary"})
        goal = read_goal("who is in the kitchen", lexicon, names)
        self.assertEqual((goal.asked, goal.relation, goal.who,
                          goal.object.kind), ("subject", "located", "people",
                                              "kitchen"))
        goal = read_goal("how many people are in the kitchen", lexicon, names)
        self.assertEqual((goal.asked, goal.who), ("count", "people"))
        goal = read_goal("how many balls are in the kitchen", lexicon, names)
        self.assertEqual((goal.asked, goal.who), ("count", "ball"))
        goal = read_goal("where did Mary go first", lexicon, names)
        self.assertEqual((goal.asked, goal.relation, goal.verb,
                          goal.sequence), ("place", "occurrence", "go", 1))
        goal = read_goal("what does Mary have", lexicon, names)
        self.assertEqual((goal.asked, goal.relation, goal.subject.name),
                         ("object", "holding", "mary"))

    def test_not_a_goal(self):
        lexicon = self.lexicon()
        self.assertIsNone(read_goal("can a dog swim", lexicon))
        self.assertIsNone(read_goal("who is he", lexicon))
        self.assertIsNone(read_goal("is anyone here", lexicon))


class LocatedTests(unittest.TestCase):
    def test_who_is_somewhere(self):
        _, turns = talk("Mary went to the kitchen.", "John went to the garden.",
                        "who is in the kitchen", "who is in the garden",
                        "John went to the kitchen.", "who is in the kitchen")
        self.assertEqual(said(turns[2]), "Mary")
        self.assertEqual(said(turns[3]), "John")
        self.assertEqual(said(turns[5]), "Mary and John")

    def test_anyone_and_how_many(self):
        _, turns = talk("Mary went to the kitchen.", "John went to the kitchen.",
                        "is anyone in the kitchen", "is anyone in the office",
                        "how many people are in the kitchen",
                        "how many people are in the office")
        self.assertEqual(said(turns[2]), "yes")
        self.assertEqual(turns[3].answer["outcome"], "unknown")
        self.assertEqual(said(turns[4]), "two")
        self.assertEqual(said(turns[5]), "none")

    def test_what_is_somewhere_carried_or_not(self):
        _, turns = talk("Mary went to the kitchen.", "Mary got the football.",
                        "what is in the kitchen", "who is in the kitchen")
        self.assertEqual(said(turns[2]), "the football")
        self.assertEqual(said(turns[3]), "Mary")


class HoldingTests(unittest.TestCase):
    def test_who_has_it_and_what_they_have(self):
        _, turns = talk("Mary went to the kitchen.", "Mary got the football.",
                        "who is carrying the football", "who has the football",
                        "what does Mary have")
        self.assertEqual(said(turns[2]), "Mary")
        self.assertEqual(said(turns[3]), "Mary")
        self.assertEqual(said(turns[4]), "the football")

    def test_given_away(self):
        _, turns = talk("Mary got the football.",
                        "Mary gave the football to John.",
                        "who has the football")
        self.assertEqual(said(turns[2]), "John")

    def test_a_pattern_still_answers_what_it_did(self):
        _, turns = talk("Mary went to the kitchen.", "Mary got the football.",
                        "what is Mary carrying", "who went to the kitchen")
        self.assertEqual(turns[2].act, "carrying")
        self.assertEqual(said(turns[2]), "the football")
        self.assertEqual(said(turns[3]), "Mary")


class OccurrencePlaceTests(unittest.TestCase):
    def test_where_did_someone_go(self):
        _, turns = talk("Mary went to the kitchen.", "Mary went to the garden.",
                        "where did Mary go", "where did Mary go first",
                        "where did Mary go last")
        self.assertEqual(said(turns[2]), "the garden")
        self.assertIn("before that, the kitchen", turns[2].answer["text"])
        self.assertEqual(said(turns[3]), "the kitchen")
        self.assertEqual(said(turns[4]), "the garden")

    def test_nowhere_told(self):
        _, turns = talk("Mary went to the kitchen.", "where did John go")
        self.assertNotEqual(turns[1].answer.get("outcome"), "retrieved")


if __name__ == "__main__":
    unittest.main()
