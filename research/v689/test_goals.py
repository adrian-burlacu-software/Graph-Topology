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

    def test_the_grammar_reads_the_cell_it_fills(self):
        from research.v689.reading import read
        lexicon = self.lexicon()
        names = frozenset({"mary", "fred"})
        for text, cell in (
                ("where is Mary", ("place", "located")),
                ("what is Mary carrying", ("object", "holding")),
                ("how many objects is Mary carrying", ("count", "holding")),
                ("who went to the kitchen", ("subject", "occurrence")),
                ("what did Mary drop", ("object", "occurrence")),
                ("who did Fred give the milk to", ("recipient", "occurrence")),
                ("when did Mary go to the kitchen", ("time", "occurrence")),
                ("how many times did Mary go", ("times", "occurrence")),
                ("what was Mary doing", ("verb", "occurrence")),
                ("what is north of Mary", ("subject", "dimension")),
                ("what is Mary north of", ("object", "dimension")),
                ("how do you go from Mary to Fred", ("path", "dimension")),
                ("what color is Mary", ("value", "attribute")),
                ("what is Mary afraid of", ("object", "attribute")),
                ("where will Mary go", ("place", "motive")),
                ("how many dogs are there", ("count", "is_a")),
                ("which dog is black", ("which", "is_a")),
                ("what kind of person is Mary", ("kind", "is_a")),
                ("what happened", ("events", "story")),
                ("what did Mary do first", ("events", "story")),
                ("what did i tell you", ("events", "told")),
                ("what will happen", ("events", "future")),
                ("what do you know about Mary", ("facts", "any")),
                ("what can Mary do", ("facts", "any")),
                ("how do you know that", ("grounds", "answer")),
                ("what about a person", ("again", "question"))):
            self.assertEqual(read(text, lexicon, names).cell, cell, text)
        self.assertIsNone(read("can a dog swim", lexicon, names).cell)

    def test_every_cell_the_grammar_reads_is_answered(self):
        from research.v689.grammar import ACTS
        from research.v689.session import Session
        self.assertEqual(set(ACTS),
                         set(Session(people.PeopleAsker())._cells()))

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


class EveryCellTests(unittest.TestCase):
    """Each relation answers every question its slots make."""

    def test_when_someone_was_somewhere(self):
        _, turns = talk("Mary went to the kitchen.", "Mary went to the garden.",
                        "when was Mary in the kitchen")
        self.assertEqual(turns[2].answer["outcome"], "retrieved")
        self.assertIn("“Mary went to the kitchen”", turns[2].answer["text"])

    def test_holding_any_whether_count(self):
        _, turns = talk("Mary went to the kitchen.", "Mary got the football.",
                        "Mary got the apple.",
                        "is anyone carrying the football",
                        "does anyone have the apple",
                        "does Mary have the football",
                        "is Mary carrying the apple",
                        "how many things does Mary have")
        self.assertEqual([said(one) for one in turns[3:]],
                         ["yes", "yes", "yes", "yes", "two"])

    def test_with_someone_else_is_not_with_her(self):
        _, turns = talk("Mary got the football.",
                        "Mary gave the football to John.",
                        "does Mary have the football")
        self.assertEqual(said(turns[2]), "no")
        self.assertEqual(turns[2].answer["outcome"], "denied")
        self.assertIn("with John", turns[2].answer["text"])

    def test_occurrences_any_count_and_place_with_object(self):
        _, turns = talk("Mary went to the kitchen.",
                        "Mary dropped the football in the kitchen.",
                        "John went to the kitchen.", "John went to the garden.",
                        "did anyone go to the garden",
                        "did anybody go to the office",
                        "how many people went to the kitchen",
                        "where did Mary drop the football")
        self.assertEqual(said(turns[4]), "yes")
        self.assertNotEqual(turns[5].answer.get("outcome"), "verified")
        self.assertEqual(said(turns[6]), "two")
        self.assertEqual(said(turns[7]), "the kitchen")


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
