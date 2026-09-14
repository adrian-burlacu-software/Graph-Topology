"""The encoder reader: roles, and readings built back from them.

No model is needed: every reading here is taken apart by the teacher
(`teach_reader.py`) and built again by `reader.build`, which must give the
grammar's reading back exactly. Over the people store, as `test_people` reads.
The model's own test needs `llm/reader` and torch.
"""
from __future__ import annotations

import unittest

from research.v689 import reader, test_people as people

NAMES = frozenset({"mary", "john", "fred", "bill", "sumit", "gertrude",
                   "greg"})


def setUpModule() -> None:                          # noqa: N802
    people.setUpModule()


def tearDownModule() -> None:                       # noqa: N802
    people.tearDownModule()


def lexicon():
    from research.v689.session import Session, Taught

    session = Session(people.PeopleAsker())
    return Taught(session.asker, session.memory.kinds)


class RoleTests(unittest.TestCase):
    def test_spans_and_marks(self):
        roles = ["O"] * 6
        reader.mark(roles, (1, 3), "SUBJ")
        reader.mark(roles, (4, 5), "OBJ")
        self.assertEqual(roles, ["O", "B-SUBJ", "I-SUBJ", "O", "B-OBJ", "O"])
        self.assertEqual(reader.spans(roles, "SUBJ"), [(1, 3)])
        self.assertEqual(reader.spans(roles, "OBJ"), [(4, 5)])
        self.assertEqual(reader.spans(["B-SUBJ", "B-SUBJ"], "SUBJ"),
                         [(0, 1), (1, 2)])

    def test_names_and_taught_kinds_are_said_as_ones_it_knows(self):
        first, second = reader.NAMED[:2]
        self.assertEqual(
            reader.said_as(["did", "inessa", "give", "kwame", "the", "ball",
                            "or", "did", "kwame", "keep", "it"],
                           frozenset({"inessa", "kwame"})),
            ["did", first, "give", second, "the", "ball", "or", "did",
             second, "keep", "it"])
        self.assertEqual(reader.said_as(["mary", "is", "here"], frozenset()),
                         ["mary", "is", "here"])
        kinds = {"wemble": "wemble.n.01"}
        self.assertEqual(
            reader.said_as(["what", "is", "a", "wemble"], frozenset(), kinds,
                           lambda word: word.rstrip("s")),
            ["what", "is", "a", "dog"])
        self.assertEqual(
            reader.said_as(["can", "wembles", "fly"], frozenset(), kinds,
                           lambda word: word.rstrip("s")),
            ["can", "dogs", "fly"])

    def test_no_placeholder_is_a_word_said_without_its_name(self):
        from research.v689 import teach_reader
        self.assertFalse(set(reader.NAMED) & set(teach_reader.PHRASES))


class RoundTripTests(unittest.TestCase):
    """What the grammar reads is what the roles build."""

    def assertRoundTrips(self, text: str, act: str, slots: str = "none"):
        from research.v689 import teach_reader

        record, why = teach_reader.label(text, lexicon(), NAMES)
        self.assertIsNotNone(record, f"{text}: {why}")
        self.assertEqual((record["act"], record["slots"]), (act, slots), text)

    def test_questions_of_each_relation(self):
        for text, act, slots in (
                ("where is Mary", "place located", "none"),
                ("what is Mary on", "place located", "none"),
                ("who is in the kitchen", "generic", "subject located"),
                ("where did Mary go first", "generic", "place occurrence"),
                ("who has the football", "subject occurrence",
                 "subject holding"),
                ("how many objects is Mary carrying", "count holding",
                 "none"),
                ("how many people are in the kitchen", "generic",
                 "count located"),
                ("who did Fred give the football to",
                 "recipient occurrence", "none"),
                ("what is the kitchen north of", "object dimension", "none"),
                ("how do you go from the kitchen to the garden",
                 "path dimension", "none"),
                ("what color is Greg", "value attribute", "none"),
                ("which dog is black", "which is_a", "none"),
                ("what did i tell you first", "events told", "none"),
                ("is Mary in the kitchen", "ask", "none"),
                ("who is Mary", "what", "none"),
                ("why can Mary swim", "why", "none"),
                ("can a dog swim", "generic", "none")):
            self.assertRoundTrips(text, act, slots)

    def test_a_statement_is_left_to_the_statement_reader(self):
        self.assertRoundTrips("Mary moved to the bathroom", "statement")

    def test_a_phrase_that_names_nobody_is_not_built(self):
        roles = ["O", "O", "B-SUBJ"]
        self.assertIsNone(reader.build(
            "place located", "none", "", roles, ["O"] * 3,
            ["where", "is", "shanda"], lexicon(), NAMES, "where is Shanda"))
        self.assertIsNotNone(reader.build(
            "place located", "none", "", roles, ["O"] * 3,
            ["where", "is", "mary"], lexicon(), NAMES, "where is Mary"))


@unittest.skipUnless(reader.enabled(), "needs llm/reader and torch")
class ModelTests(unittest.TestCase):
    def test_the_model_reads_a_question_it_was_taught(self):
        found, guess = reader.reading(["where", "is", "mary"], lexicon(),
                                      NAMES, "where is Mary")
        self.assertEqual(guess.acts[0][0], "place located")
        self.assertEqual(found.cells, [("place", "located")])


if __name__ == "__main__":
    unittest.main()
