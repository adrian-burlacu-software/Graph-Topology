"""The encoder reader: roles, and readings built back from them.

No model is needed: every reading here is taken apart by the teacher
(`teach_reader.py`) and built again by `reader.build`, which must give the
grammar's reading back exactly. Over the people store, as `test_people` reads.
The model's own test needs `llm/reader` and torch, as everything that reads
does.
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

    def test_statements_of_each_act(self):
        for text, act in (
                ("Mary moved to the bathroom", "tell"),
                ("Fred picked up the football", "tell"),
                ("Mary and Daniel went to the kitchen", "tell"),
                ("there is a beagle that can't swim", "introduce"),
                ("i have a beagle", "introduce owned"),
                ("my name is Adrian", "name"),
                ("beagles can not swim", "teach")):
            self.assertRoundTrips(text, act)

    def test_several_claims_are_read_claim_by_claim(self):
        from research.v689 import teach_reader

        record, why = teach_reader.label(
            "beagles can swim, and they can bark", lexicon(), NAMES)
        self.assertIsNotNone(record, why)
        self.assertEqual(record["act"], "teach")
        self.assertIn("B-teach", record["clauses"])

    def test_a_request_is_said_back_as_its_question(self):
        from research.v689 import teach_reader

        for text in ("do you know if a dog can swim", "describe a cat",
                     "can you cut bread with a knife", "name three birds",
                     "why can't a penguin fly", "can a dog swim"):
            record, why = teach_reader.label_ask(text)
            self.assertIsNotNone(record, f"{text}: {why}")

    def test_an_utterance_is_placed_as_the_rules_place_it(self):
        from research.v689 import teach_reader

        for text in ("yesterday Mary went to the kitchen",
                     "after Mary went to the kitchen, John slept",
                     "then she went to the garden",
                     "can you tell me where Mary is",
                     "the apple was given to Fred by Bill"):
            record, why = teach_reader.label_place(text, lexicon(), NAMES)
            self.assertIsNotNone(record, f"{text}: {why}")

    def test_a_question_is_parsed_as_the_cues_parse_it(self):
        from research.v689 import teach_reader

        for text in ("can a mouse fall into a hole", "is a mouse an animal",
                     "what is a ball made of", "where do you find a ball",
                     "does a mouse have a tail"):
            record, why = teach_reader.label_parse(text,
                                                   people.STORE["parser"])
            self.assertIsNotNone(record, f"{text}: {why}")

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

    def test_the_model_asks_places_and_parses(self):
        from research.v688.rephrase import rephrase

        self.assertEqual(rephrase("do you know if a dog can swim").text,
                         "can a dog swim")
        found = reader.place("yesterday Mary went to the kitchen", lexicon(),
                             frozenset({"mary"}))
        self.assertEqual((found.tokens, found.when.frame.key),
                         (["mary", "went", "to", "the", "kitchen"],
                          "yesterday"))
        parse = people.STORE["parser"].parse("is a mouse an animal")
        self.assertEqual((parse.subject, parse.relation, parse.target),
                         ("mouse", "is_a", "animal"))


if __name__ == "__main__":
    unittest.main()
