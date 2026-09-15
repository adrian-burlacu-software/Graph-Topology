"""Who is who, where, and with what: names, groups, pronouns, places and
possession in v689.

Run: python -m unittest research.v689.test_people -v

The language of bAbI's stories (`babi.py`), and nothing of bAbI in the code:
someone new is named as they are talked about, `she` is the woman last
talked about, `Mary and Daniel` are each told what they did together, and
`they` is the two of them. A place is where someone is until they go
somewhere else (T3, one place at a time), `no longer` and `not` deny one, and
`either ... or` leaves two open. What someone gets is with them, and goes
where they go, until they drop it or give it away (T4, VerbNet's
`ch_of_poss`). Like `test_time`, it runs on v687's own `Reasoner` and `Parser`
over a small store built here; the verbs are WordNet's own synsets, because
VerbNet is joined to them by sense.
"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from research.v687.language import Parser
from research.v687.reason import Reasoner
from research.v689 import change, reader, reading
from research.v689.asker import Asker
from research.v689.session import Session
from research.v689.test_v689 import SCHEMA

#: (id, lemmas, parent)
CONCEPTS = (
    ("entity.n.01", ("entity",), None),
    ("organism.n.01", ("organism",), "entity.n.01"),
    ("person.n.01", ("person",), "organism.n.01"),
    ("animal.n.01", ("animal",), "organism.n.01"),
    ("mouse.n.01", ("mouse",), "animal.n.01"),
    ("room.n.01", ("room",), "entity.n.01"),
    ("kitchen.n.01", ("kitchen",), "room.n.01"),
    ("bathroom.n.01", ("bathroom",), "room.n.01"),
    ("bedroom.n.01", ("bedroom",), "room.n.01"),
    ("hallway.n.01", ("hallway",), "entity.n.01"),
    ("office.n.01", ("office",), "room.n.01"),
    ("garden.n.01", ("garden",), "entity.n.01"),
    ("school.n.01", ("school",), "entity.n.01"),
    ("park.n.02", ("park",), "entity.n.01"),
    ("ball.n.01", ("ball", "football"), "entity.n.01"),
    ("milk.n.01", ("milk",), "entity.n.01"),
    ("apple.n.01", ("apple",), "entity.n.01"),
    ("travel.v.01", ("travel", "go", "move"), None),
    ("travel.v.02", ("journey",), "travel.v.01"),
    ("get.v.01", ("get",), None),
    ("transport.v.02", ("carry",), None),
    ("drop.v.01", ("drop",), None),
    ("discard.v.01", ("discard",), None),
    ("leave.v.01", ("leave",), None),
    ("leave.v.02", ("leave",), None),
    ("pick.v.02", ("pick",), None),
    ("put.v.01", ("put",), None),
    ("give.v.03", ("give",), None),
    ("pass.v.05", ("pass", "hand"), "give.v.03"))

STORE: dict = {}


def setUpModule() -> None:                      # noqa: N802
    folder = tempfile.TemporaryDirectory()
    path = Path(folder.name) / "people.sqlite"
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


class PeopleAsker(Asker):
    """v687 over the people store; v688 knows nothing."""

    def __init__(self) -> None:
        super().__init__(STORE["reasoner"], STORE["parser"])
        self.asked: list[str] = []

    def run(self, question: str) -> dict:
        self.asked.append(question)
        return {"summary": {"outcome": "unknown", "trust": "",
                            "lines": [f"unknown — {question}"]}}


def talk(*lines: str):
    asker = PeopleAsker()
    session = Session(asker)
    return session, [session.say(line) for line in lines]


def said(turn) -> str:
    """What a reply answers: the part before its reasons."""
    return (turn.answer.get("text") or "").split(" — ", 1)[0]


def verbnet() -> bool:
    return bool(change.frames())


class NameReadingTests(unittest.TestCase):
    """Someone new is named by being talked about; a question names no one."""

    @classmethod
    def setUpClass(cls):
        cls.lexicon = PeopleAsker()

    def read(self, text: str, names=frozenset()):
        return reading.read(text, self.lexicon, names)

    def test_a_capitalised_subject_is_someone_new(self):
        found = self.read("Mary moved to the bathroom.")
        self.assertEqual(found.act, "tell")
        self.assertEqual(found.mention.form, "name")
        self.assertTrue(found.mention.fresh)

    def test_a_question_names_no_one(self):
        found = self.read("Where is Mary?")
        self.assertIsNone(found.mention)

    def test_a_named_kind_is_still_naming(self):
        found = self.read("Winona is a mouse")
        self.assertEqual(found.act, "introduce")
        self.assertEqual(found.name, "Winona")

    def test_a_plural_kind_is_not_a_name(self):
        self.assertEqual(reader.place("Mice are afraid of rooms",
                                      self.lexicon).fresh, frozenset())

    def test_two_names_are_a_group(self):
        found = self.read("Mary and Daniel went to the kitchen")
        self.assertEqual(found.mention.form, "group")
        self.assertEqual([one.name for one in found.mention.members],
                         ["mary", "daniel"])
        self.assertTrue(all(one.fresh for one in found.mention.members))
        self.assertEqual(found.rest[:2], ["went", "to"])

    def test_no_longer_denies(self):
        found = self.read("Mary is no longer in the bedroom", {"mary"})
        self.assertFalse(found.holds)
        self.assertEqual(found.rest, ["in", "the", "bedroom"])

    def test_a_trailing_there_is_where_the_subject_is(self):
        found = self.read("Mary got the milk there", {"mary"})
        self.assertEqual(found.rest[-1], "milk")
        self.assertEqual(found.obj.kind, "milk")

    def test_following_that_is_a_link(self):
        found = self.read("Following that she went back to the kitchen")
        self.assertEqual(found.when.link, "after")
        self.assertEqual(found.mention.text, "she")

    def test_where_before_a_place(self):
        found = self.read("Where was Julie before the school?", {"julie"})
        self.assertIn(("place", "located"), found.cells)
        self.assertEqual(found.rest, ["before", "the", "school"])

    def test_what_someone_is_carrying(self):
        found = self.read("What is Mary carrying?", {"mary"})
        self.assertIn(("object", "holding"), found.cells)
        self.assertEqual((found.rest, found.count), (["carry"], False))
        found = self.read("How many objects is Mary carrying?", {"mary"})
        self.assertIn(("count", "holding"), found.cells)
        self.assertTrue(found.count)

    def test_who_it_went_to(self):
        found = self.read("Who did Fred give the milk to?", {"fred"})
        self.assertIn(("recipient", "occurrence"), found.cells)
        self.assertEqual(found.rest, ["give", "the", "milk", "to"])
        self.assertEqual(found.obj.kind, "milk")


class WhoTests(unittest.TestCase):
    """Names, pronouns and groups resolve to the people talked about."""

    def test_a_name_used_again_is_the_same_one(self):
        session, turns = talk("Mary moved to the bathroom.",
                              "Mary went to the kitchen.")
        self.assertEqual(len(session.discourse.referents), 3)
        self.assertEqual(turns[0].resolution.referent.id,
                         turns[1].resolution.referent.id)
        self.assertEqual(turns[0].resolution.referent.name, "Mary")

    def test_she_is_the_woman_not_the_man_talked_about_since(self):
        _, turns = talk("Mary went to the kitchen.", "John went to the garden.",
                        "Then she went to the office.", "Where is Mary?",
                        "Where is John?")
        self.assertEqual(said(turns[3]), "Mary: the office")
        self.assertEqual(said(turns[4]), "John: the garden")

    def test_a_name_that_says_nothing_of_gender_is_still_he(self):
        _, turns = talk("Daniel went to the garden.",
                        "After that he went to the hallway.",
                        "Where is Daniel?")
        self.assertEqual(said(turns[2]), "Daniel: the hallway")

    def test_they_are_the_two_talked_about_together(self):
        _, turns = talk("Mary and Daniel went to the kitchen.",
                        "John went to the office.",
                        "Then they journeyed to the hallway.",
                        "Where is Daniel?", "Where is John?")
        self.assertEqual(said(turns[3]), "Daniel: the hallway")
        self.assertEqual(said(turns[4]), "John: the office")

    def test_together_is_at_the_same_time(self):
        session, _ = talk("Mary and Daniel went to the kitchen.")
        first, second = session.timeline.occurrences
        self.assertEqual(session.timeline.relation(first.id, second.id),
                         "during")

    def test_they_with_no_one_together_is_not_about_individuals(self):
        _, turns = talk("can they swim")
        self.assertEqual(turns[0].act, "ask")
        self.assertIsNone(turns[0].resolution)


class WhereTests(unittest.TestCase):
    """T3: one place at a time, until told otherwise."""

    def test_the_last_place_gone_to(self):
        _, turns = talk("Mary moved to the bathroom.",
                        "Mary went to the kitchen.", "Where is Mary?")
        self.assertEqual(said(turns[2]), "Mary: the kitchen")

    def test_somewhere_else_is_no(self):
        _, turns = talk("Daniel moved to the hallway.",
                        "Is Daniel in the bathroom?",
                        "Is Daniel in the hallway?")
        self.assertEqual(turns[1].answer["outcome"], "denied")
        self.assertEqual(turns[2].answer["outcome"], "verified")

    def test_told_somewhere_else_is_no(self):
        _, turns = talk("Sandra is in the bathroom.",
                        "Is Sandra in the office?")
        self.assertEqual(turns[1].answer["outcome"], "denied")

    def test_no_longer_there_is_no(self):
        _, turns = talk("Mary is in the bedroom.",
                        "Mary is no longer in the bedroom.",
                        "Is Mary in the bedroom?")
        self.assertEqual(turns[2].answer["outcome"], "denied")

    def test_either_is_maybe_here_and_no_elsewhere(self):
        _, turns = talk("Fred is either in the school or the park.",
                        "Is Fred in the park?", "Is Fred in the office?")
        self.assertTrue(said(turns[1]).startswith("maybe"))
        self.assertEqual(turns[2].answer["outcome"], "denied")

    def test_going_somewhere_settles_either(self):
        _, turns = talk("Fred is either in the school or the park.",
                        "Fred moved to the kitchen.", "Is Fred in the park?")
        self.assertEqual(turns[2].answer["outcome"], "denied")

    def test_one_story_told_in_two_tenses(self):
        _, turns = talk("John is in the garden.",
                        "John moved to the bathroom.",
                        "Is John in the bathroom?")
        self.assertEqual(turns[2].answer["outcome"], "verified")

    def test_where_before_a_place_across_days(self):
        _, turns = talk("Yesterday Julie went to the office.",
                        "Julie went to the school this morning.",
                        "This afternoon Julie went to the park.",
                        "This evening Julie went to the school.",
                        "Where was Julie before the school?")
        self.assertEqual(said(turns[4]), "the park")

    def test_where_before_the_first_place_is_not_told(self):
        _, turns = talk("Julie went to the school.",
                        "Where was Julie before the school?")
        self.assertTrue(said(turns[1]).startswith("not told"))

    def test_replayed_it_is_still_there(self):
        session, _ = talk("Mary and Daniel went to the kitchen.",
                          "Then they went to the garden.")
        again = Session.rebuild(PeopleAsker(), session.conversation,
                                session.memory.log.conversation.events)
        turn = again.say("Where is Daniel?")
        self.assertEqual(said(turn), "Daniel: the garden")


@unittest.skipUnless(verbnet(), "needs data/verbnet3.3")
class PossessionTests(unittest.TestCase):
    """T4: what someone gets is with them, and where they are, until they
    let it go or give it away."""

    def test_what_she_got_goes_where_she_goes(self):
        _, turns = talk("Mary moved to the bathroom.",
                        "Mary got the football there.",
                        "Mary went to the garden.", "Where is the football?")
        self.assertTrue(said(turns[3]).startswith("the football: the garden"),
                        said(turns[3]))

    def test_what_she_dropped_stays_where_she_dropped_it(self):
        _, turns = talk("Mary moved to the bathroom.",
                        "Mary got the football there.",
                        "Mary dropped the football.",
                        "Mary went to the garden.", "Where is the football?")
        self.assertTrue(said(turns[4]).startswith(
            "the football: the bathroom"), said(turns[4]))

    def test_what_she_is_carrying_and_how_many(self):
        _, turns = talk("Mary went to the kitchen.",
                        "Mary got the football there.",
                        "Mary picked up the milk there.",
                        "What is Mary carrying?",
                        "How many objects is Mary carrying?")
        self.assertIn("football", said(turns[3]))
        self.assertIn("milk", said(turns[3]))
        self.assertTrue(said(turns[4]).startswith("two"), said(turns[4]))

    def test_discarded_and_put_down_are_let_go_of(self):
        _, turns = talk("John went to the kitchen.", "John got the milk.",
                        "John got the apple.", "John discarded the milk.",
                        "John put down the apple.", "What is John carrying?",
                        "How many objects is John carrying?")
        self.assertTrue(said(turns[5]).startswith("nothing"), said(turns[5]))
        self.assertTrue(said(turns[6]).startswith("none"), said(turns[6]))

    def test_left_is_chosen_by_what_the_story_holds(self):
        """`leave` is going away from a place (escape-51.1) and letting go of
        something (future_having-13.3); John is not at the apple, and has it."""
        session, turns = talk("John went to the kitchen.",
                              "John got the apple.", "John left the apple.",
                              "Where is John?", "What is John carrying?")
        self.assertEqual(said(turns[3]), "John: the kitchen")
        self.assertTrue(said(turns[4]).startswith("nothing"), said(turns[4]))

    def test_giving(self):
        _, turns = talk("Bill went to the office.", "Bill got the football.",
                        "Bill gave the football to Fred.",
                        "Fred passed the football to Jeff.",
                        "Who received the football?", "Who gave the football?",
                        "Who did Fred give the football to?",
                        "What did Bill give to Fred?",
                        "Who gave the football to Jeff?",
                        "Where is the football?")
        self.assertEqual(said(turns[4]), "Jeff")
        self.assertEqual(said(turns[5]), "Fred")
        self.assertEqual(said(turns[6]), "Jeff")
        self.assertEqual(said(turns[7]), "the football")
        self.assertEqual(said(turns[8]), "Fred")
        # Jeff was never said to be anywhere: it is with Jeff, and no more.
        self.assertEqual(said(turns[9]), "the football: Jeff")

    def test_where_it_was_before_it_was_carried_on(self):
        _, turns = talk("Mary went to the kitchen.", "Mary got the milk.",
                        "Mary went to the garden.",
                        "Where was the milk before the garden?")
        self.assertEqual(said(turns[3]), "the kitchen")

    def test_replayed_it_is_still_with_her(self):
        session, _ = talk("Mary went to the kitchen.", "Mary got the milk.",
                          "Mary went to the garden.")
        again = Session.rebuild(PeopleAsker(), session.conversation,
                                session.memory.log.conversation.events)
        turn = again.say("Where is the milk?")
        self.assertTrue(said(turn).startswith("the milk: the garden"),
                        said(turn))


if __name__ == "__main__":
    unittest.main()
