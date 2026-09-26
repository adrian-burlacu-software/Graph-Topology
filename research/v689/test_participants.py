"""What the two taking part have and know (`participants.py`): told of the
program, asked of either, on the kinds store `test_kinds` builds."""
from __future__ import annotations

import unittest

from research.v689 import participants
from research.v689.test_kinds import (  # noqa: F401
    setUpModule, tearDownModule, talk)


def spoken(turn) -> str:
    return turn.answer.get("spoken") or ""


class ReadingTests(unittest.TestCase):

    def test_what_you_or_i_have_is_read_off_the_parse(self):
        owning, asked = participants.read("do you have a dog")
        self.assertEqual((owning.who, owning.verb, owning.kind, asked),
                         ("addressee", "have", "dog", True))
        owning, _ = participants.read("do i have any pets")
        self.assertEqual((owning.who, owning.kind, owning.plural),
                         ("speaker", "pet", True))
        self.assertEqual(participants.read("what do you have")[0].kind, "")

    def test_statements_about_others_are_not_its(self):
        for text in ("do dogs have fleas", "does john have a dog",
                     "i have a beagle", "can you cut bread with a knife",
                     "do you know if a dog can swim"):
            self.assertIsNone(participants.read(text), text)

    def test_a_state_of_mine_told_or_asked(self):
        owning, asked = participants.read("you like cake")
        self.assertEqual((owning.verb, owning.state, asked),
                         ("like", "verb.emotion", False))
        owning, asked = participants.read("have you ever seen a whale")
        self.assertEqual((owning.verb, owning.thing, asked),
                         ("see", "a whale", True))


class OwningTests(unittest.TestCase):

    def test_i_know_what_i_have(self):
        _, turns = talk("do you have a cat", "you have a cat",
                        "do you have an animal", "do you have a wolf",
                        "what do you have")
        self.assertEqual(turns[0].answer["outcome"], "denied")
        self.assertEqual(spoken(turns[1]), "Noted: I have a cat.")
        self.assertEqual(turns[2].answer["outcome"], "verified")
        self.assertEqual(turns[3].answer["outcome"], "denied")
        self.assertEqual(spoken(turns[4]), "I have the cat.")

    def test_what_you_have_is_only_what_you_said(self):
        _, turns = talk("i have a cat", "do i have an animal",
                        "do i have a wolf")
        self.assertEqual(turns[1].answer["outcome"], "verified")
        self.assertEqual(turns[2].answer["outcome"], "unknown")

    def test_my_likes_are_what_i_was_told(self):
        _, turns = talk("do you like apples", "you like apples",
                        "do you like apples")
        self.assertEqual(turns[0].answer["outcome"], "denied")
        self.assertEqual(turns[2].answer["outcome"], "verified")

    def test_nothing_said_of_me_reads_as_of_i(self):
        from research.v689.discourse import objective
        self.assertEqual(objective("I"), "me")
        self.assertEqual(objective("the cat"), "the cat")


if __name__ == "__main__":
    unittest.main()
