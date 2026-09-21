"""Telling the agent what to do, and what it says back.

Fast for the same reason `test_v691` is: no reader, no store. What is read
here is twelve phrasings, and the point of keeping it that small is that a
failure in this suite is the planner's and not the parser's.
"""
from __future__ import annotations

import unittest

from research.v691 import acting, talking, world as W


def consistent(world: W.World) -> list:
    """Every way a set of block facts can be nonsense, as a list of
    complaints -- because `Table._put` edits the world by hand and a hand
    that forgets `clear` is how the conversation broke the first time."""
    facts = set(world.facts)
    blocks = {one.split()[1] for one in facts
              if one.split()[0] in ("table", "clear", "held")}
    blocks |= {one.split()[1] for one in facts if one.startswith("on ")}
    blocks |= {one.split()[2] for one in facts if one.startswith("on ")}
    wrong = []
    held = {one.split()[1] for one in facts if one.startswith("held ")}
    if len(held) > 1:
        wrong.append(f"holding {len(held)} blocks")
    if ("empty" in facts) == bool(held):
        wrong.append("`empty` disagrees with what is held")
    for block in sorted(blocks):
        places = [one for one in facts
                  if one == f"table {block}" or one == f"held {block}"
                  or (one.startswith("on ") and one.split()[1] == block)]
        if len(places) != 1:
            wrong.append(f"{block} is in {len(places)} places: {places}")
        above = [one for one in facts
                 if one.startswith("on ") and one.split()[2] == block]
        if len(above) > 1:
            wrong.append(f"{len(above)} blocks on {block}")
        clear = f"clear {block}" in facts
        if clear != (not above and block not in held):
            wrong.append(f"`clear {block}` is {clear} with {above}")
    return wrong


class ReadingTests(unittest.TestCase):
    """What an utterance is taken to be."""

    KNOWN = ["red", "green", "blue"]

    def act(self, text):
        return talking.read(text, self.KNOWN).act

    def test_each_phrasing_finds_its_act(self):
        for text, act in (
                ("there is a red block on the table", "describe"),
                ("there are three blocks", "describe"),
                ("the red block is on the green block", "describe"),
                ("put the red block on the green block", "want"),
                ("move the blue block onto the table", "want"),
                ("make a tower of red on green on blue", "want"),
                ("what is on the red block", "on top of"),
                ("where is the green block", "where"),
                ("what do you see", "look"),
                ("why did you move the red block", "why"),
                ("actually the red block is on the table now", "meddle"),
                ("reset", "reset"),
                ("quit", "leave"),
                ("sing me a song", "puzzled")):
            with self.subTest(said=text):
                self.assertEqual(self.act(text), act)

    def test_a_tower_chains_and_two_placings_do_not(self):
        """`and` is the whole of the difference, and it has to be, because
        `red on green on blue` and `red on green and blue on yellow` are the
        same words in the same order otherwise."""
        self.assertEqual(talking._placings("red on green on blue", []),
                         ["on red green", "on green blue"])
        self.assertEqual(
            talking._placings("red on green and blue on yellow", []),
            ["on red green", "on blue yellow"])

    def test_a_block_it_has_never_heard_of_is_refused_by_name(self):
        heard = talking.read("put the yellow block on the red block",
                             self.KNOWN)
        self.assertEqual(heard.act, "want")
        self.assertIn("yellow", heard.trouble)

    def test_an_order_with_no_blocks_in_it_says_so(self):
        heard = talking.read("put it over there", self.KNOWN)
        self.assertTrue(heard.trouble)


class NarratingTests(unittest.TestCase):
    """Saying back what was done."""

    def named(self, *names):
        return [W.Action(one, frozenset(), frozenset(), frozenset())
                for one in names]

    def test_a_pick_up_and_a_stack_are_one_thing_said(self):
        """Nobody says `I picked up the red block, then I put the red block
        on the green block`."""
        self.assertEqual(talking.narrate(self.named("take red",
                                                    "stack red green")),
                         "I put the red block on the green block")

    def test_an_unstack_and_a_stack_are_a_move(self):
        self.assertEqual(
            talking.narrate(self.named("unstack red green",
                                       "stack red blue")),
            "I moved the red block from the green block to the blue block")

    def test_a_tower_is_read_from_the_top_down(self):
        world = W.World(W.start_of([["blue", "green", "red"]]))
        self.assertEqual(talking.describe(world),
                         "the red block is on the green block, which is on "
                         "the blue block, which is on the table")

    def test_a_fact_has_words(self):
        self.assertEqual(talking.in_words("clear red"), "the red block clear")
        self.assertEqual(talking.in_words("on red green"),
                         "the red block on the green block")
        self.assertEqual(talking.in_words("empty"), "my hand empty")


class TalkingTests(unittest.TestCase):
    """The conversation, end to end."""

    def test_the_sussman_anomaly_in_english(self):
        """The whole of v691, in three sentences of English: a scene, a
        conjunctive goal whose parts interfere, and the actions taken."""
        table = talking.Table()
        table.say("there is a red block on a green block, and a blue block "
                  "on the table")
        said = table.say("put the green block on the blue block and the red "
                         "block on the green block")
        self.assertEqual(said,
                         "I took the red block off the green block and put "
                         "it on the table, put the green block on the blue "
                         "block, then put the red block on the green block")
        self.assertTrue(table.world.solved({"on green blue", "on red green"}))
        self.assertEqual(consistent(table.world), [])

    def test_it_answers_about_the_scene_it_has(self):
        table = talking.Table()
        table.say("there is a red block on a green block, and a blue block "
                  "on the table")
        self.assertEqual(table.say("where is the red block"),
                         "the red block is on the green block")
        self.assertEqual(table.say("what is on the green block"),
                         "the red block is on the green block")
        self.assertEqual(table.say("what is on the blue block"),
                         "nothing is on the blue block")

    def test_why_reads_the_means_ends_subgoal_back(self):
        """The planner's subgoal names already are the explanation --
        `achieve clear green for take green` -- so nothing is invented."""
        table = talking.Table()
        table.say("there is a red block on a green block, and a blue block "
                  "on the table")
        table.say("put the green block on the blue block")
        said = table.say("why did you move the red block")
        self.assertEqual(said,
                         "I took the red block off the green block because "
                         "I needed the green block clear before I could "
                         "pick up the green block")

    def test_why_before_anything_was_done(self):
        self.assertEqual(talking.Table().say("why did you do that"),
                         "I have not done anything yet")

    def test_meddling_leaves_a_world_that_still_makes_sense(self):
        """Editing a world by hand is fiddly in exactly the way a delete
        list makes unnecessary, which is the argument for the world being
        facts with actions over them rather than a picture."""
        table = talking.Table()
        table.say("there is a red block on a green block on a blue block")
        for said in ("actually the red block is on the table now",
                     "actually the blue block is on the red block now",
                     "now the green block is on the blue block"):
            with self.subTest(said=said):
                table.say(said)
                self.assertEqual(consistent(table.world), [])

    def test_it_plans_again_from_where_the_world_now_is(self):
        table = talking.Table()
        table.say("there is a red block on a green block, and a blue block "
                  "on the table")
        table.say("put the green block on the blue block and the red block "
                  "on the green block")
        table.say("actually the red block is on the table now")
        table.say("put the red block on the green block")
        self.assertTrue(table.world.solved({"on red green"}))
        self.assertEqual(consistent(table.world), [])

    def test_it_says_what_it_cannot_read(self):
        table = talking.Table()
        self.assertIn("blocks on a table", table.say("sing me a song"))

    def test_reset_empties_the_table(self):
        table = talking.Table()
        table.say("there is a red block on a green block")
        self.assertEqual(table.say("reset"), "the table is empty")
        self.assertEqual(table.blocks, [])


class SurpriseTests(unittest.TestCase):
    """A block falling off while the agent is working."""

    def setUp(self):
        self.table = talking.Table(talking.topple)
        self.table.say("there is a red block, a green block and a blue "
                       "block on the table")

    def test_it_notices_replans_and_says_so(self):
        said = self.table.say("put the red block on the green block and the "
                              "green block on the blue block")
        self.assertIn("something moved while I was working", said)
        self.assertIn("planned again", said)
        self.assertTrue(self.table.world.solved(
            {"on red green", "on green blue"}))
        self.assertEqual(self.table.last.surprises, 1)

    def test_the_story_is_told_in_stretches(self):
        """After a surprise the agent covers ground it has already covered,
        and one flat list of actions reads as a stutter."""
        said = self.table.say("put the red block on the green block and the "
                              "green block on the blue block")
        self.assertEqual(said.count("Then something moved"), 1)
        first = said.split(".")[0]
        self.assertEqual(first, "I put the green block on the blue block")

    def test_the_gap_is_kept_as_a_prediction_that_failed(self):
        self.table.say("put the red block on the green block and the green "
                       "block on the blue block")
        gap = self.table.last.gaps[0]
        self.assertIsInstance(gap, acting.Gap)
        self.assertIn("on green blue", gap.missing)
        self.assertIn("table green", gap.extra)


if __name__ == "__main__":
    unittest.main()
