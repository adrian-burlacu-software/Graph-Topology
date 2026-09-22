"""Counts in a world: numbers, amounts, planning over them, and talking
about them (`numbers.py`, `quantities.py`, DESIGN §10j)."""
from __future__ import annotations

import unittest
from fractions import Fraction

from research.v687 import build
from research.v687.corpora import VERBNET_DIR
from research.v691 import acting, hearing, numbers, page, quantities as Q, \
    verbs, world as W

needs_parser = unittest.skipUnless(hearing.nlp() is not None, "no spaCy")
needs_verbnet = unittest.skipUnless(VERBNET_DIR.exists(), "no VerbNet")
needs_store = unittest.skipUnless(build.DEFAULT_STORE.exists(), "no store")


class NumberTests(unittest.TestCase):

    def test_what_number_words_are_worth(self):
        for said, worth in (("twenty one", 21), ("a dozen", 12),
                            ("two and a half", Fraction(5, 2)),
                            ("three quarters", Fraction(3, 4)),
                            ("half a dozen", 6), ("1,000", 1000),
                            ("3.5", Fraction(7, 2)), ("-3", -3),
                            ("two hundred and five thousand", 205000)):
            with self.subTest(said=said):
                self.assertEqual(numbers.value(said), worth)
        for said in ("apple", "a", "", "half of"):
            with self.subTest(said=said):
                self.assertIsNone(numbers.value(said))

    def test_sums_said_aloud_are_exact(self):
        for said, worth in (("what is 17 times 4", "68"),
                            ("what is 3 plus 4 times 2", "11"),
                            ("subtract 3 from 10", "7"),
                            ("what is the sum of 3, 4 and 5", "12"),
                            ("how much is 12 divided by 5", "2.4"),
                            ("what is one third plus one third", "2/3"),
                            ("what is 1/3 times 3", "1"),
                            ("what is 50% of 80", "40"),
                            ("what is the square root of 16", "4"),
                            ("what is (3 + 4) * 2", "14"),
                            ("what is 2 to the power of 10", "1024")):
            with self.subTest(said=said):
                self.assertEqual(numbers.said(numbers.evaluate(said)), worth)

    def test_what_is_not_a_sum_is_refused(self):
        """`what is a dog` is v688's, and `what is five` is not a sum."""
        for said in ("what is a dog", "what is five", "i have 5 apples",
                     "what is 3 divided by 0"):
            with self.subTest(said=said):
                with self.assertRaises(numbers.NotArithmetic):
                    numbers.evaluate(said)

    def test_counts_are_said_with_their_noun(self):
        self.assertEqual(numbers.counted(1, "apple"), "1 apple")
        self.assertEqual(numbers.counted(3, "mouse"), "3 mice")
        self.assertEqual(numbers.counted(2, "box"), "2 boxes")


class AmountTests(unittest.TestCase):

    def test_an_amount_rides_in_the_predicate(self):
        """The arguments of a counted fact are still things."""
        found = Q.read("with=3 apple mary")
        self.assertEqual((found.predicate, found.op, found.value,
                          found.args), ("with", "=", 3, ("apple", "mary")))
        self.assertEqual("with=3 apple mary".split()[1:], ["apple", "mary"])
        self.assertIsNone(Q.read("with apple mary"))

    def test_conditions_compare_what_is_known(self):
        facts = {"with=3 apple mary", "with=2+ apple john"}
        self.assertTrue(Q.holds("with>=3 apple mary", facts))
        self.assertFalse(Q.holds("with>3 apple mary", facts))
        self.assertTrue(Q.holds("with=3 apple mary", facts))
        # A lower bound answers `at least`, and never `at most`.
        self.assertTrue(Q.holds("with>=2 apple john", facts))
        self.assertFalse(Q.holds("with<=5 apple john", facts))
        # A count nobody said is not zero.
        self.assertFalse(Q.holds("with<=5 apple sam", facts))

    def test_changes_keep_what_kind_of_knowing_it_was(self):
        facts = Q.changed({"with=3 apple mary"},
                          [("with apple mary", 2), ("with apple sam", 4),
                           ("with apple tom", -1)])
        self.assertIn("with=5 apple mary", facts)
        # Given four, nobody said how many before: at least four.
        self.assertIn("with=4+ apple sam", facts)
        # Taken from a count nobody knows: still not known.
        self.assertFalse(any("tom" in fact for fact in facts))

    def test_an_order_for_more_is_what_it_comes_to(self):
        facts = {"with=3 apple mary"}
        self.assertEqual(Q.resolved("with+=2 apple mary", facts),
                         "with=5 apple mary")
        self.assertEqual(Q.resolved("with+=2 apple sam", facts),
                         "with>=2 apple sam")
        self.assertIsNone(Q.resolved("with-=2 apple sam", facts))

    def test_lifting_an_action_on_one_thing_to_an_amount(self):
        give = W.Action("give you apple mary", frozenset({"with apple you"}),
                        frozenset({"with apple mary"}),
                        frozenset({"with apple you"}), doer="you")
        made = Q.lifted(give, "apple", 2)
        self.assertEqual(made.name, "give you apple mary #2")
        self.assertEqual(made.needs, frozenset({"with>=2 apple you"}))
        self.assertEqual(dict(made.changes), {"with apple mary": 2,
                                              "with apple you": -2})

    def test_what_a_verb_does_not_say_comes_from_the_doer(self):
        """put-9.1 says where things end up and not where they were: of
        apples, they were the doer's."""
        put = W.Action("put you apple basket", frozenset(),
                       frozenset({"at apple basket"}), frozenset(),
                       doer="you")
        made = Q.lifted(put, "apple", 3)
        self.assertEqual(dict(made.changes), {"at apple basket": 3,
                                              "with apple you": -3})

    def test_nothing_comes_from_nowhere(self):
        """A doer who gains with nobody losing is buying or finding, which
        a plan cannot do."""
        get = W.Action("get you apple", frozenset(),
                       frozenset({"with apple you"}), frozenset(),
                       doer="you")
        self.assertIsNone(Q.lifted(get, "apple", 3))

    def test_the_amounts_offered_are_what_is_short(self):
        facts = {"with=3 apple mary"}
        self.assertEqual(Q.amounts_for(["with=5 apple mary"], facts),
                         [2, 5])


def _problem(facts, goal):
    give = W.Action("give you apple mary", frozenset({"with apple you"}),
                    frozenset({"with apple mary"}),
                    frozenset({"with apple you"}), doer="you")
    take = W.Action("take you apple basket", frozenset({"at apple basket"}),
                    frozenset({"with apple you"}),
                    frozenset({"at apple basket"}), doer="you")
    ground = []
    asked = list(goal)
    for _ in range(2):
        amounts = Q.amounts_for(asked, facts)
        ground += [made for action in (give, take) for n in amounts
                   if (made := Q.lifted(action, "apple", n)) is not None]
        asked = Q.needed(ground)
    return W.Problem("counts", frozenset(facts), frozenset(goal),
                     tuple({one.name: one for one in ground}.values()))


class PlanningTests(unittest.TestCase):

    def test_it_moves_exactly_what_is_short(self):
        """Mary has 3 and should have 5; I have 1, the basket 10. Take the
        one I am short of, then give two."""
        problem = _problem({"with=1 apple you", "at=10 apple basket",
                            "with=3 apple mary"}, {"with=5 apple mary"})
        got = acting.solve(problem)
        self.assertTrue(got.solved)
        self.assertEqual([one.name for one in got.plan],
                         ["take you apple basket #1",
                          "give you apple mary #2"])
        self.assertTrue(got.shortest)

    def test_what_is_not_there_is_not_planned(self):
        problem = _problem({"with=1 apple you", "with=3 apple mary"},
                           {"with=6 apple mary"})
        self.assertFalse(acting.solve(problem).solved)


@needs_parser
@needs_verbnet
class HearingTests(unittest.TestCase):

    def heard(self, text, kind="", who=""):
        return hearing.hear(text, verbs.stated, kind=kind, who=who)

    def test_counts_stated(self):
        for said, facts in (("mary has 3 apples", ["with=3 apple mary"]),
                            ("i have five apples", ["with=5 apple you"]),
                            ("there are 12 eggs in the basket",
                             ["at=12 egg basket"])):
            with self.subTest(said=said):
                self.assertEqual(self.heard(said).facts, facts)

    def test_what_a_verb_does_to_a_count_is_verbnets(self):
        """No list of verbs that give, take, lose or eat."""
        for said, changes in (
                ("i gave 2 apples to mary", {"with apple you": -2,
                                             "with apple mary": 2}),
                ("mary gave john three apples", {"with apple mary": -3,
                                                 "with apple john": 3}),
                ("john ate 2 apples", {"with apple john": -2}),
                ("tom lost 3 marbles", {"with marble tom": -3}),
                ("mary found 4 shells", {"with shell mary": 4})):
            with self.subTest(said=said):
                self.assertEqual(dict(self.heard(said).changes), changes)

    def test_a_bare_number_counts_what_was_counted_and_she_is_who(self):
        found = self.heard("she gave 2 to john", kind="apple", who="mary")
        self.assertEqual(dict(found.changes), {"with apple mary": -2,
                                               "with apple john": 2})

    def test_an_order_for_an_amount(self):
        self.assertEqual(self.heard("give mary 2 apples").wants,
                         ["with+=2 apple mary"])
        self.assertEqual(self.heard("put 3 apples in the basket").wants,
                         ["at+=3 apple basket"])

    def test_what_a_verb_does_not_say_of_an_event_comes_from_the_doer(self):
        """VerbNet's hand says where the coins went and not where from."""
        self.assertEqual(
            dict(self.heard("peter handed 3 coins to kate").changes),
            {"with coin peter": -3, "with coin kate": 3})

    def test_a_place_said_through_two_prepositions(self):
        self.assertEqual(
            dict(self.heard("i took 4 plums out of the bowl").changes),
            {"at plum bowl": -4, "with plum you": 4})

    def test_leaving_a_place_is_one_loss(self):
        self.assertEqual(self.heard("6 people got off the bus").changes,
                         [("at people bus", -6)])

    def test_each_is_a_rate_and_is_not_one_count(self):
        self.assertEqual(self.heard("each box has 6 eggs").facts, [])

    def test_holders_told_apart_by_what_is_said_of_them(self):
        self.assertEqual(self.heard("the red box has 7 toys").facts,
                         ["with=7 toy red-box"])

    def test_something_done_that_is_not_a_move_is_doubted(self):
        found = self.heard("3 balloons popped")
        self.assertEqual((found.changes, found.unsure),
                         ([], [("balloon", "")]))

    def test_questions_of_how_many(self):
        asked = self.heard("how many more apples does john have than mary")
        self.assertEqual((asked.count["kind"], asked.count["holders"],
                          asked.count["compare"], asked.count["than"]),
                         ("apple", ["john"], "more", ["mary"]))
        asked = self.heard("how many pears does sam have")
        self.assertEqual(asked.count["kind"], "pear")
        asked = self.heard("how many apples do mary and john have together")
        self.assertTrue(asked.count["total"])


@needs_parser
@needs_verbnet
@needs_store
class ConversationTests(unittest.TestCase):

    def setUp(self):
        from research.v691.learned import Learned
        from research.v691.openworld import Open, resolver
        from research.v691.scene import Scene
        self.scene = Scene(Open(resolver(), Learned(None)))

    def say(self, text):
        return page.say_to(self.scene, text)

    def test_counting_what_was_done(self):
        self.say("mary has 3 apples")
        self.say("i have five apples")
        self.say("i gave 2 apples to mary")
        self.assertEqual(self.say("how many apples does mary have"),
                         "mary has 5 apples")
        self.say("she ate 1")
        self.assertEqual(self.say("how many apples does she have"),
                         "mary has 4 apples")
        self.assertEqual(
            self.say("how many more apples do i have than mary"),
            "you have 1 fewer apple than mary")
        self.assertIn("7 apples together",
                      self.say("how many apples do mary and i have "
                               "together"))

    def test_an_order_is_carried_out_with_what_there_is(self):
        self.say("i have 3 apples")
        self.say("there are 10 apples in the basket")
        said = self.say("give john 4 apples")
        self.assertEqual(said, "I took 1 apple from the basket, then gave "
                               "4 of your apples to john")
        self.assertEqual(self.say("how many apples are in the basket"),
                         "there are 9 apples in the basket")

    def test_it_asks_rather_than_give_away_what_is_not_mine(self):
        """Mary's apples are Mary's: short of apples, it asks."""
        self.say("mary has 5 apples")
        self.say("i have 3 apples")
        said = self.say("give john 4 apples")
        self.assertIn("where would the other 1 apple come from", said)
        self.assertEqual(self.say("how many apples does mary have"),
                         "mary has 5 apples")

    def test_a_count_nobody_said_is_asked_for_and_the_answer_resumes(self):
        said = self.say("give sam 3 pears")
        self.assertIn("How many pears do you have?", said)
        self.assertEqual(self.say("6"), "then I gave 3 of your pears to sam")
        self.assertEqual(self.say("how many pears do i have"),
                         "you have 3 pears")

    def test_a_count_it_cannot_follow_is_said_to_be_lost(self):
        """Never the old count, stated as if nothing had happened."""
        self.say("max had 11 balloons")
        self.say("3 balloons popped")
        self.assertIn("lost count",
                      self.say("how many balloons does max have"))

    def test_taken_from_nobody_knows_where_may_be_the_jar(self):
        self.say("there were 8 cookies in the jar")
        self.say("the children ate 5 cookies")
        self.assertIn("lost count",
                      self.say("how many cookies are in the jar"))

    def test_a_name_that_is_a_verb_is_not_an_order(self):
        """VerbNet has `bob`; the parse has Bob."""
        self.assertNotEqual(page.hear("bob has 5 pens", self.scene).act,
                            "want")

    def test_a_sum_is_worked_out(self):
        self.assertEqual(self.say("what is 17 times 4"), "that is 68")

    def test_how_many_is_not_taken_without_counts(self):
        """`how many legs does a spider have` is v688's."""
        heard = page.hear("how many legs does a spider have", self.scene)
        self.assertNotEqual(heard.act, "count")


if __name__ == "__main__":
    unittest.main()
