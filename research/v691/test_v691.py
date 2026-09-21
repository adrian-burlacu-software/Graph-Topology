"""v691: the world, the domain it is declared in, the plan, and the agent.

No reader of English and no store, so this suite is fast and a failure in it
is the architecture rather than the parser -- which is the whole reason
v691a was a blocks world. `test_scene.py` is where the talking is tested.
"""
from __future__ import annotations

import unittest

from research.v687.executive import (Chunks, Executive, Operator, Unwired,
                                     Working, pursuing)
from research.v691 import acting, domains, problems as P, world as W

BLOCKS = domains.DOMAINS["blocks"]


def blocks_actions(names="abc"):
    return BLOCKS.ground({one: "block" for one in names})


class DomainTests(unittest.TestCase):
    """A domain is a string, and everything else reads it."""

    def test_a_schema_grounds_into_actions(self):
        found = {one.name: one for one in blocks_actions("ab")}
        self.assertEqual(set(found), {
            "take a", "take b", "drop a", "drop b",
            "stack a b", "stack b a", "unstack a b", "unstack b a"})
        stack = found["stack a b"]
        self.assertEqual(stack.needs, frozenset({"held a", "clear b"}))
        self.assertEqual(stack.adds,
                         frozenset({"empty", "clear a", "on a b"}))
        self.assertEqual(stack.deletes, frozenset({"held a", "clear b"}))

    def test_a_schema_never_binds_one_object_twice(self):
        self.assertNotIn("stack a a",
                         [one.name for one in blocks_actions("ab")])

    def test_argument_kinds_come_from_the_schemas(self):
        """Nowhere is `in` declared to be a thing and a place; the action
        that needs it says so, and reading it off means a domain cannot
        declare its types twice and get them different."""
        self.assertEqual(domains.DOMAINS["errands"].typing()["in"],
                         ("thing", "place"))
        self.assertEqual(BLOCKS.typing()["on"], ("block", "block"))

    def test_a_line_it_cannot_read_is_an_error(self):
        """A domain that silently lost an action would fail as a planning
        result, which is the most expensive way to find a typo."""
        with self.assertRaises(ValueError):
            domains.parse("domain x\nwibble on ?x")
        with self.assertRaises(ValueError):
            domains.parse("domain x\naction go there")

    def test_every_shipped_domain_says_all_of_its_predicates(self):
        for name, domain in domains.DOMAINS.items():
            used = {one.split()[0] for schema in domain.schemas
                    for one in schema.needs + schema.adds + schema.deletes}
            with self.subTest(domain=name):
                self.assertEqual(used - set(domain.says), set())

    def test_a_fact_reads_the_way_it_is_said(self):
        self.assertEqual(BLOCKS.in_words("on red green"),
                         "the red block is on the green block")
        self.assertEqual(BLOCKS.phrase("unstack red green"),
                         "took the red block off the green block")
        self.assertEqual(BLOCKS.doing("take red"), "pick up the red block")


class WorldTests(unittest.TestCase):
    """Facts that change only by acting."""

    def setUp(self):
        self.problem = P.blocks("t", "abc", [["c", "a"], ["b"]],
                                ["on a b", "on b c"])
        self.world = self.problem.world()

    def test_a_configuration_becomes_facts(self):
        self.assertEqual(self.world.facts, frozenset({
            "table c", "on a c", "clear a", "table b", "clear b", "empty"}))

    def test_an_action_that_does_not_apply_changes_nothing(self):
        """The world refuses rather than the agent checking first: an agent
        that could only act on a world it had modelled rightly would never
        be surprised."""
        before = self.world.facts
        self.assertFalse(self.world.do(
            W.Action("take b", frozenset({"held a"}), frozenset(),
                     frozenset())))
        self.assertEqual(self.world.facts, before)
        self.assertEqual(self.world.did, [])

    def test_deletes_are_applied_before_adds(self):
        """`stack a b` deletes `clear b` and adds `clear a`. Adding first
        would let the delete undo the action's own work where they touch."""
        after = W.Action("stack a b", frozenset(), frozenset({"clear a"}),
                         frozenset({"clear a", "clear b"})).on(
                             frozenset({"clear a", "clear b"}))
        self.assertEqual(after, frozenset({"clear a"}))

    def test_an_operator_must_declare_that_it_changes_the_world(self):
        """The guard v687 put on the store, now on the thing the agent
        moves: nothing changes the world without saying so."""
        action = blocks_actions("ab")[0]
        world = W.World(P.start_of([["a"], ["b"]]))
        undeclared = Operator(name="act", apply=lambda memory: world.do(
            action) and "continue")
        with self.assertRaises(Unwired):
            Executive([undeclared], name="rogue").run(Working())


class SituationTests(unittest.TestCase):
    """Working memory over a world: the one place it must not behave like
    working memory over a belief."""

    def setUp(self):
        self.memory = acting.Situation({"clear a", "table a", "empty"})
        self.take = next(one for one in blocks_actions("ab")
                         if one.name == "take a")

    def test_a_fact_is_a_slot(self):
        self.assertIn("clear a", self.memory)
        self.assertTrue(self.memory["clear a"])
        self.assertNotIn("clear b", self.memory)

    def test_acting_retracts_as_well_as_asserts(self):
        self.memory.apply(self.take)
        self.assertIn("held a", self.memory)
        self.assertNotIn("clear a", self.memory)
        self.assertNotIn("empty", self.memory)

    def test_what_a_subgoal_did_outlives_the_subgoal(self):
        """`Working` is right for belief -- a subgoal that fails leaves
        nothing behind -- and wrong for a world. A subgoal that unstacked a
        block and then gave up has still unstacked it."""
        with self.memory.subgoal("get a"):
            self.memory.apply(self.take)
        self.assertIn("held a", self.memory)
        self.assertNotIn("clear a", self.memory)
        self.assertEqual([str(one) for one in self.memory.did], ["take a"])

    def test_a_chunk_is_keyed_on_the_world(self):
        """`keys` carries the facts, so the same impasse in a different
        arrangement is a different impasse."""
        self.assertIn("clear a", self.memory.keys())


class TasteTests(unittest.TestCase):
    """The utilities, and that none of them knows what a block is."""

    def setUp(self):
        self.problem = next(one for one in P.SUITE
                            if one.name == "four apart")

    def test_goal_facts_are_ordered_by_what_waits_on_them(self):
        """The general form of *build from the bottom*: `stack c d` leaves
        `clear c`, which `stack b c` needs."""
        self.assertEqual(
            acting.depths({"on a b", "on b c", "on c d"},
                          self.problem.actions),
            {"on c d": 0, "on b c": 1, "on a b": 2})

    def test_the_same_machinery_works_in_another_domain(self):
        """Nothing about it was ever about towers."""
        errands = domains.DOMAINS["errands"]
        objects = {"shop": "place", "home": "place", "book": "thing"}
        deep = acting.depths({"in book home"}, errands.ground(objects))
        self.assertEqual(deep, {"in book home": 0})

    def test_what_a_goal_relies_on_is_read_off_the_achievers(self):
        taste = acting.Taste.of({"on a b", "on b c", "on c d"},
                                self.problem.actions)
        # `clear b` is needed by `stack a b`, which achieves the deepest
        # goal fact, so taking it away is the worst thing to do.
        self.assertEqual(taste.relied["clear b"], 2)
        self.assertEqual(taste.relied["clear d"], 0)

    def test_no_signal_names_a_predicate_or_a_kind(self):
        for name in acting.SIGNALS:
            with self.subTest(signal=name):
                for word in ("block", "clear", "stack", "table", "tower",
                             "van", "parcel"):
                    self.assertNotIn(word, name.split())


class PlanningTests(unittest.TestCase):
    """Means-ends over a model, which is goal-stack planning."""

    def solve(self, name):
        problem = next(one for one in P.SUITE if one.name == name)
        return problem, acting.solve(problem)

    def test_the_sussman_anomaly_is_solved(self):
        """The reason this exists: achieving `on a b` and then `on b c` in
        turn undoes the first."""
        _, got = self.solve("sussman")
        self.assertTrue(got.solved)
        self.assertEqual(got.optimal, 6)
        self.assertEqual(got.acted, 6)

    def test_a_plan_is_the_actions_the_model_applied(self):
        problem, got = self.solve("three in a row")
        world = problem.world()
        for action in got.plan:
            self.assertTrue(world.do(action), f"{action} did not apply")
        self.assertTrue(world.solved(problem.goal))

    def test_the_goal_tower_is_built_from_the_bottom(self):
        _, got = self.solve("four apart")
        self.assertTrue(got.solved)
        stacked = [str(one) for one in got.plan
                   if str(one).startswith("stack")]
        self.assertEqual(stacked, ["stack c d", "stack b c", "stack a b"])
        self.assertEqual(got.acted, got.optimal)

    def test_an_action_will_not_undo_what_a_goal_has_got(self):
        """Goal protection, through `pursuing`. Outside a subgoal nothing
        is protected and `drop a` proposes; with `held a` being pursued and
        true, it does not."""
        memory = acting.Situation({"held a"})
        drop = next(one for one in blocks_actions("ab")
                    if one.name == "drop a")
        operator = acting.operator_of(drop, frozenset({"on a b"}))
        self.assertTrue(operator.proposes(memory))
        from research.v687 import executive as E
        token = E._PURSUING.set(frozenset({"held a"}))
        try:
            self.assertFalse(operator.proposes(memory))
        finally:
            E._PURSUING.reset(token)

    def test_pursuing_is_empty_outside_a_subgoal(self):
        self.assertEqual(pursuing(), frozenset())


class RegressionTests(unittest.TestCase):
    """`Executive.plan` as it stands: regression with no delete lists."""

    def test_it_finds_a_plan_that_cannot_be_executed(self):
        """Not a bug in `plan`, which says what it does -- *what could be,
        not what will* -- but the size of the gap between planning over what
        is known and planning over what is true."""
        problem = next(one for one in P.SUITE if one.name == "sussman")
        names, steps, solved = acting.by_regression(problem)
        self.assertIsNotNone(names)
        self.assertFalse(solved)
        self.assertLess(steps, len(names))

    def test_the_executive_solves_what_regression_does_not(self):
        for name in ("sussman", "three in a row", "four apart"):
            with self.subTest(problem=name):
                problem = next(one for one in P.SUITE if one.name == name)
                self.assertFalse(acting.by_regression(problem)[2])
                self.assertTrue(acting.solve(problem, optimal=False).solved)


class AgentTests(unittest.TestCase):
    """Plan, act, look -- as four operators, not as a loop."""

    def one_step(self):
        return next(one for one in P.SUITE if one.name == "one step")

    def test_the_agent_plans_acts_and_looks(self):
        problem = self.one_step()
        real = problem.world()
        report = acting.Attempt(name=problem.name)
        trace = acting.agent(problem, real, None, report).run(
            Working(goal="solve"))
        fired = [step.operator for step in trace.fired]
        self.assertEqual(trace.answered_by, "done")
        for name in ("plan it", "act", "look", "done"):
            self.assertIn(name, fired)
        self.assertTrue(real.solved(problem.goal))

    def test_every_change_to_the_world_is_recorded(self):
        problem = self.one_step()
        real = problem.world()
        trace = acting.agent(problem, real, None).run(Working(goal="solve"))
        self.assertEqual([one.what for one in trace.changes],
                         [str(one) for one in real.did])

    def test_a_world_that_moves_underneath_is_an_impasse(self):
        """Not a branch: the surprise opens a substate, which is what E2's
        mechanism was for."""
        problem = next(one for one in P.SUITE
                       if one.name == "three in a row")

        class Meddled(W.World):
            def __init__(self, facts):
                super().__init__(facts)
                self.meddled = False

            def do(self, action):
                done = super().do(action)
                if done and not self.meddled and "stack" in action.name:
                    self.meddled = True
                    block, under = action.name.split()[1:3]
                    self.facts = (self.facts - {f"on {block} {under}"}
                                  | {f"table {block}", f"clear {under}"})
                return done

        real = Meddled(problem.start)
        report = acting.Attempt(name=problem.name)
        trace = acting.agent(problem, real, None, report).run(
            Working(goal="solve"))
        self.assertTrue(real.meddled)
        self.assertEqual(report.surprises, 1)
        self.assertGreater(report.plans, 1)
        self.assertTrue(report.solved)
        self.assertIn("make sense of it",
                      [one.goal for one in trace.subgoals])
        self.assertEqual([step.operator for step in trace.subgoals[0].fired],
                         ["noticed", "plan again"])

    def test_a_surprise_keeps_the_prediction_that_failed(self):
        problem = next(one for one in P.SUITE
                       if one.name == "three in a row")
        real = problem.world()
        report = acting.Attempt(name=problem.name)
        executive = acting.agent(problem, real, None, report)

        def meddle(action, real=real):
            done = W.World.do(real, action)
            if done and len(real.did) == 1:
                block = action.name.split()[1]
                real.facts = ((real.facts - {f"held {block}"})
                              | {f"clear {block}", f"table {block}",
                                 "empty"})
            return done

        real.do = meddle
        executive.run(Working(goal="solve"))
        self.assertEqual(report.surprises, 1)
        gap = report.gaps[0]
        block = str(gap.action).split()[1]
        self.assertIn(f"held {block}", gap.missing)
        self.assertIn(f"table {block}", gap.extra)
        self.assertEqual(gap.after, 1)
        self.assertTrue(report.solved)


class SuiteTests(unittest.TestCase):
    """The numbers `DESIGN.md` quotes, pinned."""

    @classmethod
    def setUpClass(cls):
        cls.problems = P.SUITE + P.sampled(20, seed=0)
        cls.report = acting.measure(cls.problems)

    def test_the_oracle_agrees_with_the_known_optima(self):
        for name, best in (("sussman", 6), ("four apart", 6),
                           ("five inverted", 10)):
            with self.subTest(problem=name):
                problem = next(one for one in P.SUITE if one.name == name)
                self.assertEqual(len(W.shortest(problem)), best)

    def test_most_are_solved_and_most_of_those_are_shortest(self):
        self.assertEqual(self.report.total, 24)
        self.assertEqual(self.report.solved, 23)
        self.assertEqual(self.report.shortest, 19)

    def test_the_search_never_runs_away(self):
        for row in self.report.rows:
            with self.subTest(problem=row.name):
                self.assertLess(row.search.fired, 100)
                self.assertLessEqual(row.search.depth, 12)

    def test_chunking_buys_nothing_here(self):
        """Recorded rather than asserted away. A chunk is keyed on the
        impasse *and the state it arose in*, and in a world the state is
        different after every action, so the key almost never comes round
        again. A chunk that fits a world would have to be keyed on what the
        impasse turned on rather than on everything true at the time."""
        chunks = Chunks()
        acting.measure(self.problems, chunks)
        self.assertGreater(chunks.misses, 100 * max(chunks.hits, 1))
        self.assertLessEqual(chunks.hits, 5)


class OtherDomainTests(unittest.TestCase):
    """The claim that nothing in the planner is about blocks, as numbers.

    Both are solved outright and neither is solved *well*: see
    `DESIGN.md` §4. The assertion is the transfer; the shortfall is
    recorded rather than asserted away.
    """

    def test_errands_are_all_solved(self):
        report = acting.measure(P.errands(20, 0))
        self.assertEqual(report.solved, report.total)

    def test_deliveries_are_all_solved(self):
        report = acting.measure(P.delivery(20, 0))
        self.assertEqual(report.solved, report.total)

    def test_but_the_plans_are_long_where_moving_is_free(self):
        """Blocks is nearly all optimal; these are not, because `go` and
        `drive` can be repeated at no cost and means-ends counts no cost."""
        blocks = acting.measure(P.SUITE + P.sampled(20, seed=0))
        errands = acting.measure(P.errands(20, 0))
        self.assertGreater(blocks.shortest / blocks.total, 0.75)
        self.assertLess(errands.shortest / errands.total, 0.5)


if __name__ == "__main__":
    unittest.main()
