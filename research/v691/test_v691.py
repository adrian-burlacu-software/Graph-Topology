"""v691: the world, the plan, and the agent that acts on it.

No reader and no store, so this suite is fast and a failure in it is the
architecture rather than the parser -- which is the whole reason v691a is a
blocks world.
"""
from __future__ import annotations

import unittest

from research.v687.executive import (Chunks, Executive, Operator, Unwired,
                                     Working, pursuing)
from research.v691 import acting, world as W


class WorldTests(unittest.TestCase):
    """Facts that change only by acting."""

    def setUp(self):
        self.problem = W.problem("t", "abc", [["c", "a"], ["b"]],
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

    def test_the_towers_read_back(self):
        self.assertEqual(sorted(self.world.towers()), [["b"], ["c", "a"]])

    def test_an_operator_must_declare_that_it_changes_the_world(self):
        """The guard v687 put on the store, now on the thing the agent
        moves: nothing changes the world without saying so."""
        action = W.blocks("ab")[0]
        world = W.World(W.start_of([["a"], ["b"]]))
        undeclared = Operator(name="act", apply=lambda memory: world.do(
            action) and "continue")
        with self.assertRaises(Unwired):
            Executive([undeclared], name="rogue").run(Working())


class SituationTests(unittest.TestCase):
    """Working memory over a world: the one place it must not behave like
    working memory over a belief."""

    def setUp(self):
        self.memory = acting.Situation({"clear a", "table a", "empty"})

    def test_a_fact_is_a_slot(self):
        self.assertIn("clear a", self.memory)
        self.assertTrue(self.memory["clear a"])
        self.assertNotIn("clear b", self.memory)

    def test_acting_retracts_as_well_as_asserts(self):
        take = next(one for one in W.blocks("ab") if one.name == "take a")
        self.memory.apply(take)
        self.assertIn("held a", self.memory)
        self.assertNotIn("clear a", self.memory)
        self.assertNotIn("empty", self.memory)

    def test_what_a_subgoal_did_outlives_the_subgoal(self):
        """`Working` is right for belief -- a subgoal that fails leaves
        nothing behind -- and wrong for a world. A subgoal that unstacked a
        block and then gave up has still unstacked it."""
        take = next(one for one in W.blocks("ab") if one.name == "take a")
        with self.memory.subgoal("get a"):
            self.memory.apply(take)
        self.assertIn("held a", self.memory)
        self.assertNotIn("clear a", self.memory)
        self.assertEqual([str(one) for one in self.memory.did], ["take a"])

    def test_a_chunk_is_keyed_on_the_world(self):
        """`keys` carries the facts, so the same impasse in a different
        arrangement of the blocks is a different impasse."""
        self.assertIn("clear a", self.memory.keys())


class PlanningTests(unittest.TestCase):
    """Means-ends over a model, which is goal-stack planning."""

    def solve(self, name):
        problem = next(one for one in W.SUITE if one.name == name)
        return problem, acting.solve(problem)

    def test_the_sussman_anomaly_is_solved(self):
        """The reason this file exists: achieving `on a b` and then `on b c`
        in turn undoes the first."""
        problem, got = self.solve("sussman")
        self.assertTrue(got.solved)
        self.assertEqual(got.optimal, 6)

    def test_a_plan_is_the_actions_the_model_applied(self):
        problem, got = self.solve("three in a row")
        world = problem.world()
        for action in got.plan:
            self.assertTrue(world.do(action), f"{action} did not apply")
        self.assertTrue(world.solved(problem.goal))

    def test_the_goal_tower_is_built_from_the_bottom(self):
        """`on c d` before `on b c` before `on a b`: doing it the other way
        round is what makes the problem need taking apart again."""
        _, got = self.solve("four apart")
        self.assertTrue(got.solved)
        stacked = [str(one) for one in got.plan
                   if str(one).startswith("stack")]
        self.assertEqual(stacked, ["stack c d", "stack b c", "stack a b"])
        self.assertEqual(got.acted, got.optimal)

    def test_levels_number_the_goal_tower(self):
        self.assertEqual(acting.levels({"on a b", "on b c", "on c d"}),
                         {"a": 3, "b": 2, "c": 1})

    def test_an_action_will_not_undo_what_a_goal_has_got(self):
        """Goal protection, through `pursuing`. Outside a subgoal nothing
        is protected and `drop a` proposes; with `held a` being pursued and
        true, it does not."""
        memory = acting.Situation({"held a"})
        drop = next(one for one in W.blocks("ab") if one.name == "drop a")
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
        not what will* -- but the exact size of the gap between planning
        over what is known and planning over what is true."""
        problem = next(one for one in W.SUITE if one.name == "sussman")
        names, steps, solved = acting.by_regression(problem)
        self.assertIsNotNone(names)
        self.assertFalse(solved)
        self.assertLess(steps, len(names))

    def test_the_executive_solves_what_regression_does_not(self):
        for name in ("sussman", "three in a row", "four apart"):
            with self.subTest(problem=name):
                problem = next(one for one in W.SUITE if one.name == name)
                self.assertFalse(acting.by_regression(problem)[2])
                self.assertTrue(acting.solve(problem, optimal=False).solved)


class AgentTests(unittest.TestCase):
    """Plan, act, look -- as four operators, not as a loop."""

    def test_the_agent_plans_acts_and_looks(self):
        problem = next(one for one in W.SUITE if one.name == "one step")
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
        problem = next(one for one in W.SUITE if one.name == "one step")
        real = problem.world()
        trace = acting.agent(problem, real, None).run(Working(goal="solve"))
        changed = [one.what for one in trace.changes]
        self.assertEqual(changed, [str(one) for one in real.did])

    def test_a_world_that_moves_underneath_is_a_surprise(self):
        """v691b's question, asked here because the mechanism is already
        built: the agent compares what it expected with what it sees, and
        plans again when they differ."""
        problem = next(one for one in W.SUITE if one.name == "three in a row")

        class Meddled(W.World):
            def __init__(self, facts):
                super().__init__(facts)
                self.meddled = False

            def do(self, action):
                done = super().do(action)
                if done and not self.meddled and "stack" in action.name:
                    # Someone puts the block back on the table.
                    self.meddled = True
                    block = action.name.split()[1]
                    under = action.name.split()[2]
                    self.facts = (self.facts
                                  - {f"on {block} {under}"}
                                  | {f"table {block}", f"clear {under}"})
                return done

        real = Meddled(problem.start)
        report = acting.Attempt(name=problem.name)
        trace = acting.agent(problem, real, None, report).run(
            Working(goal="solve"))
        self.assertTrue(real.meddled)
        self.assertGreaterEqual(report.surprises, 1)
        self.assertGreater(report.plans, 1)
        self.assertTrue(report.solved)
        # The surprise is an impasse, not a branch: it opens a substate.
        opened = [one.goal for one in trace.subgoals]
        self.assertIn("make sense of it", opened)
        inner = [step.operator for step in trace.subgoals[0].fired]
        self.assertEqual(inner, ["noticed", "plan again"])

    def test_a_surprise_says_what_was_expected_and_what_is(self):
        """The prediction against the observation, kept where something
        could learn from it. Nothing learns from it yet."""
        problem = next(one for one in W.SUITE if one.name == "three in a row")
        real = problem.world()
        report = acting.Attempt(name=problem.name)
        executive = acting.agent(problem, real, None, report)

        # Someone puts the first block back down the moment it is picked
        # up. `Problem.actions` grounds the domain afresh each time, so the
        # planner's action is an equal object and not the same one.
        def meddle(action, real=real):
            done = W.World.do(real, action)
            if done and len(real.did) == 1:
                block = action.name.split()[1]
                real.facts = ((real.facts - {f"held {block}"})
                              | {f"clear {block}", f"table {block}", "empty"})
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
    """The numbers `DESIGN.md` §1 quotes, pinned."""

    @classmethod
    def setUpClass(cls):
        cls.problems = W.SUITE + W.sampled(20, seed=0)
        cls.report = acting.measure(cls.problems)

    def test_the_oracle_agrees_with_the_known_optima(self):
        for name, best in (("sussman", 6), ("four apart", 6),
                           ("five inverted", 10)):
            with self.subTest(problem=name):
                problem = next(one for one in W.SUITE if one.name == name)
                self.assertEqual(len(W.shortest(problem)), best)

    def test_most_are_solved_and_most_of_those_are_shortest(self):
        self.assertEqual(self.report.total, 24)
        self.assertEqual(self.report.solved, 21)
        self.assertEqual(self.report.shortest, 18)

    def test_chunking_buys_nothing_here(self):
        """Recorded rather than asserted away (E6 measured the same way on
        bAbI). A chunk is keyed on the impasse *and the state it arose in*,
        and in a world the state is different after every action, so the
        key almost never comes round again: one hit in eight hundred. A
        chunk that fits a world would have to be keyed on what the impasse
        turned on rather than on everything that was true at the time.
        """
        chunks = Chunks()
        acting.measure(self.problems, chunks)
        self.assertGreater(chunks.misses, 100 * max(chunks.hits, 1))
        self.assertLessEqual(chunks.hits, 5)

    def test_the_search_never_runs_away(self):
        """Every action is an operator and every operator repeats never;
        what bounds the run is the goal stack, not `LIMIT`."""
        for row in self.report.rows:
            with self.subTest(problem=row.name):
                self.assertLess(row.search.fired, 100)
                self.assertLessEqual(row.search.depth, 12)


if __name__ == "__main__":
    unittest.main()
