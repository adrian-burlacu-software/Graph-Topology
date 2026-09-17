"""The executive, with no engine: the cycle keeps a cascade's order,
conditions are read against what earlier operators wrote, and utilities move
only when they are allowed to."""
from __future__ import annotations

import unittest

from research.v687.executive import (ANSWERED, CONTINUE, DECLINED, Executive,
                                     Chunks, Ledger, Operator, Subgoal, Unwired,
                                     Working, attempt, effect, episode,
                                     suppressed)


class ExecutiveTests(unittest.TestCase):
    def cascade(self, answers: dict, rate: float = 0.0) -> Executive:
        return Executive([
            Operator("first", attempt(lambda: answers.get("first"))),
            Operator("second", attempt(lambda: answers.get("second"))),
            Operator("last", attempt(lambda: answers.get("last")))],
            rate=rate)

    def test_the_order_written_is_the_order_fired(self):
        memory = {}
        trace = self.cascade({"second": "yes", "last": "no"}).run(memory)
        self.assertEqual(memory["answer"], "yes")
        self.assertEqual(trace.answered_by, "second")
        self.assertEqual([(one.operator, one.outcome) for one in trace.fired],
                         [("first", DECLINED), ("second", ANSWERED)])

    def test_nothing_answering_is_an_impasse(self):
        trace = self.cascade({}).run({})
        self.assertTrue(trace.impasse)
        self.assertEqual(len(trace.fired), 3)

    def test_conditions_read_what_earlier_operators_wrote(self):
        def relation(memory):
            memory["relation"] = "is_a"
            return CONTINUE

        executive = Executive([
            Operator("relation", relation),
            Operator("taxonomy", attempt(lambda: "R1"),
                     proposes=lambda memory: memory.get("relation") == "is_a"),
            Operator("facts", attempt(lambda: "R2"),
                     proposes=lambda memory: memory.get("relation") != "is_a")])
        memory = {}
        trace = executive.run(memory)
        self.assertEqual((memory["answer"], trace.answered_by), ("R1",
                                                                 "taxonomy"))

    def test_utilities_move_only_when_allowed(self):
        still = self.cascade({"last": "x"})
        before = [one.utility for one in still.operators]
        still.reward(still.run({}), 10.0)
        self.assertEqual([one.utility for one in still.operators], before)
        learning = self.cascade({"last": "x"}, rate=0.5)
        trace = learning.run({})
        learning.reward(trace, 10.0)
        last = learning.operators[-1]
        self.assertAlmostEqual(last.utility, 1.0 + 0.5 * (10.0 - 1.0))
        # a declined operator learned nothing
        self.assertEqual(learning.operators[0].utility, 3.0)

    def test_a_learned_utility_changes_what_fires_first(self):
        executive = self.cascade({"first": "a", "last": "c"})
        executive.operators[-1].utility = 99.0
        memory = {}
        executive.run(memory)
        self.assertEqual(memory["answer"], "c")


class WorkingTests(unittest.TestCase):
    """Working memory as a stack of goals (E1): one frame deep it is the dict
    it replaced, and a subgoal reads what is beneath it, writes only its own
    frame, and hands back what it wrote."""

    def cascade(self) -> Executive:
        def relation(memory):
            memory["relation"] = "is_a"
            return CONTINUE
        return Executive([
            Operator("relation", relation),
            Operator("taxonomy", attempt(lambda: "R1"),
                     proposes=lambda memory: memory.get("relation") == "is_a"),
            Operator("facts", attempt(lambda: "R2"))])

    def test_one_frame_deep_it_is_the_dict_it_replaced(self):
        plain, working = {"why": False}, Working({"why": False}, goal="ask")
        before = self.cascade().run(plain)
        after = self.cascade().run(working)
        self.assertEqual(after.as_dict(), before.as_dict())
        self.assertEqual(dict(working), plain)
        self.assertEqual((after.goal, after.depth), ("ask", 1))

    def test_a_subgoal_reads_beneath_and_writes_only_its_own_frame(self):
        memory = Working({"referent": "rex", "relation": "is_a"}, goal="ask")
        memory.push("which sense", relation="has_part")
        self.assertEqual(memory["referent"], "rex")          # read beneath
        self.assertEqual(memory["relation"], "has_part")     # its own, first
        self.assertIn("referent", memory)
        self.assertEqual(memory.get("missing", "none"), "none")
        memory["sense"] = "dog.n.01"
        self.assertEqual((memory.depth, memory.open_goals),
                         (2, ["ask", "which sense"]))
        result = memory.pop()
        self.assertEqual(result, {"relation": "has_part",
                                  "sense": "dog.n.01"})
        # the goal beneath is as it was: nothing the subgoal wrote leaked
        self.assertEqual(dict(memory), {"referent": "rex",
                                        "relation": "is_a"})
        self.assertEqual((memory.goal, memory.depth), ("ask", 1))

    def test_the_outermost_goal_cannot_be_popped(self):
        with self.assertRaises(IndexError):
            Working(goal="ask").pop()

    def test_a_subgoal_that_fails_is_still_closed(self):
        memory = Working({"referent": "rex"}, goal="ask")
        with self.assertRaises(RuntimeError):
            with memory.subgoal("which sense"):
                memory["sense"] = "half-written"
                raise RuntimeError("the subgoal broke")
        self.assertEqual((memory.goal, dict(memory)),
                         ("ask", {"referent": "rex"}))

    def test_an_executive_runs_on_a_subgoal_and_returns_its_answer(self):
        memory = Working({"relation": "is_a"}, goal="ask")
        with memory.subgoal("resolve the relation") as result:
            trace = self.cascade().run(memory)
        self.assertEqual((trace.goal, trace.depth, trace.answered_by),
                         ("resolve the relation", 2, "taxonomy"))
        self.assertEqual(result["answer"], "R1")
        self.assertNotIn("answer", memory)


class ImpasseTests(unittest.TestCase):
    """E2: an impasse an operator names opens the subgoal of that name; what
    the subgoal returns is written back, and the goal that reached the
    impasse carries on."""

    def going(self, place_for=None) -> Executive:
        """`where will Sumit go`: no place here is for what he wants, so the
        goal reaches an impasse; a subgoal finds a place another way."""
        def here(memory):
            memory["impasse"] = "no place for it"
            return CONTINUE

        def answer(memory):
            memory["answer"] = memory["place"]
            return ANSWERED

        def not_told(memory):
            memory["answer"] = "not told"
            return ANSWERED

        def kept(memory):
            # reads what is beneath: the goal pushed it with `wanted`
            found = (place_for or {}).get(memory["wanted"])
            memory["scratch"] = "looked"
            if found is None:
                return DECLINED
            memory["place"] = found
            return ANSWERED

        finding = Executive([Operator("kept there", kept)])
        return Executive(
            [Operator("a place here", here),
             Operator("answer", answer,
                      proposes=lambda memory: "place" in memory),
             Operator("not told", not_told,
                      proposes=lambda memory: "no place for it"
                      in memory.get("resolved", ()))],
            subgoals={"no place for it": Subgoal(
                "find a place for what is wanted", finding,
                returns=("place",))})

    def test_an_impasse_opens_its_subgoal_and_the_goal_goes_on(self):
        memory = Working({"wanted": "beverage"}, goal="where will Sumit go")
        trace = self.going({"beverage": "kitchen"}).run(memory)
        self.assertEqual((memory["answer"], trace.answered_by),
                         ("kitchen", "answer"))
        self.assertEqual([(one.goal, one.depth, one.answered_by)
                          for one in trace.subgoals],
                         [("find a place for what is wanted", 2,
                           "kept there")])
        # only what the subgoal returns comes back; its scratch does not
        self.assertNotIn("scratch", memory)
        self.assertEqual((memory.goal, memory.depth),
                         ("where will Sumit go", 1))

    def test_a_subgoal_that_fails_leaves_the_goal_to_say_so(self):
        memory = Working({"wanted": "orgy"}, goal="where will Antoine go")
        trace = self.going({"beverage": "kitchen"}).run(memory)
        self.assertEqual((memory["answer"], trace.answered_by),
                         ("not told", "not told"))
        self.assertTrue(trace.subgoals[0].impasse)
        self.assertNotIn("place", memory)

    def test_an_impasse_with_no_subgoal_is_the_callers(self):
        executive = Executive([Operator(
            "stuck", lambda memory: memory.update(impasse="unknown") or
            CONTINUE)])
        trace = executive.run(Working(goal="ask"))
        self.assertTrue(trace.impasse)
        self.assertEqual(trace.subgoals, [])

    def test_a_plain_dict_has_nowhere_to_push_a_subgoal(self):
        memory = {"wanted": "beverage"}
        trace = self.going({"beverage": "kitchen"}).run(memory)
        self.assertTrue(trace.impasse)
        self.assertNotIn("answer", memory)

    def test_the_trace_shows_a_subgoal_only_when_one_was_pushed(self):
        pushed = self.going({"beverage": "kitchen"}).run(
            Working({"wanted": "beverage"}, goal="where"))
        self.assertEqual(pushed.as_dict()["subgoals"][0]["goal"],
                         "find a place for what is wanted")
        self.assertNotIn("subgoals", Executive([Operator(
            "first", attempt(lambda: "yes"))]).run(Working()).as_dict())


class CreditTests(unittest.TestCase):
    """E3: what ran for an answer is kept, and credited with what the answer
    was worth, before any utility is allowed to move."""

    def test_runs_are_kept_only_inside_an_episode(self):
        first = Executive([Operator("first", attempt(lambda: "yes"))],
                          name="asking")
        first.run(Working())
        with episode() as runs:
            first.run(Working(goal="ask"))
        first.run(Working())
        self.assertEqual([(name, trace.goal) for name, trace in runs],
                         [("asking", "ask")])

    def test_a_subgoal_is_kept_within_the_run_that_pushed_it(self):
        with episode() as runs:
            ImpasseTests().going({"beverage": "kitchen"}).run(
                Working({"wanted": "beverage"}, goal="where"))
        self.assertEqual(len(runs), 1)
        self.assertEqual(len(runs[0][1].subgoals), 1)

    def test_only_what_did_something_is_credited_subgoals_too(self):
        ledger = Ledger()
        run = {"fired": [{"operator": "a place here", "outcome": CONTINUE},
                         {"operator": "tried", "outcome": DECLINED},
                         {"operator": "answer", "outcome": ANSWERED}],
               "subgoals": [{"goal": "find", "fired": [
                   {"operator": "kept there", "outcome": ANSWERED}]}]}
        ledger.credit("where", run, 1.0, "right")
        ledger.credit("where", run, -1.0, "wrong")
        self.assertIsNone(ledger.mean("where", "tried"))
        self.assertEqual(ledger.mean("where", "kept there"), 0.0)
        rows = {(one[0], one[1]): one for one in ledger.table()}
        self.assertEqual(rows[("where", "answer")][2:],
                         (2, 0.0, {"right": 1, "wrong": 1}))

    def test_the_mean_is_where_a_learned_utility_is_heading(self):
        """ACT-R's `U += rate * (value - U)` settles on the mean reward: the
        ledger's mean is what `reward` would have taken an operator to."""
        executive = Executive([Operator("answer", attempt(lambda: "x"))],
                              rate=0.01)
        ledger = Ledger()
        values = [1.0, -1.0, 1.0, 1.0] * 2000
        for value in values:
            trace = executive.run(Working())
            executive.reward(trace, value)
            ledger.credit("", trace.as_dict(), value)
        self.assertAlmostEqual(executive.operators[0].utility,
                               ledger.mean("", "answer"), delta=0.1)


class ConflictTests(unittest.TestCase):
    """E3: a fired step keeps what it was chosen from, and the ledger says
    which choices the rewards would reverse."""

    def test_a_choice_is_kept_only_where_there_was_one(self):
        trace = Executive([
            Operator("definition", attempt(lambda: None)),
            Operator("parts", attempt(lambda: "yes"))]).run(Working())
        steps = trace.as_dict()["fired"]
        self.assertEqual(steps[0]["candidates"], ["definition", "parts"])
        # the second cycle had only `parts` left: nothing was decided there
        self.assertNotIn("candidates", steps[1])

    def test_a_reversal_is_a_passed_over_operator_that_earned_more(self):
        ledger = Ledger()
        chose_worse = {"fired": [{"operator": "definition",
                                  "outcome": ANSWERED,
                                  "candidates": ["definition", "parts"]}]}
        parts_alone = {"fired": [{"operator": "parts", "outcome": ANSWERED}]}
        for _ in range(3):
            ledger.credit("layers", chose_worse, -1.0)
            ledger.credit("layers", parts_alone, 1.0)
        self.assertEqual(ledger.reversals(),
                         [("layers", "definition", -1.0, "parts", 1.0, 3)])
        # an operator with no credit of its own is never compared
        untried = Ledger()
        untried.credit("layers", chose_worse, -1.0)
        self.assertEqual(untried.reversals(), [])


class WiringTests(unittest.TestCase):
    """E4: an operator declares the slots it needs and gives; it is proposed
    only once its needs are there, it must give what it says when it goes
    on, and an executive told what it starts with refuses operators that
    nothing could ever let fire."""

    @staticmethod
    def relation(memory) -> str:
        memory["relation"] = "is_a"
        return CONTINUE

    def test_an_operator_waits_for_what_it_needs(self):
        executive = Executive([
            Operator("taxonomy", attempt(lambda: "R1"), needs=("relation",)),
            Operator("relation", self.relation, gives=("relation",))])
        trace = executive.run({})
        # listed first, it could not be proposed until `relation` was written
        self.assertEqual([step.operator for step in trace.fired],
                         ["relation", "taxonomy"])
        self.assertEqual(trace.answered_by, "taxonomy")

    def test_going_on_without_giving_it_is_a_wiring_error(self):
        executive = Executive([Operator(
            "relation", lambda memory: CONTINUE, gives=("relation",))])
        with self.assertRaises(Unwired):
            executive.run({})

    def test_a_need_nothing_gives_is_refused_when_built(self):
        operators = [Operator("relation", self.relation, gives=("relation",)),
                     Operator("walk", attempt(lambda: "R2"),
                              needs=("relation", "target"))]
        with self.assertRaises(Unwired) as raised:
            Executive(operators, given=())
        self.assertIn("walk needs target", str(raised.exception))
        # given at the start, it is wired; unchecked, it is only never ready
        Executive(operators, given=("target",))
        self.assertEqual([one.name for one, _ in
                          Executive(operators).unreachable()], ["walk"])

    def test_reaching_one_needs_what_reaches_it_to_be_reachable(self):
        # `walk` gives `note`, but `walk` itself can never fire
        operators = [Operator("walk", self.relation, needs=("target",),
                              gives=("note",)),
                     Operator("taught", attempt(lambda: "yes"),
                              needs=("note",))]
        self.assertEqual([one.name for one, _ in
                          Executive(operators).unreachable()],
                         ["walk", "taught"])

    def test_a_subgoal_gives_what_it_returns(self):
        finding = Executive([Operator("kept there", attempt(
            lambda: "kitchen", slot="found"))])
        Executive([Operator("stuck", lambda memory: memory.update(
                       impasse="nowhere") or CONTINUE, gives=("impasse",)),
                   Operator("found", attempt(lambda: "yes"),
                            needs=("found",))],
                  subgoals={"nowhere": Subgoal("find", finding,
                                               returns=("found",))},
                  given=())


class SuppressedTests(unittest.TestCase):
    """E4: the same question asked again without an operator, to see what
    the others would have come to."""

    def cascade(self, name: str = "layers") -> Executive:
        return Executive([
            Operator("definition", attempt(lambda: "a bird")),
            Operator("parts", attempt(lambda: "wings"))], name=name)

    def test_without_the_one_that_answered_the_next_answers(self):
        with suppressed({("layers", "definition")}):
            memory: dict = {}
            trace = self.cascade().run(memory)
        self.assertEqual((trace.answered_by, memory["answer"]),
                         ("parts", "wings"))
        # and outside it, nothing is suppressed
        self.assertEqual(self.cascade().run({}).answered_by, "definition")

    def test_only_the_executive_named(self):
        with suppressed({("other", "definition")}):
            self.assertEqual(self.cascade().run({}).answered_by, "definition")

    def test_suppressions_nest(self):
        with suppressed({("layers", "definition")}):
            with suppressed({("layers", "parts")}):
                self.assertTrue(self.cascade().run({}).impasse)
            self.assertEqual(self.cascade().run({}).answered_by, "parts")


class EffectTests(unittest.TestCase):
    """E4c: what an operator changes outside working memory is declared,
    kept on its step, and taken back when the subgoal it was made for
    returns nothing."""

    @staticmethod
    def teaching(store: list, word: str):
        def apply(memory) -> str:
            store.append(word)
            effect("knowledge", f"taught {word}", undo=store.pop)
            return ANSWERED
        return apply

    def test_an_effect_is_kept_on_the_step_that_made_it(self):
        store: list = []
        trace = Executive([Operator("teach", self.teaching(store, "wemble"),
                                    effects=("knowledge",))]).run({})
        self.assertEqual(store, ["wemble"])
        self.assertEqual(trace.as_dict()["fired"][0]["effects"],
                         [{"store": "knowledge", "what": "taught wemble"}])

    def test_an_undeclared_effect_is_a_wiring_error(self):
        with self.assertRaises(Unwired):
            Executive([Operator("teach", self.teaching([], "wemble"))]
                      ).run({})

    def test_every_operator_running_must_declare_it(self):
        # the act declares nothing, though the question it asks teaches
        store: list = []
        inner = Executive([Operator("teach", self.teaching(store, "wemble"),
                                    effects=("knowledge",))])
        act = Executive([Operator("ask", lambda memory: inner.run({})
                                  and ANSWERED)])
        with self.assertRaises(Unwired) as raised:
            act.run({})
        self.assertIn(": ask changed knowledge", str(raised.exception))

    def test_outside_an_executive_nothing_is_checked(self):
        effect("knowledge", "taught at the prompt")

    def subgoal_teaching(self, store: list, finds: bool) -> Executive:
        def find(memory) -> str:
            store.append("wemble")
            effect("knowledge", "taught wemble", undo=store.pop)
            if finds:
                memory["place"] = "kitchen"
            return CONTINUE if finds else DECLINED
        finding = Executive([Operator("teach and look", find,
                                      effects=("knowledge",))])
        return Executive(
            [Operator("stuck", lambda memory: memory.update(
                impasse="nowhere") or CONTINUE, gives=("impasse",)),
             Operator("answer", attempt(lambda: "yes"), needs=("place",))],
            subgoals={"nowhere": Subgoal("find", finding,
                                         returns=("place",))})

    def test_a_subgoal_that_returns_nothing_takes_its_effects_back(self):
        store: list = []
        trace = self.subgoal_teaching(store, finds=False).run(
            Working(goal="ask"))
        self.assertEqual(store, [])
        self.assertEqual(trace.as_dict()["subgoals"][0]["undone"], 1)
        self.assertEqual(trace.changes, [])

    def test_a_subgoal_that_returns_keeps_them_as_the_goals(self):
        store: list = []
        trace = self.subgoal_teaching(store, finds=True).run(
            Working(goal="ask"))
        self.assertEqual(store, ["wemble"])
        self.assertEqual([one.what for one in trace.changes],
                         ["taught wemble"])


class MeansEndsTests(unittest.TestCase):
    """E5: stuck, an executive that plans pushes a subgoal of whatever gives
    what its most useful waiting operator needs -- no impasse named, no
    subgoal wired."""

    @staticmethod
    def writes(slot: str, value, needs=()):
        def apply(memory) -> str:
            memory[slot] = value
            return CONTINUE
        return apply

    def going(self, finds: bool = True, plans: bool = True) -> Executive:
        """Where will Antoine go: a place for what thirst moves one to, found
        where a beverage is kept -- a can, and a can is in a kitchen."""
        means = [
            Operator("kept there", self.writes("place", "kitchen"),
                     needs=("container",), gives=("place",)),
            Operator("in a container",
                     self.writes("container", "can") if finds
                     else lambda memory: DECLINED,
                     needs=("wanted",), gives=("container",))]
        return Executive(
            [Operator("wanted", self.writes("wanted", "beverage"),
                      gives=("wanted",)),
             Operator("answer", attempt(lambda: "probably the kitchen"),
                      needs=("place",)),
             Operator("not told", attempt(lambda: "not told"),
                      needs=("unachieved",))],
            means=means if plans else None, name="where will")

    def test_what_is_needed_is_achieved_by_what_gives_it_in_turn(self):
        memory = Working(goal="where will Antoine go")
        trace = self.going().run(memory)
        self.assertEqual((trace.answered_by, memory["answer"]),
                         ("answer", "probably the kitchen"))
        # `place` needs `container`, so its subgoal pushed one of its own
        outer = trace.as_dict()["subgoals"][0]
        self.assertEqual(outer["goal"], "achieve place for answer")
        self.assertEqual(outer["subgoals"][0]["goal"],
                         "achieve container for kept there")
        self.assertEqual(memory.depth, 1)

    def test_what_cannot_be_achieved_is_said_to_be(self):
        memory = Working(goal="where will Antoine go")
        trace = self.going(finds=False).run(memory)
        self.assertEqual(trace.answered_by, "not told")
        self.assertEqual(memory["unachieved"], ["place"])
        self.assertNotIn("container", memory)

    def test_an_executive_that_does_not_plan_stops_at_the_impasse(self):
        trace = self.going(plans=False).run(Working(goal="where"))
        self.assertTrue(trace.impasse)
        self.assertEqual(trace.subgoals, [])

    def test_a_circle_of_needs_is_not_pursued_forever(self):
        executive = Executive(
            [Operator("answer", attempt(lambda: "yes"), needs=("a",))],
            means=[Operator("a from b", self.writes("a", 1), needs=("b",),
                            gives=("a",)),
                   Operator("b from a", self.writes("b", 1), needs=("a",),
                            gives=("b",))])
        trace = executive.run(Working(goal="ask"))
        self.assertTrue(trace.impasse)

    def test_a_failed_means_takes_its_effects_back(self):
        store: list = []

        def looked(memory) -> str:
            store.append("looked")
            effect("conversation", "looked", undo=store.pop)
            return DECLINED
        Executive([Operator("answer", attempt(lambda: "yes"),
                            needs=("place",))],
                  means=[Operator("look", looked, gives=("place",),
                                  effects=("conversation",))]
                  ).run(Working(goal="ask"))
        self.assertEqual(store, [])

    def test_the_plan_is_the_order_the_declarations_allow(self):
        self.assertEqual(self.going().plan(("place",)),
                         ["wanted", "in a container", "kept there"])
        self.assertEqual(self.going().plan(("place",), given=("container",)),
                         ["kept there"])
        self.assertIsNone(self.going().plan(("weather",)))

    def test_what_a_means_gives_counts_as_reachable(self):
        self.assertEqual(self.going().unreachable(()), [])
        with self.assertRaises(Unwired):
            Executive([Operator("answer", attempt(lambda: "yes"),
                                needs=("place",))], given=())


class ChunkTests(unittest.TestCase):
    """E6: what a subgoal came to, kept as the operators that did it, so the
    same impasse is not searched through twice."""

    def going(self, chunks, works=lambda: True) -> Executive:
        def writes(slot, value, gated=True):
            def apply(memory) -> str:
                if gated and not works():
                    return DECLINED
                memory[slot] = value
                return CONTINUE
            return apply

        return Executive(
            [Operator("wanted", writes("wanted", "beverage", gated=False),
                      gives=("wanted",)),
             Operator("answer", attempt(lambda: "the kitchen"),
                      needs=("place",)),
             Operator("not told", attempt(lambda: "not told"),
                      needs=("unachieved",))],
            means=[Operator("kept there", writes("place", "kitchen"),
                            needs=("container",), gives=("place",)),
                   Operator("in a container", writes("container", "can"),
                            needs=("wanted",), gives=("container",))],
            name="where will", chunks=chunks)

    @staticmethod
    def pushes(trace) -> int:
        return 1 + sum(ChunkTests.pushes(one) for one in trace.subgoals)

    def test_the_second_time_the_subgoal_is_not_searched_for(self):
        chunks = Chunks()
        first = self.going(chunks).run(Working(goal="where will Antoine go"))
        second = self.going(chunks).run(Working(goal="where will Sumit go"))
        self.assertEqual(first.answered_by, second.answered_by, "answer")
        # two subgoals deep, then one that remembers both operators
        self.assertEqual((self.pushes(first), self.pushes(second)), (3, 2))
        self.assertTrue(second.as_dict()["subgoals"][0]["chunked"])
        self.assertEqual((chunks.hits, chunks.misses), (1, 2))

    def test_a_chunk_is_kept_for_every_subgoal_that_came_to_something(self):
        chunks = Chunks()
        self.going(chunks).run(Working(goal="where will Antoine go"))
        # the goal's own, and the one its subgoal reached in turn
        # only the goal whose subgoal had to push one of its own: the
        # inner step was no search, so remembering it saves nothing
        self.assertEqual(chunks.table(), [
            ("where will", ("place",), ("wanted",),
             ("kept there", "in a container"))])

    def test_a_chunk_that_no_longer_works_is_forgotten(self):
        chunks = Chunks()
        self.going(chunks).run(Working(goal="where will Antoine go"))
        broken = self.going(chunks, works=lambda: False)
        trace = broken.run(Working(goal="where will Sumit go"))
        # it was tried, it came to nothing, and the search followed it
        self.assertEqual(chunks.forgotten, 1)
        self.assertEqual(trace.answered_by, "not told")
        self.assertEqual(chunks.rules, {})

    def test_an_executive_with_no_chunks_keeps_none(self):
        trace = self.going(None).run(Working(goal="where will Antoine go"))
        self.assertEqual(self.pushes(trace), 3)


if __name__ == "__main__":
    unittest.main()
