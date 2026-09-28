"""v696: cognitive search over code. Needs Node and TypeScript, and skips
itself without them."""
from __future__ import annotations

import shutil
import unittest

HAVE = shutil.which("node") is not None
needs_node = unittest.skipUnless(HAVE, "node missing")


@needs_node
class CheckerTests(unittest.TestCase):

    def test_types_and_values_are_exact(self):
        from research.v696.checker import checker
        self.assertEqual(checker().check(
            "function f(xs: number[]): number { return xs.length; }"), [])
        self.assertTrue(checker().check(
            "function f(xs: number[]): string { return xs.length; }"))
        rows = checker().values(["xs"], [[[3, 1, 2]]],
                                ["Math.max(...xs)", "xs.nope()"])
        self.assertEqual(rows[0][0], {"value": 3})
        self.assertIn("error", rows[1][0])


@needs_node
class LibraryTests(unittest.TestCase):

    def test_read_from_the_compiler_not_written(self):
        from research.v696 import program as P
        keys = {op.key for op in P.library().ops}
        self.assertIn("method:split:string,string->string[]", keys)
        self.assertIn("method:slice:number[],number->number[]", keys)
        self.assertIn("function:Math.max:number[]->number", keys)
        # Deprecated in lib.d.ts; takes a callback; takes nothing.
        names = {op.name for op in P.library().ops}
        self.assertNotIn("substr", names)
        self.assertNotIn("map", names)
        self.assertNotIn("Math.random", names)

    def test_a_program_is_a_typed_tree(self):
        from research.v696 import program as P
        op = next(one for one in P.library().ops
                  if one.key == "function:Math.max:number[]->number")
        expr = P.apply(op, [P.param("xs", "number[]")])
        self.assertEqual((expr.type, expr.source()),
                         ("number", "Math.max(...xs)"))


@needs_node
class SpecTests(unittest.TestCase):

    def test_features_are_read_off_the_values(self):
        from research.v696.spec import Spec
        spec = Spec("t", [("s", "string")], "number",
                    [(["abc"], 3), (["hello"], 5)])
        self.assertIn("is the length of string", spec.features())

    def test_generated_tasks_keep_their_answer(self):
        from research.v696 import generating as G
        spec = G.tasks(1, 2)[0]
        self.assertEqual(len(spec.examples), G.SHOWN)
        self.assertEqual(len(spec.hidden), G.HIDDEN)
        self.assertIsNotNone(spec.answer)


@needs_node
class SearchTests(unittest.TestCase):

    def spec(self):
        from research.v696.spec import Spec
        return Spec("upper", [("s", "string")], "string",
                    [(["ab"], "AB"), (["c d"], "C D")],
                    hidden=[(["x"], "X")])

    def test_every_route_finds_a_one_step_program(self):
        from research.v696 import search as S
        for switches in (S.Switches(), S.Switches(meet=True, coarse=True)):
            got = S.Solver(switches, budget=3000).solve(self.spec())
            self.assertTrue(got.solved and got.general, switches.label())

    def test_what_was_solved_is_recognised(self):
        from research.v696 import search as S
        from research.v696.spec import Spec
        solver = S.Solver(S.Switches(recognition=True, meet=True,
                                     coarse=True), budget=3000)
        solver.solve(self.spec())
        again = Spec("upper2", [("t", "string")], "string",
                     [(["qq"], "QQ"), (["r s"], "R S")])
        got = solver.solve(again)
        self.assertEqual(got.route, "recognized")
        self.assertEqual(got.program.source(), "t.toUpperCase()")

    def test_chunks_become_operators(self):
        from research.v696 import program as P
        from research.v696 import search as S
        lib = {op.key: op for op in P.library().ops}
        inner = P.apply(lib["method:trim:string->string"],
                        [P.param("s", "string")])
        outer = P.apply(lib["method:toUpperCase:string->string"], [inner])
        op, template = S._chunk(outer)
        self.assertEqual((op.needs, op.gives), (("string",), "string"))
        rebuilt = S._instantiate(template, [P.param("t", "string")])
        self.assertEqual(rebuilt.source(), "t.trim().toUpperCase()")


@needs_node
class FormTests(unittest.TestCase):
    """Rung 2: forms read from the compiler, holes as subgoals."""

    def solve(self, spec, budget=12000):
        from research.v696 import search as S
        return S.Solver(S.Switches(meet=True, coarse=True, repair=True,
                                   forms=True), budget=budget).solve(spec)

    def test_forms_are_read_from_the_callbacks(self):
        from research.v696 import program as P
        forms = {(one.name, one.receiver, one.body)
                 for one in P.library().forms}
        self.assertIn(("map", "number[]", "U"), forms)
        self.assertIn(("filter", "string[]", "boolean"), forms)
        self.assertIn(("sort", "number[]", "number"), forms)

    def test_a_map_is_deduced_element_by_element(self):
        from research.v696.spec import Spec
        got = self.solve(Spec("double", [("xs", "number[]")], "number[]",
                              [([[1, 2, 3]], [2, 4, 6]), ([[5]], [10]),
                               ([[0, -1]], [0, -2])]))
        self.assertEqual(got.route, "deduced")
        self.assertIn(".map(", got.program.source())

    def test_a_filter_is_deduced_and_its_predicate_found(self):
        from research.v696.spec import Spec
        got = self.solve(Spec("evens", [("xs", "number[]")], "number[]",
                              [([[1, 2, 3, 4]], [2, 4]), ([[5, 6]], [6]),
                               ([[8, 1, 10]], [8, 10])]))
        self.assertEqual(got.program.source(),
                         "xs.filter((x, i) => ((x % 2) === 0))")

    def test_the_answer_is_a_function_of_it(self):
        from research.v696.search import determines
        outputs = [False, True, False, True]
        self.assertGreater(determines([{"value": 1}, {"value": 0},
                                       {"value": 1}, {"value": 0}],
                                      outputs), 0)
        self.assertEqual(determines([{"value": 1}, {"value": 1},
                                     {"value": 2}, {"value": 0}],
                                    outputs), 0)


if __name__ == "__main__":
    unittest.main()
