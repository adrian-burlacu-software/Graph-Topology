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


@needs_node
class MeaningTests(unittest.TestCase):
    """The exact channels the reader of meaning is taught and checked by."""

    def test_structure_is_read_by_the_compiler(self):
        from research.v696 import meaning as M
        uses, root = M.structure(
            "function f(xs: number[]): number { return xs.filter(x => "
            "x % 2 == 0).reduce((a, x) => a + x, 0); }", "f")
        self.assertEqual(root, "Array.reduce")
        self.assertTrue({"Array.filter", "%", "===", "+"} <= uses)
        uses, _ = M.structure("function g(n: number): number { return n "
                              "< 2 ? n : g(n - 1) + g(n - 2); }", "g")
        self.assertIn("recursion", uses)

    def test_a_library_operator_is_named_as_the_compiler_names_it(self):
        from research.v696 import meaning as M
        from research.v696 import program as P
        words = {M.word(op) for op in P.library().ops + P.library().forms}
        self.assertTrue({"String.split", "Array.filter", "Math.max", "%",
                         "Array.length"} <= words)

    def test_behaviour_is_read_off_values(self):
        from research.v696.meaning import behaviour
        said = behaviour([([[3, 1, 2]], [1, 2, 3]), ([[5, 4]], [4, 5])])
        self.assertTrue({"sorted", "reordered input 1",
                         "as long as input 1"} <= said)
        said = behaviour([([[1, 2, 3, 4]], 6), ([[5, 6]], 6),
                          ([[8, 1]], 8)])
        self.assertNotIn("the sum of input 1", said)
        self.assertIn("the sum of input 1",
                      behaviour([([[1, 2]], 3), ([[4]], 4)]))

    def test_tests_are_pairs(self):
        from research.v696.meaning import test_pairs, values_of
        tests = ("  assert.deepEqual(candidate([1, 2], \"a,b\"),[2, 1]);\n"
                 "  assert.deepEqual(candidate([], \")\"),[]);\n")
        self.assertEqual(values_of(test_pairs(tests)),
                         [([[1, 2], "a,b"], [2, 1]), ([[], ")"], [])])

    def test_code_is_read_back_into_the_search_s_trees(self):
        from research.v696.parse import parse
        tree = parse("function f(xs: number[]): number { return xs.filter("
                     "(n) => n % 2 === 0).reduce((a, b) => a + b, 0); }",
                     "f", [("xs", "number[]")])
        self.assertEqual(tree.source(), "xs.filter((x, i) => ((x % 2) === "
                                        "0)).reduce((acc, x, i) => (acc + "
                                        "x), 0)")
        self.assertEqual(parse("function f(xs: number[]): number { return "
                               "xs.pop() + 1; }", "f",
                               [("xs", "number[]")]).source(),
                         "(xs.pop() + 1)")
        # a step -- a name bound once -- is a shared node (rung 3a)
        self.assertEqual(parse("function f(s: string): number { const w = "
                               "s.split(\" \"); return w.length + w.length; "
                               "}", "f", [("s", "string")]).source(),
                         '(s.split(" ").length + s.split(" ").length)')
        # a loop that changes a number as it goes is not a fold (yet)
        self.assertIsNone(parse("function f(n: number): number { let s = 0; "
                                "while (n > 0) { s += n % 10; n = Math.floor("
                                "n / 10); } return s; }", "f",
                                [("n", "number")]))

    def test_statements_are_executed_into_one_tree(self):
        """Rung 3b: guards are ternaries, a loop carrying one value is a
        fold, a counting loop goes over a range, a loop that returns when
        it finds is `some` and `find`, `push` is an append."""
        from research.v696.parse import parse
        xs = [("xs", "number[]")]
        self.assertEqual(parse(
            "function f(xs: number[]): number { if (xs.length === 0) return "
            "-1; let t = 0; for (const x of xs) { if (x > 0) t += x; } "
            "return t; }", "f", xs).source(),
            "((xs.length === 0) ? -1 : xs.reduce((acc, x, i) => ((x > 0) ? "
            "(acc + x) : acc), 0))")
        self.assertEqual(parse(
            "function f(n: number): number { let s = 0; for (let i = 1; i <= "
            "n; i++) s += i; return s; }", "f", [("n", "number")]).source(),
            "Array.from({ length: ((n + 1) - 1) }, (_, i) => (1 + i))"
            ".reduce((acc, x, i) => (acc + x), 0)")
        self.assertEqual(parse(
            "function f(xs: number[]): number { for (const x of xs) { if (x "
            "> 9) return x * 2; } return 0; }", "f", xs).source(),
            "(xs.some((x, i) => (x > 9)) ? (xs.find((x, i) => (x > 9)) * 2)"
            " : 0)")
        self.assertEqual(parse(
            "function f(xs: number[]): number[] { const out: number[] = []; "
            "for (const x of xs) { if (x % 2 === 0) out.push(x * x); } "
            "return out; }", "f", xs).source(),
            "xs.reduce((acc, x, i) => (((x % 2) === 0) ? [...acc, (x * x)] "
            ": acc), [])")
        # two values carried (a running maximum and a list) is not one fold
        self.assertIsNone(parse(
            "function f(xs: number[]): number[] { let m = -Infinity; const "
            "out: number[] = []; for (const x of xs) { m = Math.max(m, x); "
            "out.push(m); } return out; }", "f", xs))

    def test_a_helper_is_an_operator_carrying_its_body(self):
        from research.v696 import program as P
        from research.v696.checker import checker
        from research.v696.parse import parse
        tree = parse("function odd(n: number): boolean { return n % 2 !== 0; "
                     "}\nfunction f(xs: number[]): number { const kept = "
                     "xs.filter((x) => odd(x)); return kept.length; }", "f",
                     [("xs", "number[]")])
        self.assertEqual(tree.source(), "xs.filter((x, i) => odd(x)).length")
        self.assertIn("function odd(n: number): boolean",
                      P.prelude([tree]))
        self.assertEqual(checker().values(["xs"], [[[1, 2, 3]]],
                                          [tree.source()],
                                          prelude=P.prelude([tree])),
                         [[{"value": 2}]])

    def test_a_program_that_only_fits_its_examples_is_refused(self):
        from research.v696.search import Result, Solver, Switches
        from research.v696 import program as P
        from research.v696.spec import Spec
        spec = Spec("sort", [("xs", "number[]")], "number[]",
                    [([[2, 1]], [1, 2]), ([[4, 3]], [3, 4])],
                    expected={"uses": {}, "behaviour": {"sorted": 0.95}})
        solver, result = Solver(Switches(meet=True)), Result("sort")
        reverse = next(op for op in P.library().ops if op.name == "reverse")
        fits = P.apply(reverse, [P.param("xs", "number[]")])
        self.assertFalse(solver._accepts(spec, fits, result))
        self.assertEqual(result.rejected, 1)


if __name__ == "__main__":
    unittest.main()
