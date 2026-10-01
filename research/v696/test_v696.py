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

    def test_what_json_would_lose_is_kept(self):
        """A Set's or Map's contents are its value (JSON says `{}` for
        both); a function is not a value to compare, and says so."""
        from research.v696.checker import checker
        rows = checker().values(["xs"], [[[1, 2, 2]]],
                                ["new Set(xs)", "new Map([[1, 2]])",
                                 "String", "1 / 0"])
        self.assertEqual(rows[0][0], {"value": {"$set": [1, 2]}})
        self.assertEqual(rows[1][0], {"value": {"$map": [[1, 2]]}})
        self.assertIn("error", rows[2][0])
        self.assertEqual(rows[3][0], {"value": "Infinity"})


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

    def test_a_fold_is_found_by_induction(self):
        """Rung 3: examples whose lists differ by one element at the end
        give the fold's step as a subgoal."""
        from research.v696.spec import Spec
        got = self.solve(Spec("triangle", [("n", "number")], "number",
                              [([3], 6), ([4], 10), ([6], 21), ([7], 28),
                               ([0], 0)]))
        self.assertEqual(got.route, "deduced")
        self.assertIn(".reduce(", got.program.source())

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

    def test_a_value_built_from_itself_is_bound_not_copied(self):
        """Values built from values built from values (each guard reading
        what the one before set) are bound once where they are made, not
        copied where read: the tree is the size the program was written,
        and runs as it did -- the bindings lazy, so what a guard kept from
        being worked out is not worked out."""
        from research.v696 import program as P
        from research.v696.checker import checker
        from research.v696.parse import parse
        steps = "".join(f"  if (m > {k} && xs[m - 1] < m) m = m + xs[m - 1] "
                        f"- 1;\n" for k in range(20))
        source = ("function f(xs: number[]): number {\n  let m = xs.length;\n"
                  + steps + "  return m;\n}\n")
        tree = parse(source, "f", [("xs", "number[]")])
        self.assertIsNotNone(tree)
        self.assertLess(len(tree.source()), 40 * len(source))
        inputs = [[[1, 2, 3]], [[5, 0, 1, 2]], [[]], [[9]]]
        self.assertEqual(checker().values(["xs"], inputs, [tree.source()],
                                          prelude=P.prelude([tree]))[0],
                         checker().run(source, "f", inputs))

    def test_what_the_tree_does_not_model_is_read_as_written(self):
        """Widened reading: a Set, an object, a regular expression are
        opaque -- their own text, their variables filled in; a container
        changed in place is the change made on a copy; a helper typed by
        what it is given, one made inside, one that calls itself. Each
        checked by running the tree against the source."""
        from research.v696 import program as P
        from research.v696.checker import checker
        from research.v696.parse import parse
        cases = {
            "function f(xs: number[]): number { return new Set(xs).size; }":
                [[[1, 2, 2, 3]], [[]]],
            "function f(xs: number[]): number { const seen = {}; let n = 0; "
            "for (const x of xs) { if (!(x in seen)) n++; seen[x] = true; } "
            "return n; }": [[[1, 2, 2, 3]], [[5, 5]]],
            "function f(xs: number[]): number[] { const out = [...xs]; "
            "out.sort((a, b) => b - a); return out; }": [[[3, 1, 2]], [[]]],
            "function f(xs: number[]): number { const odd = (n) => n % 2 === "
            "1; return xs.filter(odd).length; }": [[[1, 2, 3]], [[4]]],
            "function fib(n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }\n"
            "function f(xs: number[]): number[] { return xs.map((x) => "
            "fib(x)); }": [[[0, 1, 5, 10]]],
        }
        for source, inputs in cases.items():
            tree = parse(source, "f", [("xs", "number[]")])
            self.assertIsNotNone(tree, source)
            want = checker().run(source, "f", inputs)
            got = checker().values(["xs"], inputs, [tree.source()],
                                   prelude=P.prelude([tree]))[0]
            self.assertEqual(got, want, source)

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

    def test_any_loop_is_read_and_means_what_it_did(self):
        """Rung 3, finished: several values carried are a tuple; `while`
        and any `for` are the language's own loop; return, break and
        continue inside a loop are carried as values. Checked by running
        the tree against the source on the same inputs."""
        from research.v696 import program as P
        from research.v696.checker import checker
        from research.v696.parse import parse
        cases = {
            "function f(xs: number[]): number[] { let m = -Infinity; const "
            "out: number[] = []; for (const x of xs) { m = Math.max(m, x); "
            "out.push(m); } return out; }": [[[1, 3, 2, 5]], [[4]], [[]]],
            "function f(n: number): number { let s = 0; while (n > 0) { s "
            "+= n % 10; n = Math.floor(n / 10); } return s; }":
                [[1234], [0], [7]],
            "function f(xs: number[]): number { let t = 0; for (const x of "
            "xs) { if (x < 0) continue; if (x > 50) break; t += x; } return "
            "t; }": [[[1, -2, 3, 60, 4]], [[5, 6]], [[]]],
            "function f(l: number[], s: number[]): boolean { for (let i = 0; "
            "i <= l.length - s.length; i++) { let j = 0; for (; j < "
            "s.length; j++) { if (l[i + j] !== s[j]) break; } if (j === "
            "s.length) return true; } return false; }":
                [[[2, 4, 3, 5, 7], [4, 3]], [[2, 4, 3, 5, 7], [3, 7]],
                 [[1], []]],
        }
        for source, inputs in cases.items():
            params = [("xs", "number[]")] if "xs" in source.split("{")[0] \
                else [("n", "number")] if "(n:" in source \
                else [("l", "number[]"), ("s", "number[]")]
            tree = parse(source, "f", params)
            self.assertIsNotNone(tree, source)
            names = [name for name, _ in params]
            want = checker().run(source, "f", inputs)
            got = checker().values(names, inputs, [tree.source()],
                                   prelude=P.prelude([tree]))[0]
            self.assertEqual(got, want, source)

    def test_a_wrong_or_loose_program_is_read_as_it_is(self):
        """What people's JavaScript leaves unsaid or gets wrong is read as
        it runs: falling off the end, a helper's untyped parameter, a name
        never declared, a list begun empty and never settled, two names
        for one list, an element set at any depth, `== null`, a loop that
        does nothing. Checked by running the tree against the source."""
        from research.v696 import program as P
        from research.v696.checker import checker
        from research.v696.parse import parse
        number, numbers = [("n", "number")], [("xs", "number[]")]
        cases = [
            # falls off the end where nothing in the loop returned
            ("function f(n: number): number { for (let i = n - 1; i >= 0; "
             "i--) if (n % i == 0) return i; }", number, [[12], [7], [0]]),
            # a guard inside a block, the rest going on past it
            ("function f(n: number): number { let z = 1; if (n > 2) { if (n "
             "> 5) return 9; z = 2; } return z + n; }", number,
             [[1], [4], [8]]),
            # what the language has had since; a test that is a number
            ("function f(xs: number[]): number { let t = 0; let k = "
             "xs.length; while (k) { k -= 1; t += xs.at(-1) + k; } return t; "
             "}", numbers, [[[1, 2, 3]], [[5]]]),
            # a helper whose parameter says no type; a loop that does nothing
            ("function f(xs: number[]): number { var twice = function (v) { "
             "let d = 0; for (const c of v) { d += c; } return d * 2; }; for "
             "(let i = 0; i < 3; i++) { let w = i; w += 1; } return "
             "twice(xs); }", numbers, [[[1, 2, 3]], [[]]]),
            # a name never declared; a list begun empty, sorted by swapping
            ("function f(xs: number[]): number[] { p = []; for (let i = 0; i "
             "< xs.length; i++) { if (xs[i] > 0) { p.push(xs[i]) } } for "
             "(let j = 0; j < p.length; j++) { let ind = j; for (let k = j + "
             "1; k < p.length; k++) { if (p[k] < p[ind]) { ind = k } } if "
             "(ind > j) { let tmp = p[j]; p[j] = p[ind]; p[ind] = tmp } } "
             "return p }", numbers, [[[3, -1, 2, 9, 1]], [[]]]),
            # two names, one list: changed by one, returned by the other
            ("function f(xs: number[]): number[] { let p = xs; for (let j = "
             "0; j < p.length; j++) { p[j] = p[j] * 2; } return xs; }",
             numbers, [[[1, 2, 3]], [[]]]),
            # an element set two deep; the same change on both ways through
            ("function f(n: number): number { const dp = Array.from({ "
             "length: n + 1 }, () => new Array(n + 1).fill(0)); let c = 0; "
             "for (let i = 1; i <= n; i++) { for (let j = 1; j <= n; j++) { "
             "if (i === j) { c++; dp[i][j] = dp[i - 1][j - 1] + 1; } else { "
             "c++; dp[i][j] = dp[i - 1][j]; } } } return dp[n][n] + c; }",
             number, [[3], [1], [0]]),
            # `== null` is true of what is undefined; keys gone over; `c[k]++`
            ("function f(xs: number[]): number { var best; const seen: "
             "{[key: string]: number} = {}; for (const x of xs) { if (best "
             "== null) best = x; if (x in seen) seen[x]++; else seen[x] = "
             "1; } let most = 0; for (let key in seen) { if (seen[key] > "
             "most) most = seen[key]; } return best + most; }", numbers,
             [[[4, 4, 5, 4]], [[7]]]),
        ]
        for source, params, inputs in cases:
            tree = parse(source, "f", params)
            self.assertIsNotNone(tree, source)
            names = [name for name, _ in params]
            want = checker().run(source, "f", inputs)
            got = checker().values(names, inputs, [tree.source()],
                                   prelude=P.prelude([tree]))[0]
            self.assertEqual(got, want, source)
        # a name nothing declares fails in the tree as it did in the source
        tree = parse("function f(xs: number[]): number[] { return "
                     "xs.filter(x > 1); }", "f", numbers)
        self.assertIsNotNone(tree)
        self.assertIn("error", checker().values(
            ["xs"], [[[1, 2]]], [tree.source()])[0][0])

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

    def test_a_few_examples_are_not_the_judge(self):
        """One example is met by a constant. With a reading of the request
        to judge by, a constant is the last thing chosen: a program the
        decoder wrote that meets the example comes first, then one the
        search found; without a reading, the first that meets it."""
        from research.v696.parse import parse
        from research.v696.search import Solver, Switches
        from research.v696.spec import Spec
        shown = [([[1, 2, 3]], False)]
        params = [("xs", "number[]")]
        switches = Switches(meet=True, coarse=True, proposals=True)
        plain = Solver(switches, budget=800).solve(
            Spec("any_negative", params, "boolean", shown))
        self.assertEqual(plain.program.source(), "false")
        wrote = parse("function f(xs: number[]): boolean { for (const x of "
                      "xs) { if (x < 0) return true; } return false; }", "f",
                      params)
        reading = {"uses": {}, "behaviour": {}}
        judged = Solver(switches, budget=800).solve(
            Spec("any_negative", params, "boolean", shown, expected=reading,
                 proposals=[wrote]))
        self.assertEqual(judged.route, "proposed")
        self.assertEqual(judged.program.source(), wrote.source())
        alone = Solver(switches, budget=800).solve(
            Spec("any_negative", params, "boolean", shown, expected=reading))
        self.assertIn("xs", alone.program.source())


@needs_node
class EditingTests(unittest.TestCase):
    """Rung 4: a program that exists is repaired by edits made in its
    source, the rest of it untouched."""

    def bug(self, source, cases, wanted):
        from research.v696.editing import Bug
        return Bug("t", source, "f", [("xs", "number[]"), ("t", "number")],
                   "boolean", cases, wanted)

    def test_a_missing_absolute_value_is_wrapped_in(self):
        from research.v696.editing import repair
        source = ("function f(xs: number[], t: number): boolean {\n"
                  "  // any two closer than t\n"
                  "  for (let i = 0; i < xs.length; i++)\n"
                  "    for (let j = 0; j < xs.length; j++)\n"
                  "      if (i != j && xs[i] - xs[j] < t) return true;\n"
                  "  return false;\n}\n")
        cases = [[[1, 5, 2.1], 0.5], [[1, 1.3], 0.5], [[3, 1], 1]]
        fix = repair(self.bug(source, cases, [False, True, False]))
        self.assertEqual(fix.route, "fixed")
        self.assertEqual(len(fix.edits), 1)
        self.assertIn("Math.abs(xs[i] - xs[j]) < t", fix.source)
        # the comment and the layout are as they were
        self.assertIn("  // any two closer than t\n", fix.source)

    def test_a_wrong_operator_is_swapped(self):
        from research.v696.editing import repair
        source = ("function f(xs: number[], t: number): boolean {\n"
                  "  return xs.every((x) => x > t);\n}\n")
        cases = [[[3, 4], 3], [[5], 5], [[1, 9], 2]]
        fix = repair(self.bug(source, cases, [True, True, False]))
        self.assertEqual(fix.route, "fixed")
        self.assertEqual(fix.source, source.replace("x > t", "x >= t"))


@needs_node
class ProjectTests(unittest.TestCase):
    """Rung 5: a project read whole; a fault found in another file; an API
    change followed by its callers, the compiler's errors the impasses."""

    FILES = {
        "/p/digits.ts": "export function digitSum(n: number): number {\n"
                        "  let s = 0;\n  while (n > 0) { s += n % 10; "
                        "n = Math.floor(n / 10); }\n  return s;\n}\n",
        "/p/main.ts": 'import { digitSum } from "./digits";\n'
                      "export function main(xs: number[]): number {\n"
                      "  return Math.max(...xs.map((x) => digitSum(x)));\n"
                      "}\n",
        "/p/main.test.ts": 'import { main } from "./main";\n'
                           "declare var require: any;\n"
                           'const assert = require("node:assert");\n'
                           "assert.deepEqual(main([19, 5, 100]), 10);\n"
                           "assert.deepEqual(main([7]), 7);\n",
    }

    def task(self, files, kind):
        from research.v696.projects import Task
        return Task("t", kind, files, "/p/main.ts", "main",
                    [["xs", "number[]"]], "number", [[[19, 5, 100]], [[7]]],
                    [10, 7])

    def test_a_project_is_read_whole(self):
        from research.v696.checker import checker
        from research.v696.parse import parse_project
        tree = parse_project(self.FILES, "/p/main.ts", "main",
                             [("xs", "number[]")])
        self.assertIn("digitSum(x)", tree.source())
        self.assertIsNone(checker().project(self.FILES, "/p/main.test.ts"))

    def test_a_fault_in_another_file_is_fixed_there(self):
        from research.v696.changing import fix
        broken = dict(self.FILES)
        broken["/p/digits.ts"] = broken["/p/digits.ts"].replace("n % 10",
                                                                "n % 9")
        done = fix(self.task(broken, "bug"))
        self.assertEqual(done.route, "solved")
        self.assertEqual(done.files["/p/main.ts"], self.FILES["/p/main.ts"])
        self.assertIn("n % 10", done.files["/p/digits.ts"])

    def test_a_renamed_export_is_followed(self):
        from research.v696.changing import change
        renamed = dict(self.FILES)
        renamed["/p/digits.ts"] = renamed["/p/digits.ts"].replace(
            "digitSum", "sumOfDigits")
        done = change(self.task(renamed, "change"))
        self.assertEqual(done.route, "solved")
        self.assertEqual(done.impasses, 1)
        self.assertEqual(len(done.edits), 1)


if __name__ == "__main__":
    unittest.main()
