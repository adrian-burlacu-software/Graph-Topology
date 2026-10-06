"""v699: Python through every layer -- checked, read, printed, searched,
held to the ways asked, in a project, in the conversation.

    python -m unittest research.v699.test_v699
"""
from __future__ import annotations

import shutil
import unittest

HAVE_NODE = shutil.which("node") is not None


class CheckerTests(unittest.TestCase):
    """`pycheck.py`: Python run, its values as TypeScript's are sent."""

    @classmethod
    def setUpClass(cls):
        from research.v696.checker import checker
        cls.c = checker("python")

    def test_run_shapes_values_and_stops_what_runs_away(self):
        source = ('def f(xs: tuple[int, int], s: set[str]) -> list[int]:\n'
                  '    print("hi")\n    return [xs[1], xs[0]]\n\n'
                  'def slow(n: int) -> int:\n    while True:\n'
                  '        n += 1\n')
        row = self.c.run(source, "f", [[[1, 2], {"$set": ["a"]}]])[0]
        self.assertEqual(row, {"value": [2, 1], "printed": ["hi"]})
        self.assertIn("timed out", self.c.run(source, "slow", [[1]],
                                              timeout=200)[0]["error"])

    def test_values_are_encoded_as_typescript_sends_them(self):
        rows = self.c.values(["xs"], [[[3, 1]]],
                             ["sorted(xs)", "{1, 2}", "{(1, 2): 3}",
                              "float('inf')", "2.0", "2 ** 80",
                              "iter(xs)"])
        got = [row[0].get("value", "error") for row in rows]
        self.assertEqual(got[:6], [[1, 3], {"$set": [1, 2]},
                                   {"$map": [[[1, 2], 3]]}, "Infinity", 2,
                                   float(2 ** 80)])
        self.assertIn("iterator", rows[6][0]["error"])

    def test_tests_and_diagnostics_in_a_package(self):
        self.assertIsNone(self.c.tests("def g(x):\n    return x + 1\n"
                                       "assert g(1) == 2\n"))
        self.assertIn("AssertionError",
                      self.c.tests("assert 1 == 2\n") or "")
        # twice: the second is told by mypy's cache, at the first's paths
        for _ in range(2):
            found = self.c.diagnose({
                "/ws/app/__init__.py": "",
                "/ws/app/main.py": "def broken() -> int:\n    return 'x'\n"})
            self.assertEqual([(one["file"], one["line"]) for one in found],
                             [("/ws/app/main.py", 2)])


class ReadingAndPrintingTests(unittest.TestCase):
    """`pyparse.py` reads Python into the search's trees, `pyprint.py`
    writes them back as people write Python: the same behaviour."""

    CASES = [
        ("def f(xs: list[int]) -> list[int]:\n    out = []\n"
         "    for x in xs:\n        if x % 2 == 0:\n"
         "            out.append(x * 10)\n    return out\n",
         [("xs", "number[]")], [[[1, 2, 3, 4]]],
         "[(x * 10) for x in xs if ((x % 2) == 0)]"),
        ("def f(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n"
         "        a, b = b, a + b\n    return a\n",
         [("n", "number")], [[0], [1], [10]], None),
        ("def f(n: int) -> int:\n    while n > 10:\n        n -= 10\n"
         "    return n\n", [("n", "number")], [[25], [3]], None),
        ("def f(n: int) -> int:\n    return 1 if n < 2 else n * f(n - 1)\n",
         [("n", "number")], [[0], [5]], None),
    ]

    def test_read_trees_behave_as_their_source(self):
        from research.v696 import pyparse, pyprint
        from research.v696.checker import checker
        for source, params, cases, printed in self.CASES:
            tree = pyparse.parse(source, "f", params)
            self.assertIsNotNone(tree, pyparse.why(source, "f", params))
            text = pyprint.text(tree)
            if printed:
                self.assertEqual(text, printed)
            want = checker("python").run(source, "f", cases)
            got = checker("python").values([n for n, _ in params], cases,
                                           [text], prelude=pyprint.prelude(
                                               [tree]))[0]
            self.assertEqual(got, want, source)

    def test_what_a_tree_cannot_carry_is_refused_not_passed_over(self):
        from research.v696 import pyparse
        for source in (
                "def f(xs: list[int]) -> list[int]:\n    for i in range(3):"
                "\n        xs[i] += 1\n    return xs\n",
                "def f(xs: list[int]) -> int:\n    for x in xs:\n"
                "        if x > 2:\n            break\n    return 0\n"):
            self.assertIsNone(pyparse.parse(source, "f",
                                            [("xs", "number[]")]))

    def test_structure_in_the_words_both_languages_share(self):
        from research.v696.checker import checker
        uses = checker("python").structure(
            "def f(w: list[str], n: int) -> str:\n"
            "    return ' '.join(x.upper() for x in w if len(x) > n)\n",
            "f")["uses"]
        for word in ("String.toUpperCase", "String.length", ">",
                     "Array.join", "Array.filter"):
            self.assertIn(word, uses)


class SearchTests(unittest.TestCase):
    """The search composes Python's library and writes Python."""

    def test_python_specs_are_solved_and_printed_as_python(self):
        from research.v696.experiment import CONFIGS
        from research.v696.search import Solver
        from research.v696.spec import Spec
        spec = Spec("evens", [("xs", "number[]")], "number[]",
                    [([[1, 2, 3, 4]], [2, 4]), ([[7, 8]], [8])],
                    language="python", entry="evens",
                    written={"xs": "list[int]", "return": "list[int]"})
        got = Solver(CONFIGS["meet+repair+forms"], budget=3000).solve(spec)
        self.assertIsNotNone(got.program)
        written = spec.function(got.program)
        self.assertTrue(written.startswith(
            "def evens(xs: list[int]) -> list[int]:"))
        self.assertIn(" for x in xs if ", written)


class WaysTests(unittest.TestCase):
    """A way of writing, held in the language the code is in."""

    def test_each_language_its_own_test(self):
        from research.v696.checker import checker
        from research.v698 import ways as W
        source = ("def f(xs: list[int]) -> list[str]:\n    match len(xs):\n"
                  "        case 0:\n            return []\n"
                  "    return [f'{i}:{x}' for i, x in enumerate(xs) if x]\n")
        shape = checker("python").shape(source, "f")
        self.assertTrue(W.fits(shape, ["switch", "template", "enumerate",
                                       "comprehension", "filter"]))
        self.assertEqual(W.said(["switch"], "python"),
                         "with a match statement")
        self.assertEqual(W.held(["const", "python", "switch"], "python"),
                         ["switch"])
        self.assertEqual(W.asked_language(["switch", "python"]), "python")

    def test_restyled_where_it_is_syntax_alone(self):
        from research.v696.checker import checker
        from research.v698 import ways as W
        c = checker("python")
        source = ("def day(n: int) -> str:\n    if n == 0:\n"
                  "        return 'Sun'\n    elif n == 1 or n == 7:\n"
                  "        return 'Mon'\n    return '?'\n")
        for way in ("switch", "ternary"):
            out = c.restyle(source, "day", way)
            self.assertTrue(W.fits(c.shape(out, "day"), [way]))
            self.assertEqual(c.run(out, "day", [[0], [7], [3]]),
                             c.run(source, "day", [[0], [7], [3]]))


class ConversationTests(unittest.TestCase):
    """A request read in Python; a project of Python files."""

    def test_a_python_request_is_read_as_python(self):
        from research.v697 import coding
        asked = coding.read('count the vowels\ndef vowels(s: str) -> int:\n'
                            'vowels("hello") == 2\nvowels("sky") == 0')
        self.assertEqual((asked["language"], asked["entry"],
                          asked["params"], asked["signature"]),
                         ("python", "vowels", [["s", "string"]],
                          "def vowels(s: str) -> int:"))
        examples = coding.read("pairs\nf((1, 2), None) == [2, 1]",
                               "python")["examples"]
        self.assertEqual(examples[0]["args"], [[1, 2], None])
        mine = coding.read("what about\ndef day(n: int) -> str:\n"
                           "    return 'x'\nthanks")
        self.assertEqual(mine["yours"],
                         "def day(n: int) -> str:\n    return 'x'\n")

    def test_restated_in_another_language(self):
        from research.v697 import conversation as C
        said = C.restated({"english": "flags", "entry": "f",
                           "examples": [{"args": [True], "value": None}]},
                          "python")
        self.assertEqual(said, "flags\nf(True) == None")

    @unittest.skipUnless(HAVE_NODE, "node missing")
    def test_a_project_of_both_languages(self):
        from research.v698.project import Project
        held = Project("mixed")
        refused = held.put({
            "app/__init__.py": "",
            "app/maths.py": "def double(x: int) -> int:\n    return x * 2\n",
            "app/main.py": "from .maths import double\n\n"
                           "def run(n: int) -> int:\n"
                           "    return double(n) + 1\n",
            "web/index.ts": "export function hi(): string { return 'hi'; }\n",
            "notes.md": "x"})
        self.assertEqual(list(refused), ["notes.md"])
        self.assertEqual(held.languages(), ["python", "typescript"])
        self.assertEqual(held.calls(), [{"from": "app/main.py#run",
                                         "to": "app/maths.py#double",
                                         "call": "double"}])

    def test_a_taught_concept_in_python(self):
        from research.v696 import program as P
        from research.v696 import pyprint
        from research.v698 import knowledge as K
        name, source = K.helper_python("vowel", ["a", "e", "i", "o", "u"])
        op = K._python_operator(name, source)
        used = P.apply(op, [P.param("c", "string")])
        self.assertEqual(pyprint.text(used), "is_vowel(c)")
        self.assertIn("def is_vowel(x: str) -> bool:",
                      pyprint.prelude([used]))


class RepairTests(unittest.TestCase):
    """Rung 4 in Python: edits read off the syntax (`pyediting.py`), never
    in annotations; a bug fixed by the one edit it needs."""

    def test_edits_leave_annotations_alone(self):
        from research.v696 import pyediting
        source = ("from typing import List\n\n"
                  "def f(xs: List[int], k: int) -> List[int]:\n"
                  "    return [x for x in xs if x < k]\n")
        made = pyediting.edits(source, "f")
        kinds = {one.kind for one in made}
        self.assertTrue({"operator", "name", "swap"} <= kinds)
        self.assertIn("<=", {one.text for one in made})
        line = source.index("def f")
        self.assertFalse([one for one in made
                          if line <= one.start < source.index(":\n", line)])

    def test_a_bug_fixed_by_one_edit(self):
        from research.v696.editing import Bug, repair
        bug = Bug("off-by-one", "def first(xs: list) -> int:\n"
                                "    return xs[1]\n",
                  "first", [("xs", "number[]")], "number",
                  [[[1, 2, 3]], [[5, 6]]], [1, 5], language="python")
        fix = repair(bug, rewrite=False)
        self.assertEqual((fix.route, [one.kind for one in fix.edits]),
                         ("fixed", ["constant"]))
        self.assertIn("xs[0]", fix.source)


if __name__ == "__main__":
    unittest.main()
