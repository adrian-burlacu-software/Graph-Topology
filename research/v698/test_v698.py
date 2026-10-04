"""v698: a project read whole, asked about; code pasted and explained.

    python -m unittest research.v698.test_v698
"""
from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

HAVE = shutil.which("node") is not None
needs_node = unittest.skipUnless(HAVE, "node missing")
TOOLS = Path(__file__).resolve().parents[2] / "tools" / "vscode-graph-topology"

FILES = {
    "src/digits.ts": "export function digitSum(n: number): number {\n"
                     "  let s = 0;\n  while (n > 0) { s += n % 10; "
                     "n = Math.floor(n / 10); }\n  return s;\n}\n",
    "src/main.ts": 'import { digitSum } from "./digits";\n'
                   "export function main(xs: number[]): number {\n"
                   "  return Math.max(...xs.map((x) => digitSum(x)));\n}\n"
                   "export const twice = (n: number): number => main([n]) * 2;\n",
    "src/lib/index.ts": "export class Box {\n  open(): number {\n"
                        "    return 1;\n  }\n}\n",
    "README.md": "not read",
}


@needs_node
class ProjectTests(unittest.TestCase):
    def held(self):
        from research.v698.project import Project
        held = Project("t")
        self.refused = held.put(FILES)
        return held

    def test_outline_and_calls_across_files(self):
        held = self.held()
        self.assertIn("README.md", self.refused)
        names = {one["name"] for one in held.functions()}
        self.assertEqual(names, {"digitSum", "main", "twice", "Box.open"})
        calls = {(one["from"], one["to"]) for one in held.calls()}
        self.assertIn(("src/main.ts#main", "src/digits.ts#digitSum"), calls)
        self.assertIn(("src/main.ts#twice", "src/main.ts#main"), calls)
        self.assertEqual([one["from"] for one in held.callers("digitSum")],
                         ["src/main.ts#main"])
        self.assertEqual(held.find("open")[0]["file"], "src/lib/index.ts")

    def test_diagnostics_and_a_change(self):
        held = self.held()
        self.assertEqual(held.diagnostics(), [])
        held.put({"src/main.ts": FILES["src/main.ts"].replace(
            "digitSum(x)", "digitsum(x)")})
        found = held.diagnostics()
        self.assertEqual(found[0]["file"], "src/main.ts")
        self.assertEqual(found[0]["line"], 3)
        held.put({"src/lib/index.ts": None})
        self.assertNotIn("src/lib/index.ts", held.files)

    def test_a_subject_and_an_aspect(self):
        from research.v697.conversation import Workspace
        from research.v698 import asking
        held, space = self.held(), Workspace()
        asking.FOCUS.clear()
        for text, aspect, kind in (
                ("what is in the project", "explain", "project"),
                ("what is the project?", "explain", "project"),
                ("describe the codebase", "explain", "project"),
                ("Are there any bugs in this project?", "bugs", "project"),
                ("can you find any bugs in the code?", "bugs", "project"),
                ("which files", "files", "project"),
                ("where is digitSum", "where", "function"),
                ("who calls digitSum", "callers", "function"),
                ("what does main call", "calls", "function"),
                ("what does twice do", "explain", "function"),
                ("how big is it", "size", "function"),
                ("what is in src/main.ts", "explain", "file"),
                ("does the project compile", "bugs", "project"),
                # not about code: a subject of their own
                ("what is a project", None, None),
                ("do bugs have legs", None, None),
                ("where is Mary", None, None),
                ("can it swim", None, None)):
            found, subject = asking.route(text, held, space, 3, "k")
            self.assertEqual((found, subject and subject.kind),
                             (aspect, kind), text)
            if found:
                asking.FOCUS["k"] = (subject, 3)
        said = asking.answered("who calls digitSum", held, space, 4,
                               "k")["spoken"]
        self.assertIn("main (src/main.ts)", said)
        said = asking.answered("what does main call", held, space, 5,
                               "k")["spoken"]
        self.assertIn("digitSum (src/digits.ts)", said)

    def test_gaps(self):
        from research.v697.conversation import Workspace
        from research.v698 import asking
        held = self.held()
        held.put({"src/bad.ts": "export function label(n: number): string "
                                "{\n  return n;\n}\n"
                                "function never(): number {\n  return 1;\n}\n"})
        found = asking.answered("are there any bugs in the project", held,
                                Workspace(), 1, "g")
        looked = found["code"]["project"]["looked"]
        self.assertEqual(looked["contradicted"][0]["where"], "src/bad.ts:2")
        self.assertIn("never", [one["in"] for one in looked["unused"]])
        self.assertTrue(any(one["question"].startswith("digitSum(")
                            for one in looked["open"]))


@needs_node
class KnowledgeTests(unittest.TestCase):
    """A concept taught is kept, found in a request that names it, told to
    the writers, and offered to the search -- inside a lambda too."""

    def test_taught_kept_and_used(self):
        import tempfile
        from pathlib import Path
        from research.v696 import search as S
        from research.v696.experiment import CONFIGS
        from research.v696.spec import Spec
        from research.v698 import knowledge as K
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "k.json"
            K.teach("vowel", ["a", "e", "i", "o", "u"], "taught", path)
            self.assertEqual(K.relevant("count the vowels in it", path),
                             [("vowel", ["a", "e", "i", "o", "u"])])
            self.assertEqual(K.relevant("count the words", path), [])
            known = K.context({"english": "the vowels of a string"}, path)
        self.assertIn("a vowel is one of a, e, i, o, u", known["english"])
        self.assertEqual([op.name for op in known["library"]], ["isVowel"])
        spec = Spec("v", [("s", "string")], "string[]",
                    [(["hello"], ["e", "o"]), (["sky"], []),
                     (["banana"], ["a", "a", "a"])], entry="vowelsOf",
                    library=list(known["library"]))
        got = S.Solver(CONFIGS["meet+repair+forms"], budget=8000).solve(spec)
        self.assertIn("isVowel(x)", got.program.source())


@needs_node
class PastedTests(unittest.TestCase):
    def test_pasted_code_is_read_kept_and_run(self):
        from research.v697.conversation import Workspace, classify
        from research.v698.asking import pasted, read_pasted
        text = ("explain this code\n```ts\nfunction twice(xs: number[]): "
                "number[] {\n  return xs.map((x) => x * 2);\n}\n```")
        code = pasted(text)
        self.assertIn("function twice", code)
        self.assertIsNone(pasted("```ts\nlet a = 1;\n```"))
        space = Workspace()
        said = read_pasted(text, code, space)
        self.assertIn("Array.map", said["spoken"])
        self.assertEqual(classify("twice([1, 2])", space, 1), ("call", "[1, 2]"))


SWITCHED = ('function day(n: number): string {\n  switch (n) {\n'
            '    case 0: return "Sun";\n    case 1: case 7: return "Mon";\n'
            '    default: return "?";\n  }\n}\n')


@needs_node
class WaysTests(unittest.TestCase):
    """Ways of writing (`ways.py`): a switch read as the chain it means,
    each way checked on how code is written, restyled where that is syntax
    alone, held across a conversation's changes."""

    def test_a_switch_is_read_as_the_chain_it_means(self):
        from research.v696.parse import parse
        tree = parse(SWITCHED, "day", [("n", "number")])
        self.assertEqual(tree.source(), '((n === 0) ? "Sun" : (((n === 1) || '
                                        '(n === 7)) ? "Mon" : "?"))')
        falls = ('function f(n: number): string {\n  let s = "";\n'
                 '  switch (n) {\n    case 1: s = "a";\n    default: s += "b";'
                 '\n  }\n  return s;\n}\n')
        self.assertIsNone(parse(falls, "f", [("n", "number")]))

    def test_each_way_is_checked_on_the_code(self):
        from research.v696.checker import checker
        from research.v698 import ways as W
        shape = checker().shape(SWITCHED, "day")
        self.assertTrue(W.fits(shape, ["switch", "iterative", "declaration"]))
        self.assertEqual(W.missing(shape, ["switch", "recursive", "arrow"]),
                         ["recursive", "arrow"])
        fact = "const fact = (n: number): number => n < 2 ? 1 : n * fact(n - 1);"
        shape = checker().shape(fact, "fact")
        self.assertTrue(W.fits(shape, ["recursive", "arrow", "one-liner",
                                       "ternary", "no-loop", "const"]))

    def test_restyled_where_it_is_syntax_alone(self):
        from research.v696.checker import checker
        from research.v698 import ways as W
        chained = ('function day(n: number): string {\n'
                   '  return n === 0 ? "Sun" : n === 1 || n === 7 ? "Mon" '
                   ': "?";\n}\n')
        switched = checker().restyle(chained, "day", "switch")
        self.assertIn("case 7:", switched)
        rows = checker().run(switched, "day", [[0], [7], [3]])
        self.assertEqual([row["value"] for row in rows], ["Sun", "Mon", "?"])
        ifs = checker().restyle(SWITCHED, "day", "ifs")
        self.assertTrue(W.fits(checker().shape(ifs, "day"), ["ifs"]))
        arrow = checker().restyle(SWITCHED, "day", "arrow")
        self.assertTrue(W.fits(checker().shape(arrow, "day"),
                               ["arrow", "switch"]))
        self.assertIsNone(checker().restyle(
            "function f(n: number): number {\n  return n + 1;\n}\n", "f",
            "switch"))

    def test_ways_held_across_changes(self):
        from research.v698 import ways as W
        held = W.merged([], "switch")
        held = W.merged(held, "recursive")
        self.assertEqual(W.merged(held, "iterative"), ["switch", "iterative"])
        self.assertEqual(W.merged(held, "none"), held)
        self.assertEqual(W.said(["switch", "reduce"]),
                         "with a switch statement and with .reduce")


@needs_node
class ExtensionTests(unittest.TestCase):
    def test_the_extension_parses_and_its_library_holds(self):
        for name in ("extension.js", "lib.js"):
            done = subprocess.run(["node", "--check", str(TOOLS / name)],
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
        done = subprocess.run(["node", str(TOOLS / "test.js")],
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == "__main__":
    unittest.main()
