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

    def test_questions(self):
        from research.v698.asking import answered, classify
        held = self.held()
        for text, kind in (("what is in the project", "overview"),
                           ("which files", "files"),
                           ("where is digitSum", "where"),
                           ("who calls digitSum", "callers"),
                           ("what does main call", "calls"),
                           ("what does twice do", "explain"),
                           ("what is in src/main.ts", "in file"),
                           ("does the project compile", "errors"),
                           ("where is Mary", None),
                           ("can it swim", None)):
            self.assertEqual(classify(text, held)[0], kind, text)
        said = answered("who calls digitSum", held)["spoken"]
        self.assertIn("main (src/main.ts)", said)
        said = answered("what does main call", held)["spoken"]
        self.assertIn("digitSum (src/digits.ts)", said)


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
