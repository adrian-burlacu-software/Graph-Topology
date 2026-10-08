"""v700: a change asked of the project, made in its files.

    python -m unittest research.v700.test_v700
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class EditsTests(unittest.TestCase):
    """`teach_editor.py`: a commit said as blocks, an answer put in."""

    def test_a_commit_said_as_blocks_and_put_back(self):
        from research.v700 import teach_editor as T
        old = ("def f(x):\n    a = 1\n    return a\n\n"
               "def g(x):\n    a = 1\n    return x\n").splitlines(True)
        new = ("def f(x):\n    a = 1\n    return a\n\n"
               "def g(x):\n    a = 2\n    return x\n").splitlines(True)
        part = T.python_part("".join(old), [5])
        self.assertEqual(part, (4, 7))
        blocks = T.blocks_of(old, new, *part)
        # `a = 1` alone is in g once: no wider block is needed
        self.assertEqual(blocks, [(["    a = 1\n"], ["    a = 2\n"])])
        shown = "".join(old[4:7])
        self.assertEqual(T.applied(shown, T.said(blocks)), "".join(new[4:7]))

    def test_an_insertion_is_said_beside_a_line(self):
        from research.v700 import teach_editor as T
        old = ["def f():\n", "    return 1\n"]
        new = ["def f():\n", "    print(1)\n", "    return 1\n"]
        blocks = T.blocks_of(old, new, 0, 2)
        self.assertEqual(blocks, [(["def f():\n"],
                                   ["def f():\n", "    print(1)\n"])])

    def test_an_answer_not_found_once_is_refused(self):
        from research.v700 import teach_editor as T
        part = "x = 1\nx = 1\n"
        self.assertIsNone(T.applied(part, "<<<\nx = 1\n===\nx = 2\n>>>"))
        self.assertIsNone(T.applied(part, "<<<\ny = 1\n===\nx = 2\n>>>"))
        self.assertIsNone(T.applied(part, "no blocks at all"))


class ProjectOnDiskTests(unittest.TestCase):
    """A project with a root: its files changed there, and nowhere else."""

    def test_written_inside_the_root_only(self):
        from research.v698.project import Project
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "a.py").write_text("x = 1\n", encoding="utf-8")
            held = Project("t", root)
            held.put({"a.py": "x = 1\n"})
            self.assertTrue(held.write("a.py", "x = 2\n"))
            self.assertEqual((Path(root) / "a.py").read_text(), "x = 2\n")
            self.assertEqual(held.files["a.py"], "x = 2\n")
            self.assertIsNone(held.on_disk("../outside.py"))
            with self.assertRaises(ValueError):
                held.write("b.py", "y = 1\n")
        self.assertFalse(Project("no root").on_disk("a.py"))


class PartTests(unittest.TestCase):
    """What is changed: the function named, its decorators with it."""

    def test_a_function_with_its_decorators(self):
        from research.v698.asking import Subject
        from research.v698.project import Project
        from research.v700 import fixing
        held = Project("t")
        held.put({"a.py": "import x\n\n@x.cached\ndef f(a: int) -> int:\n"
                          "    return a\n\n\ndef g() -> int:\n"
                          "    return 1\n"})
        self.assertEqual(fixing.part_of(held, Subject("function", "f",
                                                      "a.py")),
                         ("a.py", 3, 5))
        self.assertEqual(fixing.part_of(held, Subject("file", "a.py",
                                                      "a.py")),
                         ("a.py", 1, 9))


class WindowTests(unittest.TestCase):
    """A long file named: the lines around the code the statement names,
    looked up in the code -- not in its docs."""

    def test_the_imports_named_not_the_docstring(self):
        from research.v698.asking import Subject
        from research.v698.project import Project
        from research.v700 import fixing
        doc = '"""Uses math and checker\n' + "words\n" * 80 + '"""\n'
        body = "".join(f"def f{n}() -> int:\n    return {n}\n\n"
                       for n in range(40))
        held = Project("t")
        held.put({"m.py": doc + "import math\nfrom x import checker\n\n"
                  + body})
        path, start, end = fixing.part_of(
            held, Subject("file", "m.py", "m.py"),
            "m.py imports math and checker but uses neither; drop them")
        self.assertTrue(start <= 83 <= 84 <= end, (start, end))
        self.assertLessEqual(end - start + 1, fixing.LONGEST_PART)
        self.assertEqual(fixing.names_in(
            "const a = 'x y'; /* b\nc */\nlet zed = f(\"q\");", "a.ts"),
            {"a": [1], "zed": [3], "f": [3]})


class CommitReadingTests(unittest.TestCase):
    """Commit messages taught as changes: of the function named, where the
    name is written as code is."""

    def test_names_written_as_code(self):
        from research.v698 import teach_code_talk as T
        self.assertTrue(T._codey("load_data_frame", "add a flag"))
        self.assertTrue(T._codey("onOff", "make onOff faster"))
        self.assertTrue(T._codey("run", "make run() faster"))
        self.assertFalse(T._codey("default", "build with PIC by default"))
        self.assertEqual(T._function_named(
            "    @x\n    async def handle(self, a):\n        pass\n"),
            "handle")
        self.assertEqual(T._function_named(
            "export function parseConfig(text: string): Config {\n"),
            "parseConfig")


if __name__ == "__main__":
    unittest.main()
