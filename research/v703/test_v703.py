"""v703: a change across a project -- what joins its files, what a plan is
shown, what a plan left undone."""
from __future__ import annotations

import unittest

from research.v703 import planning, structure, teach_plans


def _project(files: dict):
    return structure.subset(files)


class StructureTests(unittest.TestCase):
    """Files joined by calls, by imports anywhere and calls through them,
    and by a route both say."""

    FILES = {
        "srv/server.py": ('"""The server."""\n\ndef health():\n'
                          '    return {"path": "/api/health"}\n'),
        "tools/bridge.py": ('"""The bridge."""\n\ndef ask():\n'
                            '    return get("/api/health")\n'),
        "srv/page.py": ('"""The page."""\n\ndef act():\n'
                        '    from srv import shell\n'
                        '    return shell.run("ls")\n'),
        "srv/shell.py": '"""Bash."""\n\ndef run(command):\n    return 0\n',
        "srv/__init__.py": "",
    }

    def test_joined(self):
        held = _project(self.FILES)
        self.assertIn("tools/bridge.py",
                      structure.linked(held, ["srv/server.py"], 5))
        self.assertIn("srv/shell.py",
                      structure.linked(held, ["srv/page.py"], 5))

    def test_said_as_what_it_is(self):
        held = _project(self.FILES)
        said = structure.summary(held, "srv/shell.py",
                                 ["srv/shell.py", "srv/page.py"])
        self.assertIn("Bash", said)
        self.assertIn("defines run", said)
        self.assertIn("joined to srv/page.py", said)

    def test_callers_left_the_old_way(self):
        held = _project({"a.py": "def f(x):\n    return x\n",
                         "b.py": "from a import f\n\ndef g():\n"
                                 "    return f(1)\n"})
        before = structure.signatures(held, ["a.py"])
        held.put({"a.py": "def f(x, y):\n    return x + y\n"})
        self.assertEqual(structure.broken_callers(held, before, ["a.py"]),
                         ["b.py#g -> a.py#f"])
        self.assertEqual(structure.broken_callers(held, before,
                                                  ["a.py", "b.py"]), [])


class PlanTests(unittest.TestCase):
    def test_steps_of_files_shown(self):
        answer = ("srv/server.py: add the shell's asking to health\n"
                  "tools/elsewhere.py: nothing\n"
                  "new srv/notes.md: say it\n")
        steps = planning._steps(answer, ["srv/server.py"])
        self.assertEqual([one["path"] for one in steps],
                         ["srv/server.py", "srv/notes.md"])
        self.assertTrue(steps[1]["new"])

    def test_a_step_names_its_change(self):
        diff = "+PREFERRED = ('reader-code30',)\n-PREFERRED = ()\n"
        self.assertTrue(teach_plans._names_its_change(
            "Change the PREFERRED list to start with reader-code30", diff))
        self.assertFalse(teach_plans._names_its_change(
            "The file is touched by the following changes:", diff))
        self.assertFalse(teach_plans._names_its_change(
            "Update the file as needed", "+the file\n"))


if __name__ == "__main__":
    unittest.main()
