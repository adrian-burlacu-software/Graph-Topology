"""v703: a change across a project -- what joins its files, what a plan is
shown, what a plan left undone."""
from __future__ import annotations

import unittest

from research.v703 import planning, structure, teach_parts, teach_plans


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


class FocusTests(unittest.TestCase):
    """What a request names outright, what code says it, who calls it."""

    FILES = {
        "srv/server.py": ('"""The server."""\n\nclass Handler:\n'
                          '    def do_GET(self):\n'
                          '        if self.path == "/api/health":\n'
                          '            return {}\n'),
        "srv/notes.py": ('"""Says /api/health in its doc only."""\n\n'
                         'def note():\n    """About /api/health."""\n'
                         '    return 1\n'),
        "srv/shell.py": "def reads_only(command):\n    return True\n",
        "srv/page.py": ("def act():\n    from srv import shell\n"
                        "    return shell.reads_only('ls')\n"),
        "srv/__init__.py": "",
        "srv/test_srv.py": ("def test_health():\n"
                            "    assert get('/api/health') == {}\n"),
        "docs/protocol.md": "# The protocol\n",
    }

    def test_a_test_says_it_of_the_code(self):
        held = _project(self.FILES)
        outright, _ = planning.focus("show the shell in /api/health", held)
        self.assertNotIn("srv/test_srv.py", outright)
        self.assertTrue(planning._is_test("srv/test_srv.py"))
        self.assertFalse(planning._is_test("srv/testing.py"))

    def test_what_the_code_says(self):
        held = _project(self.FILES)
        server = held.files["srv/server.py"]
        self.assertIn("/api/health", planning._code_says(server, 4, 6))
        notes = held.files["srv/notes.py"]
        self.assertNotIn("/api/health", planning._code_says(notes, 1, 6))

    def test_named_outright(self):
        held = _project(self.FILES)
        outright, _ = planning.focus(
            "show whether the shell asks in /api/health, in protocol.md",
            held)
        self.assertIn("docs/protocol.md", outright)
        self.assertIn("srv/server.py", outright)
        self.assertNotIn("srv/notes.py", outright)

    def test_callers_through_an_import_in_a_function(self):
        held = _project(self.FILES)
        self.assertEqual(structure.callers_of(held, "srv/shell.py",
                                              "reads_only"),
                         [("srv/page.py", "act")])


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


class PartTests(unittest.TestCase):
    """A function's part of a commit, what it could use, what is imported
    for it."""

    def test_a_functions_changes_only(self):
        import difflib
        from research.v700 import teach_editor as T
        old = T._lines("def f():\n    return 1\n\n\ndef g():\n"
                       "    return 2\n")
        new = T._lines("def f():\n    return 1\n    # more\n\n\ndef g():\n"
                       "    return 3\n")
        spans = teach_parts.functions("".join(old))
        groups = [one for one in difflib.SequenceMatcher(
            None, old, new, autojunk=False).get_opcodes() if one[0] != "equal"]
        owners = [teach_parts._owner(spans, one, new)[0] for one in groups]
        # added under f, indented: f's; g's line changed: g's
        self.assertEqual(owners, ["f", "g"])
        blocks = teach_parts._blocks(old, new, groups[1:], 4, 6)
        self.assertEqual(T.said(blocks),
                         "<<<\n    return 2\n===\n    return 3\n>>>")

    def test_what_it_could_use(self):
        files = {"srv/server.py": "from srv import shell\n\nVERSION = 1\n",
                 "srv/shell.py": "ASK = True\n\ndef run(command):\n"
                                 "    return 0\n",
                 "srv/__init__.py": "",
                 "tools/my-bridge/server.py": "def health():\n    return 1\n"}
        units = teach_parts.units_of("srv/server.py",
                                     files["srv/server.py"], files,
                                     set(files))
        said = [one[0] for one in units]
        self.assertIn("VERSION", said)
        self.assertIn("shell.ASK", said)
        self.assertIn("shell.run", said)
        # a program of its own, no module: nothing of it to use
        self.assertFalse(any("health" in one for one in said))
        self.assertEqual(teach_parts.import_line("research/v702/shell.py"),
                         "from research.v702 import shell")

    def test_imported_at_the_top(self):
        text = ("import json\nfrom srv import page\n\n\ndef main():\n"
                "    from srv import shell\n    return shell\n")
        self.assertEqual(teach_parts.top_names(text), {"json", "page"})

    def test_the_import_looked_up(self):
        from research.v700 import fixing
        text = ('"""A server."""\nimport json\n\n\ndef health():\n'
                '    return {"asks": shell.ASK}\n')
        made = fixing._with_imports("srv/server.py", text, {
            "shell": "from srv import shell"})
        self.assertIn("import json\nfrom srv import shell\n", made)
        self.assertEqual(fixing._with_imports("srv/server.py", text, {}),
                         text)


if __name__ == "__main__":
    unittest.main()
