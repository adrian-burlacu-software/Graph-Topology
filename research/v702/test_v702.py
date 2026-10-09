"""v702: Bash -- what only reads, the command cut from a message, a run."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research.v702 import shell


class ReadsOnlyTests(unittest.TestCase):
    """What runs without asking: every program a command runs only reads,
    and it writes no file."""

    def test_reading(self):
        for command in ("ls research/v701", "git log --oneline -3",
                        "du -sh data/ | sort -h", "find . -name '*.py' | wc -l",
                        "ls 2>/dev/null", "grep -r foo . 2>&1 | head",
                        "git branch -a", "git remote show origin"):
            self.assertTrue(shell.reads_only(command)[0], command)

    def test_changing(self):
        for command in ("rm -rf x", "echo hi > out.txt",
                        "find . -name x -delete", "sed -i s/a/b/ f",
                        "git branch new", "git push", "cat $(ls)",
                        "pip install x", "env rm x", "git config a.b c",
                        "python -c 'import os'", "awk '{print}' f"):
            self.assertFalse(shell.reads_only(command)[0], command)


class CutTests(unittest.TestCase):
    """The command as the message wrote it -- its case, its quoting -- not
    the reader's lowered words."""

    def test_as_written(self):
        text = "can you run git log --oneline -3 please"
        said = ["can", "you", "run", "git", "log", "--oneline", "-", "3",
                "please"]
        self.assertEqual(shell._cut(text, said, 3, 7),
                         "git log --oneline -3")
        text = "run `cat README.md`"
        said = ["run", "`cat", "readme.md`"]
        self.assertEqual(shell._cut(text, said, 1, 2), "cat README.md")


class GroundedTests(unittest.TestCase):
    """A path the request names, as it names it: not at the computer's
    root where the writer, taught `/path/to/dir`, put it."""

    def test_the_request_s_paths(self):
        self.assertEqual(shell._grounded("find /research/v701",
                                         "what files are in research/v701?"),
                         "find research/v701")
        self.assertEqual(shell._grounded("ls -la /tmp", "list /tmp"),
                         "ls -la /tmp")
        self.assertEqual(shell._grounded("cat /etc/hosts", "show hosts"),
                         "cat /etc/hosts")


class UnsetTests(unittest.TestCase):
    """A command reading a variable nobody set is not the one asked for
    (`du -sh $DATA_DIR` measured the folder it ran in)."""

    def test_unset(self):
        self.assertEqual(shell._unset("du -sh $DATA_DIR"), ["DATA_DIR"])
        self.assertEqual(shell._unset("cat ${FILES}"), ["FILES"])
        self.assertEqual(shell._unset("ls $HOME"), [])
        self.assertEqual(shell._unset("for f in *; do echo $f; done"), [])
        self.assertEqual(shell._unset("X=1; echo $X"), [])


@unittest.skipIf(shell.bash() is None, "no bash")
class RunTests(unittest.TestCase):
    def test_in_the_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "a.txt").write_text("x", encoding="utf-8")
            found = shell.run("ls", folder)
            self.assertEqual(found["code"], 0)
            self.assertIn("a.txt", found["out"])
            self.assertTrue(shell.parses("ls -la"))
            self.assertFalse(shell.parses("ls ("))

    def test_stopped_whole(self):
        """Out of time, the command stops and all it started with it: a
        `find` under a stopped bash kept its output open."""
        import time
        started = time.time()
        found = shell.run("sleep 30 | cat", ".", timeout=2)
        self.assertIsNone(found["code"])
        self.assertLess(time.time() - started, 15)


if __name__ == "__main__":
    unittest.main()
