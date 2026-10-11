"""v704: the conversation test's scoring, offline (no server)."""
from __future__ import annotations

import unittest

from research.v704 import converse as C
from research.v704.suite import SUITE


def turn(text, stance=None, status=None, code=None):
    reply = {"text": text}
    if stance:
        reply["message"] = {"stance": stance}
    answer = {"code": {"answer": {"status": status, "code": code,
                                  "entry": "f"}}}
    return {"reply": reply, "answer": answer}


class ScoringTests(unittest.TestCase):
    def test_what_is_said(self):
        self.assertTrue(C.says("Dune")(turn("The longest is dune."), None))
        self.assertTrue(C.number(7)(turn("There are 7 books."), None))
        self.assertFalse(C.number(7)(turn("There are 17 books."), None))
        self.assertTrue(C.number(4.75)(turn("It's 4.75."), None))
        self.assertTrue(C.stance("no")(turn("No.", stance="no"), None))
        self.assertFalse(C.stance("yes")(turn("Yes, ...", stance="no"),
                                         None))

    def test_unanswered(self):
        self.assertTrue(C.unanswered(turn("I am not sure of the command")))
        self.assertTrue(C.unanswered(turn("x", status="unsure")))
        self.assertFalse(C.unanswered(turn("There are 7 books.")))

    def test_code_run(self):
        made = turn("Here is f", code="def f(n):\n    return n * 3\n")
        self.assertTrue(C.code_does([([2], 6)])(made, None))
        self.assertFalse(C.code_does([([2], 5)])(made, None))

    def test_the_suite(self):
        self.assertGreaterEqual(len(SUITE), 190)
        for one in SUITE:
            self.assertTrue(one.turns and all(one.turns))
            self.assertTrue(callable(one.check))
        self.assertTrue(C._same_as_fixture(C.FIXTURE))


class SelfModelTests(unittest.TestCase):
    """What it knows of itself, from a run: how each area does, the part
    that answers it, what went to another part."""

    def test_routing_found(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest import mock
        from research.v704 import selfmodel as M

        def row(verdict, part, said="what is 2 plus 2?"):
            return {"area": "maths", "verdict": verdict, "part": part,
                    "turns": [said], "said": ["4"]}
        runs = Path(tempfile.mkdtemp())
        rows = [row("right", "mathematics"), row("right", "mathematics"),
                row("unsure", "shell", "what is 7 squared?"),
                row("wrong", "mathematics")]
        (runs / "one.jsonl").write_text(
            "".join(json.dumps(one) + "\n" for one in rows),
            encoding="utf-8")
        with mock.patch.object(M, "RUNS", runs), \
                mock.patch.object(M, "SELF", runs / "self.json"):
            found = M.build("one")
        maths = found["areas"]["maths"]
        self.assertEqual((maths["right"], maths["usual part"]),
                         (2, "mathematics"))
        self.assertEqual([one["said"] for one in found["routing"]],
                         ["what is 7 squared?"])
        self.assertEqual(len(maths["failed"]), 1)


class FirstStepTests(unittest.TestCase):
    """A yes to a goal's plan: a look at what fails, never a change."""

    def test_what_fails_listed(self):
        from research.v698 import page
        cases = [{"said": "what is 7 squared?", "part": "shell",
                  "answer": "I would run `7`"}]
        page.GOALS["k"] = {"goal": "communicate better", "plan": {},
                           "weak": [{"what": "sending it to the right part",
                                     "cases": cases, "examples": cases}]}
        found = page._first_step("k", None)
        self.assertIn('"what is 7 squared?" went to shell', found["text"])
        self.assertNotIn("k", page.GOALS)


class HistoryTests(unittest.TestCase):
    """A goal's plan, chosen from goals reached like it."""

    def test_most_taken_in_order(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest import mock
        from research.v704 import history as H

        def goal(name, kinds):
            return {"version": name, "goal": name, "steps": [
                {"said": f"{name} {kind}", "kind": kind} for kind in kinds]}
        rows = [goal("speak plainly to people", ["build", "teach",
                                                 "measure"]),
                goal("say replies people understand", ["teach", "build",
                                                       "measure", "tidy"]),
                goal("read the words people type", ["build", "teach",
                                                    "document"])]
        path = Path(tempfile.mkdtemp()) / "history.jsonl"
        path.write_text("".join(json.dumps(one) + "\n" for one in rows),
                        encoding="utf-8")
        with mock.patch.object(H, "HISTORY", path), \
                mock.patch("research.v701.likeness.nearest",
                           lambda text, phrases: [
                               (0.9 - at / 10, one)
                               for at, one in enumerate(phrases)]):
            found = H.plan("communicate better")
        self.assertEqual([one["kind"] for one in found["steps"]],
                         ["build", "teach", "measure"])
        self.assertIn("1. Build", H.said(found))


if __name__ == "__main__":
    unittest.main()
