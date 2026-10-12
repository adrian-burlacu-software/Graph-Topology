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


class CarryingTests(unittest.TestCase):
    """A goal's plan held, and each yes its next step carried out: the
    look diagnoses each case from its trace, a change is asked of itself
    from what the look found, the cases kept and measured."""

    CASE = {"said": "what is my dog called?", "part": "kind", "n": 7,
            "turns": ["my dog is called Rex", "what is my dog called?"],
            "fired": [{"operator": "noting", "outcome": "continue"},
                      {"operator": "ask_name", "outcome": "declined"},
                      {"operator": "generic", "outcome": "answered"}]}
    PLACES = {"ask_name": [{"file": "s.py", "rule": "", "proposes": None,
                            "apply": {"function": "_ask_name",
                                      "line": 3}}]}

    def state(self):
        steps = planned("change", "check", "measure", "teach")
        steps = {"steps": [{**one, "said": one["kind"].title()}
                           for one in steps["steps"]]}
        return {"goal": "remember better", "plan": steps, "weak": [
            {"what": "remembering", "cases": [self.CASE]}]}

    def test_carried_step_by_step(self):
        from unittest import mock
        from research.v704 import doing as D
        state, asked, measured = self.state(), [], []

        class Held:
            files = {"s.py": ""}
        with mock.patch.object(D.selfmodel, "places",
                               lambda: self.PLACES), \
                mock.patch.object(D, "readings", lambda text: [
                    ("conversation", "ask_name", 0.9),
                    ("code talk", "none", 1.0)]), \
                mock.patch.object(D, "measure", lambda numbers, label:
                                  measured.append(numbers)):
            look = "\n".join(D.carry(state))
            self.assertIn('"what is my dog called?" (after "my dog is '
                          'called Rex"): read by conversation as ask_name '
                          '(90%); tried: ask_name (_ask_name, s.py) '
                          'declined -> generic answered', look)
            self.assertIn("Most telling: ask_name", look)
            # the steps left said again of what the look found
            self.assertIn('Next: 1. Change _ask_name in s.py, where "what '
                          'is my dog called?" goes wrong', look)
            change = "\n".join(D.carry(state, change=lambda text, *where: (
                asked.append((text, *where)) or "unsure"), held=Held()))
            self.assertEqual(asked, [('change s._ask_name so that it '
                                      'answers "what is my dog called?" '
                                      'after "my dog is called Rex"',
                                      "_ask_name", "s.py")])
            self.assertIn("unsure", change)
            self.assertIn("1 cases are this goal's checks",
                          "\n".join(D.carry(state)))
            self.assertIn("running its 1 cases", "\n".join(D.carry(state)))
            self.assertEqual(measured, [[7]])
            last = "\n".join(D.carry(state))
            self.assertIn("not done", last)
            self.assertIn("last step", last)
            self.assertTrue(state["done"])

    def test_a_reading_lost(self):
        from unittest import mock
        from research.v704 import doing as D
        case = {**self.CASE, "acts": {"read": [["ask_name", 0.66]],
                                      "taken": "generic"},
                "fired": [{"operator": "generic", "outcome": "answered"}]}
        with mock.patch.object(D.selfmodel, "places",
                               lambda: self.PLACES), \
                mock.patch.object(D, "readings", lambda text: []):
            found = [D.diagnose(case)]
        self.assertIn("taken as generic, so ask_name (_ask_name) was "
                      "never tried", D.said_diagnosis(found[0]))
        shared = D.finding(found)
        self.assertEqual((shared["function"], shared["how"]),
                         ("build", "never tried"))
        self.assertEqual(shared["file"], "research/v689/reader.py")
        self.assertEqual(D.request_for(shared),
                         'change reader.build so that "what is my dog '
                         'called?" is read as ask_name, as the reader '
                         'reads it')


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


WEAK = {"what": "sending each message to the right part", "count": 3,
        "examples": [{"said": "what is 7 squared?", "part": "shell"}],
        "cases": [{"said": "list the books", "by": {"operator": "data"}},
                  {"said": "what is 7 squared?",
                   "by": {"operator": "shell"},
                   "should": {"operator": "mathematics"}}],
        "where": [{"operator": "shell", "function": "asks_shell",
                   "file": "page.py", "line": 1, "role": "took",
                   "cases": 2},
                  {"operator": "mathematics", "function": "proposes",
                   "file": "maths.py", "line": 2, "role": "should take",
                   "cases": 2}]}


def planned(*kinds):
    return {"like": [{"version": "v1", "goal": "g", "likeness": 0.5}],
            "steps": [{"kind": kind, "said": "", "as in": {
                "version": "v1", "step": f"v1: {kind} it"}}
                for kind in kinds]}


class GroundingTests(unittest.TestCase):
    """Each step said of what it is done to: the weakness's place in the
    architecture's own code and its case, or the project's things."""

    def test_of_a_weakness(self):
        from research.v704 import history as H
        found = H.ground(planned("change", "teach", "check", "tidy"), WEAK)
        said = [one["said"] for one in found["steps"]]
        self.assertEqual(said[0], "Change asks_shell in page.py, which "
                         "took 2 messages that were not its own, and fix "
                         "what fails")
        # the case said is one the place took, and where it should go
        self.assertIn('"what is 7 squared?" goes to mathematics', said[1])
        self.assertIn("and the 1 other cases", said[2])
        self.assertEqual(said[3], H.SAYING["tidy"])
        self.assertEqual(found["steps"][0]["slots"]["VERSION"], "v1")

    def test_of_a_project(self):
        from research.v704 import history as H
        found = H.ground(planned("change", "teach"), None, ["a.f", "b.g"])
        self.assertEqual(found["steps"][0]["said"],
                         "Change a.f, b.g, and fix what fails")
        # nothing known to teach of: said of nothing in particular
        self.assertEqual(found["steps"][1]["said"], H.SAYING["teach"])


class PlacesTests(unittest.TestCase):
    """Where each operator lives in the architecture's own code."""

    def test_found_by_construction(self):
        from research.v704 import selfmodel as S
        found = S.places()
        shell = S.place_of("shell", "", found)
        self.assertEqual(shell["file"], "research/v698/page.py")
        self.assertEqual(shell["proposes"]["function"], "asks_shell")
        self.assertEqual(shell["apply"]["function"], "do_shell")


class SayingTests(unittest.TestCase):
    """A goal's reply in pieces: each a frame and its slots, labelled by
    construction, and a saying read back to its piece."""

    def test_labelled(self):
        from research.v704 import teach_saying as T
        words, parts = T.labelled('so "{CASE}" goes to {OPERATOR}',
                                  {"CASE": "what is 7?",
                                   "OPERATOR": "maths"})
        self.assertEqual(words, ["so", '"', "what", "is", "7", "?", '"',
                                 "goes", "to", "maths"])
        self.assertEqual(parts, ["O", "O", "B-CASE", "I-CASE", "I-CASE",
                                 "I-CASE", "O", "O", "O", "B-OPERATOR"])

    def test_pieces(self):
        from research.v704 import history as H
        from research.v704 import teach_saying as T
        steps = H.ground(planned("change", "tidy"), WEAK)
        pieces = T.pieces_of([WEAK], {"built": "2026-10-12"}, steps)
        self.assertEqual([one["kind"] for one in pieces],
                         ["goal", "worst", "weak-routing", "like", "change",
                          "tidy", "ask"])
        said = T.joined(pieces, [T.said_as_held(one) for one in pieces])
        self.assertIn("1) sending each message", said)
        self.assertIn("\n1. Change asks_shell", said)
        self.assertIn("\n2. Clear what is left over", said)

    def test_a_frame_kept(self):
        from research.v704 import teach_saying as T
        frame = 'Teach it, so that "{CASE}" goes to {OPERATOR}'
        self.assertTrue(T.kept_frame(
            frame, 'Teach it so "{CASE}" moves to {OPERATOR}', set()))
        # a slot lost, a slot twice, a fact added
        self.assertIsNone(T.kept_frame(frame, "Teach it to {OPERATOR}",
                                       set()))
        self.assertIsNone(T.kept_frame(
            frame, 'Teach it, so "{CASE}" goes to {OPERATOR} and '
                   '{OPERATOR}', set()))
        self.assertIsNone(T.kept_frame(
            frame, 'Teach it, since users complained, so "{CASE}" goes to '
                   '{OPERATOR}', set()))

    def test_read_back(self):
        from unittest import mock
        from research.v704 import teach_saying as T
        one = {"kind": "teach",
               "frame": 'Teach it so "{CASE}" goes to {OPERATOR}',
               "slots": {"CASE": "what is 7?", "OPERATOR": "maths"}}
        text = 'Teach it so "what is 7?" goes to maths'
        words, parts = T.labelled('Teach it so "{CASE}" goes to '
                                  '{OPERATOR}', one["slots"])

        def read(said, kind="teach", roles=parts, **_):
            return {"plan_kind": [(kind, 0.9)], "plan_part": roles}
        framing = {"teach", "it", "so", "goes", "to", '"'}
        with mock.patch.object(T, "_framing", lambda kind: framing), \
                mock.patch("research.encoder.read", read):
            self.assertIsNone(T.read_back(one, text))
        with mock.patch.object(T, "_framing", lambda kind: framing), \
                mock.patch("research.encoder.read",
                           lambda *a, **k: read(a, kind="check")):
            self.assertIn("read as check", T.read_back(one, text))
        with mock.patch.object(T, "_framing", lambda kind: framing - {"goes"}), \
                mock.patch("research.encoder.read", read):
            self.assertIn("says goes", T.read_back(one, text))


if __name__ == "__main__":
    unittest.main()
