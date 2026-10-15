"""v705: debugging -- a run of a function kept step by step, in Python and
TypeScript alike.

    python -m unittest research.v705.test_v705
"""
from __future__ import annotations

import shutil
import unittest

PY = '''def oldest(books):
    best = None
    for book in books:
        if best is None or book["year"] < best["year"]:
            best = book
    return best["title"]
'''
TS = '''function oldest(books: {title: string, year: number}[]): string {
  let best: {title: string, year: number} | null = null;
  for (const book of books) {
    if (best === null || book.year < best.year) {
      best = book;
    }
  }
  return best!.title;
}
'''
BOOKS = [{"title": "A", "year": 1990}, {"title": "B", "year": 1961}]
BOOKS_TS = '''export interface Book { title: string; year: number }
export function byYear(a: Book, b: Book): number {
  return a.year - b.year;
}
'''
SHELF_TS = '''import { Book, byYear } from "./books";
export function oldest(books: Book[]): string {
  const sorted = [...books].sort(byYear);
  return sorted[sorted.length - 1].title;
}
'''


class TraceTests(unittest.TestCase):
    """Each language's checker keeps where a run goes: the lines, the
    values that changed at each, what is returned or raised."""

    def traced(self, language, source):
        from research.v696.checker import checker
        held = checker(language)
        try:
            return (held.trace(source, "oldest", [BOOKS]),
                    held.trace(source, "oldest", [[]]))
        finally:
            held.close()

    def check(self, found, empty, returned_at, raised_at):
        self.assertEqual(found["output"], {"value": "B"})
        self.assertEqual(found["steps"][-1],
                         {"line": returned_at, "in": "oldest",
                          "returns": '"B"'})
        changed = [one["values"].get("best") for one in found["steps"]
                   if "values" in one and "best" in one["values"]]
        self.assertEqual(changed[0], "null")
        self.assertIn("1961", changed[-1])
        self.assertIn("error", empty["output"])
        self.assertEqual(empty["steps"][-1]["line"], raised_at)
        self.assertIn("raises", empty["steps"][-1])
        # the loop never ran: no step on its body
        self.assertNotIn(5, [one["line"] for one in empty["steps"]])

    def test_python(self):
        self.check(*self.traced("python", PY), returned_at=6, raised_at=6)

    @unittest.skipUnless(shutil.which("node"), "needs node")
    def test_typescript(self):
        self.check(*self.traced("typescript", TS), returned_at=8,
                   raised_at=8)


class ProjectTraceTests(unittest.TestCase):
    """A call as written, run in a project's module with its imports, each
    step of the project's own code kept, and whether it gives what it
    should."""

    def test_python(self):
        from pathlib import Path
        from research.v696.checker import checker
        root = Path(__file__).resolve().parents[1] / "v704" / "fixtures" / \
            "library"
        files = {"/ws/" + one.relative_to(root).as_posix():
                 one.read_text(encoding="utf-8")
                 for one in root.rglob("*.py")}
        held = checker("python")
        try:
            found = held.trace_project(
                files, "/ws/library/loans.py",
                "due_date(date(2026, 1, 1))", "date(2026, 1, 15)")
        finally:
            held.close()
        self.assertTrue(found["output"]["right"])
        self.assertEqual(found["steps"][-1]["returns"],
                         "datetime.date(2026, 1, 15)")

    @unittest.skipUnless(shutil.which("node"), "needs node")
    def test_typescript(self):
        from research.v696.checker import checker
        files = {"/ws/src/books.ts": BOOKS_TS, "/ws/src/shelf.ts": SHELF_TS}
        held = checker("typescript")
        try:
            found = held.trace_project(
                files, "/ws/src/shelf.ts",
                'oldest([{title: "A", year: 1990}, {title: "B", '
                'year: 1961}])', '"B"')
        finally:
            held.close()
        self.assertEqual(found["output"], {"value": '"A"', "right": False})
        self.assertIn("1961", found["steps"][1]["values"]["sorted"])


def _asked(question):
    return {"outcome": "unknown", "question": question}


def _summary(run):
    return run["outcome"], run["outcome"].upper()


def _parsed(text):
    return text.split()


def _built(words):
    return None


class _Turn:
    answer = None
    run = None


class _Handler:
    def ask(self, question):
        return _asked(question)

    def generic(self, said, turn):
        turn.run = self.ask(said)
        outcome, headline = _summary(turn.run)
        turn.answer = {"outcome": outcome, "text": headline}

    def build(self, text):
        words = _parsed(text)
        return _built(words)

    def lost(self, text):
        found = _built(text)
        if found is None:
            return None
        return found


class _Pinned(_Handler):
    def ask(self, question):
        return super().ask(question)


class _Word(str):
    pass


def _word_at(words: list[str], at: int, upper: bool = False) -> _Word | None:
    if at < len(words):
        return _Word(words[at].upper() if upper else words[at])
    return None


def _first_word(words: list[str]) -> _Word | None:
    if len(words) == 1:
        return _Word(words[0])
    return None


def _named(words: list[str]):
    found = _first_word(words) or _word_at(words, 0)
    return found


def _shouted(words: list[str]):
    found = _first_word(words)
    return found.upper() if found else None


class SubgoalTests(unittest.TestCase):
    """A goal's cases taken apart into the places they fail -- each a
    subgoal carried in turn, and looked at again when a round met any."""

    def found(self, said, function, how="answered"):
        return {"said": said, "lost": None, "tried": [
            {"operator": "op", "outcome": how, "function": function,
             "file": "research/x.py"}]}

    def test_places(self):
        from research.v704 import doing
        subs = doing.findings([self.found("a", "f"), self.found("b", "g"),
                               self.found("c", "g"),
                               self.found("d", "h", "declined")])
        self.assertEqual([(one["function"], one["cases"]) for one in subs],
                         [("h", 1), ("g", 2), ("f", 1)])

    def test_rounds(self):
        from research.v704 import doing
        changed = []

        def change(asked, function, file, case):
            changed.append(function)
            return f"changed {function}", function == "g"
        state = {"goal": "x", "weak": [], "subgoals": [
            {"function": "g", "file": "research/x.py", "how": "answered",
             "cases": 1, "case": {"said": "b", "before": []},
             "operator": "op", "all": ["b"]}], "sub": 0, "at": 0,
            "again": lambda: []}
        lines = doing.carry(state, change=change, held=_Held())
        self.assertEqual(changed, ["g"])
        self.assertTrue(state["done"])
        self.assertIn("1 met", lines[-1])
        self.assertIn("nothing fails now", lines[-1])


class _Held:
    files = {"research/x.py": ""}


class CompositionTests(unittest.TestCase):
    def test_precedent(self):
        # _named gets past _first_word's nothing with `or _word_at(...)`:
        # _shouted, which gives up on it, is offered the same
        import inspect
        from pathlib import Path
        from research.v705 import inside
        source = Path(__file__).read_text(encoding="utf-8")
        lines, first = inspect.getsourcelines(_shouted)
        made = inside.precedents(_shouted, {
            "line": first + 2, "returns": "None",
            "deeper": {"function": "_first_word", "returns": "None"}},
            source)
        self.assertEqual([line for _, line in made],
                         ["found = _first_word(words) or "
                          "_word_at(words, 0)"])

    """What a function that gives up should give, built from what in reach
    gives it, filled from what the function holds."""

    def test_filled(self):
        import inspect
        from pathlib import Path
        from research.v705 import inside
        source = Path(__file__).read_text(encoding="utf-8")
        lines, first = inspect.getsourcelines(_first_word)
        gave_up = first + len(lines) - 1
        made = inside.compositions(_first_word, {"line": gave_up,
                                                 "returns": "None"}, source)
        calls = [call for _, call in made]
        self.assertEqual(calls[:2], ["_word_at(words, 0)",
                                     "_word_at(words, 0, upper=True)"])
        namespace: dict = {}
        exec(compile(made[0][0], "made", "exec"), namespace)   # noqa: S102
        self.assertEqual(namespace["_first_word"](["a", "b"]), "a")


class SliceTests(unittest.TestCase):
    """Where a function's give-up came from, followed back through what it
    computed: a handler's answer to the call it was made of, a nothing to
    the call that gave it."""

    def callee(self, name, *args):
        from research.v705 import inside
        function = getattr(_Handler, name)
        _, calls = inside.watch(function,
                                lambda: function(_Handler(), *args))
        return inside._callee(function, calls[-1])

    def test_handler(self):
        # the answer is summary_of what self.ask gave: the first of them
        self.assertIs(self.callee("generic", "x?", _Turn()), _Handler.ask)

    def test_returned(self):
        # parse gave words; build gave back the nothing
        self.assertIs(self.callee("build", "a b"), _built)

    def test_bound(self):
        self.assertIs(self.callee("lost", "a b"), _built)

    def test_super(self):
        from research.v705 import inside
        _, calls = inside.watch(_Pinned.ask,
                                lambda: _Pinned().ask("x?"))
        self.assertIs(inside._callee(_Pinned.ask, calls[-1]), _Handler.ask)

    def test_changed_method(self):
        # a changed method made in its class: its super() still the class's
        from pathlib import Path
        from research.v705 import inside
        text = Path(__file__).read_text(encoding="utf-8").replace(
            "        return super().ask(question)",
            "        return {**super().ask(question), \"pinned\": True}", 1)
        made = inside.changed_function(_Pinned.ask, text)
        self.assertEqual(made(_Pinned(), "x?"),
                         {"outcome": "unknown", "question": "x?",
                          "pinned": True})
        self.assertIs(_Pinned.__dict__["ask"].__name__, "ask")


class DebugTalkTests(unittest.TestCase):
    """A debugging request answered: the call run in its project and said,
    the module's values it reads with it; right where it gives what it
    should."""

    def held(self):
        from pathlib import Path
        from research.v698.project import Project
        root = Path(__file__).resolve().parents[1] / "v704" / "fixtures" / \
            "library"
        held = Project("v705-debug-talk")
        held.put({one.relative_to(root).as_posix():
                  one.read_text(encoding="utf-8")
                  for one in root.rglob("*.py")})
        return held

    def test_run_said(self):
        from research.v705 import talk
        found = talk.debug(self.held(), "due_date(date(2026, 1, 1))", None,
                           None)
        self.assertEqual(found["status"], "traced")
        self.assertIn("LOAN_DAYS = 14", found["said"])
        self.assertIn("datetime.date(2026, 1, 15)", found["said"])

    def test_right_already(self):
        from research.v705 import talk
        found = talk.debug(self.held(), "due_date(date(2026, 1, 1))", None,
                           "date(2026, 1, 15)")
        self.assertEqual(found["status"], "right")

    def test_nowhere(self):
        from research.v705 import talk
        found = talk.debug(self.held(), "nothing_here(1)", None, "2")
        self.assertEqual(found["status"], "unplaced")


class MeasureTests(unittest.TestCase):
    """A goal about its replies (`measures`): the changes made by
    construction, and what a reply is made from."""

    def test_smaller(self):
        from research.v704 import history
        from research.v705 import measures
        text = open(history.__file__, encoding="utf-8").read()
        made = {what: changed for changed, what in
                measures.smaller(history.ground, text)}
        changed = made["`frame + PRECEDENT` without `PRECEDENT`"]
        compile(changed, "history.py", "exec")
        self.assertIn("step.update(frame=frame, slots=slots,", changed)
        self.assertEqual(len(changed.splitlines()), len(text.splitlines()))

    def test_dropped(self):
        from research.v704 import history
        from research.v705 import measures
        text = open(history.__file__, encoding="utf-8").read()
        for changed, _ in measures.dropped(history.ground, text):
            compile(changed, "history.py", "exec")

    def test_feeding(self):
        from research.v698 import page
        from research.v705 import measures
        names = {one.__name__ for one in measures.feeding(page._goal)}
        self.assertTrue({"pieces_of", "joined", "plan", "say"} <= names)

    def test_measures(self):
        from research.v705 import measures
        self.assertEqual(measures.length("one two three"), 3)
        self.assertEqual(measures.questions("why? how?"), 2)
        self.assertGreater(measures.repetition("a b c a b c"), 0)


class DeduceTests(unittest.TestCase):
    """Deduction (`deduce`): a wanted value broken down by its shape, each
    part found, the whole put back -- most general first."""

    REQUESTS = [({"text": 'count the vowels\ndef vowels(s: str) -> int:\n'
                          'vowels("hello") == 2'},
                 "def vowels(s: str) -> int:"),
                ({"text": "add two\ndef add(a: int, b: int) -> int:\n"
                          "add(1, 2) == 3"},
                 "def add(a: int, b: int) -> int:")]

    def solved(self, examples, library=None):
        from research.v705.deduce import solve
        found = solve(examples, library)
        self.assertTrue(found, examples)
        return found[0].expr

    def test_a_piece_by_what_it_begins_with(self):
        said = self.solved(self.REQUESTS[:1])
        self.assertIn("^(def", said)
        self.assertNotIn("vowe", said)

    def test_pieces_put_one_after_another(self):
        self.assertEqual(self.solved(
            [({"name": "vowels", "ret": "int"}, "def vowels() -> int:"),
             ({"name": "add", "ret": "str"}, "def add() -> str:")]),
            "'def ' + name + '() -> ' + ret + ':'")

    def test_a_list_made_of_another(self):
        self.assertEqual(self.solved(
            [({"sig": "s: str"}, [["s", "str"]]),
             ({"sig": "a: int, b: int"}, [["a", "int"], ["b", "int"]])]),
            "[one.split(': ') for one in sig.split(', ')]")

    def test_a_list_kept_of_another(self):
        self.assertEqual(self.solved(
            [({"xs": [1, 2, 3, 4]}, [2, 4]), ({"xs": [5, 6]}, [6])]),
            "[one for one in xs if one % 2 == 0]")

    def test_joined(self):
        self.assertEqual(self.solved(
            [({"xs": ["a", "b"]}, "a, b"), ({"xs": ["c", "d", "e"]},
                                            "c, d, e")]),
            "', '.join(xs)")

    def test_its_own_table(self):
        from research.v696 import pytypes
        from research.v705.deduce import Library
        library = Library({"BACK": (pytypes.BACK, "from research.v696."
                                                  "pytypes import BACK")})
        self.assertEqual(self.solved([({"kind": "string"}, "str"),
                                      ({"kind": "number"}, "int")], library),
                         "BACK.get(kind, kind)")

    def test_a_condition_by_a_word(self):
        from research.v705.deduce import condition
        self.assertEqual(condition([{"text": "def f(x): pass"}],
                                   [{"text": "function f(x) {}"}]).expr,
                         "'def' in text")


class MeansTests(unittest.TestCase):
    """Designing by means-ends (`means`): goals from how a test falls,
    moves put in, and what a change may not write."""

    def test_crash_goals(self):
        from research.v705 import means
        self.assertEqual(means.crashed(
            "x\nTypeError: run() got an unexpected keyword argument "
            "'timeout'"), {"needs": "parameter", "function": "run",
                           "name": "timeout"})
        self.assertEqual(means.crashed("x\nKeyError: 'language'"),
                         {"needs": "key", "name": "language"})
        self.assertEqual(means.crashed(
            "TypeError: _framing() missing 1 required positional "
            "argument: 'kind'")["needs"], "argument")

    def test_inserted_before_each_return(self):
        from research.v705 import means
        made = means._inserted(
            "def f(x):\n    if x:\n        return 1\n    return 2\n",
            ["log(x)"], "before a return")
        self.assertEqual(made.count("log(x)"), 2)

    def test_scope_is_what_is_bound_before(self):
        from research.v705 import means
        source = "def f(text):\n    out = {}\n    args = 1\n    return out\n"
        self.assertNotIn("args", means._scope(source, 2))
        self.assertIn("args", means._scope(source))

    def test_a_change_may_not_write_what_the_check_wants(self):
        from research.v705 import commits

        def spec():
            return "x"
        text = "def spec():\n    return 'for x in xs if (x % 2 == 0):'\n"
        case = {"tests": [], "wanted values": ["' for x in xs if '"]}
        self.assertTrue(commits.copies_test(case, spec, text))


class VerifyTests(unittest.TestCase):
    """What a change is judged by, beyond its tests passing."""

    def test_a_read_some_way_leaves_unset(self):
        from research.v705.commits import unbound
        self.assertEqual(unbound(
            "def run(self, key):\n"
            "    with self._lock:\n"
            "        if key not in self._cache:\n"
            "            answer = self.loop.run(key)\n"
            "            return answer\n"
            "        else:\n"
            "            return answer\n"), {"answer"})
        self.assertEqual(unbound(
            "def f(a):\n"
            "    if not a:\n"
            "        return 0\n"
            "    else:\n"
            "        y = 1\n"
            "    try:\n"
            "        z = g()\n"
            "    except E:\n"
            "        raise\n"
            "    return [x for x in y if x] + z\n"), set())

    def test_a_thread_that_dies_fails_its_test(self):
        import sys
        import threading
        from research.v705 import commits

        class Dies(unittest.TestCase):
            def test_it(self):
                one = threading.Thread(target=lambda: 1 / 0)
                one.start()
                one.join()
        here = sys.modules[__name__]
        here._Dies = Dies
        try:
            self.assertFalse(commits.passes(
                {"file": "research/v705/test_v705.py",
                 "test": "_Dies.test_it"}))
            self.assertIn("ZeroDivisionError", commits.LAST["failure"])
        finally:
            del here._Dies


if __name__ == "__main__":
    unittest.main()
