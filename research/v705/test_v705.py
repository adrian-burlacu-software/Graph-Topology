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


if __name__ == "__main__":
    unittest.main()
