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


if __name__ == "__main__":
    unittest.main()
