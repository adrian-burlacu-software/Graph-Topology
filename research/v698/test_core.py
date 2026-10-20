"""The core made reliable (2026-10-20): the project's index, pasted code
run, data asked of by what it holds, a change checked by what was asked --
each without a model, on small projects of their own."""
import tempfile
import unittest
from pathlib import Path

BOOKS = '''"""Books."""
import csv
from dataclasses import dataclass

#: how many books one may hold
MOST = 3


@dataclass
class Book:
    title: str
    pages: int = 0


def load(path):
    """Every book."""
    return [Book(row["title"], int(row["pages"])) for row in csv.DictReader(
        open(path))]


def unused(books):
    return books
'''
CLI = '''import sys
from shelf import books as B


def main(argv):
    if argv[0] == "list":
        print(B.load("b.csv"))
    elif argv[0] == "most":
        print(B.MOST)
    else:
        return 2
    return 0
'''
WEB = '''from shelf.books import Book, load


def _all():
    return load("b.csv")


ROUTES = {"/all": _all}


def serve(path):
    return ROUTES[path]()
'''
TEST = '''import unittest
from shelf import books


class T(unittest.TestCase):
    def test_load(self):
        self.assertTrue(books.Book("a").title)
'''


def held():
    from research.v698.project import Project
    project = Project("shelf")
    project.put({"shelf/__init__.py": "", "shelf/books.py": BOOKS,
                 "shelf/cli.py": CLI, "shelf/web.py": WEB,
                 "tests/test_books.py": TEST})
    return project


class IndexTests(unittest.TestCase):
    def test_uses_followed_through_modules_and_imports(self):
        index = held().index()
        by = {(use.file, use.by, use.how) for use in index.used_by("load")}
        self.assertIn(("shelf/cli.py", "main", "call"), by)
        self.assertIn(("shelf/web.py", "_all", "call"), by)
        self.assertIn(("shelf/web.py", "", "import"), by)
        self.assertEqual(index.used_by("unused"), [])
        self.assertIn(("shelf/cli.py", "main", "read"),
                      {(use.file, use.by, use.how)
                       for use in index.used_by("MOST")})
        tests = [use for use in index.used_by("Book")
                 if index.is_test(use.file)]
        self.assertEqual(tests[0].by, "T.test_load")

    def test_what_is_defined_and_answered_to(self):
        index = held().index()
        [most] = index.find("MOST")
        self.assertEqual((most.kind, most.value, most.doc),
                         ("constant", "3", "how many books one may hold"))
        self.assertEqual([one.name for one in index.fields("Book")],
                         ["Book.title", "Book.pages"])
        self.assertEqual(index.answers_to("main"), ["list", "most"])
        self.assertEqual(index.answers_to("serve"), ["/all"])

    def test_answers_from_the_index(self):
        from research.v698 import asking as A
        project = held()
        said = A.about("callers", A._function(project, "load"), "q",
                       project, None, 1, "k")["text"]
        self.assertIn("main (shelf/cli.py)", said)
        said = A.about("callers", A._function(project, "unused"), "q",
                       project, None, 1, "k")["text"]
        self.assertIn("is not called", said)
        said = A.about("value", A._symbol(project, "MOST"), "q", project,
                       None, 1, "k")["text"]
        self.assertIn("MOST is 3", said)
        said = A.about("fields", A._symbol(project, "Book"), "q", project,
                       None, 1, "k")["text"]
        self.assertIn("title, pages (by default 0)", said)
        said = A.about("cases", A.Subject("file", "shelf/cli.py",
                                          "shelf/cli.py"), "q", project,
                       None, 1, "k")["text"]
        self.assertIn("main answers to list, most", said)
        said = A.about("tests", A._symbol(project, "Book"), "q", project,
                       None, 1, "k")["text"]
        self.assertIn("T.test_load", said)


class ObservedTests(unittest.TestCase):
    def test_pasted_python_run(self):
        from research.v698 import observe
        seen = observe.observed("what does this return for 5?",
                                "def g(n):\n    return n * 2 + 1\n")
        self.assertEqual(seen["facts"][0], "g(5) gives 11.")
        seen = observe.observed("is there a bug?",
                                "def mean(xs):\n    return sum(xs) / "
                                "len(xs)\n")
        self.assertTrue(any("ZeroDivisionError" in one and "empty" in one
                            for one in seen["facts"]))
        seen = observe.observed("how many times?",
                                "for i in range(2, 10, 2):\n    pass\n")
        self.assertIn("The loop on line 1 runs 4 times.", seen["facts"])
        seen = observe.observed("what does it print?",
                                "for c in 'ab':\n    print(c * 2)\n")
        self.assertTrue(seen["facts"][0].startswith("It prints aa, bb"))
        seen = observe.observed("what's wrong?",
                                "def first(xs):\n    return xs[1]\n")
        self.assertTrue(any("the second element (index 1)" in one
                            for one in seen["facts"]))

    def test_pasted_needs_no_word_of_asking(self):
        from research.v698.asking import pasted
        self.assertIsNotNone(pasted("how many times does this loop run?\n"
                                    "```python\nfor i in range(3):\n    "
                                    "pass\n```"))
        self.assertIsNone(pasted("```python\nx = 1\n```"))


class DataAskedTests(unittest.TestCase):
    def library(self):
        from research.v698.project import Project
        project = Project("lib")
        project.put({
            "data/books.csv": "id,title,author,pages,year\n"
                              "1,Dune,Frank Herbert,412,1965\n"
                              "2,Kindred,Octavia E. Butler,264,1979\n"
                              "3,Wild Seed,Octavia E. Butler,248,1980\n",
            "data/members.json": '[{"id": 1, "name": "Ada Lovelace", '
                                 '"email": "ada@x.org", "joined": '
                                 '"2021-03-02"}, {"id": 2, "name": "Grace '
                                 'Hopper", "email": "grace@x.org", '
                                 '"joined": "2019-11-20"}]\n'})
        return project

    def asked(self, act, op, spans, words):
        from research.v701.querying import Asked
        return Asked(act, 1.0, op, spans, words)

    def test_what_the_data_holds_said_whole(self):
        from research.v701 import querying as Q
        project = self.library()
        # a value read in pieces: the one the data holds
        found = Q.carry(self.asked("count", "eq", {
            "TARGET": ["books"], "VALUE": ["octavia", "e . butler"]},
            "how many books by octavia e . butler are there ?".split()),
            project)
        self.assertEqual(found.value, 2)
        # no target: the records holding it
        found = Q.carry(self.asked("list", "eq", {
            "FIELD": ["email"], "VALUE": ["grace@x.org"]},
            "who has the email grace@x.org ?".split()), project)
        self.assertEqual(found.value, ["Grace Hopper"])
        # how many of a field of numbers: its amount
        found = Q.carry(self.asked("count", "none", {
            "TARGET": ["pages"], "FIELD": ["books"]},
            "how many pages are there across all the books ?".split()),
            project)
        self.assertEqual(found.value, 924)
        # the most held, and a date's least
        found = Q.carry(self.asked("max", "none", {
            "TARGET": ["author"], "FIELD": ["books"]},
            "which author has the most books ?".split()), project)
        self.assertEqual(found.value, "Octavia E. Butler")
        found = Q.carry(self.asked("min", "none", {
            "TARGET": ["member"], "FIELD": ["joined"]},
            "which member joined first ?".split()), project)
        self.assertEqual(found.value, "Grace Hopper")
        # a comparison's field named after its value
        found = Q.carry(self.asked("list", "gt", {
            "TARGET": ["books"], "VALUE": ["250"]},
            "which books have more than 250 pages ?".split()), project)
        self.assertEqual(found.value, ["Dune", "Kindred"])

    def test_a_follow_up_of_the_record(self):
        from research.v701 import querying as Q
        found = Q.of_record("who wrote it?", self.library(),
                            "data/books.csv",
                            {"id": 1, "title": "Dune",
                             "author": "Frank Herbert", "pages": 412})
        self.assertEqual(found.value, "Frank Herbert")


class RequestedTests(unittest.TestCase):
    def test_a_change_held_by_tests_of_what_was_asked(self):
        from unittest import mock
        from research.v698 import requested
        project = held()
        written = [
            # passes as it is: says nothing of the change
            "from shelf.books import MOST\nassert MOST == 3",
            "from shelf.books import MOST\nassert MOST == 5",
            "from shelf.books import MOST\nassert MOST > 4",
            "def (:",
        ]
        with mock.patch.object(requested, "tests", return_value=written):
            check = requested.checker("let one hold 5 books", project,
                                      "shelf/books.py")
        self.assertEqual(len(check["kept"]), 2)
        self.assertEqual(check["own"], ["tests/test_books.py"])
        right = BOOKS.replace("MOST = 3", "MOST = 5")
        self.assertEqual(check["verify"](right), "")
        self.assertIn("0 of 2", check["verify"](BOOKS.replace(
            "MOST = 3", "MOST = 4")))
        # a change that breaks the project's own tests is no change
        broken = right.replace("    title: str\n", "    title: str = ''\n"
                               "    def __post_init__(self): 1 / 0\n")
        self.assertIn("breaks the project's tests",
                      check["verify"](broken))

    def test_a_rename_everywhere_it_is_used(self):
        from unittest import mock
        from research.v698 import requested
        project = held()
        [(old, new, changed)] = [one for one in requested.renames(
            "rename load to read_all everywhere", project)
            if one[1] == "read_all"]
        self.assertEqual(sorted(changed), ["shelf/books.py", "shelf/cli.py",
                                           "shelf/web.py"])
        self.assertIn("B.read_all(", changed["shelf/cli.py"])
        self.assertIn("import Book, read_all", changed["shelf/web.py"])
        written = ["from shelf.books import read_all\nassert read_all"]
        with mock.patch.object(requested, "tests", return_value=written):
            check = requested.checker("rename load to read_all", project,
                                      "shelf/books.py")
        self.assertEqual(check["verify files"](changed), "")
        # the docstring and the test's words untouched: code alone
        self.assertIn('"""Every book."""', changed["shelf/books.py"])


if __name__ == "__main__":
    unittest.main()
