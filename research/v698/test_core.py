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


class DescribedTests(unittest.TestCase):
    def test_a_subject_said_by_what_it_does(self):
        from research.v698 import asking as A
        from research.v698.reading import Reading
        project = held()
        read = Reading("ask", 1.0, "value", 1.0, "described",
                       "how many books may one hold ?".split(),
                       ["O"] * 7)
        found = A.resolve(read, project, None, 1, "k")
        self.assertEqual((found.kind, found.name), ("symbol", "MOST"))


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

    def test_the_call_asked_as_it_is_written(self):
        from research.v698 import observe
        code = ("def f(a, b, c):\n    out = dict()\n    for one in (a, b, c):"
                "\n        out.update(dict.fromkeys(one))\n    return out\n")
        seen = observe.observed("what does f((1, ), (1, ), (1, 2)) return?",
                                code)
        self.assertEqual(seen["facts"][0],
                         "f((1, ), (1, ), (1, 2)) gives {1: None, 2: None}.")
        seen = observe.observed("what does f(range(3), 'u') return?",
                                "def f(xs, s):\n    return [s] * len(xs)\n")
        self.assertIn("['u', 'u', 'u']", seen["facts"][0])

    def test_an_input_found_is_one_run(self):
        from research.v698 import observe

        class Proposes:
            def write(self, prompts, longest, samples, temperature):
                return [["f(2)", "```python\nf('zebra42')\n```", "f(nope"]]
        # searched alone: 11 // 2 tried, no model asked
        code = "def f(n):\n    return n * 2 + 1\n"
        self.assertEqual(observe.found_input(code, "f", 11, None)["call"],
                         "f(5)")
        self.assertNotIn("call", observe.found_input(code, "f", 4,
                                                     Proposes()))
        # ',saw' from a function turning text round: the wanted turned round
        code = "def f(text):\n    return text[::-1]\n"
        self.assertEqual(observe.found_input(code, "f", ",saw", None)["call"],
                         "f('was,')")
        # what the search cannot reach, the model's calls begin from
        code = ("def f(s):\n    return 7 if s == ''.join(['zeb', 'ra']) + "
                "str(6 * 7) else 0\n")
        self.assertEqual(observe.found_input(code, "f", 7, Proposes())[
            "call"], "f('zebra42')")
        self.assertEqual(observe.literals("what input to f gives [(4, 1), "
                                          "(2, 3)]?"), [[(4, 1), (2, 3)]])
        self.assertEqual(observe.literals("what input to f gives False?"),
                         [False])

    def test_code_fenced_is_no_pasted_table(self):
        from research.v701.pasting import found
        self.assertIsNone(found("```python\ndef f(text, pref):\n    a, b = "
                                "1, 2\n    c, d = 3, 4\n    return a, b\n```"
                                "\nwhat does f('a', 'b') return?"))

    def test_pasted_needs_no_word_of_asking(self):
        from research.v698.asking import pasted
        self.assertIsNotNone(pasted("how many times does this loop run?\n"
                                    "```python\nfor i in range(3):\n    "
                                    "pass\n```"))
        self.assertIsNone(pasted("```python\nx = 1\n```"))


class CheckerWornTests(unittest.TestCase):
    def test_a_worn_python_checker_is_started_again(self):
        from unittest import mock
        from research.v696.pychecker import PyChecker
        replies = [{"ok": False, "error": "MemoryError"},
                   {"ok": True, "errors": []}]
        with mock.patch.object(PyChecker, "__init__", lambda self: None), \
                mock.patch.object(PyChecker, "_ask_once",
                                  lambda self, request, wait:
                                  replies.pop(0)), \
                mock.patch.object(PyChecker, "restart") as restart:
            reply = PyChecker()._ask({"op": "diagnose"}, wait=1)
        self.assertEqual(reply, {"ok": True, "errors": []})
        restart.assert_called_once()


class PlacedTests(unittest.TestCase):
    def test_a_file_named_alone_found_in_its_folder(self):
        from research.v702.shell import _placed
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "data").mkdir()
            (Path(root) / "data" / "books.csv").write_text("id\n")
            (Path(root) / "a.txt").write_text("")
            self.assertEqual(_placed("head -1 books.csv", root),
                             "head -1 data/books.csv")
            self.assertEqual(_placed("wc -l a.txt | cat", root),
                             "wc -l a.txt | cat")
            self.assertEqual(_placed("cat none.csv", root), "cat none.csv")


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

    def test_what_the_reader_read_kept_below_its_floor(self):
        from unittest import mock
        from research.v698.project import Project
        from research.v701 import querying as Q
        root = Path(__file__).resolve().parents[1] / "v704" / "fixtures" / \
            "library"
        project = Project("library", str(root))
        project.put({one.relative_to(root).as_posix(): one.read_text(
            encoding="utf-8") for one in root.rglob("*") if one.is_file()
            and "__pycache__" not in one.parts})
        for act, words in (("max", "which book is the oldest ?"),
                           ("none", "what does oldest return when there "
                                    "are no books ?")):
            asked = Q.Asked(act, 0.55, "none", {"TARGET": ["book"]},
                            words.rstrip(" ?").split())
            asked.acts = [act, "min", "list", "value"]
            with mock.patch.object(Q, "read", return_value=asked):
                found = Q.answer(words, project)
            self.assertEqual(found.value, "Solaris")
            self.assertEqual(found.looked.get("read act"), act)

    def test_asked_of_the_last_answers_records(self):
        from unittest import mock
        from research.v701 import querying as Q
        project = self.library()
        first = Q.carry(self.asked("list", "eq", {
            "TARGET": ["books"], "VALUE": ["octavia e . butler"]},
            "which books are by octavia e . butler ?".split()), project)
        kept = first.looked["kept"]
        self.assertEqual((kept["file"], kept["at"]), ("data/books.csv",
                                                      [1, 2]))
        # `how many are there?`: no target said, of the last -- those two
        asked = Q.Asked("count", 0.9, "none", {}, "how many are there".split(),
                        of="last")
        with mock.patch.object(Q, "read", return_value=asked):
            found = Q.answer("how many are there?", project, kept)
        self.assertEqual(found.value, 2)
        self.assertIn("of the 2 just answered", found.said)
        # `which of them came out after 1979?`: a filter of those alone
        asked = Q.Asked("list", 0.9, "gt", {"TARGET": ["title"],
                                             "FIELD": ["year"],
                                             "VALUE": ["1979"]},
                        "which of them came out after 1979".split(),
                        of="last")
        with mock.patch.object(Q, "read", return_value=asked):
            found = Q.answer("which of them came out after 1979?", project,
                             kept)
        self.assertEqual(found.value, ["Wild Seed"])
        # read as not of the last: every record
        asked = Q.Asked("count", 0.9, "none", {"TARGET": ["books"]},
                        "how many books are there".split())
        with mock.patch.object(Q, "read", return_value=asked):
            self.assertEqual(Q.answer("how many books are there?", project,
                                      kept).value, 3)

    def test_asked_beyond_what_is_looked_up(self):
        from unittest import mock
        from research.v701 import querying as Q
        project = self.library()
        asked = Q.Asked("beyond", 0.9, "none", {},
                        "sort them by year".split())
        with mock.patch.object(Q, "read", return_value=asked):
            found = Q.answer("sort them by year", project)
        self.assertTrue(found.looked.get("beyond"))
        # read too unsurely: not taken as beyond
        asked = Q.Asked("beyond", 0.4, "none", {}, "sort them by year".split())
        with mock.patch.object(Q, "read", return_value=asked):
            found = Q.answer("sort them by year", project)
        self.assertFalse(found is not None and found.looked.get("beyond"))

    def test_records_followed_across_links(self):
        from unittest import mock
        from research.v698.project import Project
        from research.v701 import querying as Q
        project = Project("pets")
        project.put({
            "data/student.csv": "StuID,name,age\n1,Ada,19\n2,Bo,20\n"
                                "3,Cy,21\n4,Di,22\n",
            "data/has_pet.csv": "StuID,PetID\n1,10\n2,11\n3,12\n",
            "data/pets.csv": "PetID,kind,weight\n10,dog,9.5\n11,cat,4.0\n"
                             "12,dog,12.0\n"})
        links = {(one["from"][0], one["from"][2], one["to"][0], one["to"][2])
                 for one in project.relations()}
        self.assertEqual(links, {
            ("data/has_pet.csv", "StuID", "data/student.csv", "StuID"),
            ("data/has_pet.csv", "PetID", "data/pets.csv", "PetID")})
        # `which students have a dog?`: the value held by pets, the
        # students' names through has_pet
        asked = Q.Asked("list", 0.9, "eq", {"TARGET": ["students"],
                                             "VALUE": ["dog"]},
                        "which students have a dog".split())
        with mock.patch.object(Q, "read", return_value=asked):
            found = Q.answer("which students have a dog?", project)
        self.assertEqual(found.value, ["Ada", "Cy"])
        self.assertIn("by", found.said)
        # `how much do their pets weigh?`: of the last, in another
        # collection -- those students' pets
        asked = Q.Asked("list", 0.9, "none", {"TARGET": ["weight"]},
                        "how much do their pets weigh".split(), of="last")
        with mock.patch.object(Q, "read", return_value=asked):
            found = Q.answer("how much do their pets weigh?", project,
                             found.looked["kept"])
        self.assertEqual(found.value, [9.5, 12.0])

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

    def test_a_field_without_its_records_and_a_place_said_whole(self):
        from research.v701 import querying as Q
        project = self.library()
        # `joined` read, `member` not: the one collection with that field
        found = Q.carry(self.asked("list", "none", {"FIELD": ["joined"]},
                                   "which member joined first ?".split()),
                        project)
        self.assertEqual(found.value, "Grace Hopper")
        project.put({"settings.yaml": "web:\n  port: 5000\n"})
        found = Q._place_said(self.asked("none", "none", {},
                                         "which port does the web answer on "
                                         "?".split()), project)
        self.assertEqual(found.value, 5000)

    def test_a_number_written_as_text_is_one(self):
        from research.v701.querying import _field_of
        self.assertEqual(_field_of({"v": "17,956"}, "v"), 17956)
        self.assertEqual(_field_of({"v": "$1,200.50"}, "v"), 1200.5)
        self.assertEqual(_field_of({"v": "02139"}, "v"), "02139")
        self.assertEqual(_field_of({"v": "1.96 m"}, "v"), "1.96 m")

    def test_the_most_of_a_time_is_the_latest(self):
        from research.v701 import querying as Q
        project = self.library()
        # no field said: the year, and of a time the most is the latest
        found = Q.carry(self.asked("min", "none", {"TARGET": ["book"]},
                                   "what is the newest book ?".split()),
                        project)
        self.assertEqual(found.value, "Wild Seed")
        self.assertIn("latest year", found.said)
        found = Q.carry(self.asked("max", "none", {"TARGET": ["book"]},
                                   "which book is the oldest ?".split()),
                        project)
        self.assertEqual(found.value, "Dune")
        # a field of times named, read as every record's: the earliest
        found = Q.carry(self.asked("list", "none", {
            "TARGET": ["member"], "FIELD": ["joined"]},
            "which member joined first ?".split()), project)
        self.assertEqual(found.value, "Grace Hopper")
        # and every record's where that is asked
        found = Q.carry(self.asked("list", "none", {
            "TARGET": ["members"], "FIELD": ["joined"]},
            "list the members and their joined dates".split()), project)
        self.assertEqual(len(found.value), 2)

    def test_a_record_named_by_a_value_it_holds(self):
        from research.v701 import querying as Q
        project = self.library()
        found = Q._by_value("how many pages does Kindred have?", project)
        self.assertEqual(found.value, 264)
        # a value told of it is no question of it
        self.assertIsNone(Q._by_value("Kindred has 300 pages now", project))

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

    def test_changes_chosen_as_codet_chooses(self):
        from research.v700.fixing import _agreed_by_tests
        right = [{"own ok": True, "passes": (0, 2)},
                 {"own ok": True, "passes": (0, 2)}]
        alone = {"own ok": True, "passes": (1,)}
        broken = {"own ok": False, "passes": ()}
        # test 3 is wrong -- no change passes it -- and weighs nothing
        self.assertEqual(_agreed_by_tests(right + [alone, broken], 4), right)
        # one change alone, passing a test of four: no choice
        self.assertEqual(_agreed_by_tests([alone, broken], 4), [])
        # one change alone passing half of them: a guess
        self.assertEqual(_agreed_by_tests([right[0]], 4), [])
        # one alone passing every test there is: chosen
        self.assertEqual(_agreed_by_tests([right[0]], 2), [right[0]])
        # the same tests passed, but giving differently: apart, and neither
        # group of one is chosen
        # answers agreeing on what passes none of the tests there are: no
        # choice (one model's answers agreeing are no evidence)
        other = {"own ok": True, "passes": (), "gives": ("7",)}
        self.assertEqual(_agreed_by_tests([other, dict(other), dict(other)],
                                          9, ("5",)), [])
        strip = {"own ok": True, "passes": (0, 2), "gives": ("'978.014'",)}
        whole = {"own ok": True, "passes": (0, 2), "gives": ("'978014'",)}
        self.assertEqual(_agreed_by_tests([strip, whole], 4), [])

    def test_what_a_request_writes_out_is_asked_of_the_code(self):
        from research.v698 import requested
        project = held()
        asked = requested.interface(
            "add a method `count_by(author: str, year=None)` and a field "
            "`shelved`; keep `load` as it is; the heuristic h(n)", project)
        # (`load`, the project's own, named bare: nothing asked of it)
        self.assertEqual(asked, {"count_by": 2, "shelved": None})
        right = ("class Shelf:\n    def __init__(self):\n        self.shelved"
                 " = []\n    def count_by(self, author, year=None, *, x=1):\n"
                 "        return 0\ndef load():\n    pass\n")
        self.assertEqual(requested._defines(right, asked), "")
        self.assertIn("no shelved", requested._defines(
            right.replace("self.shelved", "self.kept"), asked))
        self.assertIn("count_by does not take 2", requested._defines(
            right.replace("author, year=None", "author"), asked))

    def test_samples_that_do_not_fit_are_written_in_halves(self):
        import torch
        from research.v696 import teach_meaning

        class Tokens(dict):
            def to(self, device):
                return self

        class Tokenizer:
            pad_token_id = 0

            def apply_chat_template(self, chat, **kwargs):
                return chat[0]["content"]

            def __call__(self, texts, **kwargs):
                return Tokens(input_ids=torch.zeros((len(texts), 3),
                                                    dtype=torch.long))

            def decode(self, row, **kwargs):
                return "sampled"

        class Model:
            def __init__(self):
                self.asked = []

            def generate(self, input_ids, max_new_tokens, pad_token_id,
                         **settings):
                many = settings.get("num_return_sequences", 1)
                self.asked.append((many, settings["do_sample"]))
                if many > 2:
                    raise torch.cuda.OutOfMemoryError("CUDA out of memory")
                return torch.zeros((many, 5), dtype=torch.long)
        writer = teach_meaning.Teacher.__new__(teach_meaning.Teacher)
        writer.torch, writer.tokenizer, writer.model = torch, Tokenizer(), \
            Model()
        got = writer.write(["write tests"], longest=10, samples=6)
        # six did not fit: three and three, then two and one twice -- all
        # sampled, none greedy
        self.assertEqual(len(got[0]), 6)
        self.assertTrue(all(sampled for _, sampled in writer.model.asked))
        self.assertEqual(sum(many for many, _ in writer.model.asked
                             if many <= 2), 6)

    def test_a_server_whose_card_is_lost_says_so(self):
        from research.v698 import server
        self.assertTrue(server.wedged(RuntimeError(
            "CUDA error: an illegal memory access was encountered")))
        self.assertFalse(server.wedged(IndexError("list index out of range")))
        self.assertFalse(server.wedged(RuntimeError(
            "CUDA out of memory. Tried to allocate 20 MiB")))

    def test_what_code_gives_is_read_past_what_it_prints(self):
        import tempfile
        from pathlib import Path
        from research.v698 import requested
        root = Path(tempfile.mkdtemp())
        (root / "m.py").write_text("print(42)\ndef f(x):\n    print(7)\n"
                                   "    return x * 2\n", encoding="utf-8")
        self.assertEqual(requested._gives(root, ["f(3)", "g(1)"], "m"),
                         ("6", "NameError"))

    def test_a_turn_knows_the_time_it_has_left(self):
        import threading
        from research.v697 import page
        found = {}

        def turn():
            found["before"] = page.left()
            page.said_as_typed("make it faster")
            found["in"] = page.left()
        one = threading.Thread(target=turn)
        one.start()
        one.join()
        self.assertIsNone(found["before"])
        self.assertTrue(page.TURN_SECONDS - 5 < found["in"] <=
                        page.TURN_SECONDS)

    def test_a_change_of_a_body_changes_its_function(self):
        from research.v700.fixing import _changed_names, _enclosing
        # `return text[::-1]` for `pass` names nothing on its lines, and
        # changes `reverse`
        self.assertIn("reverse", _changed_names(
            "def reverse(text):\n    pass\n",
            "def reverse(text):\n    return text[::-1]\n"))
        lines = ["class Shelf:", "    def count(self):", "        if x:",
                 "            y = 1", "    def add(self):", "        pass",
                 "def load():", "    return 2"]
        self.assertEqual(_enclosing(lines, 3), {"Shelf", "count"})
        self.assertEqual(_enclosing(lines, 5), {"Shelf", "add"})
        self.assertEqual(_enclosing(lines, 7), {"load"})

    def test_the_errors_a_request_writes_out_are_asked(self):
        from research.v698 import requested
        statement = ("Raise a `ValueError` with a meaningful message:\n\n"
                     "```python\n# for input.\nraise ValueError(\"input base "
                     "must be >= 2\")\n```\n\nand a picture of it:\n\n```text"
                     "\n[ ][ ]\n```")
        self.assertEqual(requested.raised(statement),
                         ["input base must be >= 2"])
        right = ("def rebase(a, digits, b):\n    if a < 2:\n        raise "
                 "ValueError(\"input base must be >= 2\")\n")
        self.assertEqual(requested._unraised(right, ["input base must be >= "
                                                     "2"]), [])
        self.assertEqual(requested._unraised(
            right.replace("input base", "base"), ["input base must be >= 2"]),
            ["it never says 'input base must be >= 2'"])

    def test_a_name_no_change_has_was_a_word(self):
        from research.v700.fixing import _agreed_by_tests
        # every change lacks `exception` (Exercism: "raise an `exception`"):
        # a word of the request, refusing none; one lacking `shelved` that
        # another change has is no change asked
        has = {"own ok": True, "passes": (0,), "gives": ("1",), "by": {1, 2},
               "lacks": ["it has no exception"]}
        lacks = {"own ok": True, "passes": (0,), "gives": ("2",),
                 "by": {3, 4, 5},
                 "lacks": ["it has no exception", "it has no shelved"]}
        self.assertEqual(_agreed_by_tests([has, lacks], 1), [has])
        self.assertEqual(lacks["lacks what"], ["it has no shelved"])

    def test_the_examples_a_request_says_are_its_true_tests(self):
        from research.v698 import requested
        from research.v700.fixing import _agreed_by_tests
        said = requested.stated("I want heronArea to return the heron area."
                                " For example, heronArea(3, 4, 5) should "
                                "return 6.0.", {"heronArea"})
        self.assertEqual(len(said), 1)
        space = {}
        exec("def heronArea(a, b, c):\n    s = (a + b + c) / 2\n    return "
             "(s * (s - a) * (s - b) * (s - c)) ** 0.5\n" + said[0], space)
        with self.assertRaises(AssertionError):
            exec("def heronArea(a, b, c):\n    return a * b\n" + said[0], {})
        # a string said in a sentence is a string, its full stop no number
        said = requested.stated('reverseString should output it in reverse.'
                                ' For example, reverseString("hi") should '
                                'return "ih".', {"reverseString"})
        exec("def reverseString(s):\n    return s[::-1]\n" + said[0], {})
        # one answer holding what the person said: chosen
        one = {"own ok": True, "passes": (0,), "gives": ("6.0",)}
        self.assertEqual(_agreed_by_tests([one], 4, ("0",), stated=1), [one])
        self.assertEqual(_agreed_by_tests([one], 4, ("0",)), [])

    def test_a_change_keeping_what_it_gives_is_held_to_the_code(self):
        from research.v700.fixing import _agreed_by_tests
        original = ("1", "4", "9", "TypeError")
        same = {"own ok": True, "passes": (), "gives": original}
        other = {"own ok": True, "passes": (), "gives": ("1", "4", "8",
                                                         "TypeError")}
        self.assertEqual(_agreed_by_tests([same, other, dict(other)], 3,
                                          original, keeps=True), [same])
        # too little the code gives to be an oracle: no choice
        self.assertEqual(_agreed_by_tests([same], 3, ("1", "TypeError",
                                                      "TypeError"),
                                          keeps=True), [])

    def test_a_test_made_a_probe_of_what_code_gives(self):
        from research.v698 import requested
        probe = requested._probe("from shelf.books import Book\n"
                                 "b = Book('a', 3)\nassert b.pages == 99\n"
                                 "assert b.title\n")
        project = held()
        root = Path(tempfile.mkdtemp())
        requested._copy(project, root)
        self.assertEqual(requested._probed(root, [probe], "shelf.books"),
                         ("3; 'a'",))

    def test_a_name_written_as_code(self):
        from research.v698.requested import code_shaped
        self.assertTrue(all(map(code_shaped, ("clean_isbn", "cleanIsbn",
                                              "v2", "read_all"))))
        self.assertFalse(any(map(code_shaped, ("to", "everywhere", "Volume",
                                               "_private"))))

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


class DescribedTests(unittest.TestCase):

    def test_a_javascript_function_found_by_what_it_does(self):
        from research.v698 import asking
        from research.v698.project import Project
        project = Project("js")
        project.put({"lib/util.js": (
            "function reverseWords(text) {\n"
            "  return text.split(' ').reverse().join(' ');\n}\n\n"
            "function totalPrice(items) {\n"
            "  let sum = 0;\n"
            "  for (const item of items) { sum += item.price * item.count; }\n"
            "  return sum;\n}\n")})
        names = {one.name for one in asking._symbols(
            project, ("function", "method"))}
        self.assertEqual(names, {"reverseWords", "totalPrice"})
        found = asking._described(
            project, "which function adds up what all the items cost, "
                     "each price times how many", "where")
        self.assertEqual(found.name, "totalPrice")
        found = asking._described(
            project, "which function puts the words of a sentence in the "
                     "opposite order", "where")
        self.assertEqual(found.name, "reverseWords")


class DataWrittenTests(unittest.TestCase):

    def test_a_data_file_changed_only_by_what_was_said(self):
        from research.v700.fixing import _unsaid
        rows = "id,name,email\n1,Ada,ada@x.org\n"
        self.assertEqual(_unsaid(
            "add a customer with id 4 named Charles Darwin, email "
            "charles@example.com", rows,
            rows + "4,Charles Darwin,charles@example.com\n"), set())
        self.assertEqual(_unsaid("remove Ada", rows, "id,name,email\n"),
                         set())
        # a question read as a change: what it writes, no one said
        self.assertEqual(_unsaid(
            "show the number for each shop", "shop_id,name\n1,FC Haka\n",
            "shop_id,name,number\n1,FC Haka,2\n"), {"2"})
        # a column put in, no row changed: nothing of the data said
        self.assertTrue(_unsaid(
            "show the number for each shop", "shop_id,name\n1,FC Haka\n",
            "shop_id,name,number\n1,FC Haka\n", "data/shop.csv"))
        # a row of a cell more than the columns
        self.assertTrue(_unsaid(
            "how about those in year 2014?", "id,year\n1,2014\n2,2015\n",
            "id,year\n1,2014,2014\n2,2015\n", "data/concert.csv"))


class CodeTalkTests(unittest.TestCase):

    def test_an_answer_of_what_was_typed_answers_the_turn(self):
        from types import SimpleNamespace
        from research.v687.executive import ANSWERED, CONTINUE, Operator
        from research.v698 import page
        # an operator of the page answers what was typed, whole: the claims
        # an utterance was read as after it are not acted on again
        turn = SimpleNamespace()
        page._whole(Operator("x", lambda memory: CONTINUE)).apply(
            {"turn": turn})
        self.assertFalse(getattr(turn, "whole", False))
        page._whole(Operator("x", lambda memory: ANSWERED)).apply(
            {"turn": turn})
        self.assertTrue(turn.whole)

    def test_a_long_request_is_read_by_the_part_that_asks(self):
        from research.v698 import reading
        if not reading.taught():
            self.skipTest("no reader taught code talk")
        # past what the encoder reads: an exercise's instructions, its
        # example, then what to do of which file, last
        text = ("# Instructions\n\n" + "Given a list of numbers, each one "
                "is squared and the squares are kept in the order they came"
                " in, so that the list given back is as long. " * 4 +
                "\n\n```text\n1, 2, 3 -> 1, 4, 9\n```\n\nUse the above "
                "instructions to modify the supplied files: squares.py")
        read = reading.read(text, {"squares.py", "squares"})
        self.assertEqual(read.act, "change")
        self.assertIn(read.subject, reading.NAMED)
        self.assertIn("squares.py", read.spans.get("SUBJ", []))

    def test_a_long_requests_ways_are_of_its_parts_that_say_how(self):
        from unittest import mock
        from research.v698 import asked_ways, reading, roles
        # the problem names a zipper, a tip names enumerate: only the part
        # that says how the code is to be written holds it to a way
        parts = ["Creating a zipper for a binary tree.",
                 "You can use enumerate() to get the index.",
                 "This exercise is designed to be completed using "
                 "recursion, rather than loops."]
        role = {"creating": "describes", "you": "suggests",
                "this": "how"}
        read = {"creating": [("zip", 0.99)],
                "you": [("enumerate", 0.98)],
                "this": [("recursive", 0.99), ("no-loop", 1.0)]}
        with mock.patch.object(roles, "roles", lambda texts: [
                role[text.split()[0]] for text in texts]), \
                mock.patch.object(asked_ways, "read", lambda text: read[
                    text.split()[0]]):
            self.assertEqual(sorted(reading._ways_of(parts, ["python"])),
                             ["no-loop", "python", "recursive"])
            # the asking part's way of a family stands
            self.assertEqual(reading._ways_of(parts, ["iterative"]),
                             ["iterative", "no-loop"])

    def test_a_doing_sure_together_is_code_talk(self):
        from unittest import mock
        from research.v698 import reading

        def reads(act, acts):
            return reading.Reading(act, acts[0][1], "none", 1.0, "none",
                                   [], [], acts=acts)
        cases = [("change", [("change", 0.53), ("goal", 0.46)], True),
                 ("change", [("change", 0.5), ("none", 0.5)], False),
                 ("ask", [("ask", 0.5), ("change", 0.45)], False),
                 ("none", [("none", 0.9), ("change", 0.1)], False)]
        for act, acts, taken in cases:
            with mock.patch.object(reading, "read",
                                   return_value=reads(act, acts)):
                self.assertEqual(reading.code_talk("clean this up")
                                 is not None, taken, acts)


class TaughtValuesTests(unittest.TestCase):

    def test_the_value_a_question_ends_on_said_another_way(self):
        import random
        from research.v698 import teach_code_ask as T
        rng = random.Random(1)
        line = "what input to f gives [3, 5, 4]?"
        self.assertEqual(T._literals(line), [(22, 31)])
        said = T._revalued(line, rng)
        self.assertTrue(said.startswith("what input to f gives "))
        self.assertTrue(said.endswith("?"))
        # a value inside the words is no value the question ends on
        for kept in ("which input to g returns [x for x in range(3)]?",
                     "what does h give for 'a/1' when it is empty?",
                     "find the input for g that makes 'a/1"):
            self.assertEqual(T._revalued(kept, rng), kept)


if __name__ == "__main__":
    unittest.main()
