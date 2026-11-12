"""v706: answers as queries, found by search, kept as data. No model runs."""
from __future__ import annotations

import unittest
from pathlib import Path

from research.v706 import queries as Q
from research.v706 import synth, ways

RENTAL = Path(__file__).resolve().parents[1] / "v704" / "fixtures" / "rental"


def says_all(*words):
    return lambda said: all(one in said for one in words)


class QueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.held = ways.project_of(RENTAL)
        cls.world = Q.world(cls.held)

    def test_a_word_says_a_value_by_lookup(self):
        self.assertEqual(Q.said_as("list the classes in hires.py", "class"),
                         "classes")
        self.assertEqual(Q.said_as("in hires.py?", "rental/hires.py"),
                         "hires.py")
        self.assertIsNone(Q.said_as("the classic one", "class"))
        # as written, where capitals tell two apart
        self.assertIsNone(Q.said_as("does Hire raise anything?", "hire"))
        self.assertEqual(Q.said_as("does hire raise?", "hire"), "hire")

    def test_a_way_is_bound_run_and_said(self):
        way = {"from": "symbols", "steps": [["is", "kind", "?"],
                                            ["is", "file", "?"]],
               "give": ["list", "name"]}
        bound = Q.bind(way, self.world, "list the classes in hires.py")
        self.assertEqual(bound["said"], 2)
        rows, value = Q.run(bound, self.world)
        self.assertEqual(sorted(value), ["Hire", "TooManyHires"])
        self.assertIn("TooManyHires", Q.said(bound, rows, value))
        self.assertIsNone(Q.bind(way, self.world, "where is hires.py?"))

    def test_the_library_is_used_by_name(self):
        library = {"kind_file": {"steps": [["is", "kind", "?"],
                                           ["is", "file", "?"]]}}
        way = {"from": "symbols", "steps": [["use", "kind_file"]],
               "give": ["count"]}
        bound = Q.bind(way, self.world, "the functions of tools.py", library)
        self.assertEqual(Q.run(bound, self.world)[1], 5)

    def test_a_yes_or_no_says_no_of_nothing(self):
        way = {"from": "symbols", "steps": [
            ["is", "name", "?"],
            ["linked", "raises", "name", "name", [["has", "raise"]]]],
            "give": ["any"]}
        bound = Q.bind(way, self.world, "does clean_code raise anything?")
        rows, value = Q.run(bound, self.world)
        self.assertFalse(Q.empty(bound, value))
        self.assertTrue(Q.said(bound, rows, value).startswith("No"))


class SynthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = Q.world(ways.project_of(RENTAL))

    def test_the_shortest_way_saying_most_is_found(self):
        way, said, _ = synth.solve(self.world, "list the classes in hires.py",
                                   says_all("TooManyHires", "Hire"))
        self.assertEqual(way["steps"], [["is", "kind", "?"],
                                        ["is", "file", "?"]])
        self.assertEqual(way["give"], ["list", "name"])

    def test_a_join_is_found(self):
        way, said, _ = synth.solve(
            self.world, "which functions in tools.py have tests?",
            lambda said: "clean_code" in said and "dearest" not in said)
        self.assertEqual(way["steps"][-1][:2], ["linked", "uses"])

    def test_what_no_field_holds_is_not_found(self):
        way, _, _ = synth.solve(self.world, "which tool is the dearest?",
                                says_all("Cement Mixer"))
        self.assertIsNone(way)


class WaysTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = Q.world(ways.project_of(RENTAL))
        cls.found = {"library": {}, "ways": [
            {"from": "symbols", "steps": [["is", "kind", "?"],
                                          ["is", "file", "?"]],
             "give": ["list", "name"]},
            {"from": "symbols", "steps": [
                ["is", "kind", "?"], ["is", "file", "?"],
                ["linked", "uses", "name", "name", [["has", "test"]]]],
             "give": ["list", "name"]}]}

    def test_the_way_saying_most_of_a_message_answers_it(self):
        way, _ = ways.choose("which functions in tools.py have tests?",
                             self.world, self.found)
        self.assertIs(way, self.found["ways"][1])
        way, _ = ways.choose("list the functions in tools.py", self.world,
                             self.found)
        self.assertIs(way, self.found["ways"][0])

    def test_a_way_leaving_a_name_unused_does_not_answer(self):
        self.assertIsNone(ways.choose(
            "where is the function dearest in tools.py?", self.world,
            self.found))

    def test_shared_steps_are_given_a_name(self):
        import copy
        found = copy.deepcopy(self.found)
        self.assertEqual(ways.abstract(found), ["kind_file"])
        self.assertEqual(found["ways"][0]["steps"], [["use", "kind_file"]])
        way, bound = ways.choose("list the classes in hires.py", self.world,
                                 found)
        self.assertEqual(sorted(Q.run(bound, self.world)[1]),
                         ["Hire", "TooManyHires"])


class CurriculumTests(unittest.TestCase):
    def test_a_value_is_said_as_the_word_said_its_own(self):
        from research.v706.curriculum import _form
        self.assertEqual(_form("function", "classes", "class"), "functions")
        self.assertEqual(_form("class", "functions", "function"), "classes")
        self.assertEqual(_form("rental/tools.py", "hires.py",
                               "rental/hires.py"), "tools.py")

    def test_its_questions_have_the_projects_answers(self):
        from research.v706 import curriculum
        found = {"library": {}, "ways": [
            {"from": "symbols", "steps": [["is", "kind", "?"],
                                          ["is", "file", "?"]],
             "give": ["list", "name"], "fixture": "rental",
             "learnt from": "list the classes in hires.py"}]}
        asked = curriculum.questions(found, ("rental",))
        by = {one["turns"][0]: one["truth"] for one in asked}
        self.assertEqual(sorted(by["list the constants in hires.py"]),
                         ["HIRE_DAYS", "LATE_FEE", "MAX_HIRES"])


if __name__ == "__main__":
    unittest.main()
