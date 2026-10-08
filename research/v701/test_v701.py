"""v701: data as information -- a JSON, YAML or CSV file held as what it
holds and its schema, and questions about it answered from it.

    python -m unittest research.v701.test_v701
"""
from __future__ import annotations

import unittest

SERVICE = """name: graph-topology
server:
  host: 127.0.0.1
  port: 8697
  timeout_seconds: 900
users:
  - name: adrian
    role: owner
    projects: [graph-topology, stark-db]
  - name: reviewer
    role: reader
    projects: [graph-topology]
"""
USERS = "name,role,age\nadrian,owner,41\nreviewer,reader,29\nsam,reader,35\n"


class ModelTests(unittest.TestCase):
    """`datamodel.py`: the schema read off what a file holds."""

    def test_places_collections_and_keys(self):
        from research.v701 import datamodel as D
        model = D.read("s.yaml", SERVICE)
        places = {one["path"]: one for one in model.schema}
        self.assertEqual(places["server.port"]["types"], {"integer": 1})
        self.assertEqual(model.collections["users[]"],
                         {"count": 2, "fields": ["name", "role",
                                                 "projects"]})
        self.assertEqual(places["users[].role"]["in"], "2 of 2")
        rows = D.read("u.csv", USERS)
        self.assertEqual(rows.records()[""][0],
                         {"name": "adrian", "role": "owner", "age": 41})
        self.assertEqual(rows.keys(""), ["name"])

    def test_csv_columns_typed_as_columns_and_headers_told(self):
        from research.v701 import datamodel as D
        words = D.read("w.csv", "a\nabout\ntrue\nzone\n")
        self.assertEqual([one["column1"] for one in words.records()[""]],
                         ["a", "about", "true", "zone"])
        gaps = D.read("g.csv", "name,age\nann,\nbo,7\ncy,8\n")
        self.assertEqual([one["age"] for one in gaps.records()[""]],
                         [None, 7, 8])

    def test_what_does_not_parse_is_said(self):
        from research.v701 import datamodel as D
        self.assertIn("JSONDecodeError", D.read("x.json", "{").error)


class ProjectTests(unittest.TestCase):
    """Data files held beside code, never sent to the compiler."""

    def test_held_beside_code(self):
        from research.v698.project import Project
        held = Project("t")
        refused = held.put({"a.py": "x = 1\n", "s.yaml": SERVICE,
                            "n.md": "#"})
        self.assertEqual(list(refused), ["n.md"])
        self.assertEqual(held.languages(), ["python"])
        self.assertEqual(list(held.checked()), ["/ws/a.py"])
        self.assertEqual(held.summary()["data"], {"s.yaml": {"users[]": 2}})


class QueryTests(unittest.TestCase):
    """What a question is read as, carried out: its fields looked up."""

    @classmethod
    def setUpClass(cls):
        from research.v698.project import Project
        cls.held = Project("t")
        cls.held.put({"conf/service.yaml": SERVICE, "users.csv": USERS})

    def said(self, act, op, spans, text):
        from research.v701.querying import Asked, carry
        return carry(Asked(act, 1.0, op, spans, text.split()), self.held)

    def test_a_value_by_its_name(self):
        found = self.said("value", "none", {"TARGET": ["port"]},
                          "which port does the server listen on")
        self.assertEqual((found.value, found.file),
                         (8697, "conf/service.yaml"))
        found = self.said("value", "none", {"TARGET": ["timeout seconds"]},
                          "what is the timeout seconds")
        self.assertEqual(found.value, 900)

    def test_the_collection_that_has_the_field(self):
        found = self.said("count", "gt", {"TARGET": ["users"],
                                          "FIELD": ["age"],
                                          "VALUE": ["30"]},
                          "how many users are older than 30")
        self.assertEqual((found.value, found.file), (2, "users.csv"))
        found = self.said("list", "contains", {"TARGET": ["users"],
                                               "FIELD": ["projects"],
                                               "VALUE": ["stark-db"]},
                          "which users can see stark-db")
        self.assertEqual(found.value, ["adrian"])

    def test_both_where_both_fit_and_the_file_said_decides(self):
        found = self.said("count", "none", {"TARGET": ["users"]},
                          "how many users are there")
        self.assertEqual(found.value, [2, 3])
        found = self.said("count", "none", {"TARGET": ["users"]},
                          "how many users are there in service.yaml")
        self.assertEqual(found.value, 2)

    def test_a_field_not_named_is_found_by_its_values(self):
        found = self.said("list", "eq", {"TARGET": ["users"],
                                         "FIELD": ["see"],
                                         "VALUE": ["stark-db"]},
                          "which users can see stark-db")
        self.assertEqual(found.value, ["adrian"])
        self.assertIn("whose projects has stark-db", found.said)
        found = self.said("count", "gt", {"TARGET": ["users.csv"],
                                          "FIELD": ["older"],
                                          "VALUE": ["30"]},
                          "how many users.csv are older than 30")
        self.assertEqual(found.value, 2)
        found = self.said("mean", "eq", {"TARGET": ["age"],
                                         "VALUE": ["readers"]},
                          "what is the average age of the readers")
        self.assertEqual(found.value, 32)

    def test_a_value_named_by_the_places_around_it(self):
        from research.v698.project import Project
        from research.v701.querying import Asked, carry
        held = Project("t")
        held.put({"s.yaml": "models:\n  editor: llm/editor3\n"
                            "  judge: llm/change-judge2\n"})
        found = carry(Asked("list", 1.0, "eq", {"TARGET": ["model"],
                                                "FIELD": ["editor"]},
                            "what model is the editor".split()), held)
        self.assertEqual(found.value, "llm/editor3")

    def test_what_is_not_there_is_said(self):
        found = self.said("value", "none", {"TARGET": ["shoe size"]},
                          "what is the shoe size")
        self.assertIn("nothing called shoe size", found.said)


TASKS = ("tasks:\n  - id: t1\n    owner: adrian\n    watchers: [sam, reviewer]\n"
         "  - id: t2\n    owner: sam\n    watchers: [adrian]\n"
         "  - id: t3\n    owner: reviewer\n    watchers: []\n")


class RelationTests(unittest.TestCase):
    """Links between data, found in what it holds, and followed."""

    @classmethod
    def setUpClass(cls):
        from research.v698.project import Project
        cls.held = Project("t")
        cls.held.put({"users.csv": USERS, "tasks.yaml": TASKS})

    def test_links_to_a_collections_first_key(self):
        self.assertEqual(
            [(one["from"][2], one["to"][0], one["to"][2])
             for one in self.held.relations()],
            [("owner", "users.csv", "name"),
             ("watchers", "users.csv", "name")])

    def test_followed_the_link_the_message_says(self):
        from research.v701.querying import Asked, carry
        found = carry(Asked("value", 1.0, "eq", {"TARGET": ["role"],
                                                 "VALUE": ["t1"]},
                            "what role does the owner of t1 have".split()),
                      self.held)
        self.assertEqual(found.value, "owner")
        found = carry(Asked("list", 1.0, "eq", {"TARGET": ["age"],
                                                "VALUE": ["t1"]},
                            "how old are the watchers of t1".split()),
                      self.held)
        self.assertEqual(sorted(found.value), [29, 35])


class PastedTests(unittest.TestCase):
    """Data said in a message is found by parsing it, named by the name
    said beside it, and a request's examples are not data."""

    def test_found_and_named(self):
        from research.v701 import pasting
        found = pasting.found("here is users.csv:\n" + USERS +
                              "how many users are older than 30?")
        self.assertEqual((found["format"], found["name"], found["rest"]),
                         ("csv", "users.csv", "here is users.csv:\n"
                          "how many users are older than 30?"))
        found = pasting.found('what fields?\n```json\n{"a": 1}\n```')
        self.assertEqual((found["format"], found["rest"]),
                         ("json", "what fields?"))
        self.assertIsNone(pasting.found(
            "add two numbers\nadd(1, 2) == 3\nadd(2, 5) == 7\n"
            "add(0, 0) == 0"))

    def test_held_in_the_conversation(self):
        from research.v698 import project as Pj
        from research.v701 import pasting
        held = pasting.hold("```yaml\n" + SERVICE + "```\nwhat port?",
                            "test-v701-pasted")
        self.assertEqual(held["path"], "pasted-1.yaml")
        self.assertIn("pasted-1.yaml",
                      Pj.project("test-v701-pasted").data())


class DataChangeTests(unittest.TestCase):
    """A change of a data file, checked as data: it still reads, and puts
    in no place the file already names elsewhere."""

    def test_checked_as_data(self):
        from research.v700.fixing import data_wrong
        self.assertIsNone(data_wrong("s.yaml", SERVICE, SERVICE.replace(
            "port: 8697", "port: 9000")))
        self.assertIn("server.server.port", data_wrong(
            "s.yaml", SERVICE, SERVICE.replace(
                "port: 8697", "port: 9000\n  server:\n    port: 9000")))
        self.assertIn("no longer reads", data_wrong(
            "s.yaml", SERVICE, SERVICE.replace("port: 8697", "port: [")))
        self.assertIsNone(data_wrong("s.yaml", SERVICE, SERVICE.replace(
            "  - name: reviewer", "  - name: sam\n    role: reader\n"
            "    projects: [x]\n  - name: reviewer")))


class ExtremeTests(unittest.TestCase):
    """The record with the most of a field, where the records are asked
    for: the field named, else the one of numbers; of several, asked."""

    def held(self):
        from research.v698.project import Project
        held = Project("t")
        held.files["x/kids.csv"] = ("name,age,height\na,3,100\nb,5,90\n"
                                    "c,4,120\n")
        return held

    def asked(self, act, spans):
        from research.v701 import querying as Q
        return Q.Asked(act, 0.99, "none", spans, ["which", "kids"], ["kids"])

    def test_the_field_named(self):
        from research.v701 import querying as Q
        found = Q.carry(self.asked("max", {"TARGET": ["kids"],
                                           "FIELD": ["height"]}), self.held())
        self.assertEqual(found.value, "c")
        found = Q.carry(self.asked("min", {"TARGET": ["kids"],
                                           "FIELD": ["age"]}), self.held())
        self.assertEqual(found.value, "a")

    def test_of_several_fields_none_guessed(self):
        from research.v701 import querying as Q
        found = Q.carry(self.asked("max", {"TARGET": ["kids"]}), self.held())
        self.assertIsNone(found.value)
        self.assertIn("say which", found.said)


class RecordsKeptTests(unittest.TestCase):
    """A change of data keeps its records records: no field left out, no
    column's cells moved, no key said twice."""

    CSV = "name,age,city\nadrian,34,Toronto\nlee,41,Vancouver\n"

    def test_a_row_a_cell_short(self):
        from research.v700.fixing import data_wrong
        self.assertIn("string", data_wrong(
            "p.csv", self.CSV, self.CSV + "31,Ottawa\n"))
        self.assertIsNone(data_wrong("p.csv", self.CSV,
                                     self.CSV + "nia,31,Ottawa\n"))

    def test_a_key_said_twice(self):
        from research.v700.fixing import data_wrong
        self.assertIn("repeats", data_wrong(
            "p.csv", self.CSV, self.CSV + "lee,31,Ottawa\n"))
        users = "users:\n  - name: a\n    role: x\n  - name: b\n    role: y\n"
        self.assertIn("repeats", data_wrong(
            "s.yaml", users, users + "  - name: b\n    role: x\n"))
        self.assertIn("without role", data_wrong(
            "s.yaml", users, users + "  - name: c\n"))


class PlacingTests(unittest.TestCase):
    """A change of a data file is given the editor by the block the
    statement is of, as its changes were taught."""

    TEXT = ("# The page and the reader\nname: graph-topology\nserver:\n"
            "  port: 9000\n  timeout_seconds: 900\nmodels:\n"
            "  judge: llm/change-judge2\n  teacher:\n    offline: true\n"
            "users:\n  - name: adrian\n    role: owner\n"
            "  - name: reviewer\n    role: reader\n")

    def test_the_block_holding_what_is_said(self):
        from research.v701 import placing
        lines = self.TEXT.splitlines()
        self.assertEqual(placing.block(lines, "add a user sam, role reader"),
                         (9, 13))
        self.assertEqual(placing.block(lines, "set the timeout seconds to 1"),
                         (4, 4))
        # `change` is no part of `llm/change-judge2`, `the` of a comment
        self.assertEqual(placing.block(lines, "change the teacher to offline"),
                         (7, 8))
        self.assertIsNone(placing.block(lines, "add a widget"))

    def test_a_change_outside_it_refused(self):
        from research.v700.fixing import _outside
        part = "users:\n  - name: a\n"
        self.assertFalse(_outside((0, 1), 1, part, part + "  - name: b\n"))
        self.assertTrue(_outside((1, 1), 1, part, "people:\n  - name: a\n"))


if __name__ == "__main__":
    unittest.main()
