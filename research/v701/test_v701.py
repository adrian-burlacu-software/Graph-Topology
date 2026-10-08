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

    def test_what_is_not_there_is_said(self):
        found = self.said("value", "none", {"TARGET": ["shoe size"]},
                          "what is the shoe size")
        self.assertIn("nothing called shoe size", found.said)


if __name__ == "__main__":
    unittest.main()
