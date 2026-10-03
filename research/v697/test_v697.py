"""v697: requests for code read, the graph a turn touched, the account.

    python -m unittest research.v697.test_v697
"""
from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

HAVE = shutil.which("node") is not None
needs_node = unittest.skipUnless(HAVE, "node missing")
HERE = Path(__file__).resolve().parent


@needs_node
class RequestTests(unittest.TestCase):
    """A request for a function taken apart: its signature as given or
    made from the examples, the examples, the English left."""

    def test_signature_and_examples(self):
        from research.v697.coding import read
        said = read('reverse the words\nfunction rw(s: string): string\n'
                    'rw("a b") == "b a"')
        self.assertEqual(said["signature"], "function rw(s: string): string")
        self.assertEqual(said["examples"][0]["args"], ["a b"])
        self.assertEqual(said["examples"][0]["value"], "b a")
        self.assertEqual(said["english"], "reverse the words.")
        self.assertTrue(said["asked"])

    def test_signature_made_from_examples(self):
        from research.v697.coding import read
        said = read("sum a list: total([1,2,3]) == 6, total([]) == 0")
        self.assertEqual(said["signature"],
                         "function total(xs: number[]): number")
        self.assertEqual(said["made"], ["signature"])
        self.assertEqual(len(said["examples"]), 2)

    def test_what_is_not_code_is_left_alone(self):
        from research.v697.coding import read
        for text in ("let f(x) = x^2 + 1", "what is f(3)",
                     "is a dog an animal", "there is a beagle"):
            self.assertFalse(read(text)["asked"], text)

    def test_a_request_without_a_call_asks_for_one(self):
        from research.v697.coding import answered
        said = answered("write a function that doubles a number")
        self.assertEqual(said["code"]["answer"]["status"], "missing")
        self.assertIn("example", said["spoken"])


class GraphTests(unittest.TestCase):
    """What a turn touched, each node and edge saying where it came from."""

    def test_walk_memory_said_run(self):
        from research.v697.graph import graph_of
        turn = {
            "walk": {"chain": ["r1", "beagle.n.01", "dog.n.01"],
                     "steps": [{"concept": "dog.n.01", "rule": "R2"}],
                     "evidence": [{"concept": "dog.n.01",
                                   "relation": "capable_of",
                                   "object": "swim", "source": "seed",
                                   "confidence": 0.4}]},
            "memory": {"individuals": [{"id": "r1",
                                        "parent": "beagle.n.01"}]},
            "resolution": {"expression": "it", "referent": "r1"},
            "run": {"buffer": {"activation": {"table": {"beagle": 1.0}}},
                    "summary": {"doubts": [{"concept": "dog.n.01",
                                            "predicate": "swim",
                                            "relation": "capable_of",
                                            "reason": "inherited"}]}}}
        graph = graph_of(turn)
        nodes = {one["id"]: one for one in graph["nodes"]}
        self.assertEqual(nodes["beagle.n.01"]["label"], "beagle")
        self.assertEqual(nodes["beagle.n.01"]["activation"], 1.0)
        self.assertEqual(nodes["dog.n.01"]["rules"], ["R2"])
        self.assertTrue(nodes["r1"]["individual"])
        found = {(e["from"], e["relation"], e["to"], e["via"])
                 for e in graph["edges"]}
        self.assertIn(("r1", "is_a", "beagle.n.01", "walk"), found)
        self.assertIn(("dog.n.01", "capable_of", "swim", "walk"), found)
        self.assertIn(("dog.n.01", "capable_of", "swim", "run"), found)
        self.assertIn(("“it”", "refers to", "r1", "said"), found)

    def test_nothing_touched(self):
        from research.v697.graph import graph_of
        self.assertEqual(graph_of({}), {"nodes": [], "edges": []})


class StepTests(unittest.TestCase):
    def test_programmed_comes_before_answered(self):
        from research.v697.steps import steps_of
        turn = {"said": "x", "answer": {"outcome": "acted",
                                        "source": "programming",
                                        "code": {"answer": {
                                            "status": "confirmed"},
                                            "writers": [], "search": {
                                                "evaluated": 12}}}}
        names = [one["step"] for one in steps_of(turn, None)]
        self.assertLess(names.index("programmed"), names.index("answered"))
        line = next(one["line"] for one in steps_of(turn, None)
                    if one["step"] == "programmed")
        self.assertIn("searched 12 candidates", line)


@needs_node
class PageTests(unittest.TestCase):
    def test_the_page_script_parses(self):
        page = (HERE / "app.html").read_text(encoding="utf-8")
        script = page[page.index("<script>") + 8:page.rindex("</script>")]
        found = subprocess.run(["node", "--check", "-"], input=script,
                               capture_output=True, text=True,
                               encoding="utf-8")
        self.assertEqual(found.returncode, 0, found.stderr)


if __name__ == "__main__":
    unittest.main()
