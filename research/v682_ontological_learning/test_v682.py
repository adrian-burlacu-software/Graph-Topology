import json
import tempfile
import unittest
from pathlib import Path

from research.v682_ontological_learning.ontology import Ontology, build_demo_ontology
from research.v682_ontological_learning.run_v682 import run


class V682OntologyTests(unittest.TestCase):
    def setUp(self):
        self.ontology = build_demo_ontology()

    def test_dog_organism_is_inferred_without_changing_is_a_semantics(self):
        result = self.ontology.query("dog", "type", "organism")
        self.assertEqual(result.status, "INFERRED")
        self.assertEqual(result.proof.rule, "type + is_a -> type")
        self.assertEqual(self.ontology.query("dog", "is_a", "organism").status, "UNVERIFIED")

    def test_deep_type_propagation_and_property_inheritance(self):
        self.assertEqual(self.ontology.query("dog", "type", "organism").status, "INFERRED")
        self.assertEqual(self.ontology.query("dog", "has_property", "warm_blooded").status, "INFERRED")
        self.assertEqual(self.ontology.query("dog", "has_property", "living").status, "INFERRED")

    def test_is_a_transitivity_generalizes(self):
        ontology = Ontology()
        ontology.add_fact("a", "is_a", "b")
        ontology.add_fact("b", "subclass_of", "c")
        self.assertEqual(ontology.query("a", "is_a", "c").status, "INFERRED")

    def test_aliases_are_semantic_and_normalized(self):
        self.assertEqual(self.ontology.query("dog", "type_of", "mammal").status, "DIRECT")
        self.assertEqual(self.ontology.query("dog", "has_attribute", "warm_blooded").status, "INFERRED")
        self.assertEqual(self.ontology.query("mammal", "hypernym", "organism").status, "INFERRED")

    def test_negative_queries_do_not_invent_relationships(self):
        self.assertEqual(self.ontology.query("dog", "type", "mineral").status, "UNVERIFIED")
        self.assertEqual(self.ontology.query("mammal", "type", "plant").status, "UNVERIFIED")

    def test_grounded_questions_are_graph_queries(self):
        for question in ("is dog an organism?", "so is dog an organism?"):
            result = self.ontology.query_natural_language(question)
            self.assertEqual(result.status, "INFERRED")
            self.assertEqual(result.fact.relation, "type")

    def test_run_writes_inspectable_actual_ontology_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = run(output)
            self.assertEqual(result["passed"], result["total"])
            for name in ("ontology.json", "inferred_facts.json", "proofs.json", "evaluation.json", "ontology.html"):
                self.assertTrue((output / name).is_file())
            facts = json.loads((output / "inferred_facts.json").read_text(encoding="utf-8"))
            dog_organism = next(item for item in facts if item["subject"] == "dog" and item["object"] == "organism")
            self.assertEqual(dog_organism["proof"]["kind"], "INFERRED")
            html = (output / "ontology.html").read_text(encoding="utf-8")
            self.assertIn('"kind": "INFERRED"', html)
            self.assertIn("Focused entity", html)
