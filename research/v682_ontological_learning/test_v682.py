import json
import tempfile
import unittest
from pathlib import Path

from research.v682_ontological_learning.ontology import SemanticGraph
from research.v682_ontological_learning.run_v682 import DEFAULT_DATABASE, run


class V682RealGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = SemanticGraph(DEFAULT_DATABASE)
        cls.discovery = cls.graph.discover_rules()
        cls.accepted = [rule for rule in cls.discovery["rules"] if rule["status"] == "ACCEPTED"]

    def test_loads_real_focused_database_and_all_direct_relationships(self):
        self.assertEqual(self.graph.database, DEFAULT_DATABASE.resolve())
        self.assertEqual(len(self.graph.edges), 4276)
        self.assertGreater(len(self.graph.relations), 20)
        self.assertIn("related_to", self.graph.relations)
        self.assertIn("is_a", self.graph.relations)
        self.assertTrue(all(edge.source for edge in self.graph.edges))

    def test_discovers_all_relation_pair_composition_candidates(self):
        self.assertGreater(self.discovery["observed_two_hop_paths"], len(self.graph.edges))
        self.assertGreater(self.discovery["candidate_rules"], 0)
        self.assertTrue(self.accepted)
        self.assertTrue(all(rule["status"] in {"ACCEPTED", "REJECTED"} for rule in self.discovery["rules"]))
        self.assertTrue(all("testing" in rule and "contradictions" in rule for rule in self.discovery["rules"]))

    def test_inference_uses_only_accepted_empirical_rules_with_proofs(self):
        inferred = self.graph.infer(self.accepted)
        self.assertTrue(inferred)
        self.assertTrue(all(fact not in self.graph.direct for fact in inferred))
        self.assertTrue(all(proof.kind == "INFERRED" and proof.premises for proof in inferred.values()))

    def test_clean_graph_is_a_smaller_canonical_projection_with_provenance(self):
        clean = self.graph.build_clean_graph(self.discovery)
        self.assertLess(len(clean.nodes), len(self.graph.nodes))
        self.assertLess(len(clean.edges), len(self.graph.edges))
        dog = clean.resolve_node("dog")
        self.assertIsNotNone(dog)
        self.assertIn("dog", [clean.nodes[dog]["label"], *clean.nodes[dog]["aliases"]])
        self.assertTrue(all(edge.source for edge in clean.edges))

    def test_grounded_dog_query_is_evidence_backed_when_real_graph_supports_it(self):
        result = self.graph.query_natural_language("is dog a mammal?", self.accepted)
        self.assertEqual(result["status"], "VERIFIED")
        self.assertEqual(result["query"]["subject"], "en:dog")
        follow_up = self.graph.query_natural_language("so is dog a mammal?", self.accepted)
        self.assertEqual(follow_up["status"], "VERIFIED")

    def test_run_generates_real_graph_artifacts_and_interactive_canvas(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = run(DEFAULT_DATABASE, output)
            self.assertEqual(result["stats"]["source_database"], str(DEFAULT_DATABASE.resolve()))
            for name in (
                "clean_graph.json", "concepts.json", "relations.json", "rules.json",
                "inferred_facts.json", "proofs.json", "evaluation.json", "knowledge_globe.html",
            ):
                self.assertTrue((output / name).is_file())
            rules = json.loads((output / "rules.json").read_text(encoding="utf-8"))
            self.assertEqual(rules["candidate_rules"], result["discovery"]["candidate_rules"])
            clean = json.loads((output / "clean_graph.json").read_text(encoding="utf-8"))
            self.assertTrue(clean["stats"]["raw_database_mutated"] is False)
            self.assertLess(clean["stats"]["canonical_direct_relationships"],
                            clean["stats"]["raw_edges_considered"])
            page = (output / "knowledge_globe.html").read_text(encoding="utf-8")
            self.assertIn("Interactive 3D clean semantic graph", page)
            self.assertIn("Canonical concepts and verified relationships", page)
            self.assertIn('"aliases"', page)
