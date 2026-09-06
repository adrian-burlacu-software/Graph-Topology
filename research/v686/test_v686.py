"""Regression suite. Run: python -m unittest research.v686.test_v686 -v

Data-dependent tests skip themselves when the norms have not been pulled in.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from research.v683.measure import measure
from research.v683.ordering import ORDERINGS, coverage, optimal
from research.v683.substrate import Corpus
from research.v684 import build
from research.v686 import corpora
from research.v686.identify import Identifier
from research.v686.server import DESCRIBES

STORE = build.DEFAULT_STORE.with_name("v684_reasoning_compressed.sqlite")
HAVE_NORMS = (corpora.XCSLB_DIR / "comps_base.jsonl").exists() and \
             (corpora.AWA2_DIR / "classes.txt").exists()
requires_norms = unittest.skipUnless(HAVE_NORMS, "feature norms not downloaded")
requires_store = unittest.skipUnless(STORE.exists(), f"no store at {STORE}")


@requires_norms
class CorpusTests(unittest.TestCase):
    """The norms, read as Appendix 3 corpora."""

    def test_awa2_is_dense_and_closed(self):
        awa = corpora.load_awa2()
        self.assertEqual(len(awa), 50)
        self.assertEqual(awa.predicates, 85)

    def test_xcslb_scale(self):
        xcslb = corpora.load_xcslb()
        self.assertEqual(len(xcslb), 521)
        self.assertGreater(xcslb.predicates, 3500)

    def test_the_matrix_misalignment_that_made_this_read_the_pairs(self):
        """`budgie` must carry bird properties, not `can be covered in lip balm`."""
        birds = dict(corpora.load_xcslb(category="bird").items)
        self.assertIn("budgie", birds)
        self.assertIn("can chirp", birds["budgie"])
        self.assertNotIn("can be covered in lip balm", birds["budgie"])
        self.assertIn("has a red breast", dict(corpora.load_xcslb().items)["robin"])

    def test_slicing_by_feature_type(self):
        visual = corpora.load_xcslb(kinds=("visual perceptual",))
        self.assertLess(visual.cells, corpora.load_xcslb().cells)
        types = corpora.feature_types()
        for _, properties in visual.items:
            for prop in properties:
                self.assertEqual(types.get(prop), "visual perceptual")

    def test_every_feature_ships_a_negation(self):
        """What v684's R3 never had: the denial of each property."""
        negated = corpora.negations()
        self.assertIn("has keys", negated)
        self.assertEqual(negated["has keys"], "does not have keys")

    def test_the_wordnet_join_exists(self):
        keys = corpora.senses()
        self.assertGreater(len(keys), 500)
        self.assertTrue(keys["robin"].startswith("robin%"))


@requires_norms
class CompressionTests(unittest.TestCase):
    """Appendix 3's trie, unmodified, on both corpora."""

    def nodes(self, corpus, ordering):
        return measure(corpus, ordering, ORDERINGS[ordering](corpus)).nodes

    def test_the_headline_numbers_are_reproducible(self):
        """Pinned because an earlier ad-hoc slice was not.

        `Counter.most_common` broke ties by insertion order, which varies with
        the hash seed, so a top-8 slice differed between runs. The full corpora
        do not: these numbers held across four hash seeds.
        """
        awa = corpora.load_awa2()
        self.assertEqual(self.nodes(awa, "adaptive_coverage"), 883)
        self.assertEqual(self.nodes(awa, "global_coverage"), 1177)
        self.assertEqual(self.nodes(awa, "lexical"), 1405)
        self.assertEqual(self.nodes(awa, "anti_coverage"), 1521)
        self.assertEqual(self.nodes(corpora.load_xcslb(), "adaptive_coverage"),
                         10794)

    def test_the_figure_beats_the_text_on_every_corpus(self):
        """Appendix 3's branch-local figure against its one-global-order text."""
        for corpus in (corpora.load_awa2(), corpora.load_xcslb(),
                       corpora.load_xcslb(category="bird"),
                       corpora.load_xcslb(kinds=("visual perceptual",))):
            adaptive = self.nodes(corpus, "adaptive_coverage")
            global_ = self.nodes(corpus, "global_coverage")
            self.assertLess(adaptive, global_, corpus.name)

    def test_coverage_ordering_beats_its_own_reversal(self):
        for corpus in (corpora.load_awa2(), corpora.load_xcslb()):
            self.assertLess(self.nodes(corpus, "global_coverage"),
                            self.nodes(corpus, "anti_coverage"), corpus.name)

    def test_dense_and_closed_compresses_far_better_than_sparse(self):
        awa = measure(corpora.load_awa2(), "a",
                      ORDERINGS["adaptive_coverage"](corpora.load_awa2()))
        xcslb = measure(corpora.load_xcslb(), "a",
                        ORDERINGS["adaptive_coverage"](corpora.load_xcslb()))
        self.assertGreater(awa.reuse_rate, 0.40)
        self.assertLess(xcslb.reuse_rate, 0.20)

    def test_branch_local_ordering_beats_the_best_global_one(self):
        """`optimal` is exhaustive over global orders; adaptive is not global.

        Tie-broken by name so the slice is the same on every run.
        """
        awa = corpora.load_awa2()
        counts = coverage(awa)
        top = {p for p, _ in sorted(counts.items(),
                                    key=lambda kv: (-kv[1], kv[0]))[:8]}
        trimmed = Corpus("awa2/top8", tuple(
            (i, frozenset(p for p in s if p in top)) for i, s in awa.items))
        trimmed = Corpus("awa2/top8", tuple(x for x in trimmed.items if x[1]))
        best_global = measure(trimmed, "optimal", optimal(trimmed)).nodes
        adaptive = self.nodes(trimmed, "adaptive_coverage")
        self.assertEqual(best_global, 42)
        self.assertEqual(adaptive, 34)
        self.assertLess(adaptive, best_global)

    def test_compression_improves_with_more_individuals(self):
        awa = corpora.load_awa2()
        small = Corpus("awa2[:12]", awa.items[:12])
        self.assertLess(
            measure(small, "a", ORDERINGS["adaptive_coverage"](small)).reuse_rate,
            measure(awa, "a", ORDERINGS["adaptive_coverage"](awa)).reuse_rate)


class RoutingTests(unittest.TestCase):
    """Which questions identification takes, and which it leaves alone."""

    def test_descriptions_are_taken(self):
        for question in ("what kind of dog has spots",
                         "what kind of cat has stripes",
                         "what is red and flies"):
            self.assertTrue(DESCRIBES.search(question), question)

    def test_naming_questions_are_left_to_v684_and_v685(self):
        for question in ("what can a violin player do",
                         "what does a dog owner need",
                         "what can a violin do", "is a dog an animal",
                         "can a dog fall into a hole",
                         "what is a hammer used for"):
            self.assertFalse(DESCRIBES.search(question), question)


@requires_norms
@requires_store
class IdentificationTests(unittest.TestCase):
    """Describe a thing; be told what it is."""

    @classmethod
    def setUpClass(cls):
        cls.identifier = Identifier(STORE)

    @classmethod
    def tearDownClass(cls):
        cls.identifier.close()

    def test_the_reported_question(self):
        found = self.identifier.identify("what kind of dog has spots")
        self.assertEqual(found.verdict, "IDENTIFIED")
        self.assertEqual([c.name for c in found.candidates], ["dalmatian"])
        self.assertIn("stated", found.candidates[0].matched["spots"])

    def test_it_generalises_to_other_classes(self):
        found = self.identifier.identify("what kind of cat has stripes")
        self.assertEqual([c.name for c in found.candidates], ["tiger"])

    def test_a_correct_ambiguity_is_reported_as_one(self):
        """Two birds are red, and both really are."""
        found = self.identifier.identify("what kind of bird is red")
        names = {c.name for c in found.candidates}
        self.assertEqual(found.verdict, "AMBIGUOUS")
        self.assertEqual(names, {"robin", "cockerel"})

    def test_stemming_matches_without_matching_too_much(self):
        """Prefix matching put cheetahs under stripes and desks under trunks."""
        self.assertEqual(Identifier.stem("flies"), "fly")
        self.assertEqual(Identifier.stem("spots"), "spot")
        self.assertNotEqual(Identifier.stem("striven"), Identifier.stem("stripes"))
        self.assertNotEqual(Identifier.stem("truncated"), Identifier.stem("trunk"))

    def test_the_class_constrains_but_does_not_describe(self):
        terms, among = self.identifier.terms_of("what kind of dog has spots")
        self.assertEqual(among, "dog")
        self.assertEqual(terms, ["spots"])

    def test_the_narrowing_is_recorded_step_by_step(self):
        found = self.identifier.identify("what kind of dog has spots")
        self.assertGreaterEqual(len(found.steps), 2)
        counts = [step["remaining"] for step in found.steps]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertEqual(counts[-1], 1)

    def test_awa2_classes_reach_the_taxonomy(self):
        """Without this join `dalmatian` is not a kind of dog and is invisible."""
        self.assertIn("dalmatian", self.identifier.synset)
        self.assertIn("tiger", self.identifier.synset)

    def test_nothing_matching_is_reported_as_absent(self):
        found = self.identifier.identify("what kind of dog has feathers")
        self.assertEqual(found.verdict, "NO_MATCH")
        self.assertIn("Absent", found.note)


if __name__ == "__main__":
    unittest.main()
