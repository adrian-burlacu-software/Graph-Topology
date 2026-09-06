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
from research.v686.identifiability import cue_validity, depths
from research.v686.identifiability import measure as identifiability
from research.v686.identify import Identifier

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


@requires_norms
class TradeoffTests(unittest.TestCase):
    """Storing and asking pull in opposite directions on the same trie."""

    def rows(self, corpus):
        plans = dict(ORDERINGS)
        plans["cue_validity"] = cue_validity
        return [identifiability(corpus, name, build(corpus))
                for name, build in plans.items()]

    def test_compression_and_questions_are_perfectly_rank_inverted(self):
        """rho = +1.000 on every corpus: an ordering, not a tendency."""
        for corpus in (corpora.load_awa2(), corpora.load_xcslb(),
                       corpora.load_buchanan()):
            rows = sorted(self.rows(corpus), key=lambda r: r.reuse)
            questions = [r.mean_depth for r in rows]
            self.assertEqual(questions, sorted(questions), corpus.name)

    def test_the_best_compressor_asks_the_most_questions(self):
        rows = {r.ordering: r for r in self.rows(corpora.load_awa2())}
        best = max(rows.values(), key=lambda r: r.reuse)
        fewest = min(rows.values(), key=lambda r: r.mean_depth)
        self.assertEqual(best.ordering, "adaptive_coverage")
        self.assertEqual(fewest.ordering, "anti_coverage")
        self.assertGreater(best.mean_depth, fewest.mean_depth * 3)

    def test_distinctiveness_ordering_is_anti_coverage(self):
        """McRae's distinctiveness is 1/(concepts carrying it), so ranking by
        it is ranking by ascending coverage -- which is already implemented."""
        awa = corpora.load_awa2()
        counts = coverage(awa)
        by_distinctiveness = sorted(counts, key=lambda p: (1 / counts[p], str(p)))
        by_anti = sorted(counts, key=lambda p: (-counts[p], str(p)))
        self.assertEqual(by_distinctiveness, by_anti)

    def test_identification_depth_is_the_unique_prefix(self):
        corpus = Corpus("tiny", (
            ("a", frozenset({"x", "y"})),
            ("b", frozenset({"x", "z"})),
            ("c", frozenset({"w"}))))
        found, never = depths([("a", ("x", "y")), ("b", ("x", "z")),
                               ("c", ("w",))])
        self.assertEqual(never, 0)
        self.assertEqual(sorted(found), [1, 2, 2])

    def test_identical_predicate_sets_are_reported_not_scored(self):
        found, never = depths([("a", ("x",)), ("b", ("x",))])
        self.assertEqual(never, 2)
        self.assertEqual(found, [])

    def test_the_only_thing_buchanan_cannot_tell_apart_is_a_synonym_pair(self):
        rows = {r.ordering: r for r in self.rows(corpora.load_buchanan())}
        self.assertEqual(rows["anti_coverage"].never_unique, 2)


@requires_norms
@requires_store
class RoutingTests(unittest.TestCase):
    """Which questions identification takes, and which it leaves alone.

    Routing is grammatical, not a pattern list: a description leaves the thing
    unnamed and says what it is like -- a relative clause, an adjectival
    complement, or `what` used as a determiner. A naming question has a
    subject and asks what it does, and belongs to v684.
    """

    @classmethod
    def setUpClass(cls):
        cls.identifier = Identifier(STORE)

    @classmethod
    def tearDownClass(cls):
        cls.identifier.close()

    def test_descriptions_are_taken(self):
        for question in ("what kind of dog has spots",
                         "what kind of cat has stripes",
                         "what is an object that is round with spots",
                         "what is round with spots",
                         "what animal has stripes",
                         "which animal is big and furry",
                         "what is a bird that is red"):
            self.assertTrue(self.identifier.describes(question), question)

    def test_naming_questions_are_left_to_v684_and_v685(self):
        for question in ("what can a violin player do",
                         "what does a dog owner need",
                         "what does a dog's owner need",
                         "what can a violin do", "is a dog an animal",
                         "can a dog fall into a hole", "what is a dog",
                         "where do you find a hammer",
                         "what is a hammer used for"):
            self.assertFalse(self.identifier.describes(question), question)

    def test_the_class_is_read_from_three_shapes(self):
        for question, among, terms in (
                ("what kind of dog has spots", "dog", ["spots"]),
                ("what is an object that is round with spots", "object",
                 ["round", "spots"]),
                ("what animal has stripes", "animal", ["stripes"])):
            got_terms, got_among = self.identifier.terms_of(question)
            self.assertEqual(got_among, among, question)
            self.assertEqual(sorted(got_terms), sorted(terms), question)

    def test_an_unknown_class_is_dropped_rather_than_searched_for(self):
        self.assertIsNone(self.identifier.terms_of("what zzzqqq has stripes")[1])

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

    def test_the_narrowing_has_depth_when_several_properties_are_asked(self):
        """One property is one level; the funnel needs more than that.

        The page drew a flat two-row picture for "dog has spots" because
        every candidate was placed at depth 1 whatever eliminated it. Depth
        is now *when* a candidate fell out.
        """
        found = self.identifier.identify(
            "what kind of animal has stripes and eats meat and is fast")
        self.assertEqual(found.verdict, "IDENTIFIED")
        self.assertEqual([c.name for c in found.candidates], ["tiger"])
        depths = {entry["depth"] for entry in found.considered}
        self.assertGreaterEqual(len(depths), 3)
        survivor = next(e for e in found.considered if e["survived"])
        self.assertEqual(survivor["depth"], max(depths))

    def test_buchanan_scales_the_norms_up(self):
        buchanan = corpora.load_buchanan()
        self.assertGreater(len(buchanan), 3500)
        self.assertGreater(buchanan.cells, 24000)

    def test_root_forms_are_not_two_predicates(self):
        """`leaving` and `leave` are one property, not two."""
        rooted = corpora.load_buchanan(root_forms=True)
        surface = corpora.load_buchanan(root_forms=False)
        self.assertLess(rooted.predicates, surface.predicates / 2)

    def test_each_attribute_owns_the_rivals_it_removed(self):
        """`depth` is the index of the step that removed the candidate.

        Counting from the terms alone is off by one exactly when a class was
        named, which drew the rivals of `stripes` hanging off `animal`.
        """
        found = self.identifier.identify(
            "what kind of animal has stripes and eats meat and is fast")
        terms = [step["term"] for step in found.steps]
        self.assertEqual(terms[0], "animal")           # the class step
        by_step = {}
        for entry in found.considered:
            if not entry["survived"]:
                by_step.setdefault(entry["depth"], []).append(entry["name"])
        self.assertNotIn(0, by_step)                   # the class removed none
        removed_by_stripes = by_step.get(terms.index("stripes"), [])
        self.assertTrue(removed_by_stripes)
        for name in removed_by_stripes:
            self.assertNotIn("stripes", self.identifier.stated.get(name, ()))

    def test_the_first_attribute_owns_rivals_at_depth_zero(self):
        """Depth is a step index, so the first attribute's rivals are depth 0.

        The page read that as `entry.depth || 1`, and zero is falsy, so every
        rival of the first attribute was drawn on the second: `round` appeared
        to rule out nothing and `hexagons` carried ten.
        """
        found = self.identifier.identify("what is round with hexagons")
        self.assertEqual([step["term"] for step in found.steps],
                         ["round", "hexagons"])
        depths = {e["depth"] for e in found.considered if not e["survived"]}
        self.assertIn(0, depths)
        self.assertIn(1, depths)

    def test_properties_are_taken_general_before_specific(self):
        """Broadest first, so each step narrows visibly instead of the first
        one answering the whole question."""
        found = self.identifier.identify("what is round with hexagons")
        remaining = [step["remaining"] for step in found.steps]
        self.assertEqual(remaining, sorted(remaining, reverse=True))
        self.assertGreater(remaining[0], remaining[-1])

    def test_rivals_are_the_nearest_misses_not_the_alphabet(self):
        """Sorting by name gave `ambulance, accordion, antelope`."""
        found = self.identifier.identify("what is round with hexagons")
        self.assertEqual([c.name for c in found.candidates], ["football"])
        rivals = {e["name"] for e in found.considered if not e["survived"]}
        self.assertTrue(rivals & {"ball", "balloon", "frisbee"}, rivals)
        self.assertNotIn("ambulance", rivals)
        self.assertNotIn("accordion", rivals)

    def test_nothing_matching_is_reported_as_absent(self):
        found = self.identifier.identify("what kind of dog has feathers")
        self.assertEqual(found.verdict, "NO_MATCH")
        self.assertIn("Absent", found.note)


if __name__ == "__main__":
    unittest.main()
