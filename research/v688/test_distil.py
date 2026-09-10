"""Tests for norm distillation and the R19 simulation.

No model is loaded. What can be checked without a GPU is the apparatus: that
the gold matrix is closed, that the simulation's arithmetic is R19's, that
the free-listing arm joins to features that exist, and that the 50 classes
resolve to animals rather than to people and pointing devices. What cannot be
checked here -- whether SmolLM3's norms are any good -- is what
`--validate` and `--corroborate` measure and what `AUDIT.md` §17 records.

`R19sConstants` is the one that will fail first and should. `norms.py` copies
`FLOOR` and `MIN_KINDS` out of `profile.py` so the simulation can run without
building a reasoner, and a copied constant that drifts is a simulation of a
rule nobody has.
"""
from __future__ import annotations

import csv
import unittest

from research.v688 import norms, screen


class R19sConstants(unittest.TestCase):
    """The copies must equal the originals or the simulation is fiction."""

    def test_floor_and_minimum_match_profile(self):
        from research.v687 import profile

        self.assertEqual(norms.FLOOR, profile.CORROBORATION_FLOOR)
        self.assertEqual(norms.MIN_KINDS, profile.CORROBORATION_MIN_KINDS)


class TheGoldMatrixIsClosed(unittest.TestCase):
    """The whole experiment rests on AwA2 stating what is false."""

    def test_every_class_is_scored_on_every_attribute(self):
        cells = norms.gold()
        self.assertEqual(len(norms.classes()), 50)
        self.assertEqual(len(norms.attributes()), 83)
        self.assertEqual(len(cells), 50 * 83)

    def test_it_carries_both_answers(self):
        cells = norms.gold()
        self.assertTrue(any(cells.values()))
        self.assertTrue(any(not held for held in cells.values()))

    def test_the_binary_matrix_agrees_with_the_continuous_one(self):
        """The published threshold sits near 20.8; a cell far either side of
        it must land on the matching side of the binary split."""
        binary, scores = norms.gold(), norms.confidence_of_gold()
        for key, score in scores.items():
            if score <= 5.0:
                self.assertFalse(binary[key], key)
            elif score >= 60.0:
                self.assertTrue(binary[key], key)

    def test_the_two_excluded_attributes_are_the_geographic_ones(self):
        every = norms.read(norms.AWA2 / "predicates.txt")
        self.assertEqual(sorted(set(every) - set(norms.attributes())),
                         ["newworld", "oldworld"])


class TheQuestions(unittest.TestCase):
    """One phrasing, shared with the screen, so a judgement caches once."""

    def test_the_grid_phrases_through_screen(self):
        self.assertEqual(norms.question("beaver", "flys"), "can a beaver fly")
        self.assertEqual(norms.question("antelope", "black"),
                         "is an antelope black")

    def test_every_cell_has_a_question(self):
        blank = [(name, attribute) for name in norms.classes()
                 for attribute in norms.attributes()
                 if not norms.question(name, attribute)]
        self.assertEqual(blank, [])

    def test_the_attributes_are_the_screens(self):
        self.assertTrue(set(norms.attributes()) <= set(screen.ATTRIBUTES))


class TheFreeListingArm(unittest.TestCase):
    """The comparison must join to features XCSLB actually has."""

    @classmethod
    def setUpClass(cls) -> None:
        path = norms.ROOT / "data" / "xcslb" / "feature_lexicon.csv"
        with path.open(encoding="utf-8") as handle:
            cls.lexicon = {row["feature"] for row in csv.DictReader(handle)}

    def test_every_mapped_feature_exists(self):
        missing = [(attribute, feature)
                   for attribute, features in norms.EQUIVALENT.items()
                   for feature in features if feature not in self.lexicon]
        self.assertEqual(missing, [])

    def test_it_maps_only_attributes_awa2_has(self):
        stray = set(norms.EQUIVALENT) - set(norms.attributes())
        self.assertEqual(stray, set())

    def test_it_is_a_subset_and_says_so(self):
        """It covers a minority of the grid on purpose; the report labels the
        free-listing arm as a subset because of this."""
        self.assertLess(len(norms.EQUIVALENT), len(norms.attributes()) / 2)

    def test_free_listing_misses_most_of_what_is_true(self):
        """The finding this file was built on: people list what is
        distinctive, not what is typical."""
        report = norms.free_listing_density()
        self.assertGreater(report["awa2_says_yes"], 100)
        self.assertLess(report["recall_of_free_listing"], 0.5)


class TheSenses(unittest.TestCase):
    """`sheep` is not a docile person and `mouse` is not a pointing device."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.resolved = norms.synset_of(norms.classes())

    def test_every_class_resolves(self):
        missing = [name for name in norms.classes()
                   if name not in self.resolved]
        self.assertEqual(missing, [])

    def test_the_traps_resolve_to_animals(self):
        for name, wrong in (("sheep", "person"), ("mouse", "device"),
                            ("seal", "stamp"), ("bat", "baseball"),
                            ("mole", "spy")):
            with self.subTest(name):
                self.assertIn(name, self.resolved)
                self.assertNotIn(wrong, self.resolved[name])

    def test_nothing_resolves_above_animal(self):
        """A class that resolved to `entity` would silently join everything."""
        for name, concept in self.resolved.items():
            with self.subTest(name):
                self.assertNotIn(concept, ("entity.n.01", "object.n.01",
                                           "whole.n.02", "organism.n.01"))


class TheR19Simulation(unittest.TestCase):
    """`verdict` is R19's arithmetic and nothing else."""

    def source(self, **holds):
        return {(name, "furry"): value for name, value in holds.items()}

    def test_it_refuses_to_speak_below_the_minimum(self):
        members = [f"c{i}" for i in range(7)]
        source = self.source(**{name: True for name in members})
        self.assertIsNone(norms.verdict(members, "furry", source))

    def test_a_third_is_enough(self):
        members = [f"c{i}" for i in range(9)]
        source = self.source(**{name: index < 3
                                for index, name in enumerate(members)})
        self.assertEqual(norms.verdict(members, "furry", source), "believed")

    def test_below_a_third_is_refused(self):
        members = [f"c{i}" for i in range(9)]
        source = self.source(**{name: index < 2
                                for index, name in enumerate(members)})
        self.assertEqual(norms.verdict(members, "furry", source), "refused")

    def test_a_class_the_source_does_not_cover_leaves_the_denominator(self):
        """`profile.corroboration` draws its kinds from the norms, so an
        uncovered class is absent rather than counted as not bearing it."""
        members = [f"c{i}" for i in range(12)]
        partial = self.source(**{name: True for name in members[:8]})
        self.assertEqual(norms.verdict(members, "furry", partial), "believed")
        thin = self.source(**{name: True for name in members[:7]})
        self.assertIsNone(norms.verdict(members, "furry", thin))

    def test_sparsity_alone_can_flip_the_verdict(self):
        """The finding, as arithmetic: same truth, fewer listings."""
        members = [f"c{i}" for i in range(12)]
        dense = self.source(**{name: True for name in members})
        sparse = dict(dense)
        for name in members[3:]:
            sparse[(name, "furry")] = False       # nobody mentioned it
        self.assertEqual(norms.verdict(members, "furry", dense), "believed")
        self.assertEqual(norms.verdict(members, "furry", sparse), "refused")


class TheTally(unittest.TestCase):
    """Losing a true inheritance and believing a false one are not the same."""

    def test_it_separates_the_two_errors(self):
        members = [f"c{i}" for i in range(9)]
        truth = {(name, "furry"): True for name in members}
        truth.update({(name, "flys"): False for name in members})
        wrong = {(name, "furry"): False for name in members}
        wrong.update({(name, "flys"): True for name in members})
        scored = norms.tally({"a": members}, ["furry", "flys"], truth, wrong)
        self.assertEqual(scored["pairs"], 2)
        self.assertEqual(scored["counts"]["lost"], 1)
        self.assertEqual(scored["counts"]["over"], 1)
        self.assertEqual(scored["agreed"], 0.0)

    def test_a_source_that_cannot_speak_is_silent_not_wrong(self):
        members = [f"c{i}" for i in range(9)]
        truth = {(name, "furry"): True for name in members}
        scored = norms.tally({"a": members}, ["furry"], truth, {})
        self.assertEqual(scored["silent"], 1.0)
        self.assertEqual(scored["agreed"], 0.0)


if __name__ == "__main__":
    unittest.main()
