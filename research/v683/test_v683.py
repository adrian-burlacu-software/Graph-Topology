"""Regression suite. Run: python -m unittest research.v683.test_v683 -v"""
from __future__ import annotations

import unittest

from research.v683 import measure, ordering, substrate
from research.v683.substrate import PAPER_TABLE_1
from research.v683.trie import ROOT, PredicateTrie


class TrieTests(unittest.TestCase):
    def test_ensure_allocates_once_then_reuses(self):
        trie = PredicateTrie()
        first, allocated = trie.ensure(ROOT, "Big")
        second, again = trie.ensure(ROOT, "Big")
        self.assertTrue(allocated)
        self.assertFalse(again)
        self.assertEqual(first, second)
        self.assertEqual((trie.allocated, trie.reused), (1, 1))

    def test_same_predicate_in_two_branches_is_two_nodes(self):
        """Figure 20 draws `Spots` twice; node identity is the path, not the symbol."""
        trie = PredicateTrie()
        trie.insert("Dalmatian", ("Big", "Spots"))
        trie.insert("Jack Russell Terrier", ("Spots",))
        under_big = trie.lookup(("Big", "Spots"))
        under_root = trie.lookup(("Spots",))
        self.assertNotEqual(under_big, under_root)
        self.assertEqual(trie.node_count, 3)

    def test_lookup_of_absent_route_is_none(self):
        trie = PredicateTrie()
        trie.insert("Briard", ("Big", "Long Hair"))
        self.assertIsNone(trie.lookup(("Big", "Spots")))
        self.assertIsNone(trie.lookup(("Loud",)))

    def test_path_round_trips(self):
        trie = PredicateTrie()
        result = trie.insert("Saint Bernard", ("Big", "Long Hair", "Loud"))
        self.assertEqual(trie.path(result.node), ("Big", "Long Hair", "Loud"))
        self.assertEqual(trie.individuals_at(result.node), ("Saint Bernard",))
        self.assertEqual(result.depth, 3)


class PaperFigure20Tests(unittest.TestCase):
    """Appendix 3 is the specification; Figure 20 is the expected output."""

    EXPECTED = {
        "Saint Bernard": ("Big", "Long Hair", "Loud"),
        "Briard": ("Big", "Long Hair"),
        "Dalmatian": ("Big", "Spots"),
        "Border Collie": ("Big",),
        "Jack Russell Terrier": ("Spots",),
    }

    def test_coverage_counts_match_table_1(self):
        counts = ordering.coverage(PAPER_TABLE_1)
        self.assertEqual(
            dict(counts), {"Big": 4, "Long Hair": 2, "Spots": 2, "Loud": 1}
        )

    def test_adaptive_ordering_reproduces_figure_20(self):
        plan = dict(ordering.adaptive_coverage(PAPER_TABLE_1))
        self.assertEqual(plan, self.EXPECTED)

    def test_global_ordering_also_reproduces_figure_20(self):
        """The paper's example is too small to separate its two descriptions."""
        plan = dict(ordering.global_coverage(PAPER_TABLE_1))
        self.assertEqual(plan, self.EXPECTED)

    def test_figure_20_is_optimal_at_five_nodes(self):
        best = measure.measure(PAPER_TABLE_1, "optimal", ordering.optimal(PAPER_TABLE_1))
        self.assertEqual(best.nodes, 5)
        for name in ("adaptive_coverage", "global_coverage"):
            result = measure.measure(
                PAPER_TABLE_1, name, ordering.ORDERINGS[name](PAPER_TABLE_1)
            )
            self.assertEqual(result.nodes, best.nodes)

    def test_table_2_column_order_is_the_depth_first_walk(self):
        """Table 2's columns read Big, Long Hair, Loud, Spots -- the trie's own walk."""
        trie = measure.store(PAPER_TABLE_1, ordering.adaptive_coverage(PAPER_TABLE_1))
        walk: list[str] = []

        def visit(node: int) -> None:
            for symbol, child in sorted(
                trie.children(node).items(), key=lambda item: item[1]
            ):
                if symbol not in walk:
                    walk.append(symbol)
                visit(child)

        visit(ROOT)
        self.assertEqual(walk, ["Big", "Long Hair", "Loud", "Spots"])


class OrderingTests(unittest.TestCase):
    def test_every_ordering_is_a_permutation_of_the_predicate_set(self):
        """An ordering may reorder predicates; it may never add or drop one."""
        for name, fn in ordering.ORDERINGS.items():
            for individual, path in fn(PAPER_TABLE_1):
                original = dict(PAPER_TABLE_1.items)[individual]
                self.assertEqual(set(path), set(original), name)
                self.assertEqual(len(path), len(set(path)), name)

    def test_orderings_are_deterministic(self):
        for name, fn in ordering.ORDERINGS.items():
            self.assertEqual(fn(PAPER_TABLE_1), fn(PAPER_TABLE_1), name)

    def test_anti_coverage_is_never_cheaper_than_coverage(self):
        results = {
            name: measure.measure(PAPER_TABLE_1, name, fn(PAPER_TABLE_1)).nodes
            for name, fn in ordering.ORDERINGS.items()
        }
        self.assertLessEqual(results["global_coverage"], results["anti_coverage"])

    def test_optimal_refuses_intractable_input(self):
        corpus = substrate.Corpus(
            "wide", tuple((f"i{n}", frozenset({f"p{n}"})) for n in range(12))
        )
        with self.assertRaises(ValueError):
            ordering.optimal(corpus)


class MeasurementTests(unittest.TestCase):
    def test_reuse_rate_is_relative_to_flat_storage(self):
        result = measure.measure(
            PAPER_TABLE_1, "adaptive_coverage",
            ordering.adaptive_coverage(PAPER_TABLE_1),
        )
        self.assertEqual(result.flat_cells, 9)
        self.assertEqual(result.nodes, 5)
        self.assertAlmostEqual(result.reuse_rate, 1 - 5 / 9, places=6)

    def test_every_individual_is_reachable_after_storage(self):
        plan = ordering.adaptive_coverage(PAPER_TABLE_1)
        trie = measure.store(PAPER_TABLE_1, plan)
        for individual, path in plan:
            node = trie.lookup(path)
            self.assertIsNotNone(node, individual)
            self.assertIn(individual, trie.individuals_at(node))

    def test_depth_is_invariant_across_orderings(self):
        """Ordering permutes an individual's predicates; it cannot change how
        many it has. The report leans on this, so it is asserted."""
        depths = {
            name: measure.measure(PAPER_TABLE_1, name, fn(PAPER_TABLE_1)).mean_depth
            for name, fn in ordering.ORDERINGS.items()
        }
        self.assertEqual(len(set(depths.values())), 1, depths)

    def test_median_resists_an_outlier_that_moves_the_mean(self):
        """`en:person` carries 6,231 predicates; H3 must not ride on it."""
        corpus = substrate.Corpus(
            "skewed",
            tuple([("hub", frozenset(f"p{n}" for n in range(400)))]
                  + [(f"i{n}", frozenset({"shared"})) for n in range(99)]),
        )
        result = measure.measure(
            corpus, "global_coverage", ordering.global_coverage(corpus)
        )
        self.assertEqual(result.median_depth, 1)
        self.assertGreater(result.mean_depth, 4)

    def test_scaling_curve_grows_monotonically(self):
        curve = measure.scaling_curve(
            PAPER_TABLE_1, ordering.adaptive_coverage, steps=5
        )
        self.assertEqual(curve[-1].individuals, len(PAPER_TABLE_1))
        sizes = [point.individuals for point in curve]
        self.assertEqual(sizes, sorted(sizes))

    def test_scaling_curve_samples_rather_than_taking_an_ordered_prefix(self):
        """A prefix of a name-sorted corpus is a sample of the dictionary.

        Here the first half of the corpus carries one predicate each and the
        second half carries five. An ordered prefix sees only the shallow half
        and reports mean depth 1.0; a real sample sees both and lands near 3.
        """
        corpus = substrate.Corpus(
            "front_loaded",
            tuple(
                [(f"a{n:03d}", frozenset({f"p{n}"})) for n in range(100)]
                + [(f"z{n:03d}", frozenset(f"q{n}_{k}" for k in range(5)))
                   for n in range(100)]
            ),
        )
        curve = measure.scaling_curve(corpus, ordering.global_coverage, steps=4)
        self.assertEqual(curve[0].individuals, 50)
        self.assertGreater(
            curve[0].mean_depth, 2.0,
            "first scaling step saw only the shallow half of the corpus",
        )


if __name__ == "__main__":
    unittest.main()
