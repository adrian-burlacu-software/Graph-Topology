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

    def test_the_floor_is_enough(self):
        """Written against the constant, not against a third. §19 moved the
        floor and a test that spells the number out tests the number."""
        members = [f"c{i}" for i in range(10)]
        enough = int(norms.FLOOR * 10) + 1
        source = self.source(**{name: index < enough
                                for index, name in enumerate(members)})
        self.assertEqual(norms.verdict(members, "furry", source), "believed")

    def test_below_the_floor_is_refused(self):
        members = [f"c{i}" for i in range(10)]
        under = max(int(norms.FLOOR * 10) - 1, 0)
        source = self.source(**{name: index < under
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


class ThePlainForm(unittest.TestCase):
    """A sense-tagged object is not something to put to a model."""

    def test_it_strips_a_synset_suffix(self):
        from research.v688 import densify

        self.assertEqual(densify.plain("animal tissue.n.01"), "animal tissue")
        self.assertEqual(densify.plain("fly.v.01"), "fly")

    def test_it_leaves_free_text_alone(self):
        from research.v688 import densify

        self.assertEqual(densify.plain("hunt at night"), "hunt at night")
        self.assertEqual(densify.plain("3.5 inches"), "3.5 inches")
        self.assertEqual(densify.plain(""), "")

    def test_the_question_reads_as_english(self):
        from research.v688 import densify

        self.assertEqual(densify.question("leopard", "capable_of", "hunt"),
                         "can a leopard hunt")
        self.assertEqual(densify.question("bat", "has_a",
                                          "animal tissue.n.01"),
                         "does a bat have animal tissue")
        self.assertEqual(densify.question("owl", "has_property", "nocturnal"),
                         "is an owl nocturnal")


class WhatIsStoredIsFoundAgain(unittest.TestCase):
    """§17's condition 3: `_hit` has to reach a distilled norm."""

    def test_the_stored_predicate_contains_the_term(self):
        from research.v687.identify import Identifier
        from research.v688 import densify, teacher

        for relation, obj, term in (("has_a", "wing", "wings"),
                                    ("capable_of", "hunt at night", "hunt"),
                                    ("has_property", "nocturnal", "nocturnal"),
                                    ("has_a", "animal tissue.n.01", "tissue")):
            with self.subTest(obj):
                predicate = teacher.stated(relation, densify.plain(obj))
                self.assertIsNotNone(
                    Identifier._hit(term, frozenset({predicate})),
                    f"{term!r} did not reach {predicate!r}")

    def test_a_bare_term_would_have_worked_too_but_reads_badly(self):
        """Recorded because it is the tempting shortcut: `_hit` matches any
        word, so the term alone matches -- and then the profile display and
        the trie carry `wing` as if somebody had said it."""
        from research.v687.identify import Identifier

        self.assertIsNotNone(Identifier._hit("wings", frozenset({"wing"})))


class TheCorroborationChange(unittest.TestCase):
    """R19 with a second evidence base. The denominator must not move."""

    class Stub:
        """Enough of `Profiles` to run `corroboration` without a store."""

        def __init__(self, stated, distilled):
            from research.v687.identify import Identifier

            self.stated = stated
            self.distilled = distilled
            self.identifier = Identifier
            self._ancestors = {name: {"bird.n.01"} for name in stated}
            # `corroboration` consults the distilled witnesses too; none
            # here, so this class still measures the norms half alone.
            self.distilled_kinds: dict = {}
            self._kind_lineage: dict = {}

        def _lineage(self):
            return self._ancestors

        def witnesses(self, ancestor, term):
            """The real one, so this stub exercises the real arithmetic."""
            from research.v687.profile import Profiles

            return Profiles.witnesses(self, ancestor, term)

        def bears(self, term, predicates):
            from research.v687.profile import Profiles

            return Profiles.bears(self, term, predicates)

    def run_it(self, stated, distilled, term="fly"):
        from research.v687.profile import Profiles

        stub = self.Stub(stated, distilled)
        return Profiles.corroboration(stub, "bird.n.01", term)

    #: A kind must carry *something* -- `corroboration` skips a concept
    #: whose norms are empty, because `stated.get(name)` is falsy for an
    #: empty frozenset. So the fixtures give every kind one filler property.
    FILLER = frozenset({"exists"})

    def test_distilled_evidence_adds_bearing(self):
        stated = {f"b{i}": self.FILLER for i in range(9)}
        stated["b0"] = frozenset({"can fly"})
        bare, _ = self.run_it(stated, {})
        dense, _ = self.run_it(
            stated, {f"b{i}": frozenset({"can fly"}) for i in range(1, 6)})
        self.assertEqual(bare, 1)
        self.assertEqual(dense, 6)

    def test_the_denominator_is_untouched(self):
        """Distillation adds properties to kinds the norms already cover; it
        must not invent kinds, or the ratio moves for two reasons at once."""
        stated = {f"b{i}": self.FILLER for i in range(9)}
        _, without = self.run_it(stated, {})
        _, with_extra = self.run_it(
            stated, {"stranger": frozenset({"can fly"}),
                     "b1": frozenset({"can fly"})})
        self.assertEqual(without, 9)
        self.assertEqual(with_extra, 9)

    def test_a_concept_the_norms_do_not_cover_is_not_a_kind(self):
        stated = {f"b{i}": frozenset({"x"}) for i in range(4)}
        stated["empty"] = frozenset()
        _, kinds = self.run_it(stated, {})
        self.assertEqual(kinds, 4)

    def test_sparsity_refuses_where_density_believes(self):
        """§17 as arithmetic, through the real method."""
        stated = {f"b{i}": self.FILLER for i in range(9)}
        stated["b0"] = frozenset({"can fly"})
        bearing, kinds = self.run_it(stated, {})
        self.assertLess(bearing / kinds, 1 / 3)          # refused
        dense = {f"b{i}": frozenset({"can fly"}) for i in range(7)}
        bearing, kinds = self.run_it(stated, dense)
        self.assertGreaterEqual(bearing / kinds, 1 / 3)  # believed


class TheProvenanceStaysSeparable(unittest.TestCase):
    """Condition 1 of §17, asserted rather than trusted."""

    def test_identify_does_not_merge_distilled_norms(self):
        """If it ever does, the trie, the profile display and the audit's
        `shipped` control all silently change meaning."""
        import inspect

        from research.v687 import identify

        source = inspect.getsource(identify)
        self.assertNotIn("load_distilled", source)

    def test_a_missing_file_is_not_an_error(self):
        from research.v687 import corpora

        self.assertEqual(corpora.load_distilled(
            norms.ROOT / "data" / "no-such-file.json"), {})

    def test_the_ablation_switch_exists(self):
        from research.v687 import profile

        self.assertIn("V687_NO_DISTILLED_NORMS",
                      inspect_source(profile))


def inspect_source(module):
    import inspect

    return inspect.getsource(module)


class TheDerivedArtifactsAreTracked(unittest.TestCase):
    """A repository whose answers depend on an untracked file is not one.

    `distilled_norms.json` changes what `Profiles.corroboration` counts, so it
    belongs in git beside the code that reads it rather than in `data/`, which
    is ignored wholesale and holds corpora somebody else published.
    """

    def test_the_paths_point_at_derived_and_not_at_data(self):
        from research.v687 import corpora
        from research.v688 import densify, prune, record_r19

        for name, path in (("DISTILLED", corpora.DISTILLED),
                           ("DEMOTED", corpora.DEMOTED),
                           ("densify.NORMS", densify.NORMS),
                           ("densify.CALLS", densify.CALLS),
                           ("prune.DEMOTED", prune.DEMOTED),
                           ("record_r19.CALLS", record_r19.CALLS)):
            with self.subTest(name):
                self.assertEqual(path.parent.name, "derived", name)

    def test_the_writers_and_the_readers_agree(self):
        """`densify.py` writes what `corpora.load_distilled` reads, and
        `prune.py` writes what `corpora.load_demoted` reads. They were
        separate literals until they disagreed."""
        from research.v687 import corpora
        from research.v688 import densify, prune

        self.assertEqual(densify.NORMS, corpora.DISTILLED)
        self.assertEqual(prune.DEMOTED, corpora.DEMOTED)

    def test_a_missing_artifact_is_an_ablation_and_not_a_crash(self):
        from research.v687 import corpora

        missing = norms.ROOT / "derived" / "no-such-file.json"
        self.assertEqual(corpora.load_distilled(missing), {})
        self.assertEqual(corpora.load_demoted(missing), frozenset())

    def test_the_rebuild_order_is_written_down(self):
        """Step 1 has to run before step 2 and the reason is expensive to
        rediscover: without `r19-calls.json`, `densify` has 824,431 cells."""
        readme = (norms.ROOT / "derived" / "README.md").read_text(
            encoding="utf-8")
        for step in ("record_r19", "densify --build", "prune --build",
                     "record_inherited"):
            self.assertIn(step, readme)


class TheDistilledKinds(unittest.TestCase):
    """A witness counts only where it has testimony."""

    def stub(self, distilled_kinds, lineage):
        from research.v687 import profile
        from research.v687.identify import Identifier

        class Stub:
            pass

        one = Stub()
        one.identifier = Identifier
        one.stated = {}
        one.distilled = {}
        one.distilled_kinds = distilled_kinds
        one._kind_lineage = lineage
        one.bears = lambda term, predicates: profile.Profiles.bears(
            one, term, predicates)
        one._ancestors = {}
        one._lineage = lambda: {}
        return one

    def witnesses(self, kinds, lineage, term="fly"):
        from research.v687.profile import Profiles

        return Profiles.witnesses(self.stub(kinds, lineage), "bird.n.01", term)

    def test_a_witness_asked_and_affirming_counts_both_ways(self):
        kinds = {"a.n.01": {"asked_at": frozenset(),
                            "asked": frozenset({"can fly"}),
                            "predicates": frozenset({"can fly"})}}
        self.assertEqual(self.witnesses(kinds, {"a.n.01": {"bird.n.01"}}),
                         (1, 1))

    def test_a_witness_asked_and_denying_counts_against(self):
        """It is in the denominator and not the numerator, which is the whole
        value of having asked it."""
        kinds = {"a.n.01": {"asked_at": frozenset(),
                            "asked": frozenset({"can fly"}),
                            "predicates": frozenset()}}
        self.assertEqual(self.witnesses(kinds, {"a.n.01": {"bird.n.01"}}),
                         (1, 0))

    def test_a_witness_never_asked_about_the_term_does_not_count_at_all(self):
        """The design: padding the denominator with silence would bias R19
        toward refusal, which is §17's trap self-inflicted."""
        kinds = {"a.n.01": {"asked_at": frozenset(),
                            "asked": frozenset({"can swim"}),
                            "predicates": frozenset({"can swim"})}}
        self.assertEqual(self.witnesses(kinds, {"a.n.01": {"bird.n.01"}}),
                         (0, 0))

    def test_a_witness_under_a_different_ancestor_is_not_counted(self):
        kinds = {"a.n.01": {"asked_at": frozenset(),
                            "asked": frozenset({"can fly"}),
                            "predicates": frozenset({"can fly"})}}
        self.assertEqual(self.witnesses(kinds, {"a.n.01": {"fish.n.01"}}),
                         (0, 0))

    def test_no_witnesses_is_free(self):
        self.assertEqual(self.witnesses({}, {}), (0, 0))

    def test_the_loader_drops_a_witness_with_nothing_asked(self):
        """It could only ever pad a denominator."""
        import json
        import tempfile
        from pathlib import Path

        from research.v687 import corpora

        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / "kinds.json"
            path.write_text(json.dumps({
                "kept.n.01": {"name": "kept", "asked": ["can fly"],
                              "predicates": ["can fly"]},
                "empty.n.01": {"name": "empty", "asked": [],
                               "predicates": []}}), encoding="utf-8")
            loaded = corpora.load_distilled_kinds(path)
        self.assertEqual(sorted(loaded), ["kept.n.01"])

    def test_a_missing_file_is_an_ablation(self):
        from research.v687 import corpora

        self.assertEqual(corpora.load_distilled_kinds(
            norms.ROOT / "derived" / "no-such-file.json"), {})

    def test_the_artifact_is_tracked_beside_the_others(self):
        from research.v687 import corpora
        from research.v688 import kinds

        self.assertEqual(corpora.KINDS.parent.name, "derived")
        self.assertEqual(kinds.KINDS, corpora.KINDS)

    def test_the_ablation_switch_exists(self):
        from research.v687 import profile

        self.assertIn("V687_NO_DISTILLED_KINDS", inspect_source(profile))

    def test_r19_no_longer_calls_its_denominator_the_norms(self):
        """With witnesses loaded, `the kinds of X the norms cover` is false."""
        from research.v687 import server

        self.assertNotIn("the norms cover bear that out",
                         inspect_source(server))


class TheDenseWitnessForm(unittest.TestCase):
    """`asked_at` names ancestors, not claims, and that is not a shortcut.

    R19 only consults `(ancestor, term)` when the term matched a fact on that
    ancestor. A witness asked about *every* inheritable fact there was
    therefore asked about that one, so naming the ancestor carries the same
    information as listing 640,000 claims -- and fits in a few kilobytes.
    """

    def witnesses(self, kinds, term="anything at all", dense=True):
        import unittest.mock

        from research.v687 import profile
        from research.v687.identify import Identifier

        class Stub:
            pass

        one = Stub()
        one.identifier = Identifier
        one.distilled_kinds = kinds
        one._kind_lineage = {c: {"bird.n.01"} for c in kinds}
        one.bears = lambda term, predicates: profile.Profiles.bears(
            one, term, predicates)
        with unittest.mock.patch.object(profile, "DENSE_WITNESSES", dense):
            return profile.Profiles.witnesses(one, "bird.n.01", term)

    def test_dense_testimony_is_on_and_can_be_turned_off(self):
        """§21 measured it and left it off; §24 retired the objection and §25
        turned it on. `V687_SPARSE_WITNESSES=1` still ignores `asked_at`."""
        from research.v687 import profile

        self.assertTrue(profile.DENSE_WITNESSES)
        kinds = {"a.n.01": {"asked_at": frozenset({"bird.n.01"}),
                            "asked": frozenset(),
                            "predicates": frozenset({"can fly"})}}
        self.assertEqual(self.witnesses(kinds, "fly", dense=True), (1, 1))
        self.assertEqual(self.witnesses(kinds, "fly", dense=False), (0, 0))

    def test_the_floor_leaves_a_majority_believable(self):
        """§25: `dog swim` is 9 of 13 -- 69% -- and a note that reports a
        majority and then refuses reads as the system arguing with itself.
        The floor has to sit below the majorities it means to believe."""
        from research.v687 import profile

        self.assertLess(profile.CORROBORATION_FLOOR, 9 / 13)

    def test_asked_at_lets_a_witness_speak_to_any_term_there(self):
        kinds = {"a.n.01": {"asked_at": frozenset({"bird.n.01"}),
                            "asked": frozenset(),
                            "predicates": frozenset({"can fly"})}}
        self.assertEqual(self.witnesses(kinds, "nobody asked this"), (1, 0))
        self.assertEqual(self.witnesses(kinds, "fly"), (1, 1))

    def test_asked_at_elsewhere_does_not_carry(self):
        kinds = {"a.n.01": {"asked_at": frozenset({"fish.n.01"}),
                            "asked": frozenset(),
                            "predicates": frozenset({"can fly"})}}
        self.assertEqual(self.witnesses(kinds, "fly"), (0, 0))

    def test_the_loader_reads_both_shapes(self):
        import json
        import tempfile
        from pathlib import Path

        from research.v687 import corpora

        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / "k.json"
            path.write_text(json.dumps({
                "dense.n.01": {"name": "dense", "asked_at": ["bird.n.01"],
                               "predicates": ["can fly"]},
                "cheap.n.01": {"name": "cheap", "asked": ["can fly"],
                               "predicates": []},
                "empty.n.01": {"name": "empty"}}), encoding="utf-8")
            loaded = corpora.load_distilled_kinds(path)
        self.assertEqual(sorted(loaded), ["cheap.n.01", "dense.n.01"])
        self.assertEqual(loaded["dense.n.01"]["asked_at"],
                         frozenset({"bird.n.01"}))
        self.assertEqual(loaded["cheap.n.01"]["asked_at"], frozenset())


class TheTermMatchingBug(unittest.TestCase):
    """`_hit` matches one word; `server.corroborate` passes whole targets.

    R19 has two callers. `profile.verify` passes a term `route` has already
    reduced to a content word; `server.corroborate` passes the parsed target
    as it stands -- `a blowhole` for `does a dolphin have a blowhole`. `_hit`
    stems the term as a whole and compares it to each *word* of a predicate,
    so a multi-word target matched nothing, bearing came out zero, and R19
    refused: `0 of the 14 kinds of whale on record bear that out`.

    Invisible while the norms were sparse enough that a low count looked
    ordinary. §21's witnesses made the denominator big enough to notice.
    """

    def bears(self, term, predicates):
        from research.v687 import profile
        from research.v687.identify import Identifier

        class Stub:
            pass

        one = Stub()
        one.identifier = Identifier
        return profile.Profiles.bears(one, term, frozenset(predicates))

    def test_the_bug_a_determiner_used_to_defeat_the_match(self):
        from research.v687.identify import Identifier

        self.assertIsNone(Identifier._hit("a blowhole", {"has blowhole"}))
        self.assertTrue(self.bears("a blowhole", {"has blowhole"}))

    def test_a_multi_word_target_matches_on_a_content_word(self):
        self.assertTrue(self.bears("four wheels", {"has four wheels"}))
        self.assertTrue(self.bears("a hard shell", {"has a hard shell"}))

    def test_a_single_word_term_is_unchanged(self):
        self.assertTrue(self.bears("blowhole", {"has blowhole"}))
        self.assertFalse(self.bears("blowhole", {"has a tail"}))

    def test_function_words_alone_never_match(self):
        """`does a` must not bear out every predicate containing `a`."""
        self.assertFalse(self.bears("the a an", {"has a tail"}))

    def test_it_can_only_raise_bearing_never_lower_it(self):
        """The fallback runs only after `_hit` has already failed, so a
        refusal can become an acceptance and never the reverse."""
        self.assertTrue(self.bears("fly", {"can fly"}))
        self.assertFalse(self.bears("swim", {"can fly"}))

    def test_nothing_bears_out_an_empty_set(self):
        self.assertFalse(self.bears("blowhole", set()))


class TheSharpestLevel(unittest.TestCase):
    """R19 checks where the evidence is, not where the crawl put the fact."""

    class Stub:
        """Enough of `Profiles` to run `sharpest` over a fixed taxonomy."""

        def __init__(self, counts, chain):
            self.counts = counts
            self.chain = chain

        def corroboration(self, node, term):
            return self.counts.get(node, (0, 0))

        @property
        def reasoner(self):
            chain = self.chain

            class Walk:
                @staticmethod
                def ascend(concept):
                    return [(node, index, [])
                            for index, node in enumerate(chain)]
            return Walk()

    def sharpest(self, counts, chain, concept="dog.n.01",
                 attached="animal.n.01", term="leg"):
        from research.v687.profile import Profiles

        return Profiles.sharpest(self.Stub(counts, chain), concept,
                                 attached, term)

    #: `dog have legs`: 156 of 244 animals, 8 of 8 dogs. The case that
    #: bounded the corroboration floor at 0.64 until §23.
    CHAIN = ["dog.n.01", "canine.n.02", "carnivore.n.01", "animal.n.01"]
    COUNTS = {"dog.n.01": (8, 8), "canine.n.02": (12, 12),
              "carnivore.n.01": (29, 29), "animal.n.01": (156, 244)}

    def test_it_takes_the_narrowest_level_that_can_speak(self):
        where, bearing, kinds = self.sharpest(self.COUNTS, self.CHAIN)
        self.assertEqual((where, bearing, kinds), ("dog.n.01", 8, 8))

    def test_a_level_too_thin_to_speak_is_passed_over(self):
        counts = dict(self.COUNTS, **{"dog.n.01": (3, 3)})
        where, _bearing, kinds = self.sharpest(counts, self.CHAIN)
        self.assertEqual(where, "canine.n.02")
        self.assertGreaterEqual(kinds, 8)

    def test_it_never_climbs_above_where_the_fact_was_attached(self):
        """A fact stated of animals stays a claim about animals. Otherwise
        the walk could go looking for a level that agrees."""
        chain = self.CHAIN + ["organism.n.01", "entity.n.01"]
        counts = dict(self.COUNTS, **{"dog.n.01": (0, 0),
                                      "canine.n.02": (0, 0),
                                      "carnivore.n.01": (0, 0),
                                      "entity.n.01": (900, 900)})
        where, _bearing, _kinds = self.sharpest(counts, chain)
        self.assertEqual(where, "animal.n.01")

    def test_it_stops_at_the_first_level_that_speaks_not_the_best(self):
        """Taking the *best* rung would be shopping for a verdict."""
        counts = {"dog.n.01": (0, 9), "canine.n.02": (12, 12),
                  "carnivore.n.01": (29, 29), "animal.n.01": (156, 244)}
        where, bearing, _kinds = self.sharpest(counts, self.CHAIN)
        self.assertEqual(where, "dog.n.01")
        self.assertEqual(bearing, 0)

    def test_with_the_flag_off_it_uses_the_attached_level(self):
        import unittest.mock

        from research.v687 import profile

        with unittest.mock.patch.object(profile, "SHARPEST", False):
            where, bearing, kinds = self.sharpest(self.COUNTS, self.CHAIN)
        self.assertEqual((where, bearing, kinds), ("animal.n.01", 156, 244))

    def test_the_attached_level_is_no_longer_what_bounds_the_floor(self):
        """§23's point, asserted as the claim rather than as a number.

        It used to read `CORROBORATION_FLOOR > 156/244`, because the binding
        case was `does a dog have legs` corroborated at `animal.n.01`. §25
        moved the floor to 0.6 for an unrelated reason and that assertion
        failed while the behaviour it was guarding was fine -- the fourth
        test this week to pin a reading instead of a claim. What matters is
        that the walk happens at all."""
        from research.v687 import profile

        self.assertTrue(profile.SHARPEST)
        where, bearing, kinds = self.sharpest(self.COUNTS, self.CHAIN)
        self.assertEqual(where, "dog.n.01")
        self.assertGreater(bearing / kinds, profile.CORROBORATION_FLOOR)


class TheCapabilityPrompt(unittest.TestCase):
    """§24: a better capability judge, and the wrong judge for R19."""

    def test_the_cache_is_namespaced_by_prompt(self):
        """99,000 judgements were answered under `careful`. A second prompt
        answering a different question must not read them, and `careful`
        must keep the bare key so none of them is orphaned."""
        from research.v688.teacher import Teacher

        bare = Teacher.key("", "", "can a dog fall into a hole")
        self.assertEqual(bare, "||can a dog fall into a hole")
        self.assertEqual(Teacher.key("", "", "x", "careful"),
                         "careful::||x")
        self.assertNotEqual(Teacher.key("", "", "x", "capable"),
                            Teacher.key("", "", "x"))

    def test_the_default_style_is_careful(self):
        from research.v688.teacher import CAREFUL, STYLES

        self.assertIs(STYLES[""], CAREFUL)

    def test_nothing_is_routed_to_the_capability_prompt(self):
        """It was measured and it lost: every artifact grew threefold and
        accuracy fell 92.1% to 90.9%. R19 asks whether a fact is a claim
        about the class, and typicality is the right test for that even when
        the fact is a capability."""
        from research.v688 import densify

        self.assertEqual(densify.CAPABILITY, frozenset())
        for relation in ("capable_of", "receives_action", "has_a"):
            self.assertEqual(densify.style_for(relation), "")

    def test_the_prompt_is_kept_so_it_is_not_rediscovered(self):
        from research.v688 import teacher

        self.assertIn("capable", teacher.STYLES)
        self.assertIn("ordinary member", teacher.CAPABLE)


class TheSparseBuildDoesNotDestroyTheDenseOne(unittest.TestCase):
    """`--build` covers 319 concepts and `--dense` covers 691.

    Writing only what a run touched threw away eight and a half GPU-hours
    once. It merges now.
    """

    def test_it_merges_into_what_is_already_there(self):
        import inspect

        from research.v688 import kinds

        source = inspect.getsource(kinds.build)
        self.assertIn("load_distilled_kinds", source)
        self.assertIn("setdefault", source)

    def test_the_artifact_still_carries_its_dense_testimony(self):
        """A guard on the artifact itself: the overnight run's `asked_at`
        sets are what §21 cost, and nothing since should have dropped them."""
        from research.v687 import corpora

        loaded = corpora.load_distilled_kinds()
        if not loaded:
            self.skipTest("no distilled kinds built")
        dense = [one for one in loaded.values() if one["asked_at"]]
        self.assertGreater(len(dense), 500,
                           "dense witnesses look truncated")
