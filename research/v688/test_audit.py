"""Tests for the audit apparatus.

None of these builds an engine. The audit's own correctness is a question
about a transform and a scoring rule, and both are pure -- which is the
reason they are separable from the sweep at all, and why this module runs in
under a second while the sweep it checks runs for twenty minutes.

The class that matters most is `TheGoldSetIsNotWhatItLooksLike`. It pins the
finding that came out of building this: COMPS' foils are drawn from a sparse
free-listing matrix, so they are absence and not denial. That is easy to
forget and expensive to forget, so it is asserted against the data rather
than written in a comment.
"""
from __future__ import annotations

import csv
import unittest
from pathlib import Path

from research.v687 import corpora
from research.v688 import audit


class ThePhrasing(unittest.TestCase):
    """A gold pair becomes a question by one transform over its first word."""

    @classmethod
    def setUpClass(cls) -> None:
        path = audit.ROOT / "data" / "xcslb" / "feature_lexicon.csv"
        with path.open(encoding="utf-8") as handle:
            cls.features = [row["feature"] for row in csv.DictReader(handle)
                            if row["feature"].strip()]

    def test_every_feature_in_the_lexicon_phrases(self):
        unreachable = [f for f in self.features if not audit.phrase("a bat", f)]
        self.assertEqual(unreachable, [])

    def test_every_concept_ships_an_article(self):
        article = audit.articles()
        missing = [c for c, _ in corpora.load_xcslb().items
                   if not article.get(c)]
        self.assertEqual(missing, [])

    def test_the_five_shapes(self):
        self.assertEqual(audit.phrase("a beagle", "has legs"),
                         "does a beagle have legs")
        self.assertEqual(audit.phrase("a beagle", "can fly"),
                         "can a beagle fly")
        self.assertEqual(audit.phrase("a beagle", "is a vehicle"),
                         "is a beagle a vehicle")
        self.assertEqual(audit.phrase("a beagle", "eats meat"),
                         "does a beagle eat meat")
        self.assertEqual(audit.phrase("a beagle", "slithers"),
                         "does a beagle slither")

    def test_a_bare_auxiliary_is_not_a_question(self):
        for word in ("is", "can", "has"):
            self.assertEqual(audit.phrase("a beagle", word), "")

    def test_the_de_inflection_is_right_on_every_verb_the_lexicon_opens_with(self):
        """The docstring claims 108 of 109. Hold it to that.

        `used` is the one that is not a third-person verb, and `phrase`
        routes it before `stem` is reached.
        """
        leads = {f.split()[0] for f in self.features}
        verbs = sorted(leads - audit.AUXILIARY)
        self.assertGreaterEqual(len(verbs), 100)
        wrong = [v for v in verbs if audit.stem(v).endswith("s")
                 and not audit.stem(v).endswith("ss")]
        self.assertEqual(wrong, [])
        # The shapes the rule exists for.
        self.assertEqual(audit.stem("carries"), "carry")
        self.assertEqual(audit.stem("goes"), "go")
        self.assertEqual(audit.stem("washes"), "wash")
        self.assertEqual(audit.stem("uses"), "use")
        self.assertEqual(audit.stem("contains"), "contain")
        self.assertEqual(audit.phrase("a pan", "used for cooking"),
                         "is a pan used for cooking")


class TheGoldSet(unittest.TestCase):

    def test_the_sample_is_the_same_every_run(self):
        """Two configurations are comparable only if they were asked the
        same questions, and nothing here reseeds a generator."""
        self.assertEqual([p.keys() for p in audit.pairs(20)],
                         [p.keys() for p in audit.pairs(20)])

    def test_the_limit_caps_each_rung_not_the_whole(self):
        chosen = audit.pairs(20)
        seen = {name: 0 for name in audit.LADDER}
        for pair in chosen:
            seen[pair.foil_kind] += 1
        self.assertEqual(set(seen.values()), {20})

    def test_asking_by_key_is_cheaper_than_asking_by_pair(self):
        chosen = audit.pairs(50)
        asked = audit.questions_for(chosen)
        self.assertLess(len(asked), 2 * len(chosen))
        for pair in chosen:
            for key in pair.keys():
                self.assertIn(key, asked)


class TheGoldSetIsNotWhatItLooksLike(unittest.TestCase):
    """COMPS' foils are absence, not denial, and this is why.

    `corpora.denied_xcslb` says its foils are "the only place in any of this
    data where absence is stated rather than merely observed". These tests
    say otherwise, from the shipped data.
    """

    def test_the_concept_matrix_is_sparse_so_a_zero_is_a_silence(self):
        """A closed norming -- every concept judged on every feature -- would
        be dense. 1.58% is a free listing, and its zeros are what nobody
        happened to say."""
        path = audit.ROOT / "data" / "xcslb" / "concept_matrix.txt"
        rows = [line.split() for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ones = sum(row.count("1") for row in rows)
        cells = sum(len(row) - 1 for row in rows)     # first column is a name
        self.assertLess(ones / cells, 0.05)

    def test_a_near_foil_is_routinely_true_of_the_concept(self):
        """The check a reader can do by eye, kept as a test.

        If these were denials, every one of them would be false.
        """
        foils = {(p.foil, p.prop) for p in audit.pairs()}
        for concept, prop in (("stocking", "absorbs sweat"),
                              ("potato", "absorbs water")):
            self.assertIn((concept, prop), foils)

    def test_the_audit_never_scores_a_foil_as_a_denial(self):
        """The design consequence. `score_absolute` looks only at the held
        side; the foil is used for a comparison and never for a label."""
        chosen = audit.pairs(5)
        answers = {}
        for pair in chosen:
            held, foil = pair.keys()
            answers[held] = {"outcome": "verified", "confidence": 0.9}
            answers[foil] = {"outcome": "verified", "confidence": 0.9}
        absolute = audit.score_absolute(answers, chosen)
        self.assertEqual(absolute["n"], len({f"{p.held}|{p.prop}"
                                             for p in chosen}))
        self.assertEqual(absolute["confirmed"], 1.0)


class TheScoring(unittest.TestCase):

    def test_an_unsettled_answer_is_zero_and_not_a_small_yes(self):
        self.assertEqual(audit.stance({"outcome": "unknown",
                                       "confidence": 0.4}), 0.0)
        self.assertEqual(audit.stance({"outcome": "retrieved",
                                       "confidence": 0.9}), 0.0)
        self.assertEqual(audit.stance({}), 0.0)

    def test_denied_is_below_unknown_is_below_verified(self):
        denied = audit.stance({"outcome": "denied", "confidence": 0.5})
        verified = audit.stance({"outcome": "verified", "confidence": 0.5})
        self.assertLess(denied, 0.0)
        self.assertLess(0.0, verified)

    def test_a_tie_is_reported_and_never_counted_right(self):
        """Both sides silent is the common case here, and splitting it would
        flatter every configuration by the same 50%."""
        chosen = audit.pairs(5)
        answers = {}
        for pair in chosen:
            for key in pair.keys():
                answers[key] = {"outcome": "unknown", "confidence": 0.0}
        overall = audit.score_pairs(answers, chosen)["overall"]
        self.assertEqual(overall["n"], len(chosen))
        self.assertEqual(overall["decided"], 0)
        self.assertEqual(overall["right"], 0)
        self.assertIsNone(overall["accuracy"])

    def test_the_listed_concept_winning_is_what_counts_as_right(self):
        chosen = audit.pairs(5)
        answers = {}
        for pair in chosen:
            held, foil = pair.keys()
            answers[held] = {"outcome": "verified", "confidence": 0.8}
            answers[foil] = {"outcome": "unknown", "confidence": 0.0}
        overall = audit.score_pairs(answers, chosen)["overall"]
        self.assertEqual(overall["decided"], len(chosen))
        self.assertEqual(overall["accuracy"], 1.0)

        for pair in chosen:                       # and the other way round
            held, foil = pair.keys()
            answers[held], answers[foil] = answers[foil], answers[held]
        self.assertEqual(audit.score_pairs(answers, chosen)
                         ["overall"]["accuracy"], 0.0)

    def test_a_pair_with_a_side_missing_is_dropped_not_guessed(self):
        chosen = audit.pairs(5)
        answers = {pair.keys()[0]: {"outcome": "verified", "confidence": 0.8}
                   for pair in chosen}
        self.assertEqual(audit.score_pairs(answers, chosen)["overall"]["n"], 0)


class TheAblation(unittest.TestCase):
    """The configurations differ in what they turn off, and nothing else."""

    def test_the_norms_answer_only_in_the_shipped_configuration(self):
        from research.v687.reasoning import ReasoningEngine

        self.assertIs(audit.engine_class("shipped"), ReasoningEngine)
        for config in ("crawl", "corroborated", "loop"):
            engine = audit.engine_class(config)
            self.assertTrue(issubclass(engine, ReasoningEngine))
            self.assertIsNot(engine.about, ReasoningEngine.about)

    def test_only_the_crawl_configuration_turns_corroboration_off(self):
        from research.v687.reasoning import ReasoningEngine

        self.assertIsNot(audit.engine_class("crawl").corroborate,
                         ReasoningEngine.corroborate)
        for config in ("corroborated", "loop"):
            self.assertIs(audit.engine_class(config).corroborate,
                          ReasoningEngine.corroborate)

    def test_route_is_left_alone(self):
        """Disabling `profiles.route` would change inverse-question routing
        and pin labelling, neither of which the ablation is about."""
        source = Path(audit.__file__).read_text(encoding="utf-8")
        self.assertNotIn("def route", source)


class ThePoolStillDefaultsToTheShippedEngine(unittest.TestCase):

    def test_engine_class_is_optional(self):
        """`audit.py` added a parameter to `EnginePool`. Nothing served
        passes it, so the default has to stay what the server always got."""
        import inspect

        from research.v688.pool import EnginePool

        parameter = inspect.signature(EnginePool.__init__).parameters
        self.assertIn("engine_class", parameter)
        self.assertIsNone(parameter["engine_class"].default)


if __name__ == "__main__":
    unittest.main()
