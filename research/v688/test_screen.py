"""Tests for the foil screen.

No model is loaded here. The screen is a threshold over a judgement plus some
arithmetic on rates, and both halves are checkable without a GPU: the
question sets are built from files, and `prevalence` is a formula. What
cannot be tested without the model -- whether SmolLM3 is any good at this --
is exactly what `--calibrate` measures and what `AUDIT.md` records.

The class that matters most is `TheCalibrationSetsAreActuallyGold`. The whole
argument for screening a benchmark with a model rests on the error rates
being counted rather than assumed, and that rests on AwA2's matrix being
closed. If that stops being true the screen stops meaning anything, so it is
asserted against the data.
"""
from __future__ import annotations

import unittest

from research.v688 import screen


class TheAttributeQuestions(unittest.TestCase):
    """AwA2's 85 predicates, as questions somebody could answer."""

    def test_every_attribute_is_phrased_or_deliberately_left_out(self):
        predicates = [line.split()[-1] for line in
                      (screen.AWA2 / "predicates.txt")
                      .read_text(encoding="utf-8").splitlines()
                      if line.strip()]
        missing = [name for name in predicates
                   if name not in screen.ATTRIBUTES]
        self.assertEqual(missing, ["newworld", "oldworld"])

    def test_every_template_takes_the_subject(self):
        for name, template in screen.ATTRIBUTES.items():
            with self.subTest(attribute=name):
                self.assertIn("{s}", template)
                self.assertEqual(template.format(s="a beaver").count("beaver"),
                                 1)

    def test_the_article_is_read_from_the_word(self):
        self.assertEqual(screen.article("beaver"), "a")
        self.assertEqual(screen.article("antelope"), "an")
        self.assertEqual(screen.article("otter"), "an")

    def test_a_few_read_as_english(self):
        claims = {c.question for c in screen.awa2_claims()}
        for one in ("can a bat fly", "does a zebra have stripes",
                    "is a polar bear white"):
            self.assertIn(one, claims)


class TheCalibrationSetsAreActuallyGold(unittest.TestCase):
    """The screen's error rates are counted, and this is what they rest on."""

    def test_awa2_is_closed_over_its_attributes(self):
        """Every class scored on every attribute -- which is why a zero is a
        denial here and a silence in XCSLB."""
        rows = [line.split() for line in
                (screen.AWA2 / "predicate-matrix-continuous.txt")
                .read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 50)
        self.assertEqual({len(row) for row in rows}, {85})

    def test_the_cuts_take_only_the_ends(self):
        """A claim used for calibration is one nobody would argue about."""
        self.assertLess(screen.DENIED_AT, screen.HELD_AT)
        claims = screen.awa2_claims()
        self.assertTrue(any(c.truth is False for c in claims))
        self.assertTrue(any(c.truth is True for c in claims))
        # The published binary threshold sits at about 20.8, so both cuts
        # must fall clear of it or the "gold" is the disputed middle.
        self.assertLess(screen.DENIED_AT, 20.8)
        self.assertGreater(screen.HELD_AT, 20.8)

    def test_the_four_sets_carry_the_truth_they_claim(self):
        sets = {"awa2_false": False, "awa2_true": True,
                "xcslb_true": True, "corrupted_false": False}
        built = screen.truth_sets(20)
        for name, truth in sets.items():
            with self.subTest(name):
                self.assertTrue(built[name])
                self.assertEqual({c.truth for c in built[name]}, {truth})

    def test_the_foils_claim_nothing(self):
        self.assertEqual({c.truth for c in screen.foil_claims(50)}, {None})


class TheClaimSets(unittest.TestCase):
    """The sizes and the sampling."""

    def test_both_sides_of_comps_are_reached(self):
        self.assertEqual(len(screen.comps_claims("acceptable")), 12335)
        self.assertEqual(len(screen.comps_claims("unacceptable")), 36701)

    def test_no_foil_claim_is_also_a_held_claim(self):
        """If one were, the screen would be asked to deny something COMPS
        itself lists as true elsewhere, and the label would be incoherent."""
        held = {(c.concept, c.prop) for c in screen.comps_claims("acceptable")}
        foil = {(c.concept, c.prop)
                for c in screen.comps_claims("unacceptable")}
        self.assertEqual(held & foil, set())

    def test_sampling_is_a_strided_walk_so_runs_repeat(self):
        rows = list(range(100))
        self.assertEqual(screen.sample(rows, 10),
                         [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        self.assertEqual(screen.sample(rows, 0), rows)
        self.assertEqual(screen.sample(rows, 500), rows)

    def test_a_small_sample_is_a_subset_of_a_large_one(self):
        big = {c.question for c in screen.xcslb_claims(400)}
        small = {c.question for c in screen.xcslb_claims(200)}
        self.assertTrue(small <= big)

    def test_every_foil_lands_on_a_rung(self):
        where = screen.rungs()
        stray = [c for c in screen.foil_claims(300)
                 if where.get((c.concept, c.prop)) not in screen.LADDER]
        self.assertEqual(stray, [])

    def test_the_ladder_matches_the_audit(self):
        from research.v688 import audit

        self.assertEqual(screen.LADDER, audit.LADDER)


class TheScreenItself(unittest.TestCase):
    """The threshold and the arithmetic, with judgements supplied."""

    def claims(self, *questions):
        return [screen.Claim(q, None, "foil") for q in questions]

    def test_a_denial_below_the_threshold_does_not_count(self):
        judged = {"a": (True, 0.95), "b": (True, 0.60), "c": (False, 0.99)}
        rows = self.claims("a", "b", "c")
        self.assertEqual(screen.denial_rate(rows, judged, 0.9), (3, 1 / 3))
        self.assertEqual(screen.denial_rate(rows, judged, 0.5), (3, 2 / 3))

    def test_an_unjudged_claim_is_left_out_rather_than_counted_right(self):
        judged = {"a": (True, 0.99)}
        self.assertEqual(screen.denial_rate(self.claims("a", "b"), judged,
                                            0.9), (1, 1.0))

    def test_prevalence_is_the_screening_correction(self):
        # A perfect screen reports what it sees.
        self.assertAlmostEqual(screen.prevalence(0.4, 1.0, 0.0), 0.4)
        # One that fires on a tenth of true claims has seen fewer than it
        # thinks: (0.4 - 0.1) / (0.9 - 0.1).
        self.assertAlmostEqual(screen.prevalence(0.4, 0.9, 0.1), 0.375)

    def test_prevalence_declines_to_guess_when_the_screen_cannot_separate(self):
        self.assertIsNone(screen.prevalence(0.4, 0.50, 0.48))

    def test_prevalence_is_clamped_rather_than_reported_out_of_range(self):
        self.assertEqual(screen.prevalence(0.05, 0.9, 0.1), 0.0)
        self.assertEqual(screen.prevalence(0.99, 0.9, 0.1), 1.0)


class TheAuditReadsIt(unittest.TestCase):
    """`--gold screened` is wired, and degrades rather than crashing."""

    def test_base_is_the_default(self):
        from research.v688 import audit

        self.assertEqual(audit.GOLD, "base")
        self.assertEqual(audit.gold_file("base"), audit.COMPS)

    def test_screened_falls_back_when_it_has_not_been_built(self):
        """It is a build product, so a fresh clone must still be able to run
        the audit."""
        from research.v688 import audit

        chosen = audit.pairs(20, "screened")
        self.assertTrue(chosen)
        if not audit.SCREENED.exists():
            self.assertEqual(audit.gold_file("screened"), audit.COMPS)

    def test_the_denial_score_counts_the_foil_side(self):
        from research.v688 import audit

        chosen = [audit.Pair("has feathers", "sock", "spanner", "taxonomic",
                             "visual perceptual"),
                  audit.Pair("has feathers", "sock", "hammer", "overlap",
                             "visual perceptual")]
        answers = {"spanner|has feathers": {"outcome": "verified"},
                   "hammer|has feathers": {"outcome": "unknown"}}
        scored = audit.score_denials(answers, chosen)
        self.assertEqual(scored["overall"]["n"], 2)
        self.assertEqual(scored["overall"]["asserted"], 1)
        self.assertEqual(scored["overall"]["silent_share"], 0.5)
        self.assertEqual(scored["overall"]["wrong_when_it_spoke"], 1.0)
        self.assertEqual(scored["by_foil"]["taxonomic"]["asserted"], 1)

    def test_the_denial_score_ignores_the_held_side(self):
        """The held claim is true; asserting it is not over-affirmation."""
        from research.v688 import audit

        chosen = [audit.Pair("has feathers", "sock", "spanner", "taxonomic",
                             "visual perceptual")]
        answers = {"sock|has feathers": {"outcome": "verified"},
                   "spanner|has feathers": {"outcome": "denied"}}
        scored = audit.score_denials(answers, chosen)
        self.assertEqual(scored["overall"]["n"], 1)
        self.assertEqual(scored["overall"]["asserted"], 0)
        self.assertEqual(scored["overall"]["refused"], 1)


class TheCacheHoldsOpen(unittest.TestCase):
    """`judge_all` asks tens of thousands of questions; the cache is 2.5 MB."""

    def test_batch_defers_the_write_and_nests(self):
        from research.v688.teacher import Teacher

        teacher = Teacher(load=False)
        written = []
        teacher._save = lambda: written.append(1)          # noqa: SLF001
        with teacher.batch():
            with teacher.batch():
                teacher._write_cache()                     # noqa: SLF001
                teacher._write_cache()                     # noqa: SLF001
            self.assertEqual(written, [])
        self.assertEqual(len(written), 1)

    def test_outside_a_batch_every_judgement_still_saves(self):
        from research.v688.teacher import Teacher

        teacher = Teacher(load=False)
        written = []
        teacher._save = lambda: written.append(1)          # noqa: SLF001
        teacher._write_cache()                             # noqa: SLF001
        self.assertEqual(len(written), 1)


class TheTeacherIsAskedWhatWasAsked(unittest.TestCase):
    """An unsettled question goes to the model as asked, and nothing else does.

    It used to be put the facts R28 held back instead. For `does a pig have
    wings` those were crawled sentences filed under `animal` -- "leathery
    bat-like wings" -- and the model was asked whether a sentence about bats
    supports pig wings, while the reader's own question was never put.
    """

    @staticmethod
    def answer(question, verdict="UNKNOWN", polar=True):
        from types import SimpleNamespace

        return SimpleNamespace(question=question, verdict=verdict,
                               payload={"parse": {"polar": polar}})

    @staticmethod
    def teacher(says=None):
        from unittest import mock

        from research.v688.teacher import Teacher

        # The cache is tens of megabytes and `judge` is replaced anyway.
        with mock.patch.object(Teacher, "_read_cache", lambda self: None):
            teacher = Teacher(load=False)
        teacher.asked = []

        def judge(subject, fact, claim, style=""):
            teacher.asked.append((subject, fact, claim))
            return (says or {}).get(claim, (False, 0.5)) + (False,)

        teacher.judge = judge
        return teacher

    def test_the_question_is_put_bare(self):
        teacher = self.teacher()
        teacher.review([self.answer("does a pig have wings")])
        self.assertEqual(teacher.asked, [("", "", "does a pig have wings")])

    def test_only_what_the_store_left_open(self):
        teacher = self.teacher()
        judged = teacher.review([
            self.answer("does a dog have legs", "VERIFIED"),
            self.answer("do fish run", "CONTRADICTED"),
            self.answer("what is a fish", polar=False),
            self.answer("does a pig have wings", "UNRECORDED")])
        self.assertEqual([one.question for one in judged],
                         ["does a pig have wings"])

    def test_a_question_is_put_once_per_run(self):
        teacher = self.teacher()
        done: set = set()
        teacher.review([self.answer("does a pig have wings")], done=done)
        again = teacher.review([self.answer("does a pig have wings")],
                               done=done)
        self.assertEqual(again, [])
        self.assertEqual(len(teacher.asked), 1)

    def test_an_answer_at_the_floor_settles_either_way(self):
        teacher = self.teacher({"can a dog swim": (True, 0.995),
                                "can a dog fall into a hole": (True, 0.9),
                                "do pigs fly": (False, 0.999),
                                "can a leopard hunt at night": (False, 0.9)})
        judged = teacher.review([self.answer(text) for text in (
            "can a dog swim", "can a dog fall into a hole", "do pigs fly",
            "can a leopard hunt at night")])
        self.assertEqual([one.settles for one in judged],
                         [True, False, True, False])


class TheNegationIsTheClaimDenied(unittest.TestCase):
    """`negation.negate`, whose one trap is `can`."""

    def test_can_is_ability_not_refraining(self):
        from research.v688.negation import negate

        # `can a dog not swim` asks whether it is able to refrain.
        self.assertEqual(negate("can a dog swim", "dog"),
                         "is a dog unable to swim")

    def test_the_rest_take_a_plain_not(self):
        from research.v688.negation import negate

        self.assertEqual(negate("does a lion have flippers", "lion"),
                         "does a lion not have flippers")
        self.assertEqual(negate("is a killer whale white", "killer whale"),
                         "is a killer whale not white")

    def test_it_declines_a_subject_it_cannot_find(self):
        from research.v688.negation import negate

        self.assertEqual(negate("can a coffee maker heat", "coffee_maker"),
                         "")


if __name__ == "__main__":
    unittest.main()
