"""Regression suite for the loop.

Run: python -m unittest research.v688.test_v688 -v

Tests that need the built store skip themselves when it is absent. The pool is
built once for the whole module at two workers -- enough to prove the fan-out
is parallel, cheap enough to run in a suite.
"""
from __future__ import annotations

import unittest

from research.v687 import build

from . import attention, gap, server
from .attention import Activation, Curiosity
from .buffer import Buffer
from .loop import Loop, seed_questions
from .pool import EnginePool
from .question import (Generator, MAX_FAMILY, article, phrase,
                       phrase_predicate)

STORE = build.DEFAULT_STORE
HAVE_STORE = STORE.exists()
requires_store = unittest.skipUnless(HAVE_STORE, f"no store at {STORE}")

POOL: EnginePool | None = None
CURIOSITY: Curiosity | None = None
LOOP: Loop | None = None
RUNS: dict = {}


def setUpModule() -> None:                      # noqa: N802
    global POOL, CURIOSITY, LOOP
    if not HAVE_STORE:
        return
    # Five, not two: a cycle puts out one question per worker, so a smaller
    # pool truncates the corroboration fan-out and the conflicts these tests
    # are about never surface. The pool size is part of what is under test.
    POOL = EnginePool(STORE, workers=5)
    CURIOSITY = Curiosity(POOL.engines[0].profiles.plan)
    # Six, matching the server default: a definition ladder four answers
    # deep needs five cycles to run, and capping at four hides the only
    # behaviour that is genuinely serial.
    LOOP = Loop(POOL, CURIOSITY, max_cycles=6)


def tearDownModule() -> None:                   # noqa: N802
    if POOL is not None:
        POOL.close()


def run(utterance: str):
    """One run per utterance for the whole module: the loop is deterministic
    and rebuilding a run costs seconds."""
    if utterance not in RUNS:
        RUNS[utterance] = LOOP.run(utterance)
    return RUNS[utterance]


class VerdictVocabularyTests(unittest.TestCase):
    """The five kinds a verdict can be about, and the repair each licenses."""

    def test_every_verdict_v687_can_return_is_classified(self):
        """An unclassified verdict reaches the loop as `unknown` and licenses
        nothing, which is silent failure rather than a gap."""
        known = (gap.ABOUT_THE_WORLD | gap.ABOUT_CONTENT | gap.ABOUT_COVERAGE
                 | gap.ABOUT_THE_QUESTION | gap.ABOUT_RELIABILITY)
        for verdict in known:
            with self.subTest(verdict=verdict):
                self.assertNotEqual(gap.kind_of(verdict), "unknown")

    def test_the_kinds_do_not_overlap(self):
        seen: set[str] = set()
        for group in (gap.ABOUT_THE_WORLD, gap.ABOUT_CONTENT,
                      gap.ABOUT_COVERAGE, gap.ABOUT_THE_QUESTION,
                      gap.ABOUT_RELIABILITY):
            self.assertFalse(seen & group, seen & group)
            seen |= group

    def test_only_coverage_and_question_verdicts_open_a_gap(self):
        """A truth value is not a hole. Treating `CONTRADICTED` as one is the
        confusion the v687 audit was written to end."""
        for verdict in gap.ABOUT_THE_WORLD | gap.ABOUT_CONTENT:
            with self.subTest(verdict=verdict):
                self.assertIsNone(gap.KIND.get(verdict))
        for verdict in gap.ABOUT_COVERAGE | gap.ABOUT_THE_QUESTION:
            with self.subTest(verdict=verdict):
                self.assertIsNotNone(gap.KIND.get(verdict))

    def test_a_synset_id_never_reaches_a_question(self):
        """`is a beagle a dog.n.01` was asked, and reported back as an unknown
        word: a gap the loop manufactured for itself."""
        self.assertEqual(gap.plain("dog.n.01"), "dog")
        self.assertEqual(gap.plain("german_shepherd.n.01"), "german shepherd")
        self.assertEqual(gap.plain("an agility"), "agility")
        self.assertEqual(gap.plain("the ground"), "ground")


@requires_store
class GapReadingTests(unittest.TestCase):
    """What a v687 payload says about its own incompleteness."""

    def test_an_unknown_word_names_the_word(self):
        payload = POOL.engines[0].ask("is a wemble a greeting")
        hole = gap.read_gap(payload)
        self.assertEqual(hole.kind, "word")
        self.assertEqual(hole.blocker, "wemble")
        self.assertTrue(hole.blocking)

    def test_a_coverage_gap_names_what_was_missing_not_the_subject(self):
        """`is a whale a fish` knows perfectly well about whales."""
        payload = POOL.engines[0].ask("is a whale a fish")
        hole = gap.read_gap(payload)
        self.assertEqual(hole.kind, "coverage")
        self.assertEqual(hole.blocker, "fish")
        self.assertFalse(hole.blocking)

    def test_the_beagle_yes_carries_three_reasons_to_doubt_it(self):
        """Confidence 0.42, inherited three levels, sense assumed -- and the
        verdict says VERIFIED."""
        payload = POOL.engines[0].ask("does a beagle swim")
        self.assertEqual(payload["verdict"], "VERIFIED")
        reasons = {doubt.reason for doubt in gap.read_doubts(payload)}
        self.assertEqual(reasons, {"weak", "inherited", "assumed_sense"})

    def test_a_well_recorded_answer_raises_no_doubt(self):
        """The control. If everything were doubted, nothing would be."""
        payload = POOL.engines[0].ask("does a robin fly")
        self.assertEqual(payload["verdict"], "VERIFIED")
        self.assertEqual(gap.read_doubts(payload), [])

    def test_the_relation_to_re_ask_under_comes_from_the_evidence(self):
        """`is a dog wild` routes as is_a and rests on a has_property fact;
        re-asking under the route produced `is a collie a wild`."""
        payload = POOL.engines[0].ask("is a dog wild")
        doubts = gap.read_doubts(payload)
        self.assertTrue(doubts)
        self.assertEqual(doubts[0].relation, "has_property")


class AttentionTests(unittest.TestCase):
    """Salience, gain, and the product that ranks them."""

    def test_activation_decays_and_is_pruned_at_the_floor(self):
        live = Activation()
        live.bump("dog", 1.0)
        for _ in range(12):
            live.decay()
        self.assertNotIn("dog", live.table)

    def test_a_zero_in_any_score_kills_the_question(self):
        """A product, not a sum: a maximally informative question about
        something nobody mentioned should not be asked."""
        self.assertEqual(attention.rank(0.0, 1.0, 1.0), 0.0)
        self.assertEqual(attention.rank(1.0, 0.0, 1.0), 0.0)
        self.assertEqual(attention.rank(1.0, 1.0, 0.0), 0.0)
        self.assertGreater(attention.rank(0.5, 0.5, 0.5), 0.0)

    def test_a_blocked_word_outranks_being_interested(self):
        self.assertGreater(attention.URGENCY["word"],
                           attention.URGENCY["curiosity"])
        self.assertGreater(attention.URGENCY["doubt"],
                           attention.URGENCY["curiosity"])

    def test_gain_is_highest_for_a_predicate_that_halves_the_field(self):
        """`anti_coverage` -- rarest first -- is the greedy approximation.
        A singleton identifies one individual and says nothing about the
        rest, and this is the measure that knows the difference."""
        plan = [(f"i{n}", frozenset({"half"} if n % 2 else set()) |
                 ({"all"}) | ({"one"} if n == 0 else set()))
                for n in range(20)]
        curious = Curiosity(plan)
        self.assertAlmostEqual(curious.gain("half"), 1.0, places=2)
        self.assertEqual(curious.gain("all"), 0.0)
        self.assertLess(curious.gain("one"), 0.4)


@requires_store
class QuestionShapeTests(unittest.TestCase):
    """Every generated question has to be a question v687 can read."""

    @classmethod
    def setUpClass(cls):
        cls.generator = Generator(POOL.engines[0], CURIOSITY)

    def test_articles(self):
        self.assertEqual(article("beagle"), "a")
        self.assertEqual(article("accordion"), "an")

    def test_a_stored_claim_reads_back_under_its_own_relation(self):
        self.assertEqual(phrase("collie", "capable_of", "swim"),
                         "can a collie swim")
        self.assertEqual(phrase("whale", "is_a", "fish"),
                         "is a whale a fish")
        self.assertEqual(phrase("violin", "made_of", "wood"),
                         "is a violin made of wood")

    def test_the_two_biggest_relations_have_frames(self):
        """`receives_action` is 556,871 facts and `has_a` is 86,748. Without
        frames both fell through to `is a X Y`."""
        from .question import FRAMES
        for relation in ("receives_action", "has_a", "capable_of",
                         "has_property"):
            self.assertIn(relation, FRAMES)

    def test_a_plural_object_takes_no_article(self):
        """`is a carp scales` was asked. `scales` is not a WordNet lemma and
        stripping the `es` gives `scal`, so it needs the lemmatiser."""
        self.assertEqual(self.generator.reask("carp", "has_a", "scales"),
                         "does a carp have scales")
        self.assertEqual(
            self.generator.reask("carp", "has_property",
                                 "distinctive fishy smell"),
            "does a carp have a distinctive fishy smell")

    def test_a_predicate_opening_with_its_own_verb_is_lemmatised(self):
        """`does a violin requires skill to play` is not a sentence."""
        self.assertEqual(
            phrase_predicate("violin", "requires skill to play", "",
                             "", self.generator.lemma("requires")),
            "does a violin require skill to play")

    def test_a_bare_attribute_takes_the_frame_its_part_of_speech_needs(self):
        self.assertEqual(phrase_predicate("collie", "furry", "a"),
                         "is a collie furry")
        self.assertEqual(phrase_predicate("collie", "tail", "n"),
                         "does a collie have a tail")
        self.assertEqual(phrase_predicate("collie", "ground", "n"),
                         "is a collie found in the ground")
        self.assertEqual(phrase_predicate("wolf", "meat", "n"),
                         "does a wolf eat meat")

    def test_an_awa2_coinage_is_not_asked_at_all(self):
        """`oldworld` and `quadrapedal` are real columns of a real corpus and
        nonsense as questions."""
        self.assertEqual(phrase_predicate("collie", "oldworld", "n"), "")
        self.assertEqual(phrase_predicate("collie", "quadrapedal", "n"), "")

    def test_curiosity_declines_a_word_the_store_barely_holds(self):
        """`purr` has one recorded fact and is a noun in WordNet, which was
        enough to produce `can a purr be airborne`."""
        self.assertFalse(self.generator.substantial("purr"))
        self.assertFalse(self.generator.substantial("hunt"))
        self.assertTrue(self.generator.substantial("cat"))
        self.assertTrue(self.generator.substantial("beagle"))

    def test_a_field_too_large_to_be_a_family_is_not_a_check(self):
        """`wing` resolves to the aircraft sense, whose nearest ancestor with
        kinds is `device` -- fifty-nine of them, including accordions. It
        asked `can an accordion play video`."""
        self.assertGreater(len(POOL.engines[0].profiles.subtypes("device")),
                           MAX_FAMILY)
        self.assertLessEqual(len(POOL.engines[0].profiles.subtypes("dog")),
                             MAX_FAMILY)

    def test_curiosity_compares_an_unknown_concept_against_its_kin(self):
        """`beagle` is not one of the 541, so its own rivals are all 541 and
        every predicate scores the same."""
        anchor, pool = self.generator.pool_for("beagle")
        self.assertEqual(anchor, "dog")
        self.assertIn("dog", pool)
        self.assertIn("wolf", pool)
        self.assertLess(len(pool), len(CURIOSITY.universe) / 4)


class SeedTests(unittest.TestCase):
    """What gets asked first, from what was said."""

    def test_a_question_is_asked_as_it_stands(self):
        self.assertEqual(seed_questions("does a beagle swim"),
                         ["does a beagle swim"])

    def test_a_statement_is_put_back_as_the_question_that_checks_it(self):
        self.assertEqual(seed_questions("a whale is a fish"),
                         ["is a whale a fish"])

    def test_a_statement_with_two_clauses_is_two_claims(self):
        """Checking `a beagle is a dog that hunts rabbits` as one claim checks
        neither of them."""
        asked = seed_questions("a beagle is a dog that hunts rabbits",
                               lambda word: "hunt" if word == "hunts" else word)
        self.assertEqual(asked, ["is a beagle a dog",
                                 "does a beagle hunt rabbits"])


@requires_store
class PoolTests(unittest.TestCase):
    def test_the_pool_reports_what_it_actually_built(self):
        self.assertEqual(POOL.workers, len(POOL.engines))
        self.assertGreater(POOL.build_seconds, 0)

    def test_every_engine_has_its_own_parser(self):
        """`Parser._spent` is written during a parse, so two threads sharing
        one parser corrupt each other's reading of a sentence. That is why
        this is a pool of engines and not a pool of threads."""
        parsers = {id(engine.parser) for engine in POOL.engines}
        self.assertEqual(len(parsers), POOL.workers)

    def test_a_question_that_throws_does_not_take_the_cycle_with_it(self):
        answer = POOL.ask_one("")
        self.assertTrue(answer.error or answer.verdict)


@requires_store
class LoopTests(unittest.TestCase):
    """The whole cycle, on the cases it was built for."""

    def test_a_weak_yes_is_put_to_its_own_family_and_loses(self):
        """The example the whole thing exists for. Neither answer is wrong;
        the disagreement is invisible to anything that asks once."""
        found = run("does a beagle swim")
        self.assertEqual(found.summary["verdict"], "VERIFIED")
        self.assertEqual(found.summary["trust"],
                         "contradicted by its own family")
        conflict = found.summary["conflicts"][0]
        denied = {row["question"] for row in conflict["against"]}
        self.assertIn("can a dog swim", denied)
        self.assertIn("can a collie swim", denied)

    def test_the_family_is_asked_in_one_cycle_not_one_at_a_time(self):
        """Corroboration is the breadth the pool exists for: the parent and
        every sibling go out together."""
        found = run("does a beagle swim")
        doubting = [cycle for cycle in found.cycles
                    if any(a.origin == "doubt" for a in cycle.answers)]
        self.assertTrue(doubting)
        self.assertGreaterEqual(
            len([a for a in doubting[0].answers if a.origin == "doubt"]), 4)

    def test_an_unknown_word_stops_and_asks_to_be_told(self):
        """Two asks. Nothing downstream of an unknown word means anything,
        and a third attempt would be the loop talking itself into an answer."""
        found = run("is a wemble a greeting")
        self.assertEqual(found.summary["trust"], "unreadable")
        self.assertLessEqual(found.as_dict()["asked"], 3)
        blocked = {hole["blocker"] for hole in found.summary["needs_telling"]}
        self.assertIn("wemble", blocked)

    def test_a_corroborated_answer_settles_without_a_conflict(self):
        found = run("does a robin fly")
        self.assertEqual(found.summary["trust"], "corroborated")
        self.assertEqual(found.summary["conflicts"], [])

    def test_absent_is_not_false(self):
        found = run("a whale is a fish")
        self.assertEqual(found.summary["trust"], "absent, not false")

    def test_curiosity_is_about_the_subject_not_the_object(self):
        """`does a snake have legs` asked `is a leg furry`, and `what eats
        meat` asked `can a meat walk`. Curiosity asks what a thing is like,
        and the object of a question is not what the question is about."""
        found = run("does a snake have legs")
        for cycle in found.cycles:
            for answer in cycle.answers:
                if answer.origin == "curiosity":
                    self.assertNotEqual(answer.about, "leg", answer.question)

    def test_attention_withdraws_from_a_topic_that_yields_nothing(self):
        """`meat` really is one of the corpus concepts, so the questions are
        legitimate -- and every one comes back UNKNOWN. The loop asks once
        and stops rather than spending another cycle on it."""
        found = run("what eats meat")
        curious = [answer for cycle in found.cycles
                   for answer in cycle.answers if answer.origin == "curiosity"]
        self.assertTrue(curious)
        self.assertTrue(all(a.verdict in ("UNKNOWN", "UNRECORDED", "NO_MATCH")
                            for a in curious))
        self.assertLessEqual(len(found.cycles), 3)

    def test_a_chain_never_starts_from_a_guess(self):
        """A run about whales went `does a goldfish have a gill` -> `is a
        goldfish a bony fish`: true, and about nothing."""
        for utterance in ("a whale is a fish", "does a beagle swim"):
            found = run(utterance)
            answers = {a.question: a for c in found.cycles for a in c.answers}
            for answer in answers.values():
                if answer.origin != "chain" or not answer.parent:
                    continue
                parent = answers.get(answer.parent)
                if parent is None:
                    continue
                with self.subTest(question=answer.question):
                    self.assertIn(parent.origin, ("seed", "gap", "chain"))

    def test_a_property_word_is_not_looked_up_as_a_thing(self):
        """`is a dog wild` left `wild` uncovered; `wild` has a noun sense --
        a wild region -- so `what is a wild` came back defining wilderness
        and the ladder walked off to geographical area, region, location."""
        found = run("is a dog wild")
        asked = {answer.question for cycle in found.cycles
                 for answer in cycle.answers}
        self.assertNotIn("what is a wild", asked)
        self.assertFalse({q for q in asked if "geographical" in q}, asked)

    def test_curiosity_never_asks_about_a_word_nobody_used(self):
        """Attention is the bound on curiosity. Without it the loop crawls
        the ontology: `what is a wemble` defines `greeting`, whose definition
        mentions `land`, and two cycles later it asks whether land is brown."""
        for utterance in ("does a beagle swim", "is a dog wild"):
            found = run(utterance)
            for cycle in found.cycles:
                for answer in cycle.answers:
                    if answer.origin != "curiosity":
                        continue
                    with self.subTest(question=answer.question):
                        self.assertIn(answer.about, found.buffer[
                            "activation"]["table"])

    def test_every_question_the_loop_generates_can_be_parsed(self):
        """A generated question that v687 cannot read is the loop inventing
        work for itself. UNPARSED is the failure this catches."""
        for utterance in ("does a beagle swim", "is a shark a fish",
                          "is a violin made of wood", "a whale is a fish"):
            found = run(utterance)
            for cycle in found.cycles:
                for answer in cycle.answers:
                    with self.subTest(question=answer.question):
                        self.assertNotEqual(answer.verdict, "UNPARSED")
                        self.assertFalse(answer.error, answer.error)

    def test_a_cycle_never_puts_out_more_questions_than_it_has_workers(self):
        for found in RUNS.values():
            for cycle in found.cycles:
                self.assertLessEqual(len(cycle.answers), LOOP.width)

    def test_a_family_check_is_not_cut_short_by_the_pool_size(self):
        """`can a dog fall into a hole` puts the claim to seven kinds of
        canine. At five workers the last two used to be dropped, and a
        conflict a wide pool found was one a narrow pool never saw."""
        found = run("can a dog fall into a hole")
        families: dict[str, set[str]] = {}
        for cycle in found.cycles:
            for answer in cycle.answers:
                if answer.origin == "doubt" and answer.parent:
                    families.setdefault(answer.parent, set()).add(answer.about)
        self.assertTrue(families)
        widest = max(len(kin) for kin in families.values())
        # The whole family, however many workers there were. `subtypes`
        # is the size it should have reached.
        self.assertGreater(widest, LOOP.width)


@requires_store
class ExampleTests(unittest.TestCase):
    """Every example on the page, against the claim its card makes.

    The examples are read out of `server.EXAMPLES` rather than copied here,
    so adding one to the page adds it to this test -- the same arrangement
    that keeps v687's page and its suite from drifting apart.
    """

    def test_every_example_settles_the_way_its_card_says(self):
        for example in server.EXAMPLES:
            with self.subTest(utterance=example["text"]):
                found = run(example["text"])
                self.assertEqual(found.summary["trust"], example["expect"])

    def test_every_example_has_something_to_show(self):
        for example in server.EXAMPLES:
            with self.subTest(utterance=example["text"]):
                found = run(example["text"])
                self.assertGreaterEqual(len(found.cycles), 1)
                self.assertTrue(found.summary["lines"])
                first = found.cycles[0].answers[0]
                self.assertTrue(first.steps or first.note,
                                "an example with nothing to open")

    def test_the_examples_between_them_exercise_every_generator(self):
        """A page of sixteen examples that only ever showed one source of
        questions would be a page about one third of the machinery."""
        origins: set[str] = set()
        for example in server.EXAMPLES:
            found = run(example["text"])
            for cycle in found.cycles:
                origins |= {answer.origin for answer in cycle.answers}
        self.assertEqual(origins, {"seed", "gap", "doubt", "split", "chain",
                                   "curiosity"})

    def test_some_example_reasons_in_a_line_rather_than_a_fan(self):
        """A page that only ever fanned out would be a page about breadth.
        The definition ladder is the case nineteen workers cannot shorten:
        each rung's subject is inside the previous rung's answer."""
        deepest = max(run(example["text"]).summary["depth"]
                      for example in server.EXAMPLES)
        self.assertGreaterEqual(deepest, 4)

    def test_a_divided_family_is_probed_rather_than_counted(self):
        """`1 of 3 deny it` is a count, not an answer. When the family
        disagrees the question is which side the subject is on, and it cannot
        be formed until the fan-out comes back."""
        found = run("does a cat purr")
        asked = [answer for cycle in found.cycles for answer in cycle.answers
                 if answer.origin == "split"]
        self.assertTrue(asked)
        self.assertIn("difference between", asked[0].question)

    def test_an_incidental_conflict_does_not_overturn_the_headline(self):
        """`is a shark a fish` stayed VERIFIED while something asked on the
        way past did not hold up. Reporting the second as the first says a
        shark is not a fish, which nothing here concluded."""
        found = run("does a cat purr")
        self.assertEqual(found.summary["verdict"], "VERIFIED")
        self.assertNotEqual(found.summary["trust"],
                            "contradicted by its own family")
        self.assertTrue(found.summary["conflicts"])

    def test_the_examples_cover_every_kind_of_outcome(self):
        outcomes = {example["expect"] for example in server.EXAMPLES}
        for expected in ("contradicted by its own family", "weakly held",
                         "unreadable", "absent, not false", "corroborated"):
            self.assertIn(expected, outcomes)


if __name__ == "__main__":
    unittest.main()
