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


class _QuietBuffer:
    """A buffer that raises nothing, so a factor can be tested on its own.

    The store no longer holds a family disagreement that is sound -- see
    `test_no_family_disagreement_in_the_store_is_currently_sound` -- so the
    factors that fire on one are asserted against a given conflict rather
    than a found one. Everything else about them is unchanged.
    """

    seen_doubts: tuple = ()

    def borne_out(self, question: str) -> bool:   # noqa: ARG002
        return False


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

    def test_the_beagle_yes_is_inherited_but_not_weakly_sourced(self):
        """It was called weak on a 0.60 floor that 92.5% of the store falls
        below. 0.42 is around Ascent++'s 78th percentile -- an above-average
        fact for where it came from. What is actually wrong with the answer
        is that it is inherited three levels and the family denies it, and
        saying `weak` alongside that was a claim the data does not support."""
        payload = POOL.engines[0].ask("does a beagle swim")
        self.assertEqual(payload["verdict"], "VERIFIED")
        reasons = {doubt.reason for doubt in gap.read_doubts(payload)}
        self.assertEqual(reasons, {"inherited", "assumed_sense"})
        self.assertFalse(gap.is_weak("ascentpp", 0.42))
        self.assertTrue(gap.is_weak("ascentpp", 0.10))
        self.assertFalse(gap.is_weak("conceptnet", 0.35),
                         "every conceptnet fact is 0.35; the number is not a"
                         " signal")

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
                         "is a collie on the ground")
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

    def test_the_family_machinery_works_on_a_disagreement_it_is_given(self):
        """The mechanism, asserted where it lives rather than through data.

        This test used to be `test_a_weak_yes_is_put_to_its_own_family_and_
        loses`, described as "the example the whole thing exists for", and it
        has now had two examples taken off it. `does a beagle swim` went when
        AwA2's zeros stopped being read as denials. `does a boat have sails`
        went when COMPS' foils did: `canoe NOT has sails` is a foil sampled
        out of a 1.58%-dense free listing, so nobody was ever asked whether a
        canoe has sails, and the norms positively assert `boat has sails`.

        See `test_no_family_disagreement_in_the_store_is_currently_sound` for
        why it is not simply pointed at a third example, and AUDIT.md §9.
        """
        from . import confidence

        class Fake:
            question = "does a boat have sails"
            verdict = "VERIFIED"
            payload: dict = {}

        alone = confidence.of_run(Fake(), _QuietBuffer(), [], overturned=False)
        denied = confidence.of_run(Fake(), _QuietBuffer(), [Fake()],
                                   overturned=False)
        self.assertIn("its family denied it", {one[0] for one in denied.factors})
        self.assertNotIn("its family denied it",
                         {one[0] for one in alone.factors})
        self.assertLess(denied.value, alone.value)

    def test_a_dispute_unsettles_a_yes_and_does_not_deny_it(self):
        """`AUDIT.md` §27: the teacher's no against a crawled yes nothing bore
        out leaves the claim unknown -- not a no, because a model does not
        outrank a record either."""
        from types import SimpleNamespace

        from . import confidence

        class Fake:
            question = "can a fish walk on land"
            verdict = "VERIFIED"
            payload: dict = {}

        said_no = SimpleNamespace(question=Fake.question, supports=False,
                                  confidence=0.996, settles=True)
        alone = confidence.of_run(Fake(), _QuietBuffer(), [],
                                  overturned=False)
        disputed = confidence.of_run(Fake(), _QuietBuffer(), [],
                                     overturned=False, challenged=said_no)
        self.assertEqual(alone.outcome, "verified")
        self.assertEqual(disputed.outcome, "unknown")
        self.assertIn("the teacher disputes it",
                      {one[0] for one in disputed.factors})

    @requires_store
    def test_a_yes_nothing_bore_out_is_put_to_the_teacher(self):
        """`can a fish walk on land` is VERIFIED on one Ascent++ row about
        mudskippers, and the run finds nothing for or against it. The teacher
        used to be asked only what the store left open, so nobody asked."""
        from .teacher import Teacher

        class Sure:
            available = True

            def __init__(self, supports):
                self.supports, self.asked = supports, []

            def judge(self, subject, fact, claim, style=""):
                self.asked.append(claim)
                return self.supports, 0.999, False

            review = Teacher.review
            challenge = Teacher.challenge

        for supports, trust, opens, outcome in (
                (False, "disputed by the teacher", "DISPUTED", "unknown"),
                (True, "unchallenged; the teacher agrees", "VERIFIED",
                 "verified")):
            with self.subTest(supports=supports):
                teacher = Sure(supports)
                found = Loop(POOL, CURIOSITY, max_cycles=6,
                             teacher=teacher).run("can a fish walk on land")
                self.assertIn("can a fish walk on land", teacher.asked)
                self.assertEqual(found.summary["trust"], trust)
                self.assertEqual(found.summary["outcome"], outcome)
                self.assertTrue(found.summary["lines"][0].startswith(opens))

    @requires_store
    def test_no_family_disagreement_in_the_store_is_currently_sound(self):
        """The finding, kept as a test so it is noticed if it stops being true.

        Sixty-eight questions were put through the loop looking for a family
        disagreement that is *correct*. Four fired and all four are artefacts:
        `does a rat swim`, `does a zebra run` and `does a deer run` on AwA2
        zeros -- the deer denied because `("red" and "run")` matched `red` and
        AwA2 says deer are not red -- and `does a moth fly`, denied by
        `can a tineoid fly in may`, a crawled predicate about May.

        Rats swim, zebras run, deer run and moths fly. If a genuine instance
        is ever found, put it here and give the machinery its example back.
        """
        for utterance in ("does a rat swim", "does a moth fly"):
            with self.subTest(utterance=utterance):
                found = run(utterance)
                for conflict in found.summary.get("conflicts") or []:
                    for row in conflict.get("against") or []:
                        self.assertNotEqual(
                            row.get("verdict"), "VERIFIED",
                            "a conflict that agrees is not a conflict")

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
        """The loop asks once and stops rather than spending another cycle.

        This used to assert that curiosity *fires* here, on the grounds that
        "`meat` really is one of the corpus concepts, so the questions are
        legitimate". They were not: the predicates on offer were
        `quadrapedal`, `fast` and `ground`, because the norms are 29% living
        kinds and those split the field best for anything. `is meat
        quadrapedal` is not a legitimate question, and `near_enough` now
        declines it -- which is the same withdrawal one step earlier.
        """
        found = run("what eats meat")
        curious = [answer for cycle in found.cycles
                   for answer in cycle.answers if answer.origin == "curiosity"]
        self.assertTrue(all(a.verdict in ("UNKNOWN", "UNRECORDED", "NO_MATCH")
                            for a in curious))
        self.assertLessEqual(len(found.cycles), 3)

    def test_a_predicate_is_not_asked_of_something_unlike_its_holders(self):
        """`does a person have whiskers`, reported from the page.

        `whiskers` is held by bear, buffalo, cow, elephant and fox, so it
        splits the norms beautifully and gain says it is the best question
        available. Gain measures what an answer would tell you and says
        nothing about whether the question belongs.
        """
        from .question import Generator

        maker = Generator(POOL.engines[0], CURIOSITY)
        for predicate in ("oldworld", "quadrapedal", "walks", "chewteeth"):
            with self.subTest(predicate=predicate):
                self.assertFalse(maker.near_enough("person", predicate))
        # and the field it was built for is untouched
        for predicate in ("quadrapedal", "walks"):
            self.assertTrue(maker.near_enough("dog", predicate))

    def test_the_branch_is_read_from_the_corpus_before_it_is_guessed(self):
        """Every sense guess this filter tried was wrong somewhere: `pig` is
        a foundry mould to `profiles.synset`, `mouse` a device to both it and
        `senses_of`, and `sheep` resolves to `person.n.01` -- one holder in
        44, which was enough to turn the whole filter off. XCSLB ships a
        category and it needs no resolving."""
        from .question import Generator

        maker = Generator(POOL.engines[0], CURIOSITY)
        for word in ("pig", "mouse", "sheep", "dog"):
            with self.subTest(word=word):
                self.assertEqual(maker.branch_of(word), "animal.n.01")
        self.assertEqual(maker.branch_of("apple"), "plant.n.02")
        self.assertEqual(maker.branch_of("hammer"), "artifact.n.01")

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
                    self.assertIn(parent.origin,
                                  ("seed", "gap", "chain", "require"))

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

    def test_an_action_is_checked_against_what_doing_it_needs(self):
        """`do fish run` rests on one crawled row. The store does not record
        what running needs of a body -- ConceptNet's prerequisites are about
        people, `find book` and `buy book` -- so it is derived: the things
        that can run have a leg, 12 of 90, 102 times commoner than among
        concepts at large."""
        from .graph import Requirements
        needs = Requirements(POOL.engines[0].reasoner)
        self.assertEqual(needs.of("run").part, "leg")
        self.assertEqual(needs.of("fly").part, "wing")

    def test_a_requirement_is_a_part_and_not_a_standing(self):
        """Unfiltered, running needs `reputation` and `the power`, because
        the crawl is mostly about running a business; walking needs a
        `friend` and reading needs a `parent`."""
        from .graph import Requirements
        needs = Requirements(POOL.engines[0].reasoner)
        for action in ("walk", "read"):
            found = needs.of(action)
            self.assertIsNone(found, found and found.part)

    def test_the_requirement_check_reaches_the_denial(self):
        """The payoff, and it takes two answers: derive the requirement and
        ask it of the fish.

        This used to take three. `does a fish have legs` came back VERIFIED
        -- on `fish has_part "no legs"`, whose negation lives in the object
        column where nothing read it -- and the family fan-out was what
        eventually overturned it. R3 reads the object now, so the denial
        arrives on the first answer and the fan-out is not needed. Fewer
        steps to the same place is the point; the assertion is on the
        conclusion, not on the route."""
        found = run("do fish run")
        asked = [answer for cycle in found.cycles for answer in cycle.answers
                 if answer.origin == "require"]
        self.assertEqual([a.question for a in asked], ["does a fish have legs"])
        self.assertEqual(asked[0].verdict, "CONTRADICTED")
        self.assertEqual(found.summary["trust"],
                         "not supported by the rest of the store")

    def test_a_requirement_already_settled_is_not_asked(self):
        """`does a beagle swim` derived `tooth` -- swimmers have teeth, 15 of
        54, because swimmers are animals -- and asked `does a beagle have
        teeth`. The store settles that before a worker is spent on it, and it
        grounded nothing when it came back.

        The derivation is left alone; what changed is whether the question is
        worth putting. `animal.n.01` is what makes the difference: its `has a
        leg` and `has a wing` are the rows R19 refuses to inherit, so a fish
        and its legs stay an open question while a beagle and its teeth,
        recorded on `dog.n.01`, do not."""
        found = run("does a beagle swim")
        asked = [a.question for cycle in found.cycles for a in cycle.answers
                 if a.origin == "require"]
        self.assertEqual(asked, [])
        self.assertEqual(found.summary["trust"], "corroborated")
        needs = self.requirements()
        self.assertTrue(needs.recorded_of("beagle.n.01", "tooth"))
        self.assertFalse(needs.recorded_of("fish.n.01", "leg"))

    def requirements(self):
        from .graph import Requirements
        return Requirements(POOL.engines[0].reasoner)

    def test_an_awa2_zero_is_not_a_denial(self):
        """`does a dog swim` came back denied, with all three kinds of dog
        the norms cover lined up behind it. Those three are AwA2 rows
        annotated 0 for `swims`, and a 0 there means the attribute is not
        characteristic of the class -- the same zeros deny that a collie has
        claws or muscle, or is ever black.

        A zero stands unless another source states the same thing plainly of
        the concept or a class tight enough to speak for it. `dog capable_of
        "swim"` is Ascent++ at 0.68, its 97th percentile, so the zero on
        `swims` is a disagreement between sources rather than a no. Nothing
        says a dog flies, so that zero is untouched."""
        engine = POOL.engines[0]
        self.assertEqual(engine.ask("does a dog swim")["verdict"], "VERIFIED")
        self.assertEqual(engine.ask("does a collie swim")["verdict"],
                         "VERIFIED")
        self.assertEqual(engine.ask("can a dog fly")["verdict"],
                         "CONTRADICTED")
        # Adjectives are left alone in both directions: AwA2's colours and
        # sizes are its least reliable zeros and the crawl's are no better,
        # so `is a bobcat white` keeps its no.
        self.assertEqual(engine.ask("is a bobcat white")["verdict"],
                         "CONTRADICTED")

    def test_only_a_decisive_requirement_grounds_a_denial(self):
        """Saying "the no is not about anatomy" claims to know what the
        anatomy is for. `does a dog swim` said "a dog does have teeth, which
        is what the things that do it have in common" -- true, and nothing to
        do with swimming.

        Only flying has a part that stands out from the anatomy its doers
        share, so only there is the line worth putting."""
        from .graph import Requirements
        needs = Requirements(POOL.engines[0].reasoner)
        self.assertTrue(needs.of("fly").decisive)
        for action in ("swim", "run", "climb", "jump"):
            with self.subTest(action=action):
                found = needs.of(action)
                self.assertFalse(found.decisive, f"{action} -> {found.part}")

    def test_a_denial_is_grounded_too(self):
        """A penguin cannot fly and does have wings, which says the no is not
        about anatomy. Only checking positives would have missed that."""
        found = run("can a penguin fly")
        asked = [a for c in found.cycles for a in c.answers
                 if a.origin == "require"]
        self.assertTrue(asked)
        self.assertEqual(asked[0].question, "does a penguin have wings")

    def test_a_definition_ladder_only_runs_when_one_was_asked_for(self):
        """`does a snake run` opened a gap on `snake`, whose definition v687
        read as the winding-river sense, and the ladder walked river ->
        stream -> body of water. Defining the words in your own follow-ups is
        not reasoning."""
        found = run("does a snake run")
        asked = {a.question for c in found.cycles for a in c.answers}
        self.assertFalse({q for q in asked if q.startswith("what is a river")},
                         asked)
        ladder = run("what is a beagle")
        self.assertGreaterEqual(ladder.summary["depth"], 3)

    def test_curiosity_reads_the_graph_and_not_only_the_norms(self):
        """The feature norms cover 541 of 45,219 concepts and give the same
        six AwA2 columns to all of them. The graph gives a concept its own
        relatives."""
        from .graph import GraphCuriosity
        wider = GraphCuriosity(POOL.engines[0].reasoner)
        found = wider.expectations(wider.sense_of("violin"), 4)
        self.assertTrue(found)
        self.assertIn("make music", {one.object for one in found})

    def test_a_habitat_column_is_lived_on_or_in_but_never_found_in(self):
        """`is a dog found in the ground` is a question about burial; v687
        strips the preposition before matching, so it answers `on` and `in`
        alike and the mistake is invisible from the answer. And the frame
        carries no verb, because `live` is scored as a content term of its
        own and matches `lives in a stable`, which the norms deny of dogs."""
        self.assertEqual(phrase_predicate("dog", "ground", "n"),
                         "is a dog on the ground")
        self.assertEqual(phrase_predicate("fish", "water", "n"),
                         "is a fish in the water")

    def test_a_multi_word_concept_is_not_a_single_word_question(self):
        """v687 made `pig bed.n.01` -- a mould for casting pig iron -- the
        primary sense of `pig`, because the crawl has more rows about foundry
        beds than about pigs. Fixed in v687's `senses_of`, which is the only
        change this branch makes there."""
        senses = POOL.engines[0].reasoner.senses_of("pig")
        self.assertTrue(senses)
        self.assertNotIn(" ", senses[0]["id"].split(".")[0])

    def test_a_fact_about_a_few_is_not_a_fact_about_the_class(self):
        """`do pigs fly` rests on `mammal capable_of fly`, true of bats and
        false of the other kinds the store knows. The shape is general: a
        claim inherited from an ancestor that a minority of that ancestor's
        own kinds bear out."""
        found = run("do pigs fly")
        self.assertEqual(found.summary["verdict"], "UNKNOWN")
        # R19 inside v687 is where this shape is caught now. The v688-level
        # version of it needed a conflict, and the two the page had were both
        # absence read as denial.
        note = found.cycles[0].answers[0].note or ""
        self.assertIn("kinds of", note)

    def test_a_no_another_reading_would_answer_yes_is_not_produced_any_more(self):
        """This asserted the repair. It now asserts there is nothing to repair.

        `is a mouse an animal` came back CONTRADICTED about `mouse.n.04`, the
        device, and the loop re-asked it pinned to `mouse.n.01` and led with
        the yes. v687 picks the rodent itself now: `animal` places the
        question in one branch of the taxonomy and one mouse sense is in it.

        The pin the loop used to supply and the sense v687 now chooses are the
        same one, which is the point -- the repair was real and so is its
        being unnecessary.
        """
        found = run("is a mouse an animal")
        self.assertEqual(found.summary["verdict"], "VERIFIED")
        self.assertEqual(found.cycles[0].answers[0].payload["concept"],
                         "mouse.n.01")
        self.assertFalse([a for c in found.cycles for a in c.answers
                          if a.origin == "sense"])
        # `as_asked` kept what v687 said about the sense it chose, and it
        # agrees with the headline now because there is no longer a correction
        # to record. That equality is the whole finding.
        self.assertEqual(found.summary["as_asked"], "VERIFIED")

    def test_the_same_words_under_two_readings_are_two_questions(self):
        """Keyed by text alone, a pinned re-ask would be deduplicated against
        the answer it exists to disagree with.

        Asserted on the key rather than through a run: the loop no longer
        produces a pinned re-ask anywhere, because the one example that made
        it — `is a mouse an animal` — is answered correctly by v687 now. The
        keying is still what stops the two collapsing if it ever does.
        """
        from .pool import Answer

        plain = Answer(question="is a mouse an animal", payload={}, worker=0,
                       started=0.0, elapsed=0.0)
        pinned = Answer(question="is a mouse an animal", payload={}, worker=0,
                        started=0.0, elapsed=0.0,
                        pins={"mouse": "mouse.n.01"})
        self.assertEqual(plain.key, "is a mouse an animal")
        self.assertNotEqual(pinned.key, plain.key)
        self.assertTrue(pinned.key.startswith("is a mouse an animal "))

    @requires_store
    def test_the_sense_generator_has_no_live_example(self):
        """The finding, kept so it is noticed if it stops being true.

        `sense` re-asks when v687's answer came back about a different word.
        Its only instance on the page was `is a mouse an animal`, and v687
        picking the rodent itself removed it. Twenty-four candidates were
        tried — `why does a dog bark`, `is a crane a bird`, `is a bass a
        fish`, `is a date a fruit` and so on — and none fires it.

        This is the fourth piece of v688 machinery to lose its example to a
        v687 fix or a data correction in one sitting. AUDIT.md §10.
        """
        origins = {a.origin for c in run("is a mouse an animal").cycles
                   for a in c.answers}
        self.assertNotIn("sense", origins)

    def test_a_means_never_opens_a_gap_of_its_own(self):
        """Checking the penguin family asked `is an emperor penguin strong`,
        which came back UNRECORDED -- and the loop went off to find out what
        an emperor is, what a king is, and what a jackass is."""
        found = run("can a penguin fly")
        asked = {a.question for c in found.cycles for a in c.answers}
        for stray in ("what is an emperor", "what is a king",
                      "what is a jackass"):
            self.assertNotIn(stray, asked)

    def test_the_examples_between_them_exercise_every_generator(self):
        """A page of sixteen examples that only ever showed one source of
        questions would be a page about one third of the machinery."""
        origins: set[str] = set()
        for example in server.EXAMPLES:
            found = run(example["text"])
            for cycle in found.cycles:
                origins |= {answer.origin for answer in cycle.answers}
        # `sense` is absent, and its absence is a finding rather than a gap in
        # the page. It fired on exactly one example -- `is a mouse an animal`,
        # where v687 answered CONTRADICTED about `mouse.n.04`, the device --
        # and v687 now picks the rodent itself, because `animal` places the
        # question in a branch and one mouse sense is in it. Twenty-four
        # candidate questions were tried for a replacement and none fires it.
        # See `test_the_sense_generator_has_no_live_example` and AUDIT.md §10.
        self.assertEqual(origins, {"seed", "gap", "doubt", "split",
                                   "chain", "require", "curiosity"})

    def test_some_example_reasons_in_a_line_rather_than_a_fan(self):
        """A page that only ever fanned out would be a page about breadth.
        The definition ladder is the case nineteen workers cannot shorten:
        each rung's subject is inside the previous rung's answer."""
        deepest = max(run(example["text"]).summary["depth"]
                      for example in server.EXAMPLES)
        self.assertGreaterEqual(deepest, 3)

    def test_a_divided_family_is_probed_rather_than_counted(self):
        """`1 of 3 deny it` is a count, not an answer. When the family
        disagrees the question is which side the subject is on, and it cannot
        be formed until the fan-out comes back."""
        found = run("does a cat purr")
        asked = [answer for cycle in found.cycles for answer in cycle.answers
                 if answer.origin == "split"]
        self.assertTrue(asked)
        self.assertIn("difference between", asked[0].question)

    def test_only_an_undermining_doubt_weakens_a_hold(self):
        """`is a dog an animal` -- WordNet, 0.95, two levels up -- read
        `weakly held`, because `inherited` and `assumed_sense` fired on
        almost every answer the loop ever gave. They are how a taxonomy and
        a crawl work, not defects of a particular answer."""
        for utterance in ("is a dog an animal", "does a bird have wings"):
            with self.subTest(utterance=utterance):
                found = run(utterance)
                self.assertNotEqual(found.summary["trust"], "weakly held")

    def test_corroborated_means_something_bore_it_out(self):
        """`is a dog wild` rests on one fact and the family returns two
        shrugs and a yes. Nothing contradicted it, which is not the same as
        corroboration, and calling both the same makes the word useless."""
        self.assertEqual(run("is a dog wild").summary["trust"], "unchallenged")
        self.assertEqual(run("does a robin fly").summary["trust"],
                         "corroborated")

    def test_a_no_scored_one_word_at_a_time_is_flagged(self):
        """`is a violin made of wood` is denied by scoring `made` and `wood`
        apart and failing one of them -- the norms record "can be made of
        ebony" as false, and denying one wood does not deny wood. The claim
        as asked was never put to anything, v687 says so in its own note, and
        a correct denial does not.

        This was `does a cow eat grass` until R3 learned that negation scopes
        forward. `cattle capable_of "digest grass but humans cannot"` asserts
        the grass and denies it of humans, and reading the trailing `cannot`
        as a denial made that a no. It is a yes now, so the detector needed a
        denial that is still wrong; the violin is the one left.

        The assertion is on the doubt, not on the headline. `off_target`
        fires on the violin too and outranks this one for the summary line,
        and which of two true complaints gets shown is a presentation choice
        -- that the pair was scored apart is the finding under test.

        **Third example, same cause.** The violin's denial rested on `can be
        made of ivory`, which is a COMPS foil rather than anything anyone
        judged false, so with the foils gone the violin reads absent and
        `scored_apart` has no instance either. What survives is the same
        complaint on a *yes*: `does a boat have sails` is VERIFIED on
        `can sail`, which shares no word with `sails`. One term was enough to
        reach it, which is the finding; `off_target` is the reason the loop
        files it under.
        """
        found = run("does a boat have sails")
        doubts = [one["reason"] for one in found.summary["doubts"]]
        self.assertIn("off_target", doubts, doubts)
        self.assertTrue(any("can sail" in line
                            for line in found.summary["lines"]))
        self.assertEqual(run("is a violin made of wood").summary["trust"],
                         "absent, not false")
        self.assertEqual(run("can a dog fly").summary["trust"],
                         "denied, unchallenged")

    def test_no_source_file_carries_a_mangled_escape(self):
        """Three regexes in this tree have had their word boundaries turned
        into literal backspace characters in transit, and each one silently
        stopped matching -- including one in v687 that disabled a guard
        outright. A control character in source is never intended."""
        import pathlib
        here = pathlib.Path(__file__).parent
        for folder in (here, here.parent / "v687"):
            for source in folder.glob("*.py"):
                text = source.read_text(encoding="utf-8")
                stray = [character for character in text
                         if ord(character) < 9 or 11 <= ord(character) < 32]
                with self.subTest(source=source.name):
                    self.assertFalse(stray, f"{source}: {stray!r}")

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
        for expected in ("not supported by the rest of the store",
                         "unchallenged", "unreadable", "absent, not false",
                         "corroborated"):
            self.assertIn(expected, outcomes)


class ReadingTests(unittest.TestCase):
    """Four outcomes and a number, in place of seventeen verdicts and none.

    The seventeen still decide the repair; `gap.kind_of` reads them and this
    does not. What changed is what a person reads off the page.
    """

    def test_every_verdict_reads_as_one_of_four(self):
        """Including ones this map has never seen: an unlisted verdict has
        settled nothing, and `unknown` is the reading that says so."""
        from . import confidence, gap
        every = (gap.ABOUT_THE_WORLD | gap.ABOUT_CONTENT | gap.ABOUT_COVERAGE
                 | gap.ABOUT_THE_QUESTION | gap.ABOUT_RELIABILITY)
        for verdict in sorted(every) + ["ERROR", "", "SOMETHING_NEW"]:
            with self.subTest(verdict=verdict):
                self.assertIn(confidence.outcome_of(verdict),
                              ("verified", "denied", "unknown", "retrieved"))
        self.assertEqual(confidence.outcome_of("HELD"), "verified")
        self.assertEqual(confidence.outcome_of("CONTRADICTED"), "denied")
        self.assertEqual(confidence.outcome_of("DEFINED"), "retrieved")
        self.assertEqual(confidence.outcome_of("UNRECORDED"), "unknown")

    def test_a_fact_is_placed_within_its_own_source(self):
        """The finding this rests on: 0.42 sounds low and is Ascent++'s 78th
        percentile. ConceptNet and WordNet write one number on every row, so
        for them the number is not a signal and the source is."""
        from .confidence import percentile
        self.assertAlmostEqual(percentile("ascentpp", 0.42)[0], 0.78, places=2)
        self.assertAlmostEqual(percentile("ascentpp", 0.157)[0], 0.25,
                               places=2)
        for source in ("conceptnet", "wordnet"):
            with self.subTest(source=source):
                self.assertIn("says nothing", percentile(source, 0.35)[1])

    def test_subsumption_does_not_decay_with_distance(self):
        """`is a beagle a dog` read 0.55 -- medium confidence that a beagle
        is a dog -- because the dog is three levels up and R5 decays what is
        borrowed. Subsumption is not borrowed: the walk is the proof."""
        found = run("is a beagle a dog")
        self.assertEqual(found.summary["outcome"], "verified")
        self.assertEqual(found.summary["band"], "high")
        self.assertTrue(any(one["name"] == "exact"
                            for one in found.summary["factors"]),
                        found.summary["factors"])

    def test_silence_carries_no_confidence(self):
        """A number beside `unknown` would be read as a weakly held claim,
        and there is no claim. `do pigs fly` is not a faint yes."""
        found = run("do pigs fly")
        self.assertEqual(found.summary["outcome"], "unknown")
        self.assertEqual(found.summary["confidence"], 0.0)

    def test_the_loop_is_what_moves_the_number(self):
        """The loop's own contribution to the reading: a single ask cannot
        produce it, because a single ask has nothing to compare against.

        Asserted on a doubt rather than on a conflict, because the store has
        no sound conflict left to assert on. `does a boat have sails` still
        makes the point in the direction that matters -- v687 concludes
        VERIFIED, and the number comes down because the loop found the
        verdict resting on a predicate nobody asked about.
        """
        found = run("does a boat have sails")
        self.assertEqual(found.summary["outcome"], "verified")
        named = {one["name"] for one in found.summary["factors"]}
        self.assertIn("1 doubt(s)", named)
        doubt = next(one for one in found.summary["factors"]
                     if one["name"] == "1 doubt(s)")
        self.assertLess(doubt["factor"], 1.0)

    def test_every_factor_is_reported_with_its_reason(self):
        """A confidence that cannot be argued with is one to be suspicious
        of, so nothing goes into the product without saying why."""
        for utterance in ("is a dog an animal", "does a beagle swim",
                          "what is a beagle"):
            with self.subTest(utterance=utterance):
                for one in run(utterance).summary["factors"]:
                    self.assertTrue(one["why"].strip(), one)
                    self.assertIsInstance(one["factor"], float)

    def test_every_answer_carries_its_own_reading(self):
        """Not only the headline: every row in the table is one of the four,
        with its own number."""
        found = run("does a beagle swim")
        rows = [a.as_dict(False) for c in found.cycles for a in c.answers]
        self.assertTrue(rows)
        for row in rows:
            with self.subTest(question=row["question"]):
                self.assertIn(row["outcome"],
                              ("verified", "denied", "unknown", "retrieved"))
                self.assertIn(row["band"], ("low", "medium", "high"))
                self.assertGreaterEqual(row["confidence"], 0.0)
                self.assertLessEqual(row["confidence"], 1.0)


class ServiceTests(unittest.TestCase):
    """The page is served a thread per request, over engines that are not."""

    def test_runs_do_not_overlap(self):
        """Two runs at once share `engines[0]` outside the pool's queue, and
        that is one sqlite connection: `does a beagle swim` came back
        `InterfaceError` while the page was asking something else."""
        import threading
        import time
        from types import SimpleNamespace

        service = server.Service.__new__(server.Service)
        service._lock = threading.Lock()
        service._engines = threading.Lock()
        service._cache = {}
        inside, most = [0], [0]

        class Slow:
            def run(self, utterance, pinned=None):
                inside[0] += 1
                most[0] = max(most[0], inside[0])
                time.sleep(0.05)
                inside[0] -= 1
                return SimpleNamespace(as_dict=lambda: {"said": utterance})

        service.loop = Slow()
        threads = [threading.Thread(target=service.run, args=(f"q{n}",))
                   for n in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(most[0], 1)

    def test_a_reader_who_waited_gets_the_run_it_waited_for(self):
        import threading
        import time
        from types import SimpleNamespace

        service = server.Service.__new__(server.Service)
        service._lock = threading.Lock()
        service._engines = threading.Lock()
        service._cache = {}
        calls = []

        class Slow:
            def run(self, utterance, pinned=None):
                calls.append(utterance)
                time.sleep(0.05)
                return SimpleNamespace(as_dict=lambda: {"said": utterance})

        service.loop = Slow()
        threads = [threading.Thread(target=service.run, args=("same",))
                   for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(calls, ["same"])


if __name__ == "__main__":
    unittest.main()
