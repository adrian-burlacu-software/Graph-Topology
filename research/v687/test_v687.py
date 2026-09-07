"""The reasoning the v686 audit found missing.

    python -m unittest research.v687.test_v687 -v

Data-dependent tests skip themselves when the norms have not been pulled in.
The inherited suites (`test_trie`, `test_reasoning`, `test_bridging`,
`test_norms`) carry v683 through v686 unchanged and must stay green: nothing
here is allowed to cost an answer that already worked.
"""
from __future__ import annotations

import unittest

from research.v687 import build, corpora, logic
from research.v687.analogy import Analogies
from research.v687.contrast import Contrast
from research.v687.causal import Causal
from research.v687.identify import Identifier
from research.v687.inverse import SUBJECT, Inverse
from research.v687.language import Parser
from research.v687.profile import ASIDE, Profiles
from research.v687.reason import Reasoner

STORE = build.DEFAULT_STORE.with_name("v684_reasoning_compressed.sqlite")
HAVE_NORMS = (corpora.XCSLB_DIR / "comps_base.jsonl").exists() and \
             (corpora.AWA2_DIR / "classes.txt").exists()
requires_norms = unittest.skipUnless(HAVE_NORMS, "feature norms not downloaded")
requires_store = unittest.skipUnless(STORE.exists(), f"no store at {STORE}")


class KleeneTests(unittest.TestCase):
    """R20. Three values, because silence is not falsehood."""

    def test_one_false_conjunct_settles_a_conjunction(self):
        self.assertEqual(logic.conjoin([logic.TRUE, logic.FALSE,
                                        logic.UNKNOWN]), logic.FALSE)

    def test_an_unknown_conjunct_suspends_rather_than_sinks(self):
        self.assertEqual(logic.conjoin([logic.TRUE, logic.UNKNOWN]),
                         logic.UNKNOWN)

    def test_one_true_disjunct_settles_a_disjunction(self):
        self.assertEqual(logic.disjoin([logic.UNKNOWN, logic.TRUE]),
                         logic.TRUE)

    def test_negation_leaves_the_unknown_unknown(self):
        self.assertEqual(logic.negate(logic.UNKNOWN), logic.UNKNOWN)
        self.assertEqual(logic.negate(logic.TRUE), logic.FALSE)

    def test_the_question_is_read_into_a_tree(self):
        query = logic.parse("a tail and wings", ASIDE)
        self.assertEqual(query.tree.op, "and")
        self.assertEqual(sorted(query.tree.terms()), ["tail", "wings"])
        self.assertEqual(logic.parse("furry or purple", ASIDE).tree.op, "or")
        self.assertEqual(logic.parse("not furry", ASIDE).tree.op, "not")

    def test_a_quantifier_is_not_also_a_negation(self):
        """`no birds fly` is quantified, not negated, and reading `no` as
        both answers the opposite question."""
        query = logic.parse("no fly", ASIDE)
        self.assertEqual(query.quantifier, "none")
        self.assertEqual(query.tree.op, "term")

    def test_quantifiers_are_recognised(self):
        for text, want in (("all fly", "all"), ("some fly", "some"),
                           ("most fly", "most"), ("no fly", "none"),
                           ("fly", None)):
            self.assertEqual(logic.parse(text, ASIDE).quantifier, want, text)


class GatingTests(unittest.TestCase):
    """R18. Refusing by name is the feature."""

    def test_the_constructions_that_have_no_rule_are_refused(self):
        for question in ("is a whale bigger than a dolphin",
                         "how many legs does a dog have",
                         "what is the largest animal",
                         "what if a dog had wings"):
            self.assertIsNotNone(logic.unsupported(question), question)

    def test_counting_kinds_is_not_refused_with_counting_parts(self):
        """The taxonomy can count kinds; nothing can count legs."""
        self.assertIsNone(logic.unsupported("how many kinds of dog are there"))

    def test_ordinary_questions_pass(self):
        for question in ("what can a violin do", "is a whale furry",
                         "what kind of dog has spots", "why do people sleep"):
            self.assertIsNone(logic.unsupported(question), question)


@requires_norms
@requires_store
class CompositionTests(unittest.TestCase):
    """R20 over the norms: the defect that started the backlog."""

    @classmethod
    def setUpClass(cls):
        cls.identifier = Identifier(STORE)
        cls.profiles = Profiles(cls.identifier)

    @classmethod
    def tearDownClass(cls):
        cls.identifier.close()

    def assess(self, name, text):
        return self.profiles.assess(name, logic.parse(text, ASIDE))

    def test_asking_for_two_things_is_not_answered_by_one(self):
        """v686 returned on the first term that matched, so this came back
        VERIFIED on the strength of the tail."""
        answer = self.assess("dog", "tail and wings")
        self.assertNotEqual(answer.verdict, "HELD")
        self.assertIn("tail", answer.parts)
        self.assertIn("wings", answer.parts)

    def test_a_disjunction_needs_only_one_side(self):
        self.assertEqual(self.assess("dog", "furry or purple").verdict, "HELD")

    def test_an_unknown_part_suspends_a_conjunction(self):
        answer = self.assess("dog", "furry and telepathic")
        self.assertEqual(answer.verdict, "UNRECORDED")
        self.assertIn("unsettled", answer.detail)

    def test_all_is_refuted_by_one_counterexample(self):
        answer = self.assess("bird", "all fly")
        self.assertEqual(answer.verdict, "DENIED")
        self.assertEqual(answer.quantifier, "all")
        self.assertTrue(any(m["verdict"] == "DENIED" for m in answer.members))

    def test_some_and_most_are_different_questions(self):
        self.assertEqual(self.assess("bird", "some fly").verdict, "HELD")
        self.assertEqual(self.assess("bird", "most fly").verdict, "HELD")
        self.assertEqual(self.assess("bird", "no fly").verdict, "DENIED")

    def test_a_class_the_norms_do_not_cover_is_still_quantifiable(self):
        """`bird` is not one of XCSLB's 521 concepts; 29 of them are birds."""
        self.assertFalse(self.profiles.knows("bird"))
        self.assertGreater(len(self.profiles.subtypes("bird")), 20)


@requires_norms
@requires_store
class CorroborationTests(unittest.TestCase):
    """R19. One crawled sentence is not a property of a category."""

    @classmethod
    def setUpClass(cls):
        cls.identifier = Identifier(STORE)
        cls.profiles = Profiles(cls.identifier)

    @classmethod
    def tearDownClass(cls):
        cls.identifier.close()

    def test_a_fact_its_kinds_bear_out_is_inherited(self):
        bearing, kinds = self.profiles.corroboration("bird.n.01", "fly")
        self.assertGreater(bearing / kinds, 1 / 3)
        self.assertEqual(self.profiles.verify_one("robin", "fly").verdict,
                         "INHERITED")

    def test_a_fact_its_kinds_refute_is_not(self):
        """`animal.n.01 has a wing` made every dog winged."""
        bearing, kinds = self.profiles.corroboration("animal.n.01", "wings")
        self.assertLess(bearing / kinds, 1 / 3)
        self.assertEqual(self.profiles.verify_one("dog", "wings").verdict,
                         "UNRECORDED")

    def test_a_thin_sample_cannot_refute(self):
        """Four whales are not evidence about whales in general."""
        _, kinds = self.profiles.corroboration("whale.n.02", "sing")
        self.assertLess(kinds, 8)

    def test_votes_from_below_must_be_independent(self):
        """Three breeds re-inheriting one sentence are one witness, not
        three, so induction counts stated evidence only."""
        below = self.profiles._from_below("dog", ["wings"])
        self.assertIsNone(below)


@requires_norms
@requires_store
class ContrastTests(unittest.TestCase):
    """R21. Four question types, one operation."""

    @classmethod
    def setUpClass(cls):
        cls.identifier = Identifier(STORE)
        cls.contrast = Contrast(Profiles(cls.identifier))

    @classmethod
    def tearDownClass(cls):
        cls.identifier.close()

    def test_shared_and_distinct_are_both_reported(self):
        found = self.contrast.compare("dog", "wolf")
        self.assertGreater(found.shared_total, 0)
        self.assertTrue(set(found.only_left).isdisjoint(found.only_right))
        self.assertNotIn(found.shared[0], found.only_left)

    def test_storage_order_does_not_group_by_similarity(self):
        """The finding, asserted: two whales share nearly half their
        properties and part at the first node of the stored trie."""
        found = self.contrast.compare("killer whale", "blue whale")
        self.assertGreater(found.jaccard, 0.4)
        self.assertEqual(len(found.together), 0)

    def test_nearest_is_computed_and_can_say_why(self):
        near = self.contrast.nearest("robin")
        self.assertTrue({"canary", "wren"} & {n["name"] for n in near[:4]})
        self.assertTrue(near[0]["because"])

    def test_typicality_ranks_a_member_against_its_class(self):
        found = self.contrast.typicality("dalmatian", "dog")
        self.assertTrue(found.sound)
        self.assertEqual(found.rank, 1)

    def test_typicality_says_when_the_class_has_no_core(self):
        """XCSLB elicitation is free and sparse: the most ordinary bird
        carries under half of what counts as a bird's core."""
        found = self.contrast.typicality("robin", "bird")
        self.assertFalse(found.sound)


@requires_store
class InverseTests(unittest.TestCase):
    """R22. The graph read from the object side."""

    @classmethod
    def setUpClass(cls):
        cls.reasoner = Reasoner(STORE)
        cls.inverse = Inverse(cls.reasoner)

    @classmethod
    def tearDownClass(cls):
        cls.reasoner.close()

    def test_backwards_questions_are_recognised(self):
        for question, relation in (("what is made of wood", "made_of"),
                                   ("what is found in a toolbox", "at_location"),
                                   ("what causes fire", "causes"),
                                   ("who makes a car", "created_by")):
            read = self.inverse.reads(question)
            self.assertIsNotNone(read, question)
            self.assertEqual(read[0], relation, question)

    def test_a_question_that_names_its_subject_is_left_alone(self):
        """v684 owns those, and the capability catch-all must not take them."""
        self.assertIsNone(self.inverse.reads("what can a violin do"))
        self.assertIsNone(self.inverse.reads("is a dog an animal"))

    def test_the_toolbox_is_answered_from_the_object_side(self):
        found = self.inverse.find("at_location", "toolbox")
        names = {row["concept"].split(".")[0] for row in found.subjects}
        self.assertTrue({"hammer", "screwdriver"} & names, names)

    def test_a_paired_relation_is_read_from_the_other_column(self):
        """`what has wings` must not answer with the parts of a wing."""
        found = self.inverse.find("has_part", "wings", SUBJECT)
        names = {row["concept"].split(".")[0] for row in found.subjects}
        self.assertNotIn("aileron", names)
        self.assertTrue({"bird", "bat", "angel"} & names, names)

    def test_matching_is_whole_word(self):
        self.assertTrue(Inverse._names("made of wood", ["wood"]))
        self.assertFalse(Inverse._names("wooden spoon", ["wood"]))

    def test_a_denial_does_not_answer_a_question_asking_for_the_thing(self):
        self.assertTrue(Inverse._denied("does not eat meat"))
        self.assertFalse(Inverse._denied("eat meat"))


@requires_store
class CausalTests(unittest.TestCase):
    """R23. Scripts, and abduction as a ranking."""

    @classmethod
    def setUpClass(cls):
        cls.reasoner = Reasoner(STORE)
        cls.causal = Causal(cls.reasoner,
                            parser=Parser(vocabulary=cls.reasoner.vocabulary()))

    @classmethod
    def tearDownClass(cls):
        cls.reasoner.close()

    def test_the_three_question_shapes(self):
        self.assertEqual(self.causal.reads("why do people sleep")[0], "why")
        self.assertEqual(self.causal.reads("what happens when you cook")[0],
                         "then")
        self.assertEqual(self.causal.reads("what explains a fire")[0],
                         "explain")
        self.assertIsNone(self.causal.reads("what can a violin do"))

    def test_a_script_runs_in_script_order(self):
        found = self.causal.script("drive", question="what happens when you drive")
        phases = [step["phase"] for step in found.steps]
        order = {"before": 0, "during": 1, "after": 2, "then": 3,
                 "in order to": 4, "wants": 5}
        self.assertEqual(phases, sorted(phases, key=lambda p: order[p]))

    def test_a_script_is_word_level_and_says_so(self):
        """ConceptNet records these of a word and the build attached them to
        whatever synset it could find -- `cook.n.02` is Captain Cook."""
        found = self.causal.script("cook", question="what happens when you cook")
        self.assertEqual(found.concept, "cook")
        self.assertIn("Word-level", found.note)

    def test_abduction_ranks_rather_than_lists(self):
        found = self.causal.explains("fire")
        self.assertGreater(found.considered, 5)
        scores = [h.score for h in found.hypotheses]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_a_cause_that_causes_everything_explains_nothing(self):
        found = self.causal.explains("fire")
        best = found.hypotheses[0]
        broad = [h for h in found.hypotheses if h.explains > best.explains]
        for other in broad:
            self.assertLess(other.score, best.score)

    def test_restating_the_word_is_not_explaining_it(self):
        found = self.causal.explains("fire")
        self.assertFalse([h for h in found.hypotheses
                          if h.cause.startswith("fire.")])


@requires_norms
@requires_store
class AnalogyTests(unittest.TestCase):
    """R24. Over the norms, where the vocabulary is closed."""

    @classmethod
    def setUpClass(cls):
        cls.identifier = Identifier(STORE)
        cls.analogies = Analogies(Profiles(cls.identifier))

    @classmethod
    def tearDownClass(cls):
        cls.identifier.close()

    def test_the_flagship_mapping(self):
        found = self.analogies.solve("bark", "dog", "cat")
        self.assertEqual(found.source_property, "can bark")
        names = [m.property for m in found.mappings]
        self.assertIn("can meow", names)
        self.assertIn("can purr", names)

    def test_a_mapping_keeps_the_kind_of_property(self):
        found = self.analogies.solve("fins", "fish", "bird")
        self.assertEqual(found.kind, "visual perceptual")
        for mapping in found.mappings:
            self.assertEqual(mapping.kind, found.kind)

    def test_the_answer_is_never_something_the_source_also_has(self):
        found = self.analogies.solve("bark", "dog", "cat")
        for mapping in found.mappings:
            self.assertNotIn(mapping.property, self.analogies.stated["dog"])

    def test_a_class_stands_for_what_its_kinds_share(self):
        """`fish` and `bird` are not XCSLB concepts; cod and robin are."""
        self.assertFalse(self.analogies.profiles.knows("bird"))
        self.assertTrue(self.analogies.properties("bird"))

    def test_a_concept_outside_the_norms_is_declined(self):
        found = self.analogies.solve("wheel", "zzzqqq", "cat")
        self.assertFalse(found.mappings)
        self.assertIn("norms", found.note)


@requires_norms
@requires_store
class RoutingTests(unittest.TestCase):
    """Which rule takes which question. Every bug here is a silent one: the
    answer looks fine until you notice it answered something else."""

    @classmethod
    def setUpClass(cls):
        from research.v687.reasoning import ReasoningEngine
        cls.engine = ReasoningEngine(STORE)

    def rule(self, question):
        return self.engine.ask(question)["parse"]["relation"]

    def test_a_named_subject_is_never_read_backwards(self):
        """`what is a hammer made of` came back with the two things made
        *out of* hammers."""
        self.assertEqual(self.rule("what is a hammer made of"), "made_of")
        self.assertEqual(self.rule("what is a hammer used for"), "used_for")

    def test_the_bridge_keeps_its_questions(self):
        """Judging "did anyone answer" from an empty evidence list took
        v685's two-subject questions away from it."""
        self.assertEqual(self.rule("what does a car driver need"),
                         "has_prerequisite")
        self.assertTrue(self.engine.ask("what does a dog's owner need")
                        .get("bridge"))

    def test_identification_keeps_its_questions(self):
        for question in ("what kind of dog has spots",
                         "what is round with hexagons",
                         "what kind of animal is furry and has spots and is big"):
            self.assertEqual(self.rule(question), "identify", question)

    def test_counting_kinds_is_answered_and_not_refused(self):
        """R18 refuses `how many legs` and says in the same breath that kinds
        can be counted. Nothing implemented that, so the question its own
        refusal offers fell through to a listing about the word `kind`."""
        payload = self.engine.ask("how many kinds of dog are there")
        self.assertEqual(payload["parse"]["relation"], "R25")
        counted = payload["kinds"]
        self.assertGreater(counted["total"], counted["direct"])
        self.assertGreater(counted["direct"], counted["described"])

    def test_a_backwards_question_with_no_subject_is_taken(self):
        for question in ("what is made of wood", "who makes a car",
                         "what is found in a toolbox", "what has wings"):
            self.assertEqual(self.rule(question), "R22", question)

    def test_every_new_answer_can_be_drawn_and_replayed(self):
        """The page draws from `identification` and lights nodes by the
        `concept` of each step, so a step naming something that was never
        drawn lights nothing. Deriving the steps from the tree makes that
        impossible -- this is the assertion that it stays that way."""
        for question in ("what is the difference between a dog and a wolf",
                         "what is similar to a dog",
                         "is a dalmatian a typical dog",
                         "how many kinds of dog are there",
                         "what explains a fire",
                         "what happens when you drive a car",
                         "bark is to dog as what is to cat",
                         "what is made of wood"):
            payload = self.engine.ask(question)
            tree = payload.get("identification")
            self.assertIsNotNone(tree, question)
            drawn = ({round_["term"] for round_ in tree["steps"]}
                     | {entry["name"] for entry in tree["considered"]}
                     | {c["name"] for c in tree["candidates"]})
            self.assertTrue(payload["steps"], question)
            for step in payload["steps"]:
                self.assertIn(step["concept"], drawn, question)

    def test_a_profile_does_not_repeat_itself_in_the_fact_table(self):
        """The walk and the ancestry are laid out once, on their own card."""
        payload = self.engine.ask("what attributes does a blue whale have")
        self.assertTrue(payload["profile"]["path"])
        self.assertEqual(payload["evidence"], [])


@requires_norms
@requires_store
class PinnedSenseTests(unittest.TestCase):
    """R6 the other way round: the reader choosing the sense.

    The rules have always run per sense. What was missing was the reader
    being able to see which one, or say it was wrong -- and that gap was
    widest in the newest rules, which pick a sense with a heuristic and never
    offered a way to overrule it.
    """

    @classmethod
    def setUpClass(cls):
        from research.v687.reasoning import ReasoningEngine
        cls.engine = ReasoningEngine(STORE)

    def tearDown(self):
        from research.v687 import pins
        pins.use(None)

    def test_every_content_word_offers_its_senses(self):
        words = {w["word"]: w for w in
                 self.engine.word_senses("why does a dog bark")["words"]}
        self.assertEqual(set(words), {"dog", "bark"})
        self.assertGreater(words["bark"]["total"], 1)
        self.assertTrue(any(s["default"] for s in words["bark"]["senses"]))
        self.assertTrue(words["bark"]["applies"])

    def test_grammar_words_get_no_chip(self):
        words = [w["word"] for w in
                 self.engine.word_senses("is a dog an animal")["words"]]
        for grammar in ("is", "a", "an"):
            self.assertNotIn(grammar, words)

    def test_a_pin_changes_the_event_sense(self):
        """R23 picks the sense carrying the most eventive facts, which for
        `bark` is the covering of a tree."""
        loose = self.engine.ask("why does a dog bark")
        pinned = self.engine.ask("why does a dog bark", None,
                                 {"bark": "bark.v.01"})
        self.assertTrue(loose["causal"]["steps"])
        self.assertFalse(pinned["causal"]["steps"])
        self.assertIn("pinned", pinned["note"])

    def test_a_pin_changes_the_class_a_count_walks(self):
        loose = self.engine.ask("how many kinds of mouse are there")
        pinned = self.engine.ask("how many kinds of mouse are there", None,
                                 {"mouse": "mouse.n.01"})
        self.assertEqual(loose["kinds"]["total"], 0)      # the device
        self.assertGreater(pinned["kinds"]["total"], 5)   # the animal

    def test_a_pin_changes_the_class_identification_searches(self):
        pinned = self.engine.ask("what kind of mouse has a tail", None,
                                 {"mouse": "mouse.n.01"})
        self.assertEqual(pinned["identification"]["among_concept"],
                         "mouse.n.01")

    def test_a_pin_holds_the_backwards_reading_to_one_sense(self):
        loose = self.engine.ask("what is a mouse part of")
        pinned = self.engine.ask("what is a mouse part of", None,
                                 {"mouse": "mouse.n.01"})
        self.assertGreater(len(loose["backwards"]["subjects"]),
                           len(pinned["backwards"]["subjects"]))

    def test_a_pin_on_the_subject_does_not_disable_the_rules_above_v684(self):
        """Feeding a pin in as `concept` skipped every rule above v684,
        because that parameter is a gate and not a preference: pinning `bark`
        stopped `why does a dog bark` being a why-question at all."""
        pinned = self.engine.ask("why does a dog bark", None,
                                 {"bark": "bark.v.01"})
        self.assertEqual(pinned["parse"]["relation"], "R23")

    def test_a_pin_survives_into_v684s_own_answer(self):
        loose = self.engine.ask("is a hammer a tool")
        pinned = self.engine.ask("is a hammer a tool", None,
                                 {"hammer": "hammer.n.01"})
        self.assertEqual(loose["verdict"], "VERIFIED")
        self.assertEqual(pinned["concept"], "hammer.n.01")
        self.assertEqual(pinned["verdict"], "UNKNOWN")

    def test_pins_are_request_scoped(self):
        """The server is threaded, so two readers pinning different senses of
        the same word must not see each other's choice."""
        from research.v687 import pins
        pins.use({"mouse": "mouse.n.01"})
        self.assertEqual(pins.of("mouse"), "mouse.n.01")
        pins.use(None)
        self.assertIsNone(pins.of("mouse"))


if __name__ == "__main__":
    unittest.main()
