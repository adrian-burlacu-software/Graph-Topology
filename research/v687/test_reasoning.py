"""Regression suite. Run: python -m unittest research.v687.test_v684 -v

Tests that need the built store skip themselves when it is absent, so the suite
runs on a fresh clone before `python -m research.v687.build`.
"""
from __future__ import annotations

import unittest

import collections
import sqlite3
from pathlib import Path

from research.v687 import build, compress, rules, senses
from research.v687.language import Parser
from research.v687.reason import Reasoner

STORE = build.DEFAULT_STORE
HAVE_STORE = STORE.exists()
requires_store = unittest.skipUnless(HAVE_STORE, f"no store at {STORE}")


class RuleTests(unittest.TestCase):
    def test_related_to_never_participates(self):
        """R7. It is 1,678,150 of 3.9M edges and says only 'co-occurs'."""
        self.assertIn("related_to", rules.GATED)
        self.assertFalse(rules.inheritable("related_to"))

    def test_inheritable_and_not_are_disjoint_and_reasoned(self):
        """R2. Every non-inheritable relation states why."""
        self.assertFalse(rules.INHERITABLE & set(rules.NOT_INHERITABLE))
        for relation in rules.NOT_INHERITABLE:
            self.assertTrue(rules.why_not_inheritable(relation))
            self.assertFalse(rules.inheritable(relation))

    def test_made_of_does_not_descend(self):
        """A chair is furniture; furniture is not therefore made of wood."""
        self.assertFalse(rules.inheritable("made_of"))
        self.assertTrue(rules.inheritable("capable_of"))

    def test_confidence_decays_with_distance(self):
        """R5."""
        self.assertAlmostEqual(rules.confidence_at(1.0, 0), 1.0)
        self.assertLess(rules.confidence_at(1.0, 5), rules.confidence_at(1.0, 1))
        self.assertAlmostEqual(rules.confidence_at(1.0, 2), rules.DECAY ** 2)

    def test_negation_blocks_its_positive(self):
        """R3."""
        self.assertTrue(rules.blocks("not_capable_of", "capable_of"))
        self.assertTrue(rules.blocks("capable_of", "not_capable_of"))
        self.assertFalse(rules.blocks("capable_of", "at_location"))

    def test_relation_families_are_symmetric(self):
        """R9. has_a and has_part must answer for each other, both ways."""
        self.assertEqual(rules.family("has_a"), rules.family("has_part"))
        self.assertIn("has_part", rules.family("has_a"))
        self.assertEqual(rules.family("capable_of"), ["capable_of"])

    def test_part_of_is_not_in_the_has_part_family(self):
        """It is the inverse, not a synonym; conflating them reverses facts."""
        self.assertNotIn("part_of", rules.family("has_part"))

    def test_every_rule_has_text_for_the_ui(self):
        for key in ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"):
            self.assertIn(key, rules.RULE_TEXT)


class ParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = Parser()

    def test_polar_question_yields_subject_relation_target(self):
        parse = self.parser.parse("can a dog fall into a hole")
        self.assertEqual(parse.subject, "dog")
        self.assertEqual(parse.relation, "capable_of")
        self.assertIn("fall", parse.target)
        self.assertTrue(parse.polar)

    def test_copula_plus_determiner_is_a_taxonomy_question(self):
        """`is a dog an animal` asks about kinds, not properties."""
        parse = self.parser.parse("is a dog an animal")
        self.assertEqual(parse.relation, "is_a")
        self.assertEqual(parse.target, "animal")

    def test_copula_without_determiner_stays_a_property_question(self):
        parse = self.parser.parse("is a dog friendly")
        self.assertEqual(parse.relation, "has_property")

    def test_auxiliary_verb_is_stripped_from_the_target(self):
        """`does a dog have a tail` is about a tail, not about having."""
        parse = self.parser.parse("does a dog have a tail")
        self.assertEqual(parse.relation, "has_part")
        self.assertNotIn("have", (parse.target or "").split())

    def test_open_question_has_no_target(self):
        parse = self.parser.parse("what can a violin do")
        self.assertEqual(parse.subject, "violin")
        self.assertIsNone(parse.target)
        self.assertFalse(parse.polar)

    def test_relation_cues_cover_the_documented_shapes(self):
        for question, relation in (
            ("what is a violin made of", "made_of"),
            ("what is a hammer used for", "used_for"),
            ("where do you find a hammer", "at_location"),
            ("what does a dog want", "desires"),
        ):
            self.assertEqual(self.parser.parse(question).relation, relation, question)

    def test_matcher_requires_real_overlap(self):
        matches = self.parser.matcher()
        self.assertTrue(matches("fall into hole", "fall into a hole"))
        self.assertFalse(matches("fall in love", "fall into a hole"))
        self.assertTrue(matches("anything", None))

    def test_unparseable_input_does_not_raise(self):
        parse = self.parser.parse("???")
        self.assertIsNone(parse.subject)
        self.assertTrue(parse.note)


@requires_store
class ReasonerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reasoner = Reasoner(STORE)
        cls.parser = Parser()
        # staticmethod: a bare function on a class binds as a method,
        # which would pass `self` as the first argument to the matcher.
        cls.match = staticmethod(cls.parser.matcher())

    @classmethod
    def tearDownClass(cls):
        cls.reasoner.close()

    def test_word_resolves_to_several_senses_eponymous_first(self):
        """R6. `dog` names eight synsets; dog.n.01 must lead."""
        senses = self.reasoner.senses_of("dog")
        self.assertGreater(len(senses), 1)
        self.assertEqual(senses[0]["id"], "dog.n.01")

    def test_facts_landed_on_the_eponymous_sense(self):
        """The build bug this guards: all of dog's facts went to andiron.n.01."""
        self.assertGreater(self.reasoner.fact_count("dog.n.01"), 100)

    def test_taxonomy_walk_reaches_animal_from_dog(self):
        answer = self.reasoner.classify("dog.n.01", "animal")
        self.assertEqual(answer.verdict, "VERIFIED")
        self.assertEqual(answer.evidence[0].distance, 2)

    def test_absent_classification_is_unknown_not_false(self):
        """R8. The ontology does not assert negatives by omission."""
        answer = self.reasoner.classify("dog.n.01", "vehicle")
        self.assertEqual(answer.verdict, "UNKNOWN")
        self.assertIn("Absent, not false", answer.note)

    def test_inheritance_finds_a_fact_from_an_ancestor(self):
        answer = self.reasoner.verify("dog.n.01", "capable_of",
                                      "fall into a hole", self.match)
        self.assertEqual(answer.verdict, "VERIFIED")
        self.assertGreater(answer.evidence[0].distance, 0)

    def test_inherited_confidence_is_below_direct(self):
        """R5, observable end to end."""
        answer = self.reasoner.verify("dog.n.01", "capable_of",
                                      "fall into a hole", self.match)
        self.assertLess(answer.evidence[0].confidence, 0.95)

    def test_relation_family_finds_has_a_when_asked_has_part(self):
        """R9. Ascent++ files the tail under has_a, WordNet under has_part."""
        answer = self.reasoner.verify("dog.n.01", "has_part", "tail", self.match)
        self.assertEqual(answer.verdict, "VERIFIED")

    def test_non_inheritable_relation_stops_the_walk(self):
        """R2. made_of must not climb the taxonomy."""
        answer = self.reasoner.verify("dog.n.01", "made_of", "wood", self.match)
        stops = [s for s in answer.steps if s.kind == "stop"]
        self.assertTrue(stops)
        self.assertEqual(stops[0].rule, "R2")

    def test_every_step_names_the_rule_that_produced_it(self):
        answer = self.reasoner.verify("dog.n.01", "capable_of",
                                      "fall into a hole", self.match)
        self.assertTrue(answer.steps)
        for step in answer.steps:
            self.assertIn(step.rule, rules.RULE_TEXT, step.detail)

    def test_ascent_terminates_on_the_acyclic_taxonomy(self):
        visited = [node for node, _, _ in self.reasoner.ascend("dog.n.01")]
        self.assertEqual(len(visited), len(set(visited)))
        self.assertIn("entity.n.01", visited)

    def test_describe_ranks_direct_facts_above_inherited(self):
        answer = self.reasoner.describe("violin.n.01", "capable_of")
        self.assertEqual(answer.verdict, "LISTING")
        self.assertTrue(answer.evidence)
        self.assertEqual(answer.evidence[0].distance, 0)

    def test_describe_never_repeats_a_fact_from_higher_up(self):
        """R4. The nearest statement wins; duplicates are dropped."""
        answer = self.reasoner.describe("dog.n.01", None)
        seen = [(f.relation, f.object.lower()) for f in answer.evidence]
        self.assertEqual(len(seen), len(set(seen)))

    def test_answer_serialises_for_the_ui(self):
        payload = self.reasoner.classify("dog.n.01", "animal").as_dict()
        for key in ("verdict", "steps", "evidence", "chain", "rules"):
            self.assertIn(key, payload)


@requires_store
class EngineTests(unittest.TestCase):
    """The path the browser actually takes."""

    @classmethod
    def setUpClass(cls):
        from research.v687.engine import Engine
        cls.engine = Engine(STORE)

    @classmethod
    def tearDownClass(cls):
        cls.engine.reasoner.close()

    def test_end_to_end_question(self):
        payload = self.engine.ask("is a dog an animal")
        self.assertEqual(payload["verdict"], "VERIFIED")
        self.assertTrue(payload["steps"])
        self.assertTrue(payload["senses"])

    def test_unknown_word_is_reported_not_raised(self):
        payload = self.engine.ask("is a zzzqqq an animal")
        self.assertIn(payload["verdict"], ("UNKNOWN_WORD", "UNPARSED"))

    def test_sense_can_be_overridden(self):
        payload = self.engine.ask("what can a dog do", concept="cad.n.01")
        self.assertEqual(payload["concept"], "cad.n.01")


@requires_store
class SenseChoiceTests(unittest.TestCase):
    """The join between word-level facts and WordNet senses (senses.py)."""

    @classmethod
    def setUpClass(cls):
        cls.connection = sqlite3.connect(f"file:{STORE}?mode=ro", uri=True)
        cls.senses = senses.Senses(cls.connection)

    @classmethod
    def tearDownClass(cls):
        cls.connection.close()

    def evidence(self, *phrases):
        bag = collections.Counter()
        for phrase in phrases:
            bag.update(senses.tokens(phrase))
        return bag

    def test_tokens_drop_stopwords_and_punctuation(self):
        self.assertEqual(senses.tokens("a carpenter's toolbox"),
                         ["carpenter", "toolbox"])

    def test_the_bug_that_started_this_hammer_is_a_tool(self):
        """WordNet's hammer.n.01 is a gun part; the facts describe the tool."""
        choice, _, margin = self.senses.choose(
            "hammer", self.evidence("carpenter's toolbox", "hardware store",
                                    "toolbelt", "drive a nail", "tool box"))
        self.assertEqual(choice, "hammer.n.02")
        self.assertGreater(margin, 0)

    def test_evidence_does_not_drag_a_word_to_a_verb_sense(self):
        """`dog` has chase.v.01 among its senses; these sources describe things."""
        choice, _, _ = self.senses.choose(
            "dog", self.evidence("bark at strangers", "chase a cat",
                                 "wag its tail", "bury a bone"))
        self.assertEqual(choice, "dog.n.01")

    def test_a_thin_margin_does_not_leave_the_words_own_synset(self):
        """EPONYMOUS_FACTOR.

        On the real evidence `seal` scored `navy seal.n.01` barely ahead of
        its own synsets, and `spring` reached `leap.n.01`. Requiring a
        decisive margin to abandon the synset named for the word keeps both
        home, while `bank` still leaves for `depository financial
        institution.n.01`, which it beats many times over.
        """
        recorded = dict(self.connection.execute(
            "SELECT lemma, concept FROM lemmas WHERE primary_sense = 1 "
            "AND lemma IN ('seal', 'spring', 'bank')"))
        self.assertEqual(recorded["seal"].rsplit(".", 2)[0], "seal")
        self.assertEqual(recorded["spring"].rsplit(".", 2)[0], "spring")
        self.assertEqual(recorded["bank"], "depository financial institution.n.01")

    def test_no_evidence_falls_back_to_the_prior(self):
        choice, score, _ = self.senses.choose("hammer", collections.Counter())
        self.assertEqual(score, 0.0)
        self.assertEqual(choice, "hammer.n.01")

    def test_unknown_word_chooses_nothing(self):
        self.assertIsNone(self.senses.choose("zzzqqq", collections.Counter())[0])

    def test_the_store_records_which_sense_was_chosen(self):
        chosen = [r[0] for r in self.connection.execute(
            "SELECT concept FROM lemmas WHERE lemma='hammer' AND primary_sense=1")]
        self.assertEqual(chosen, ["hammer.n.02"])


@requires_store
class BreadthGateTests(unittest.TestCase):
    """R12: word-level facts do not inherit from top-of-taxonomy concepts."""

    @classmethod
    def setUpClass(cls):
        cls.reasoner = Reasoner(STORE)

    @classmethod
    def tearDownClass(cls):
        cls.reasoner.close()

    def test_the_limit_sits_above_animal_and_below_person(self):
        """Where the threshold falls is the whole claim; pin both sides of it."""
        size = dict(self.reasoner.connection.execute(
            "SELECT id, descendants FROM concepts WHERE id IN "
            "('animal.n.01', 'plant.n.02', 'person.n.01', 'artifact.n.01')"))
        self.assertLess(size["animal.n.01"], rules.BREADTH_LIMIT)
        self.assertLess(size["plant.n.02"], rules.BREADTH_LIMIT)
        self.assertGreaterEqual(size["person.n.01"], rules.BREADTH_LIMIT)
        self.assertGreaterEqual(size["artifact.n.01"], rules.BREADTH_LIMIT)

    def test_broad_concepts_are_gated_and_ordinary_ones_are_not(self):
        self.assertTrue(self.reasoner.too_broad("person.n.01"))
        self.assertFalse(self.reasoner.too_broad("animal.n.01"))
        self.assertFalse(self.reasoner.too_broad("dog.n.01"))

    def test_r12_sets_facts_aside_without_stopping_the_walk(self):
        """The UI colours `block` and `stop` as a halt, so R12 must not be one.

        `can a dog fall into a hole` on dog.n.03 passes four gated concepts
        and still reaches object.n.01.
        """
        parser = Parser()
        answer = self.reasoner.verify("dog.n.03", "capable_of",
                                      "fall into a hole", parser.matcher())
        gated = [s for s in answer.steps if s.rule == "R12"]
        self.assertTrue(gated)
        for step in gated:
            self.assertEqual(step.kind, "skip", step.detail)
        self.assertNotIn("stop", {s.kind for s in gated})
        # the walk carried on past every one of them
        deepest = max(s.distance for s in answer.steps)
        self.assertGreater(deepest, max(s.distance for s in gated))

    def test_r12_gates_the_assumed_join_not_inheritance_itself(self):
        self.assertTrue(rules.inheritable_from("capable_of", 99999, False))
        self.assertFalse(rules.inheritable_from("capable_of", 99999, True))
        self.assertTrue(rules.inheritable_from("capable_of", 10, True))

    def test_a_hammer_is_not_found_in_a_tomb(self):
        """The reported symptom: artifact.n.01's word-level facts reaching down."""
        answer = self.reasoner.describe("hammer.n.02", "at_location")
        objects = {f.object.lower() for f in answer.evidence}
        self.assertIn("hardware store", objects)
        for junk in ("tomb", "grave", "museum", "excavation"):
            self.assertNotIn(junk, objects)


@requires_store
class SubjectDetectionTests(unittest.TestCase):
    """Finding what a question is about, when the tagger cannot.

    A noun that is also a verb makes spaCy read `a canine fall` as one compound
    noun. It then reports either no subject or the wrong end of the run, and
    neither "take the first noun" nor "take the last" is right for both
    `canine fall` and `fire truck`. The ontology settles it.
    """

    @classmethod
    def setUpClass(cls):
        reasoner = Reasoner(STORE)
        cls.parser = Parser(vocabulary=reasoner.vocabulary())
        reasoner.close()
        if cls.parser.nlp is None:
            raise unittest.SkipTest("subject detection needs spaCy")

    def subject(self, question):
        return self.parser.parse(question).subject

    def test_a_subject_that_is_also_a_verb(self):
        """The reported bug: this answered about `fall.n.01`."""
        self.assertEqual(self.subject("can a canine fall into a hole"), "canine")

    def test_the_same_shape_across_several_words(self):
        for question, expected in (
                ("can a hammer break glass", "hammer"),
                ("does a wolf howl", "wolf"),
                ("can a rock fall", "rock"),
                ("can a dog fall into a hole", "dog")):
            self.assertEqual(self.subject(question), expected, question)

    def test_a_compound_subject_is_kept_whole(self):
        """`fire truck` is one concept, so the subject does not stop at `fire`."""
        self.assertEqual(self.subject("can a fire truck move"), "fire truck")
        self.assertEqual(self.subject("can a police dog bark"), "police dog")
        self.assertEqual(self.subject("can a bird of prey fly"), "bird of prey")

    def test_a_modifier_is_not_mistaken_for_the_subject(self):
        """`large` is a lemma too; the phrase has to end on a noun."""
        self.assertEqual(self.subject("can a large dog fall"), "dog")

    def test_the_target_survives_a_multiword_subject(self):
        parse = self.parser.parse("can a fire truck move")
        self.assertEqual(parse.subject, "fire truck")
        self.assertEqual(parse.target, "move")

    def test_the_older_question_shapes_are_unchanged(self):
        for question, expected in (
                ("is a dog an animal", "dog"), ("does a dog have a tail", "dog"),
                ("what can a violin do", "violin"),
                ("where do you find a hammer", "hammer"),
                ("what is a hammer used for", "hammer")):
            self.assertEqual(self.subject(question), expected, question)

    def test_the_parser_still_works_without_a_vocabulary(self):
        bare = Parser()
        self.assertEqual(bare.parse("can a dog fall into a hole").subject, "dog")


class TargetMatchingTests(unittest.TestCase):
    """What counts as answering the question that was actually asked."""

    def setUp(self):
        self.parser = Parser()
        self.match = self.parser.matcher()

    def test_a_preposition_is_not_content(self):
        """`into` sat outside a list already holding `in`, `to`, `on`, `at`."""
        from research.v687.language import STOP
        for word in ("into", "onto", "from", "within"):
            self.assertIn(word, STOP)

    def test_particles_are_still_content(self):
        """`fall down` and `fall over` are different claims, so keep both words."""
        from research.v687.language import STOP
        for word in ("down", "up", "over", "out", "off", "through"):
            self.assertNotIn(word, STOP)

    def test_the_reported_bug_a_shared_verb_is_not_an_answer(self):
        """This verified `can a dog fall into a hole` off a ratchet catch."""
        self.assertFalse(self.match("fall into wrong hands", "fall into a hole"))
        self.assertFalse(self.match("fall into place", "fall into a hole"))
        self.assertFalse(self.match("fall into the trap", "fall into a hole"))

    def test_a_real_answer_still_matches(self):
        self.assertTrue(self.match("fall into hole", "fall into a hole"))
        self.assertTrue(self.match("fall into a deep hole", "fall into a hole"))

    def test_the_matcher_reports_how_close_it_came(self):
        self.assertEqual(self.match.score("fall into hole", "fall into a hole"), 1.0)
        near = self.match.score("fall into wrong hands", "fall into a hole")
        self.assertGreater(near, 0)
        self.assertLess(near, self.match.threshold)


@requires_store
class SuggestionTests(unittest.TestCase):
    """A partial hit is offered, not believed."""

    @classmethod
    def setUpClass(cls):
        cls.reasoner = Reasoner(STORE)
        cls.match = staticmethod(Parser().matcher())

    @classmethod
    def tearDownClass(cls):
        cls.reasoner.close()

    def test_a_near_miss_becomes_a_suggestion_not_a_verdict(self):
        answer = self.reasoner.verify("pawl.n.01", "capable_of",
                                      "fall into a hole", self.match)
        self.assertEqual(answer.verdict, "UNKNOWN")
        self.assertTrue(answer.suggestions)
        objects = {s.object for s in answer.suggestions}
        self.assertIn("fall into wrong hands", objects)
        self.assertNotIn("fall into wrong hands",
                         {e.object for e in answer.evidence})

    def test_suggestions_are_ranked_capped_and_deduplicated(self):
        answer = self.reasoner.verify("pawl.n.01", "capable_of",
                                      "fall into a hole", self.match)
        shares = [s.similarity for s in answer.suggestions]
        self.assertEqual(shares, sorted(shares, reverse=True))
        self.assertLessEqual(len(answer.suggestions), Reasoner.MAX_SUGGESTIONS)
        self.assertEqual(len(answer.suggestions),
                         len({s.object.lower() for s in answer.suggestions}))
        for share in shares:
            self.assertGreaterEqual(share, Reasoner.SUGGEST_FLOOR)
            self.assertLess(share, self.match.threshold)

    def test_an_answered_question_offers_no_suggestions(self):
        answer = self.reasoner.verify("dog.n.01", "capable_of",
                                      "fall into a hole", self.match)
        self.assertEqual(answer.verdict, "VERIFIED")
        self.assertEqual(answer.suggestions, [])

    def test_suggestions_serialise_for_the_ui(self):
        answer = self.reasoner.verify("pawl.n.01", "capable_of",
                                      "fall into a hole", self.match)
        payload = answer.as_dict()
        self.assertIn("suggestions", payload)
        self.assertIn("similarity", payload["suggestions"][0])


@requires_store
class SynonymTests(unittest.TestCase):
    """Different words for the same thing should reach the same knowledge."""

    @classmethod
    def setUpClass(cls):
        cls.reasoner = Reasoner(STORE)
        cls.parser = Parser(vocabulary=cls.reasoner.vocabulary())
        cls.match = staticmethod(cls.parser.matcher())

    @classmethod
    def tearDownClass(cls):
        cls.reasoner.close()

    def test_lemmas_of_one_synset_are_the_same_concept(self):
        """WordNet models these as synonyms, so nothing extra is needed."""
        for word in ("dog", "domestic dog", "canis familiaris"):
            self.assertEqual(self.reasoner.senses_of(word)[0]["id"], "dog.n.01",
                             word)

    def test_an_informal_term_inherits_from_the_word_it_paraphrases(self):
        """`pooch` is its own synset, but its parent is dog.n.01."""
        self.assertIn("dog.n.01", self.reasoner.parents_of("pooch.n.01"))
        self.assertEqual(self.reasoner.senses_of("doggie")[0]["id"], "pooch.n.01")

    def test_dog_canine_and_bitch_reach_the_same_fact(self):
        """The three arrive by different routes at one stored fact."""
        found = {}
        for concept in ("dog.n.01", "canine.n.02", "bitch.n.04"):
            answer = self.reasoner.verify(concept, "capable_of",
                                          "fall into a hole", self.match)
            self.assertEqual(answer.verdict, "VERIFIED", concept)
            found[concept] = (answer.evidence[0].concept,
                              answer.evidence[0].object)
        self.assertEqual(len(set(found.values())), 1, found)
        self.assertEqual(found["dog.n.01"][0], "canine.n.02")


@requires_store
class RangeTypingTests(unittest.TestCase):
    """R13: a relation's object must be the kind of thing the relation takes."""

    @classmethod
    def setUpClass(cls):
        cls.connection = sqlite3.connect(f"file:{STORE}?mode=ro", uri=True)
        parents = collections.defaultdict(set)
        for child, parent in cls.connection.execute(
                "SELECT child, parent FROM taxonomy"):
            parents[child].add(parent)
        senses_of = collections.defaultdict(list)
        for lemma, concept in cls.connection.execute(
                "SELECT lemma, concept FROM lemmas"):
            senses_of[lemma].append(concept)
        cls.ranges = senses.Ranges(parents, senses_of, rules.RANGES)

    @classmethod
    def tearDownClass(cls):
        cls.connection.close()

    def test_a_location_has_to_be_a_place(self):
        for junk in ("communication", "high quality", "accordance", "harmony"):
            self.assertFalse(self.ranges.allows("at_location", junk), junk)

    def test_real_places_pass(self):
        for place in ("garage", "hardware store", "toolbelt", "london",
                      "carpenter's toolbox", "store"):
            self.assertTrue(self.ranges.allows("at_location", place), place)

    def test_the_head_word_is_tried_when_the_phrase_is_unknown(self):
        """`carpenter's toolbox` is not a synset; `toolbox` is."""
        self.assertEqual(self.ranges.denotes("carpenter's toolbox"),
                         self.ranges.denotes("toolbox"))

    def test_any_sense_may_satisfy_the_range(self):
        """`store` resolves to a supply, but one of its senses is a shop.

        Checking only the chosen sense dropped `hammer at_location store`.
        """
        chosen = self.connection.execute(
            "SELECT concept FROM lemmas WHERE lemma='store' AND primary_sense=1"
        ).fetchone()[0]
        self.assertEqual(chosen, "store.n.02")          # "a supply of something"
        self.assertTrue(self.ranges.allows("at_location", "store"))

    def test_a_word_the_ontology_does_not_know_is_not_rejected(self):
        self.assertTrue(self.ranges.allows("at_location", "zzzqqq wumpus"))

    def test_relations_without_a_declared_range_are_untouched(self):
        self.assertNotIn("capable_of", rules.RANGES)
        self.assertTrue(self.ranges.allows("capable_of", "high quality"))

    def test_what_is_stated_outranks_what_is_borrowed(self):
        """R4 as an ordering, not only as a stopping rule.

        `violin capable_of run android`, inherited from `device` four levels
        up, used to sit above `sound beautiful` stated about violins, because
        confidence alone decided the order.
        """
        reasoner = Reasoner(STORE)
        try:
            evidence = reasoner.describe("violin.n.01", "capable_of").evidence
            self.assertTrue(evidence)
            distances = [f.distance for f in evidence]
            self.assertEqual(distances, sorted(distances))
            for fact in evidence[:10]:
                self.assertEqual(fact.distance, 0, fact.object)
        finally:
            reasoner.close()

    def test_the_store_holds_no_location_that_is_not_a_place(self):
        offenders = [obj for obj, in self.connection.execute(
            "SELECT DISTINCT object FROM facts WHERE relation = 'at_location' "
            "AND sense_assumed = 1 LIMIT 4000")
            if not self.ranges.allows("at_location", obj)]
        self.assertEqual(offenders, [])


class CompressionTests(unittest.TestCase):
    """R10/R11 on a small synthetic store, so the assertions can be exact."""

    def setUp(self):
        directory = Path(__file__).resolve().parent / "__pycache__"
        directory.mkdir(exist_ok=True)
        self.source = directory / "_test_source.sqlite"
        self.out = directory / "_test_compressed.sqlite"
        for path in (self.source, self.out):
            path.unlink(missing_ok=True)
        connection = sqlite3.connect(self.source)
        connection.executescript(build.SCHEMA)
        connection.executemany(
            "INSERT INTO concepts VALUES (?,?,?,?,?,?)",
            [(f"{n}.n.01", n, "n", 1, "", d) for n, d in
             (("animal", 3), ("mammal", 2), ("dog", 0), ("cat", 0), ("bird", 0))])
        connection.executemany("INSERT INTO taxonomy VALUES (?,?)", [
            ("mammal.n.01", "animal.n.01"), ("dog.n.01", "mammal.n.01"),
            ("cat.n.01", "mammal.n.01"), ("bird.n.01", "animal.n.01")])
        connection.executemany("INSERT INTO facts VALUES (?,?,?,?,?,?)", [
            ("animal.n.01", "capable_of", "breathe", "wordnet", 0.9, 0),
            # the children repeat what the parent already says: R10 drops these
            ("dog.n.01", "capable_of", "breathe", "ascentpp", 0.8, 1),
            ("cat.n.01", "capable_of", "breathe", "ascentpp", 0.8, 1),
            ("dog.n.01", "capable_of", "bark", "ascentpp", 0.8, 1),
            # made_of does not inherit, so repeating it is not redundant
            ("animal.n.01", "made_of", "cells", "wordnet", 0.9, 0),
            ("dog.n.01", "made_of", "cells", "ascentpp", 0.8, 1)])
        connection.commit()
        connection.close()

    def tearDown(self):
        for path in (self.source, self.out):
            path.unlink(missing_ok=True)

    def kept(self):
        connection = sqlite3.connect(self.out)
        rows = {(r[0], r[1], r[2]) for r in connection.execute(
            "SELECT concept, relation, object FROM facts")}
        connection.close()
        return rows

    def test_r10_drops_only_what_inheritance_rebuilds(self):
        stats = compress.compress(self.source, self.out, verbose=False)
        self.assertEqual(stats["dropped"], 2)          # dog and cat "breathe"
        kept = self.kept()
        self.assertNotIn(("dog.n.01", "capable_of", "breathe"), kept)
        self.assertIn(("animal.n.01", "capable_of", "breathe"), kept)
        self.assertIn(("dog.n.01", "capable_of", "bark"), kept)
        self.assertIn(("dog.n.01", "made_of", "cells"), kept)

    def test_compression_is_verified_lossless(self):
        compress.compress(self.source, self.out, verbose=False)
        result = compress.verify(self.source, self.out, verbose=False)
        self.assertTrue(result["lossless"], result["examples"])
        self.assertEqual(result["not_rederivable"], 0)

    def test_r10_respects_the_breadth_gate(self):
        """A fact a broad ancestor may not lend is not redundant below it."""
        connection = sqlite3.connect(self.source)
        connection.execute("UPDATE concepts SET descendants = ? WHERE id = ?",
                           (rules.BREADTH_LIMIT + 1, "animal.n.01"))
        connection.execute("UPDATE facts SET sense_assumed = 1 "
                           "WHERE concept = 'animal.n.01'")
        connection.commit()
        connection.close()
        stats = compress.compress(self.source, self.out, verbose=False)
        self.assertEqual(stats["dropped"], 0)
        self.assertIn(("dog.n.01", "capable_of", "breathe"), self.kept())
        self.assertTrue(compress.verify(self.source, self.out,
                                        verbose=False)["lossless"])


@requires_store
class AnswerAuditTests(unittest.TestCase):
    """The six findings of the v687 answer audit, each pinned by its own case.

    Every one of these was a question the system answered -- wrongly, or about
    something else -- with nothing in the answer saying so. They are kept
    together rather than filed by module because what they have in common is
    the thing worth not regressing: a confident answer to a question nobody
    asked.
    """

    @classmethod
    def setUpClass(cls):
        from research.v687.reasoning import ReasoningEngine
        cls.engine = ReasoningEngine(STORE)

    @classmethod
    def tearDownClass(cls):
        cls.engine.reasoner.close()

    def verdict(self, question):
        return self.engine.ask(question)["verdict"]

    # -- F1: a denial is not a denial of every word inside it --------------
    def test_a_qualified_denial_does_not_deny_the_bare_property(self):
        """The norms deny `has small ears` of a beaver. Beavers have ears."""
        for question in ("does a beaver have ears", "does a horse have teeth",
                         "does a horse have eyes", "does a chair have legs",
                         "is a wheel part of a car"):
            with self.subTest(question=question):
                self.assertNotEqual(self.verdict(question), "CONTRADICTED")

    def test_the_qualified_denial_still_answers_its_own_question(self):
        """`has small ears` denies small ears, whatever it fails to say about
        ears.

        The beaver row behind this was a COMPS foil and is gone, so the rule
        is asserted where it lives. `does a beaver have small ears` now reads
        UNKNOWN, which is the honest answer to a question nobody was asked.
        """
        hit, narrower = self.engine.identifier.denial_hit(
            ["small", "ears"], frozenset({"has small ears"}))
        self.assertEqual(hit, "has small ears")
        self.assertEqual(narrower, [])
        self.assertEqual(self.verdict("does a beaver have small ears"),
                         "UNKNOWN")

    def test_an_unqualified_denial_is_untouched(self):
        for question in ("is a whale furry", "does a penguin fly"):
            with self.subTest(question=question):
                self.assertEqual(self.verdict(question), "CONTRADICTED")

    def test_a_locative_tail_does_not_qualify_the_claim(self):
        """`has spots on its body` says where, not which, so it would still
        deny spots, while `has small ears` says which and would not.

        Asserted against `denial_hit` rather than through an answer. The only
        instance this had was `dog / spots`, which turned out to be a COMPS
        foil rather than a denial, so the data went and the rule did not --
        and a rule tested only through one row of data is a rule tested
        through that row.
        """
        hit = self.engine.identifier.denial_hit(
            ["spots"], frozenset({"has spots on its body"}))
        self.assertEqual(hit[0], "has spots on its body")
        narrower = self.engine.identifier.denial_hit(
            ["ears"], frozenset({"has small ears"}))
        self.assertIsNone(narrower[0])
        self.assertEqual(narrower[1], ["has small ears"])
        self.assertEqual(self.engine.profiles.verify(
            "dalmatian", ["spots"]).verdict, "HELD")

    def test_a_count_is_matched_with_what_it_counts(self):
        """`eight legs` is not `eight eyes` and `legs`, and a spider that has
        eight legs does not have six."""
        spider = self.engine.ask("does a spider have eight legs")
        self.assertEqual(spider["verdict"], "VERIFIED")
        self.assertIn("has eight legs", spider["note"])
        self.assertNotIn("eyes", spider["note"])
        self.assertNotEqual(self.verdict("does a spider have six legs"),
                            "VERIFIED")
        self.assertNotEqual(self.verdict("does a dog have two legs"),
                            "VERIFIED")
        self.assertEqual(self.verdict("does a dog have four legs"), "VERIFIED")
        self.assertEqual(self.engine.identifier._hit(
            "eight legs", frozenset({"has eight eyes", "has legs"})), None)

    def test_the_statement_cited_is_the_one_that_covers_the_question(self):
        cats = self.engine.ask("do cats eat mice")
        self.assertEqual(cats["verdict"], "VERIFIED")
        self.assertIn("mice", cats["note"])

    def test_a_thing_does_not_happen(self):
        """ASCENT++ records of `bark.n.01`, the covering of a tree, that it
        causes vomiting: that is not what happens when a dog barks."""
        payload = self.engine.ask("what happens when a dog barks")
        objects = [step["object"] for step in payload["causal"]["steps"]]
        self.assertIn("produce sounds", objects)
        self.assertNotIn("vomiting", objects)
        self.assertIn("not something that happens", payload["note"])
        self.assertTrue(self.engine.ask("what happens when it rains")
                        ["causal"]["steps"])

    # -- F2: the subject is the thing asked about, or nothing --------------
    def test_a_subject_that_is_not_a_noun_is_still_the_subject(self):
        for question in ("is hello a greeting", "is red a color",
                         "is chess a game", "is running a sport"):
            with self.subTest(question=question):
                self.assertEqual(self.verdict(question), "VERIFIED")

    def test_an_unknown_subject_is_named_rather_than_replaced(self):
        for question in ("is a wemble an animal", "does a blorp have wings",
                         "does a quovix have wings"):
            with self.subTest(question=question):
                payload = self.engine.ask(question)
                self.assertEqual(payload["verdict"], "UNKNOWN_WORD")
                self.assertIn(payload["parse"]["unknown"], question)

    def test_a_bare_noun_predicate_is_a_class(self):
        self.assertEqual(self.verdict("is a chair furniture"), "VERIFIED")

    def test_a_bare_property_predicate_is_still_a_property(self):
        """`white` has a noun sense too, and reading it as a class stopped the
        norms from ever being asked."""
        self.assertEqual(self.verdict("is a raccoon white"), "VERIFIED")
        self.assertEqual(self.verdict("is a bobcat white"), "CONTRADICTED")

    def test_a_coordination_is_never_one_class(self):
        self.assertEqual(self.verdict("is a dog furry or purple"), "VERIFIED")

    def test_no_false_yes_from_reading_the_other_noun(self):
        for question in ("is a dog a cat", "is a whale a fish",
                         "is a bat a bird"):
            with self.subTest(question=question):
                self.assertEqual(self.verdict(question), "UNKNOWN")

    def test_a_copula_is_not_a_compound_role(self):
        """`is a siamese cat skimmer` was bridged to skimmers, and then --
        once subsumption became reflexive -- answered yes."""
        self.assertNotEqual(self.verdict("is a siamese cat skimmer"),
                            "VERIFIED")

    # -- F3: a class is only a class within one vocabulary -----------------
    def test_typicality_does_not_rank_across_corpora(self):
        for question in ("is a dog a typical animal",
                         "is a robin a typical animal"):
            with self.subTest(question=question):
                self.assertNotEqual(self.verdict(question), "CONTRADICTED")

    def test_a_class_with_no_core_says_so(self):
        payload = self.engine.ask("is a dog a typical animal")
        self.assertEqual(payload["verdict"], "UNKNOWN")
        self.assertIn("no core", payload["note"])

    def test_typicality_still_ranks_within_one_corpus(self):
        found = self.engine.contrast.typicality("beaver", "animal")
        self.assertIsNotNone(found)
        self.assertEqual(found.corpus, "awa2")
        self.assertTrue(found.excluded)

    # -- F4: subsumption is reflexive --------------------------------------
    def test_everything_is_itself(self):
        for question in ("is a bee a bee", "is a dog a dog",
                         "is a hammer a hammer", "is a glove a glove"):
            with self.subTest(question=question):
                self.assertEqual(self.verdict(question), "VERIFIED")

    # -- F5: the doer is accounted for -------------------------------------
    def test_a_script_does_not_ignore_who_the_question_named(self):
        beaver = self.engine.ask("what happens when a beaver moves")
        piano = self.engine.ask("what happens when a piano moves")
        self.assertNotEqual(beaver["note"], piano["note"])
        for payload, actor in ((beaver, "beaver"), (piano, "piano")):
            with self.subTest(actor=actor):
                self.assertEqual(payload["causal"]["actor"], actor)

    # -- F6: refuse what cannot be answered, define what can ---------------
    def test_constructions_with_no_data_behind_them_are_refused(self):
        for question in ("what is the capital of France",
                         "when did the war end", "what does hello mean",
                         "what is the opposite of hot",
                         "who invented the telephone"):
            with self.subTest(question=question):
                self.assertEqual(self.verdict(question), "UNSUPPORTED")

    def test_a_definition_comes_from_the_taxonomy(self):
        payload = self.engine.ask("what is a robin")
        self.assertEqual(payload["verdict"], "DEFINED")
        self.assertEqual(payload["definition"]["genus"], "thrush.n.03")
        self.assertIn("songbird", payload["definition"]["gloss"])

    def test_defining_an_unknown_word_says_which_word(self):
        payload = self.engine.ask("what is a wemble")
        self.assertEqual(payload["verdict"], "UNKNOWN_WORD")
        self.assertIn("wemble", payload["note"])

    def test_define_does_not_swallow_the_questions_around_it(self):
        for question, rule in (("what is a dog made of", "made_of"),
                               ("what is similar to a dog", "R21"),
                               ("what is the difference between a dog and a "
                                "cat", "R21")):
            with self.subTest(question=question):
                self.assertEqual(
                    self.engine.ask(question)["parse"]["relation"], rule)


@requires_store
class DialogueQueryTests(unittest.TestCase):
    """The queries a teaching dialogue puts to semantic memory.

    Predicted from a conversation rather than from the code -- "Hello" / "what
    is hello" / "it is a greeting" / "who are you" -- and then asked. What a
    cognition needs before it can produce the next line is mostly taxonomic,
    and the taxonomic half works; what fails is indexical, and the point of
    these tests is that it fails *by name* rather than by answering something
    else.
    """

    @classmethod
    def setUpClass(cls):
        from research.v687.reasoning import ReasoningEngine
        cls.engine = ReasoningEngine(STORE)

    @classmethod
    def tearDownClass(cls):
        cls.engine.reasoner.close()

    def verdict(self, question):
        return self.engine.ask(question)["verdict"]

    # -- a word the cue spent is not the subject ---------------------------
    def test_a_relation_cue_does_not_become_the_subject(self):
        """`what is needed to greet` was answered about `need`, and `what
        does a greeting cause` about `cause` -- the question's own grammar
        read back as the thing it asks about."""
        for question, subject in (("what is needed to greet", "greet"),
                                  ("what does a greeting cause", "greeting"),
                                  ("what is needed to bake bread", "bread")):
            with self.subTest(question=question):
                self.assertEqual(
                    self.engine.parser.parse(question).subject, subject)

    def test_the_cue_still_leaves_the_subject_alone(self):
        """Only the words naming the relation are spent. A cue pattern can
        span half the question, and spending the match left `what is a hammer
        made of` with no subject at all."""
        for question, subject in (("what is a hammer made of", "hammer"),
                                  ("where does a penguin live", "penguin"),
                                  ("what is a hammer used for", "hammer"),
                                  ("what can a violin do", "violin")):
            with self.subTest(question=question):
                self.assertEqual(
                    self.engine.parser.parse(question).subject, subject)

    # -- indexicals ---------------------------------------------------------
    def test_an_indexical_is_refused_rather_than_answered_generically(self):
        """`what is my name`, `whose name is it` and `what is a name` were
        three questions with one answer: the properties of name.n.01."""
        for question in ("what is my name", "whose name is it", "who am i",
                         "what am i", "what is your name"):
            with self.subTest(question=question):
                self.assertEqual(self.verdict(question), "UNSUPPORTED")

    def test_the_kind_question_underneath_still_answers(self):
        self.assertEqual(self.verdict("what is a name"), "DEFINED")
        self.assertEqual(self.verdict("does a person have a name"), "VERIFIED")

    # -- what a definition has to carry for a dialogue ---------------------
    def test_a_definition_reports_the_sort(self):
        """Whether a thing is an object or an abstraction decides what is
        worth asking about it next."""
        for word, sort in (("hello", "abstraction.n.06"),
                           ("a robin", "physical entity.n.01"),
                           ("a conversation", "abstraction.n.06")):
            with self.subTest(word=word):
                payload = self.engine.ask(f"what is {word}")
                self.assertEqual(payload["definition"]["sort"], sort)

    def test_a_name_is_not_mistaken_for_a_kind(self):
        """`I am Adrian` resolves to a 20th-century physiologist, and `is
        adrian a person` says yes. The store holds no instances, and a
        definition that does not say so is how the two Adrians get confused."""
        for word in ("adrian", "mary", "peter"):
            with self.subTest(word=word):
                payload = self.engine.ask(f"what is {word}")
                self.assertTrue(
                    payload["definition"]["names_an_individual"])
                self.assertIn("names one individual", payload["note"])

    def test_an_ordinary_kind_is_not_flagged_as_an_individual(self):
        for word in ("a robin", "hello", "a person", "a dog", "a greeting"):
            with self.subTest(word=word):
                payload = self.engine.ask(f"what is {word}")
                self.assertFalse(
                    payload["definition"]["names_an_individual"])

    # -- what the dialogue actually needs and gets -------------------------
    def test_the_taxonomic_half_of_the_dialogue_answers(self):
        for question, verdict in (
                ("what is hello", "DEFINED"),
                ("is hello a greeting", "VERIFIED"),
                ("is hello a communication", "VERIFIED"),
                ("what kinds of greeting are there", "LISTING"),
                ("what is a person", "DEFINED"),
                ("does a person have a name", "VERIFIED"),
                ("can a person speak", "VERIFIED"),
                ("what is a greeting used for", "LISTING"),
                ("where do you find a greeting", "LISTING")):
            with self.subTest(question=question):
                self.assertEqual(self.verdict(question), verdict)

    def test_what_the_data_cannot_support_stays_unknown(self):
        """Not failures -- the honest half. No corpus here records what
        follows a greeting or whether one is polite."""
        for question in ("what happens after a greeting",
                         "is a greeting polite",
                         "is a greeting part of a conversation",
                         "is hello a typical greeting"):
            with self.subTest(question=question):
                self.assertEqual(self.verdict(question), "UNKNOWN")


@requires_store
class SemanticDialogueTests(unittest.TestCase):
    """The dialogue carried on past the introductions, where it turns semantic.

    A cognition being told about a dog asks what a dog is, whether it is an
    animal, whether it eats meat, how it differs from a wolf, and what would
    explain barking. Those are questions about kinds, and they are the half of
    a dialogue this memory is actually for.
    """

    @classmethod
    def setUpClass(cls):
        from research.v687.reasoning import ReasoningEngine
        cls.engine = ReasoningEngine(STORE)

    @classmethod
    def tearDownClass(cls):
        cls.engine.reasoner.close()

    def verdict(self, question):
        return self.engine.ask(question)["verdict"]

    # -- a negation denies what it scopes over, not the sentence it is in --
    def test_a_crawled_negation_about_something_else_is_not_a_denial(self):
        """ConceptNet gives dogs `capable of not eat bone of contention`, a
        mangled idiom, and it answered `does a dog eat meat` with no."""
        self.assertEqual(self.verdict("does a dog eat meat"), "VERIFIED")

    def test_a_negation_the_question_covers_still_denies(self):
        self.assertEqual(self.verdict("does a penguin fly"), "CONTRADICTED")
        self.assertEqual(self.verdict("is a whale furry"), "CONTRADICTED")

    def test_the_negation_scope_is_read_from_the_negator(self):
        from research.v687.identify import Identifier
        self.assertTrue(Identifier.denies_term(["fly"], "capable of cannot fly"))
        self.assertFalse(Identifier.denies_term(
            ["eat"], "capable of not eat bone of contention"))
        self.assertFalse(Identifier.denies_term(
            ["person"], "capable of never attacked person"))

    # -- a cue is a word, not a substring ----------------------------------
    def test_a_cue_does_not_match_inside_a_word(self):
        """`beagle` contains `be`, so `does a beagle breathe` was read as a
        property question and answered UNKNOWN, while `does a dog breathe`
        fell through to the polar default and answered correctly."""
        for question in ("does a beagle breathe", "does a dog breathe",
                         "does a beagle bark"):
            with self.subTest(question=question):
                self.assertEqual(
                    self.engine.parser.parse(question).relation, "capable_of")
                self.assertEqual(self.verdict(question), "VERIFIED")

    def test_the_cues_still_match_their_own_inflections(self):
        for question, relation in (
                ("what is needed to bake bread", "has_prerequisite"),
                ("what does a dog eat", "capable_of"),
                ("what is a hammer made of", "made_of"),
                ("where does a penguin live", "at_location"),
                ("what is a hammer used for", "used_for")):
            with self.subTest(question=question):
                self.assertEqual(
                    self.engine.parser.parse(question).relation, relation)

    # -- R27: absence is not denial, except between the top branches -------
    def test_the_top_branches_of_the_taxonomy_exclude_each_other(self):
        for question in ("is a dog a plant", "is a dog an idea",
                         "is hello an animal", "is a rose an animal",
                         "is a violin an animal"):
            with self.subTest(question=question):
                self.assertEqual(self.verdict(question), "CONTRADICTED")

    def test_exclusion_is_not_claimed_where_the_tree_has_holes(self):
        """WordNet does not record that a dog is a pet or that a whale is not
        a fish, and reading either absence as a denial would be the closed
        world mistake this whole store is careful about."""
        for question in ("is a dog a pet", "is a whale a fish",
                         "is a bat a bird", "is a person an animal"):
            with self.subTest(question=question):
                self.assertEqual(self.verdict(question), "UNKNOWN")

    def test_exclusion_needs_every_sense_of_the_target(self):
        """`plant` also means a factory and a stooge in an audience."""
        self.assertIsNotNone(
            self.engine.reasoner.excludes("dog.n.01", "plant"))
        self.assertIsNone(
            self.engine.reasoner.excludes("dog.n.01", "animal"))

    # -- a hedged reading falls back rather than answering the wrong one ---
    def test_a_hedged_is_a_falls_back_to_the_property_reading(self):
        """`is winter cold` has the shape of `is a chair furniture`, and only
        the data tells them apart: `winter has_property cold` was recorded at
        0.87 and thrown away by a taxonomy walk."""
        self.assertEqual(self.verdict("is winter cold"), "VERIFIED")
        self.assertEqual(self.verdict("is a wolf wild"), "VERIFIED")
        self.assertEqual(self.verdict("is a chair furniture"), "VERIFIED")

    def test_r27_does_not_deny_a_property_by_excluding_a_noun_sense(self):
        """Every high-confidence false denial the COMPS audit found.

        `modern`, `aquatic`, `cold` and `feminine` all carry a noun sense in
        a branch the subject cannot be in, so the taxonomy reading excluded
        and answered CONTRADICTED at 0.95 -- confident, and about a question
        nobody asked. On a bare predicate the is_a reading is a guess, so the
        exclusion is withdrawn and the property reading gets its turn.

        These are outside the 521 concepts the norms cover, which is where
        the bug bites: R17 answers the covered ones and hides it.
        """
        for question, concept in (("is a laptop modern", "laptop.n.01"),
                                  ("is a skyscraper modern", "skyscraper.n.01")):
            with self.subTest(question=question):
                self.assertIsNone(self.engine.profiles.route(question))
                self.assertIsNotNone(              # the exclusion is armed
                    self.engine.reasoner.excludes(concept, "modern"))
                self.assertEqual(self.verdict(question), "VERIFIED")

    def test_a_withdrawn_exclusion_leaves_absence_and_not_denial(self):
        """When the property reading finds nothing either, the taxonomy
        answer is still standing and still answers the wrong question."""
        answer = self.engine.ask("is a mussel aquatic")
        self.assertNotEqual(answer["verdict"], "CONTRADICTED")

    def test_the_determiner_is_what_withdraws_it_and_not_the_adjective(self):
        """`animal` owns an adjective sense too, so gating the withdrawal on
        the target having one would have taken `is a mouse an animal` -- a
        page example -- with it. `an animal` carries a determiner and is not
        hedged; `modern` is bare and is."""
        parse = self.engine.parser.parse
        self.assertTrue(parse("is a television modern").hedged)
        self.assertFalse(parse("is a mouse an animal").hedged)
        self.assertEqual(self.verdict("is a dog a plant"), "CONTRADICTED")
        self.assertEqual(self.verdict("is a dog an idea"), "CONTRADICTED")

    # -- what the dialogue asks and gets -----------------------------------
    def test_the_semantic_turn_answers_end_to_end(self):
        for question, verdict in (
                ("what is a beagle", "DEFINED"),
                ("is a beagle a dog", "VERIFIED"),
                ("is a beagle an animal", "VERIFIED"),
                ("is a dog an organism", "VERIFIED"),
                ("can a dog bark", "VERIFIED"),
                ("do all dogs bark", "VERIFIED"),
                ("is meat food", "VERIFIED"),
                ("is a stranger a person", "VERIFIED"),
                ("is a house a building", "VERIFIED"),
                ("is winter a season", "VERIFIED"),
                ("do all animals breathe", "VERIFIED"),
                ("is an animal alive", "VERIFIED")):
            with self.subTest(question=question):
                self.assertEqual(self.verdict(question), verdict)


@requires_store
class PageExampleTests(unittest.TestCase):
    """Every example question on the page, against the card it sits on.

    A card is a claim: click Analogy and every question there is answered by
    R24. An example that routes elsewhere is broken even when its answer is
    defensible, because the reader clicked the card to see that rule work.
    The questions are read out of `app.html` rather than copied here, so
    adding one to the page adds it to this test.
    """

    #: card name -> (rules it may route to, verdicts it may return)
    PROMISED = {
        "Taxonomy & facts": ({"is_a", "capable_of", "made_of", "at_location",
                              "used_for", "has_part", "has_property", "verify"},
                             {"VERIFIED", "CONTRADICTED", "LISTING"}),
        "Bridging": (None, {"LISTING"}),
        "Identification": ({"identify"}, {"IDENTIFIED", "AMBIGUOUS"}),
        "Retrieval": ({"profile", "verify"},
                      {"PROFILE", "VERIFIED", "CONTRADICTED"}),
        "Logic & quantifiers": ({"verify"},
                                {"VERIFIED", "CONTRADICTED", "UNRECORDED"}),
        "Contrast & counting": ({"R21", "R25"},
                                {"LISTING", "VERIFIED", "CONTRADICTED"}),
        "Inverse": ({"R22"}, {"LISTING"}),
        "Scripts & abduction": ({"R23"}, {"LISTING"}),
        "Analogy": ({"R24"}, {"LISTING"}),
        "Exclusion": ({"is_a"}, {"CONTRADICTED", "UNKNOWN"}),
        "Definition": ({"R26"}, {"DEFINED", "UNKNOWN_WORD"}),
        "Refused": ({"R18"}, {"UNSUPPORTED"}),
    }

    @classmethod
    def setUpClass(cls):
        from research.v687.reasoning import ReasoningEngine
        cls.engine = ReasoningEngine(STORE)

    @classmethod
    def tearDownClass(cls):
        cls.engine.reasoner.close()

    @staticmethod
    def cards():
        import re
        page = (Path(__file__).parent / "app.html").read_text(encoding="utf-8")
        opening = page.index("const REASONING = [")
        block = page[opening:page.index("\n];", opening)]
        for match in re.finditer(
                r'name:\s*"([^"]+)",\s*rule:\s*"([^"]+)",(.*?)'
                r'questions:\s*\[(.*?)\]', block, re.S):
            name, _, _, listed = match.groups()
            yield name, re.findall(r'"([^"]+)"', listed)

    def test_every_example_is_answered_by_the_rule_its_card_promises(self):
        seen = 0
        for name, questions in self.cards():
            rules, verdicts = self.PROMISED[name]
            for question in questions:
                seen += 1
                with self.subTest(card=name, question=question):
                    payload = self.engine.ask(question)
                    if rules:
                        self.assertIn(payload["parse"]["relation"], rules)
                    self.assertIn(payload["verdict"], verdicts)
                    self.assertTrue(payload["steps"],
                                    "an example with no derivation to show")
        self.assertGreaterEqual(seen, 60, "the page lost its examples")


@requires_store
class AnswerRoutingTests(unittest.TestCase):
    """Which rule answers, and which fact it cites."""

    @classmethod
    def setUpClass(cls):
        from research.v687.reasoning import ReasoningEngine
        cls.engine = ReasoningEngine(STORE)

    @classmethod
    def tearDownClass(cls):
        cls.engine.reasoner.close()

    def test_the_norms_hand_back_what_they_did_not_answer(self):
        """`can a dog fall into a hole` is answered as `verify` over feature
        norms, which have nothing of their own to say, so it is handed to the
        crawl.

        **What the crawl then says changed in §25.** The answer used to be
        VERIFIED on one sentence at `canine.n.02`, which the norms described
        seven kinds of -- too few for R19 to speak. With distilled witnesses
        there are thirteen, four bear it out, and R19 refuses. The hand-back
        is what this test is for and it still happens; the fact is still
        found and still cited as the suggestion behind the refusal."""
        payload = self.engine.ask("can a dog fall into a hole")
        self.assertEqual(payload["parse"]["relation"], "capable_of")
        self.assertEqual(payload["verdict"], "UNKNOWN")
        self.assertIn("R19", payload["note"])
        cited = [fact["object"] for fact in payload["suggestions"]]
        self.assertIn("fall into hole", cited)

    def test_a_corroborated_inheritance_stays_with_the_norms(self):
        """The hand-back must not take `does a robin fly` with it: there the
        norms really do bear on the answer, a clear majority of the birds.

        **Neither number is pinned, and the second attempt got that wrong
        too.** `derived/distilled_norms.json` moved this from `21 of 29` to
        `28 of 29`; a later pin on the denominator then broke when
        `derived/distilled_kinds.json` moved it to `30 of 35`. Both artifacts
        are build products and a fresh clone has neither, so every one of
        those readings is correct on some checkout. What is invariant is the
        claim the test is named for: R19 can speak here, a majority of the
        birds bear it out, and the note quotes whatever it actually counted.
        """
        payload = self.engine.ask("does a robin fly")
        self.assertEqual(payload["parse"]["relation"], "verify")
        self.assertEqual(payload["verdict"], "VERIFIED")
        bearing, kinds = self.engine.profiles.corroboration("bird.n.01", "fly")
        self.assertGreaterEqual(kinds, 8)          # R19 can speak at all
        self.assertGreater(bearing / kinds, 0.5)   # and a majority agrees
        self.assertIn(f"{bearing} of {kinds}", payload["note"])

    def test_an_ancestor_fact_is_ranked_by_what_it_accounts_for(self):
        """Shortest-first picked `capable of fall victim` over `capable of
        fall into hole`. Coverage first, shortness among equals.

        Asserted on the detail rather than the predicate, because the ranking
        is a sort over the candidates and survives whatever R19 then does
        with the winner -- which since §25 is refuse it. Reading it off
        `predicate` tested the sort *and* the corroboration; this tests the
        sort, which is what the docstring is about."""
        for asked, wanted in ((["fall", "hole"], "fall into hole"),
                              (["fall"], "fall victim")):
            with self.subTest(asked=asked):
                answer = self.engine.profiles.verify_one("dog", "fall",
                                                         asked=asked)
                self.assertIn(wanted, answer.detail)

    def test_a_preposition_is_not_a_property(self):
        _, _, terms = self.engine.profiles.route("can a dog fall into a hole")
        self.assertEqual(terms, ["fall", "hole"])

    def test_a_yes_no_question_is_not_a_compound_role(self):
        """spaCy tags `breathe` as a noun, so the bridge read `dog breathe`
        the way it reads `violin player` and answered about breathe.v.01."""
        for question in ("does a dog breathe", "does a beagle bark"):
            with self.subTest(question=question):
                payload = self.engine.ask(question)
                self.assertTrue(payload["concept"].startswith(
                    question.split()[2]), payload["concept"])

    def test_a_possessive_still_bridges(self):
        payload = self.engine.ask("what does a dog's owner need")
        self.assertTrue(payload.get("bridge"))
        self.assertEqual(payload["concept"], "owner.n.01")


@requires_store
class BackwardsQualityTests(unittest.TestCase):
    """What the inverse offers as an answer."""

    @classmethod
    def setUpClass(cls):
        from research.v687.reasoning import ReasoningEngine
        cls.engine = ReasoningEngine(STORE)

    @classmethod
    def tearDownClass(cls):
        cls.engine.reasoner.close()

    def test_the_word_restated_is_not_an_answer_about_itself(self):
        """`what has wings` answered `wing part_of bastard wing` -- a wing,
        offered as a thing that has wings."""
        from research.v687.inverse import SUBJECT
        found = self.engine.inverse.find("has_part", "wings", SUBJECT)
        names = {row["concept"].split(".")[0] for row in found.subjects}
        self.assertNotIn("wing", names)
        self.assertTrue({"bird", "bat", "angel"} & names, names)

    def test_a_self_contradicting_row_is_not_an_answer(self):
        """`aileron has_part wing` and `aileron part_of wing` are both stored
        and only one can be true."""
        from research.v687.inverse import SUBJECT
        found = self.engine.inverse.find("has_part", "wings", SUBJECT)
        names = {row["concept"].split(".")[0] for row in found.subjects}
        self.assertNotIn("aileron", names)
        self.assertNotIn("flight feather", names)

    def test_a_pin_that_empties_the_reading_says_so(self):
        loose = self.engine.ask("what is a mouse part of")
        pinned = self.engine.ask("what is a mouse part of", None,
                                 {"mouse": "mouse.n.01"})
        self.assertGreater(len(loose["backwards"]["subjects"]),
                           len(pinned["backwards"]["subjects"]))
        self.assertIn("pinned", pinned["note"])


@requires_store
class AnswerWordingTests(unittest.TestCase):
    """Sentences that argued with themselves."""

    @classmethod
    def setUpClass(cls):
        from research.v687.reasoning import ReasoningEngine
        cls.engine = ReasoningEngine(STORE)

    @classmethod
    def tearDownClass(cls):
        cls.engine.reasoner.close()

    def test_a_list_cut_short_says_it_was_cut_short(self):
        """`41 do not: antelope, beaver, blue whale, bobcat` names four and
        claims forty-one, which reads as arithmetic gone wrong.

        This used to ask `do all birds fly` and expect seven exceptions. Four
        of those seven were COMPS foils rather than denials, and with them
        gone the honest list is `chicken, emu, penguin` -- three named, three
        claimed, nothing to truncate. So the truncation is exercised on a list
        that is still long.
        """
        note = self.engine.ask("do all mammals fly")["note"]
        self.assertIn("41 do not", note)
        self.assertIn("more", note)

    def test_dropping_the_foils_left_the_flightless_birds_that_really_are(self):
        """The same question, as a record of what the change bought: seven
        exceptions became three, and the three are flightless."""
        note = self.engine.ask("do all birds fly")["note"]
        self.assertIn("3 do not", note)
        for bird in ("chicken", "emu", "penguin"):
            self.assertIn(bird, note)

    def test_sharing_no_prefix_is_not_a_parting_point(self):
        """`they walk 0 node(s) together before parting at X` cannot both be
        true."""
        note = self.engine.ask(
            "what is the difference between a dog and a wolf")["note"]
        self.assertNotIn("0 node(s)", note)
        self.assertIn("first node", note)

    def test_corroboration_is_reported_only_where_it_was_consulted(self):
        """Printing `0 of 7 bear it out` beside a yes reads as the answer
        arguing with itself; below the minimum R19 declines to judge."""
        note = self.engine.ask("is a wolf dangerous")["note"]
        self.assertNotIn("0 of 7", note)


class OverAffirmationTests(unittest.TestCase):
    """The audit that asked why `can a rock swim` was VERIFIED.

    Four questions were named and all four came back yes. Behind them were
    four separate defects, and this class holds one test per defect plus the
    two answers that must not be lost to the fixes.
    """

    @classmethod
    def setUpClass(cls):
        from research.v687.reasoning import ReasoningEngine
        cls.engine = ReasoningEngine(STORE)

    @classmethod
    def tearDownClass(cls):
        cls.engine.reasoner.close()

    def verdict(self, question):
        return self.engine.ask(question)["verdict"]

    def test_a_target_inside_a_longer_word_is_not_a_match(self):
        """The shortcut for "the fact names the target outright" was a raw
        substring test. `can a tree fly` was VERIFIED because a plant
        attracts a butterfly, `can a car walk` because a car blocks the
        sidewalk, `can a horse sing` because a horse goes missing."""
        from research.v687.language import whole_words
        for target, phrase in (("fly", "attract butterfly"),
                               ("walk", "block the sidewalk"),
                               ("sing", "go missing"),
                               ("run", "get drunk"),
                               ("run", "develop stronger trunk")):
            with self.subTest(target=target):
                self.assertIn(target, phrase, "the old test passed on this")
                self.assertFalse(whole_words(target, phrase))
        self.assertTrue(whole_words("swim", "go for swim"))
        self.assertTrue(whole_words("lay eggs", "lay eggs today"))
        for question in ("can a tree fly", "can a car walk", "can a horse sing"):
            with self.subTest(question=question):
                self.assertNotEqual(self.verdict(question), "VERIFIED")

    def test_a_qualified_fact_does_not_affirm_the_bare_claim(self):
        """R28. `rock capable_of "go for swim"` is about a place people swim
        and `fish capable_of "walk on land"` is about the fish that do.

        v687 already declined the mirror of this -- denying a qualified
        property does not deny the property -- and this is the same reading
        applied to yes. The note has to say which, because "the store records
        this narrower thing" and "the store records nothing" are different
        answers and only the first names what to ask next."""
        answer = self.engine.ask("can a rock swim")
        self.assertEqual(answer["verdict"], "UNKNOWN")
        self.assertIn("go for swim", answer["note"])
        self.assertIn("narrower claim", answer["note"])

    def test_a_class_fact_is_put_to_the_class_before_it_is_inherited(self):
        """R19, which existed and was wired into the norms path only.

        `do pigs fly` found nothing on hog, swine, even-toed ungulate,
        ungulate or placental, then took `mammal capable_of fly` five levels
        up. That is a fact about bats. The norms are what tell an existential
        from a universal, because they asked a fixed question of every
        concept they cover.

        **The class named in the note is not pinned.** It used to be `mammal`,
        the level the crawl attached the sentence to; `Profiles.sharpest`
        (§23) checks the narrowest level that can speak instead, so the note
        now reads `0 of the 18 kinds of even-toed ungulate` -- the same
        refusal on sharper evidence. What must hold is that R19 ran, that it
        refused, and that whatever class it names is one the concept actually
        belongs to."""
        for question in ("do pigs fly", "does a cat lay eggs"):
            with self.subTest(question=question):
                answer = self.engine.ask(question)
                self.assertEqual(answer["verdict"], "UNKNOWN")
                self.assertIn("R19", answer["note"])
                self.assertIn("kinds of", answer["note"])
                named = answer["note"].split(" is recorded as")[0].strip()
                above = {node.rsplit(".", 2)[0] for node, _distance, _parents
                         in self.engine.reasoner.ascend(answer["concept"])}
                self.assertIn(named, above,
                              f"{named!r} is not a class {question} is about")

    def test_a_negation_written_into_the_object_is_read_as_one(self):
        """R3 only ever looked at the relation column. `fish has_a "no legs"`
        put the negation in the object, lemma overlap scored it a full answer
        to "legs", and `does a fish have legs` came back VERIFIED on the fact
        that a fish has none."""
        for question in ("does a fish have legs", "does a snake have legs"):
            with self.subTest(question=question):
                self.assertEqual(self.verdict(question), "CONTRADICTED")

    def test_a_denial_does_not_outweigh_the_same_source_saying_otherwise(self):
        """`winter has_property cold` is 0.87 and `"never cold"` is 0.23,
        both from Ascent++. One sentence about a mild winter does not
        overturn twenty saying it is cold. Same source is the whole of the
        comparison -- ConceptNet's 0.35 is a constant, not a percentile."""
        self.assertEqual(self.verdict("is winter cold"), "VERIFIED")

    def test_negation_scopes_forward(self):
        """`cattle capable_of "digest grass but humans cannot"` asserts the
        grass and denies it of humans. Counting any negator anywhere in the
        phrase made `does a cow eat grass` CONTRADICTED."""
        from research.v687.profile import Profiles
        denies = Profiles._denies
        self.assertFalse(denies("digest grass but humans cannot", "grass"))
        self.assertTrue(denies("no legs", "legs"))
        self.assertTrue(denies("cannot fly", "fly"))
        self.assertTrue(denies("digest grass but humans cannot"))
        self.assertEqual(self.verdict("does a cow eat grass"), "VERIFIED")

    def test_the_answers_that_must_survive_all_of_it(self):
        """Four rules that each refuse something, and the yes they must not
        take with them. Every one of these rests on the target stated plainly
        and near: `bird capable_of "fly"`, not "fly in the sky"."""
        for question in ("can a bird fly", "can a fish swim", "can a horse run",
                         "does a bird lay eggs", "can a person speak",
                         "does a dog have legs", "can a dog bark",
                         "do all animals breathe", "is a dog an animal"):
            with self.subTest(question=question):
                self.assertEqual(self.verdict(question), "VERIFIED")


class PinnedSenseIsHonouredTests(unittest.TestCase):
    """`can a dog bark` answered VERIFIED with `bark` pinned to the covering
    of a tree, and to a three-masted sailing ship, with the same note both
    times. The reading shown was never the reading used."""

    @classmethod
    def setUpClass(cls):
        from research.v687.reasoning import ReasoningEngine
        cls.engine = ReasoningEngine(STORE)

    @classmethod
    def tearDownClass(cls):
        cls.engine.reasoner.close()

    def test_a_modal_is_completed_by_a_verb_whatever_the_tagger_says(self):
        """spaCy reads `a dog bark` as a compound noun, so every word that is
        also a common noun came back NOUN: bark, fly, run, swim. The tag is
        what the sense picker offers a reading by, which is how `bark` in
        `can a dog bark` was offered as the covering of a tree and marked the
        default."""
        for question, word in (("can a dog bark", "bark"),
                               ("can a bird fly", "fly"),
                               ("can a horse run", "run"),
                               ("can a rock swim", "swim"),
                               ("why does a dog bark", "bark"),
                               ("do all dogs bark", "bark")):
            with self.subTest(question=question):
                tags = {t["text"].lower(): t["pos"]
                        for t in self.engine.parser.parse(question).tokens}
                self.assertEqual(tags.get(word), "VERB", tags)

    def test_the_noun_after_the_verb_is_left_alone(self):
        """Reading past the verb turned `legs` into one. `is` and `have` take
        a noun and are not in the set at all."""
        for question, word, pos in (("does a dog have legs", "legs", "NOUN"),
                                    ("does a bird lay eggs", "eggs", "NOUN"),
                                    ("is a dog an animal", "animal", "NOUN"),
                                    ("is a chair furniture", "furniture",
                                     "NOUN")):
            with self.subTest(question=question):
                tags = {t["text"].lower(): t["pos"]
                        for t in self.engine.parser.parse(question).tokens}
                self.assertEqual(tags.get(word), pos, tags)

    def test_the_ontology_vetoes_splitting_a_compound(self):
        """`fire truck` is a lemma, so `can a fire truck` keeps its subject.
        `truck fly` is not, so `can a fire truck fly` splits after truck."""
        parser = self.engine.parser
        self.assertEqual(parser.parse("can a fire truck fly").subject,
                         "fire truck")
        self.assertEqual(parser.parse("can a fire truck").subject,
                         "fire truck")

    def test_the_tagger_can_swap_the_subject_and_the_verb(self):
        """`can dogs bark` comes back with `dogs` a VERB and `bark` a NOUN --
        both slots wrong at once. The picker offered `chase.v.01` as the
        reading of `dog`, and a pin checked against the tags refused
        `bark.v.04`, which is the correct reading, on a question the reader
        had already corrected twice over."""
        tags = {t["text"].lower(): t["pos"]
                for t in self.engine.parser.parse("can dogs bark").tokens}
        self.assertEqual(tags.get("dogs"), "NOUN", tags)
        self.assertEqual(tags.get("bark"), "VERB", tags)
        self.assertEqual(
            self.engine.ask("can dogs bark", None,
                            {"dog": "dog.n.01",
                             "bark": "bark.v.04"})["verdict"], "VERIFIED")

    def test_the_slot_is_read_off_the_grammar_not_off_a_tag(self):
        """What completes the auxiliary, which is the only thing a pin is
        checked against. `is` and `are` take a noun and have no such slot."""
        for question, slot in (("can dogs bark", "bark"),
                               ("can a dog bark", "bark"),
                               ("can a large dog fall", "fall"),
                               ("does a dog have legs", "have"),
                               ("can a fire truck move", "move"),
                               ("can a fire truck", ""),
                               ("is a dog an animal", "")):
            with self.subTest(question=question):
                self.assertEqual(
                    self.engine.parser.parse(question).verb_slot, slot)

    def test_a_pin_the_sentence_cannot_take_is_refused(self):
        """The bug as reported: a noun sense pinned onto the word completing
        a modal. Asking whether a dog can tough-protective-covering-of-a-tree
        is not a question, and answering it yes is worse than declining."""
        for sense in ("bark.n.01", "bark.n.03"):
            with self.subTest(sense=sense):
                answer = self.engine.ask("can a dog bark", None,
                                         {"bark": sense})
                self.assertEqual(answer["verdict"], "UNSUPPORTED")
                self.assertIn(sense, answer["note"])
                self.assertIn("as a verb", answer["note"])
        # And the same word pinned to a reading the sentence *can* take is
        # not refused.
        self.assertEqual(
            self.engine.ask("can a dog bark", None,
                            {"bark": "bark.v.04"})["verdict"], "VERIFIED")

    def test_a_pin_that_did_not_bear_on_the_answer_says_so(self):
        """R17 matches the word against its own predicates and resolves no
        sense, so every reading of `bark` gives the same answer. A control
        that looks as though it works everywhere and works in places is
        worse than no control."""
        answer = self.engine.ask("can a dog bark", None,
                                 {"bark": "bark.v.04"})
        self.assertEqual(answer["pins_unused"], {"bark": "bark.v.04"})
        self.assertEqual(answer["pins_used"], {})
        self.assertIn("did not use the reading you pinned", answer["note"])

    def test_a_pin_that_did_bear_on_the_answer_is_not_complained_about(self):
        """`mouse` pinned to the animal is what the derivation stands on."""
        answer = self.engine.ask("is a mouse an animal", None,
                                 {"mouse": "mouse.n.01"})
        self.assertEqual(answer["verdict"], "VERIFIED")
        self.assertEqual(answer["pins_used"], {"mouse": "mouse.n.01"})
        self.assertEqual(answer["pins_unused"], {})
        self.assertNotIn("did not use the reading", answer["note"])


class SenseToSenseTests(unittest.TestCase):
    """R29. A pin on the object only means something where there is a sense
    to bind it to, and in this store that is WordNet's synset-to-synset rows.
    """

    @classmethod
    def setUpClass(cls):
        from research.v687.reasoning import ReasoningEngine
        cls.engine = ReasoningEngine(STORE)

    @classmethod
    def tearDownClass(cls):
        cls.engine.reasoner.close()

    def test_the_direction_is_the_stores_and_not_the_columns_name(self):
        """`(car.n.01, part_of, accelerator.n.01)` is in the store and an
        accelerator is a part of a car, not the reverse. Both spellings are
        held and both read the same way round, so a "does X have Y" question
        looks in `part_of`. Asserted on four unambiguous pairs, because
        reading it off the column name gets it backwards."""
        rows = self.engine.reasoner.connection.execute(
            "SELECT concept, object FROM facts WHERE relation = 'part_of' "
            "AND source = 'wordnet' AND concept IN "
            "('car.n.01', 'hand.n.01', 'bird.n.01', 'tree.n.01')").fetchall()
        held = {(row["concept"], row["object"]) for row in rows}
        for whole, part in (("car.n.01", "accelerator.n.01"),
                            ("hand.n.01", "finger.n.01"),
                            ("bird.n.01", "beak.n.02"),
                            ("tree.n.01", "limb.n.02")):
            with self.subTest(whole=whole):
                self.assertIn((whole, part), held)

    def test_a_pinned_object_is_answered_between_senses(self):
        """`does a car have an accelerator` is UNKNOWN through the words --
        the accelerator is recorded only as a synset -- and VERIFIED through
        the graph. This is the pin changing an answer, which on a free-text
        relation it cannot do."""
        self.assertEqual(
            self.engine.ask("does a car have an accelerator")["verdict"],
            "UNKNOWN")
        answer = self.engine.ask("does a car have an accelerator", None,
                                 {"accelerator": "accelerator.n.01"})
        self.assertEqual(answer["verdict"], "VERIFIED")
        self.assertIn("Between senses", answer["note"])
        self.assertEqual(answer["pins_used"],
                         {"accelerator": "accelerator.n.01"})

    def test_it_inherits(self):
        """What is true of a car is true of a hatchback."""
        self.assertEqual(
            self.engine.reasoner.verify_sense(
                "hatchback.n.01", "has_part", "accelerator.n.01").verdict,
            "VERIFIED")

    def test_a_sense_the_graph_does_not_link_falls_back_to_the_words(self):
        """`beak.n.01` is the beak of an animal other than a bird, so the
        graph has no bird-to-beak.n.01 row. The words still answer, and the
        note says the graph was asked first and had nothing -- "the graph
        does not record this" and "no word matched" are different things to
        know."""
        answer = self.engine.ask("does a bird have a beak", None,
                                 {"beak": "beak.n.01"})
        self.assertEqual(answer["verdict"], "VERIFIED")
        self.assertIn("Asked between senses first", answer["note"])

    def test_the_graph_is_not_authoritative(self):
        """The synset rows are patchy: they hold a car's wheel and a dog's
        tail and not a fish's gills or a horse's legs. Letting the graph's
        silence stand as the answer would lose every one of those."""
        reasoner = self.engine.reasoner
        for concept, sense in (("fish.n.01", "gill.n.01"),
                               ("horse.n.01", "leg.n.01")):
            with self.subTest(concept=concept):
                self.assertEqual(
                    reasoner.verify_sense(concept, "has_part", sense).verdict,
                    "UNKNOWN")
        self.assertEqual(self.engine.ask("does a horse have legs")["verdict"],
                         "VERIFIED")

    def test_a_free_text_relation_has_nothing_to_bind_to(self):
        """`capable_of` is 772,890 rows of crawled text and none of them name
        a sense, so `can a dog bark` cannot be answered between senses however
        the reader pins it."""
        self.assertNotIn("capable_of", self.engine.reasoner.SENSE_TAGGED)
        self.assertEqual(
            self.engine.reasoner.verify_sense(
                "dog.n.01", "capable_of", "bark.v.04").verdict, "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
