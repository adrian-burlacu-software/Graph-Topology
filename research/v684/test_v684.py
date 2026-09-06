"""Regression suite. Run: python -m unittest research.v684.test_v684 -v

Tests that need the built store skip themselves when it is absent, so the suite
runs on a fresh clone before `python -m research.v684.build`.
"""
from __future__ import annotations

import unittest

import collections
import sqlite3
from pathlib import Path

from research.v684 import build, compress, rules, senses
from research.v684.language import Parser
from research.v684.reason import Reasoner

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
        from research.v684.server import Engine
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
        from research.v684.language import STOP
        for word in ("into", "onto", "from", "within"):
            self.assertIn(word, STOP)

    def test_particles_are_still_content(self):
        """`fall down` and `fall over` are different claims, so keep both words."""
        from research.v684.language import STOP
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


if __name__ == "__main__":
    unittest.main()
