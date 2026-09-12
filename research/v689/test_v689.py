"""Regression suite for v689: discourse referents over episodic memory.

Run: python -m unittest research.v689.test_v689 -v

Two kinds of test. Reading is tested against a fake lexicon, because what it
decides is which words pick out an individual. Everything after that runs on
v687 itself -- the real `Reasoner`, the real `Parser`, the real rules -- over a
nine-concept store built here, so what R3 and R4 do to an individual is tested
as v687 does it rather than as a stub imagines it. Only v688's loop is a
stand-in: it is asked about kinds, and the tests say what it answers.
"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from research.v687.language import Parser
from research.v687.reason import Reasoner
from research.v689 import reading
from research.v689.asker import Asker
from research.v689.definitions import (DefinitionMemory, GlossReader, pieces,
                                       question_for)
from research.v689.longterm import Archive, Keeper
from research.v689.session import Session

# -- a store small enough to read ------------------------------------------

SCHEMA = """
CREATE TABLE concepts (id TEXT PRIMARY KEY, lemma TEXT NOT NULL,
    pos TEXT NOT NULL, sense INTEGER NOT NULL, definition TEXT,
    descendants INTEGER NOT NULL DEFAULT 0);
CREATE TABLE taxonomy (child TEXT NOT NULL, parent TEXT NOT NULL,
    PRIMARY KEY (child, parent));
CREATE TABLE facts (concept TEXT NOT NULL, relation TEXT NOT NULL,
    object TEXT NOT NULL, source TEXT NOT NULL, confidence REAL NOT NULL,
    sense_assumed INTEGER NOT NULL,
    PRIMARY KEY (concept, relation, object, source));
CREATE TABLE lemmas (lemma TEXT NOT NULL, concept TEXT NOT NULL,
    primary_sense INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (lemma, concept));
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

#: (id, lemma, parent)
CONCEPTS = (("entity.n.01", "entity", None),
            ("organism.n.01", "organism", "entity.n.01"),
            ("animal.n.01", "animal", "organism.n.01"),
            ("person.n.01", "person", "organism.n.01"),
            ("doctor.n.01", "doctor", "person.n.01"),
            ("dog.n.01", "dog", "animal.n.01"),
            ("beagle.n.01", "beagle", "dog.n.01"),
            ("cat.n.01", "cat", "animal.n.01"),
            ("hog.n.03", "hog", "animal.n.01"),
            # A colour is a noun too, and that is what made v687's parser
            # read `is a beagle black` as a hedged `is_a`. The real store has
            # it; a test store without it passed while the page failed.
            ("black.n.01", "black", "entity.n.01"),
            ("airplane.n.01", "airplane", "entity.n.01"),
            ("kitten.n.01", "kitten", "cat.n.01"),
            ("gland.n.01", "gland", "entity.n.01"),
            ("testis.n.01", "testicle", "gland.n.01"),
            ("secrete.v.01", "secrete", None),
            ("produce.v.01", "produce", None))

LEMMAS = tuple((lemma, concept) for concept, lemma, _ in CONCEPTS) + (
    ("pig", "hog.n.03"), ("plane", "airplane.n.01"))

#: Real WordNet glosses, for the definitions reader; everything else is
#: glossed `a <lemma>`.
DEFINITIONS = {
    "kitten.n.01": "young domestic cat",
    "testis.n.01": ("one of the two male reproductive glands that produce "
                    "spermatozoa and secrete androgens")}

FACTS = (("dog.n.01", "capable_of", "swim", "ascentpp", 0.6, 1),
         ("dog.n.01", "capable_of", "bark", "ascentpp", 0.7, 1),
         ("dog.n.01", "has_a", "tail", "ascentpp", 0.6, 1),
         ("beagle.n.01", "has_property", "black", "ascentpp", 0.5, 1),
         ("animal.n.01", "capable_of", "breathe", "conceptnet", 0.35, 0),
         ("airplane.n.01", "capable_of", "fly", "ascentpp", 0.6, 1))

STORE: dict = {}


def setUpModule() -> None:                      # noqa: N802
    folder = tempfile.TemporaryDirectory()
    path = Path(folder.name) / "tiny.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    connection.executemany(
        "INSERT INTO concepts VALUES (?, ?, ?, 1, ?, 1)",
        [(concept, lemma, concept.split(".")[-2],
          DEFINITIONS.get(concept, f"a {lemma}"))
         for concept, lemma, _ in CONCEPTS])
    connection.executemany(
        "INSERT INTO taxonomy VALUES (?, ?)",
        [(concept, parent) for concept, _, parent in CONCEPTS if parent])
    connection.executemany("INSERT INTO lemmas VALUES (?, ?, 1)", LEMMAS)
    connection.executemany("INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?)",
                           FACTS)
    connection.commit()
    connection.close()
    reasoner = Reasoner(path)
    parser = Parser(vocabulary=reasoner.vocabulary(),
                    nouns=reasoner.noun_vocabulary())
    STORE.update(folder=folder, reasoner=reasoner, parser=parser)


def tearDownModule() -> None:                   # noqa: N802
    if STORE:
        STORE["reasoner"].connection.close()
        STORE["folder"].cleanup()


class TinyAsker(Asker):
    """v687 over the tiny store; v688 answers what the test says it does."""

    def __init__(self, outcomes: dict | None = None) -> None:
        super().__init__(STORE["reasoner"], STORE["parser"])
        self.outcomes = outcomes or {}
        self.asked: list[str] = []

    def run(self, question: str) -> dict:
        self.asked.append(question)
        outcome = self.outcomes.get(question, "unknown")
        if isinstance(outcome, dict):
            return outcome
        return {"summary": {"outcome": outcome, "trust": "",
                            "lines": [f"{outcome} — {question}"]}}


def talk(*lines: str, outcomes: dict | None = None):
    asker = TinyAsker(outcomes)
    session = Session(asker)
    return session, [session.say(line) for line in lines], asker


def who(turn) -> str | None:
    referent = turn.resolution.referent if turn.resolution else None
    return referent.id if referent else None


def rules_at(turn, distance: int = 0) -> set:
    return {step["rule"] for step in (turn.walk or {}).get("steps", [])
            if step.get("distance") == distance}


# -- reading ---------------------------------------------------------------

class FakeLexicon:
    KINDS = {"beagle", "dog", "cat", "animal", "pig", "person"}

    def subject(self, question: str):
        tokens = question.lower().split()
        for index, token in enumerate(tokens):
            pair = " ".join(tokens[index:index + 2])
            if pair in self.KINDS:
                return pair
            if token in self.KINDS:
                return token
        return None

    def lemma(self, word: str) -> str:
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        return word

    def known(self, phrase: str) -> bool:
        return phrase in self.KINDS


class ReadingTests(unittest.TestCase):
    lexicon = FakeLexicon()

    def test_contractions_give_negation_one_spelling(self):
        self.assertEqual(reading.words("It can't swim."),
                         ["it", "can", "not", "swim"])
        self.assertEqual(reading.words("There" + chr(8217) + "s a beagle"),
                         ["there", "is", "a", "beagle"])

    def test_the_second_one_has_no_kind_and_a_place(self):
        found = reading.read("does the second one swim", self.lexicon)
        self.assertEqual((found.act, found.mention.form, found.mention.ordinal,
                          found.mention.kind), ("ask", "ordinal", 2, ""))

    def test_a_modifier_describes_rather_than_names(self):
        found = reading.read("is the black beagle fast", self.lexicon)
        self.assertEqual((found.mention.kind, found.mention.modifiers),
                         ("beagle", ["black"]))

    def test_an_indefinite_question_is_about_the_kind(self):
        self.assertEqual(reading.read("does a beagle swim",
                                      self.lexicon).act, "generic")

    def test_an_introduction_can_carry_a_clause(self):
        found = reading.read("there's a beagle that can't swim",
                             self.lexicon)
        self.assertEqual(found.act, "introduce")
        self.assertFalse(found.relative.holds)

    def test_that_alone_is_the_thing_when_nothing_else_is_left(self):
        found = reading.read("is that black", self.lexicon)
        self.assertEqual((found.mention.form, found.rest),
                         ("pronoun", ["black"]))

    def test_my_name_is_a_naming_not_a_claim_about_a_kind(self):
        found = reading.read("my name is Adrian", self.lexicon)
        self.assertEqual((found.act, found.name), ("name", "Adrian"))

    def test_a_claim_about_a_kind_is_teaching(self):
        found = reading.read("a wemble is a kind of animal", self.lexicon)
        self.assertEqual((found.act, found.mention.kind), ("teach", "wemble"))
        found = reading.read("beagles can swim", self.lexicon)
        self.assertEqual((found.act, found.mention.kind), ("teach", "beagle"))

    def test_a_bare_unknown_singular_is_someone_not_a_kind(self):
        self.assertNotEqual(reading.read("Adrian can swim",
                                         self.lexicon).act, "teach")

    def test_a_question_about_a_word_v687_lacks_keeps_its_kind(self):
        found = reading.read("can a wemble fly", self.lexicon)
        self.assertEqual((found.act, found.mention.kind, found.rest),
                         ("generic", "wemble", ["fly"]))

    def test_your_is_the_one_being_talked_to(self):
        found = reading.read("what is your name", self.lexicon)
        self.assertEqual((found.act, found.mention.form),
                         ("ask_name", "addressee"))

    def test_only_a_capitalised_word_after_i_am_is_a_name(self):
        self.assertEqual(reading.read("I am Adrian", self.lexicon).act,
                         "name")
        self.assertEqual(reading.read("i am tired", self.lexicon).act,
                         "tell")


# -- v687's rules, on individuals ------------------------------------------

class RulesOnIndividualsTests(unittest.TestCase):

    def test_r3_a_told_negation_blocks_what_the_kind_does(self):
        _, turns, _ = talk("there is a beagle", "can it swim",
                           "it can't swim", "can it swim",
                           outcomes={"can a beagle swim": "verified"})
        self.assertEqual(turns[1].walk["verdict"], "VERIFIED")
        self.assertEqual(turns[1].answer["source"], "kind")
        self.assertEqual((turns[3].answer["outcome"],
                          turns[3].answer["source"]), ("denied", "told"))
        self.assertIn("R3", rules_at(turns[3]))

    def test_r4_doing_a_thing_shows_it_can(self):
        """Reported from the page: `he was flying` never met `can the pig
        fly`, and the answer came from pigs."""
        _, turns, _ = talk("there was a pig", "he was flying",
                           "can the pig fly",
                           outcomes={"does a pig fly": "denied"})
        self.assertIn("exception", turns[1].answer["text"])
        self.assertEqual((turns[2].answer["outcome"],
                          turns[2].answer["source"]), ("verified", "told"))
        self.assertIn("R4", rules_at(turns[2]))

    def test_not_doing_a_thing_is_not_being_unable_to(self):
        _, turns, _ = talk("there is a pig", "it wasn't flying",
                           "can it fly")
        self.assertEqual(turns[2].answer["source"], "kind")

    def test_not_doing_still_answers_does_it(self):
        _, turns, _ = talk("there is a pig", "it wasn't flying",
                           "does it fly")
        self.assertEqual((turns[2].answer["outcome"],
                          turns[2].answer["source"]), ("denied", "told"))

    def test_what_cannot_does_not(self):
        _, turns, _ = talk("there is a pig", "it can't fly", "does it fly")
        self.assertEqual((turns[2].answer["outcome"],
                          turns[2].answer["source"]), ("denied", "told"))

    def test_r3_reads_a_denial_written_into_the_object(self):
        _, turns, _ = talk("there is a dog", "it has no tail",
                           "does it have a tail")
        self.assertEqual((turns[2].answer["outcome"],
                          turns[2].answer["source"]), ("denied", "told"))

    def test_e1_a_quality_does_not_descend_to_one_of_them(self):
        """`beagle has_property black` is in the store; this beagle is not
        thereby black."""
        _, turns, _ = talk("there is a beagle", "is it black")
        self.assertEqual((turns[1].answer["outcome"],
                          turns[1].answer["source"]),
                         ("unknown", "tendency"))
        self.assertIn("E1", rules_at(turns[1]))

    def test_a_quality_that_is_also_a_noun_is_still_a_quality(self):
        """Reported from the page: `black` is a colour as well as a quality,
        so v687's parser reads `is a beagle black` as a hedged `is_a`. It was
        refused as a taxonomy claim, never stored, and `the black one` found
        nobody."""
        _, turns, _ = talk("there is a beagle", "it is black",
                           "is it black", "there is another beagle",
                           "is the black one fast")
        self.assertEqual(turns[1].answer["source"], "told")
        self.assertEqual((turns[2].answer["outcome"],
                          turns[2].answer["source"]), ("verified", "told"))
        self.assertEqual(who(turns[4]), "r1")

    def test_a_quality_that_is_also_a_noun_is_not_a_taxonomy_denial(self):
        """Asked, a hedged `is_a` gets the taxonomy first and the property
        reading when the taxonomy does not verify -- v687's engine's order --
        so E1 answers it rather than an exclusion."""
        _, turns, _ = talk("there is a beagle", "is it black")
        self.assertEqual((turns[1].answer["outcome"],
                          turns[1].answer["source"]),
                         ("unknown", "tendency"))

    def test_a_told_quality_is_answered_at_the_individual(self):
        _, turns, _ = talk("there is a black beagle", "is it black")
        self.assertEqual((turns[1].answer["outcome"],
                          turns[1].answer["source"]), ("verified", "told"))

    def test_r1_is_it_a_dog_walks_up_from_the_individual(self):
        _, turns, asker = talk("there is a beagle", "is it a dog")
        self.assertEqual(turns[1].answer["outcome"], "verified")
        self.assertIn("dog.n.01", turns[1].walk["chain"])
        self.assertEqual(asker.asked, [])

    def test_a_correction_replaces_rather_than_contradicts(self):
        _, turns, _ = talk("there is a beagle", "it can't swim",
                           "it can swim", "can it swim")
        self.assertEqual((turns[3].answer["outcome"],
                          turns[3].answer["source"]), ("verified", "told"))

    def test_a_narrower_kind_moves_it_down_the_taxonomy(self):
        session, turns, _ = talk("there is a dog", "it is a beagle",
                                 "does it bark")
        self.assertEqual(session.memory.parent["r1"], "beagle.n.01")
        self.assertEqual(turns[2].asked, "does a beagle bark")


# -- the trie ---------------------------------------------------------------

class EpisodicTrieTests(unittest.TestCase):

    def test_growth_is_driven_by_allocation(self):
        """A second beagle shares every predicate with the first, so it
        allocates nothing until one of them is told something."""
        _, turns, _ = talk("there is a beagle", "there is another beagle",
                           "the first beagle is black")
        self.assertGreater(turns[0].growth[-1]["allocated"], 0)
        self.assertEqual(turns[1].growth[-1]["allocated"], 0)
        self.assertEqual(turns[2].growth[-1]["allocated"], 1)

    def test_the_dog_is_identified_through_the_kinds_above_a_beagle(self):
        _, turns, _ = talk("there is a beagle", "does the dog bark")
        self.assertEqual(who(turns[1]), "r1")
        self.assertEqual(turns[1].resolution.identification["wanted"],
                         ["is_a dog"])

    def test_the_black_one_is_identified_by_what_it_was_told(self):
        _, turns, _ = talk("there is a black beagle",
                           "there is a brown beagle",
                           "is the black one fast")
        self.assertEqual(who(turns[2]), "r1")

    def test_a_description_that_fits_nothing_is_not_guessed(self):
        _, turns, _ = talk("there is a black beagle",
                           "is the white one fast")
        self.assertIsNone(who(turns[1]))


# -- attention ---------------------------------------------------------------

class ResolutionTests(unittest.TestCase):

    def test_it_is_the_most_recent(self):
        _, turns, _ = talk("there is a beagle", "there is a cat",
                           "does it purr")
        self.assertEqual(who(turns[2]), "r2")
        self.assertEqual(turns[2].asked, "does a cat purr")

    def test_it_follows_the_conversation_not_the_count(self):
        _, turns, _ = talk("there is a beagle", "it can't swim",
                           "can it swim", "i have another beagle",
                           "can it swim")
        self.assertEqual(who(turns[4]), "r2")

    def test_the_first_and_the_second(self):
        _, turns, _ = talk("there is a beagle", "there is another beagle",
                           "does the first beagle swim",
                           "is the second one fast")
        self.assertEqual((who(turns[2]), who(turns[3])), ("r1", "r2"))

    def test_the_beagle_between_two_beagles_is_a_question(self):
        _, turns, asker = talk("there is a beagle",
                               "there is another beagle",
                               "does the beagle bark")
        self.assertIsNone(who(turns[2]))
        self.assertTrue(turns[2].resolution.ambiguous)
        self.assertNotIn("does a beagle bark", asker.asked)

    def test_the_other_one(self):
        _, turns, _ = talk("there is a beagle", "there is another beagle",
                           "the first beagle is black",
                           "is the other one black")
        self.assertEqual(who(turns[3]), "r2")

    def test_the_cat_said_first_introduces_a_cat(self):
        session, turns, _ = talk("the cat is black")
        self.assertTrue(turns[0].resolution.introduced)
        self.assertTrue(session.discourse.referents[0].accommodated)

    def test_it_with_nothing_before_is_refused(self):
        _, turns, asker = talk("can it swim")
        self.assertIsNone(who(turns[0]))
        self.assertEqual(asker.asked, [])

    def test_there_is_no_third(self):
        _, turns, _ = talk("there is a beagle", "there is another beagle",
                           "is the third one fast")
        self.assertIsNone(who(turns[2]))

    def test_what_is_it(self):
        _, turns, _ = talk("there is a beagle", "it can't swim",
                           "what is it")
        self.assertIn("a beagle", turns[2].answer["text"])
        self.assertIn("it can't swim", turns[2].answer["text"])


class YouAndNamesTests(unittest.TestCase):

    def test_my_name_is(self):
        """Reported from the page: it went to v688 as a claim about a kind
        and came back `is a my name Adrian`."""
        _, turns, asker = talk("my name is Adrian", "what is my name")
        self.assertIn("Adrian", turns[1].answer["text"])
        self.assertEqual(asker.asked, [])

    def test_your_name_is_not_mine(self):
        """Reported from the page: `what is your name` answered `your name
        is Adrian`, because `your` was read as `my`."""
        _, turns, _ = talk("my name is Adrian", "what is your name")
        self.assertEqual(turns[1].resolution.referent.id, "program")
        self.assertNotIn("Adrian", turns[1].answer["text"])

    def test_this_program_can_be_named_apart_from_you(self):
        _, turns, _ = talk("your name is Vee", "what is your name",
                           "what is my name")
        self.assertIn("Vee", turns[1].answer["text"])
        self.assertNotIn("Vee", turns[2].answer["text"])

    def test_it_never_means_this_program(self):
        _, turns, _ = talk("there is a beagle", "what is your name",
                           "can it swim")
        self.assertEqual(who(turns[2]), "r1")

    def test_who_am_i(self):
        _, turns, _ = talk("I am Adrian", "who am i")
        self.assertIn("Adrian", turns[1].answer["text"])

    def test_what_you_say_of_yourself_is_about_a_person(self):
        _, turns, _ = talk("i can't swim", "can i swim")
        self.assertEqual(turns[1].asked, "can a person swim")
        self.assertEqual((turns[1].answer["outcome"],
                          turns[1].answer["source"]), ("denied", "told"))

    def test_it_never_means_you(self):
        _, turns, _ = talk("i have a beagle", "i can swim", "can it swim")
        self.assertEqual(who(turns[2]), "r1")

    def test_a_name_is_told_not_an_identity(self):
        _, turns, _ = talk("there is a beagle", "its name is Rex",
                           "there is another beagle", "its name is Rex",
                           "does Rex bark")
        self.assertIsNone(who(turns[4]))
        self.assertTrue(turns[4].resolution.ambiguous)

    def test_my_beagle_is_the_one_you_have(self):
        _, turns, _ = talk("there is a beagle", "i have another beagle",
                           "does my beagle bark")
        self.assertEqual(who(turns[2]), "r2")

    def test_a_capitalised_name_said_to_be_a_kind_introduces_one(self):
        _, turns, _ = talk("Rex is a beagle", "can Rex swim")
        self.assertEqual(turns[0].act, "introduce")
        self.assertEqual(who(turns[1]), "r1")
        self.assertEqual(turns[1].asked, "can a beagle swim")

    def test_the_dogs_name(self):
        _, turns, _ = talk("there is a dog", "the dog's name is Rex",
                           "what is its name")
        self.assertIn("Rex", turns[2].answer["text"])

    def test_you_can_be_narrowed_too(self):
        session, _, _ = talk("i am a doctor")
        self.assertEqual(session.memory.parent["you"], "doctor.n.01")


class TeachingTests(unittest.TestCase):
    """Taxonomy and norms, into episodic memory only."""

    def test_a_new_kind_joins_the_taxonomy(self):
        session, turns, asker = talk(
            "a wemble is a kind of animal", "can a wemble breathe",
            "there is a wemble", "can it breathe",
            outcomes={"can an animal breathe": "verified"})
        self.assertEqual(turns[0].act, "teach")
        self.assertEqual(session.memory.edges["wemble"], ["animal.n.01"])
        self.assertEqual(turns[1].answer["outcome"], "verified")
        self.assertEqual(who(turns[2]), "r1")
        self.assertEqual(turns[3].answer["outcome"], "verified")
        # v688 is asked about animals, never about a word it does not have.
        self.assertTrue(asker.asked)
        self.assertFalse([one for one in asker.asked if "wemble" in one])

    def test_a_taught_kind_answers_is_a(self):
        _, turns, _ = talk("a wemble is a kind of animal",
                           "is a wemble an animal")
        self.assertEqual(turns[1].answer["outcome"], "verified")

    def test_a_norm_about_a_known_kind_reaches_its_individuals_by_r3(self):
        _, turns, _ = talk("beagles can't swim", "there is a beagle",
                           "can it swim",
                           outcomes={"can a beagle swim": "verified"})
        self.assertEqual(turns[0].act, "teach")
        self.assertEqual((turns[2].answer["outcome"],
                          turns[2].answer["source"]), ("denied", "taught"))
        self.assertIn("R3", rules_at(turns[2], 1))

    def test_a_norm_answers_the_kind_itself(self):
        _, turns, _ = talk("beagles can't swim", "can a beagle swim",
                           outcomes={"can a beagle swim": "verified"})
        self.assertEqual((turns[1].answer["outcome"],
                          turns[1].answer["source"]), ("denied", "taught"))

    def test_a_norm_on_a_taught_kind(self):
        _, turns, asker = talk("wembles can fly", "there is a wemble",
                               "can it fly")
        self.assertEqual((turns[2].answer["outcome"],
                          turns[2].answer["source"]), ("verified", "taught"))
        self.assertEqual(asker.asked, [])

    def test_a_taught_kind_asks_the_nearest_kind_the_store_has(self):
        """Reported from the page: `can a wemble breathe` was UNKNOWN,
        because the walk into animal found nothing plain and v688 was never
        asked."""
        _, turns, asker = talk("a wemble is a kind of animal",
                               "can a wemble swim",
                               outcomes={"can an animal swim": "verified"})
        self.assertEqual((turns[1].answer["outcome"],
                          turns[1].answer["source"]), ("verified", "kind"))
        self.assertEqual(asker.asked, ["can an animal swim"])

    def test_an_individual_of_a_taught_kind_asks_it_too(self):
        _, turns, _ = talk("a wemble is a kind of animal",
                           "there is a wemble", "can it swim",
                           outcomes={"can an animal swim": "verified"})
        self.assertEqual((turns[2].answer["outcome"],
                          turns[2].answer["source"]), ("verified", "kind"))

    def test_a_store_row_reached_through_a_taught_kind_is_judged(self):
        """Reported from the page: a wemble could fly before anyone said
        so, on `animal capable_of fly` -- a crawled row about bats, trusted
        raw because the walk came through a taught kind."""
        _, turns, _ = talk("a wemble is a kind of animal",
                           "there is a wemble", "can it breathe",
                           outcomes={"can an animal breathe": "denied"})
        self.assertEqual((turns[2].answer["outcome"],
                          turns[2].answer["source"]), ("denied", "kind"))

    def test_the_store_is_never_written(self):
        before = STORE["reasoner"].fact_count("beagle.n.01")
        talk("beagles can't swim", "a wemble is a kind of animal")
        self.assertEqual(STORE["reasoner"].fact_count("beagle.n.01"), before)


class CompoundTests(unittest.TestCase):
    """Several claims in one statement, split by the dependency parse.

    Reported from the page twice. `testicles shrink in cold temperatures and
    expand in warm ones` went to v688 as a question; with `, and they expand`
    it was stored as one claim, `shrink in cold temperatures and they expand
    in warm ones`, which no question could ever match.
    """

    @classmethod
    def setUpClass(cls):
        cls.lexicon = TinyAsker()

    def parts(self, text):
        found = reading.read(text, self.lexicon)
        return [found] + list(found.more)

    def test_they_is_the_kind_before_it(self):
        parts = self.parts("testicles shrink in cold temperatures, and they "
                           "expand in warm ones")
        self.assertEqual(
            [(one.act, one.mention.kind, one.rest) for one in parts],
            [("teach", "testicle", ["shrink", "in", "cold", "temperatures"]),
             ("teach", "testicle", ["expand", "in", "warm", "temperatures"])])

    def test_a_shared_subject_keeps_its_auxiliary(self):
        self.assertEqual(
            [(one.aux, one.rest, one.holds)
             for one in self.parts("beagles can swim and bark")],
            [("can", ["swim"], True), ("can", ["bark"], True)])

    def test_but_they_can(self):
        self.assertEqual(
            [(one.aux, one.rest, one.holds)
             for one in self.parts("beagles can't swim but they can run")],
            [("can", ["swim"], False), ("can", ["run"], True)])

    def test_and_between_nouns_is_one_claim(self):
        self.assertEqual([one.rest for one in
                          self.parts("dogs eat meat and bones")],
                         [["eat", "meat", "and", "bones"]])

    def test_an_individual_and_then_it(self):
        self.assertEqual(
            [(one.act, one.mention.form) for one in
             self.parts("there is a beagle and it can't swim")],
            [("introduce", "indefinite"), ("tell", "pronoun")])

    def test_several_claims_the_parse_cannot_split_are_refused(self):
        self.assertEqual(reading.read("dogs bark and cats purr",
                                      self.lexicon).act, "compound")

    def test_without_a_parse_the_parser_still_finds_the_kind(self):
        found = reading.read("dogs bark", FakeLexicon())
        self.assertEqual((found.act, found.mention.kind, found.rest),
                         ("teach", "dog", ["bark"]))

    def test_what_happened_is_not_a_claim_about_a_kind(self):
        self.assertNotEqual(reading.read("a dog chased me",
                                         self.lexicon).act, "teach")

    def test_both_are_taught_and_the_opposite_is_denied(self):
        session, turns, _ = talk(
            "testicles shrink in cold temperatures, and they expand in warm "
            "ones",
            "do testicles expand in warm temperatures",
            "do testicles expand in cold temperatures",
            "do testicles shrink")
        self.assertEqual(sorted(fact.object for fact in
                                session.memory.facts["testis.n.01"]),
                         ["expand in warm temperatures",
                          "shrink in cold temperatures"])
        self.assertEqual((turns[1].answer["outcome"],
                          turns[1].answer["source"]), ("verified", "taught"))
        self.assertEqual((turns[2].answer["outcome"],
                          turns[2].answer["source"]), ("denied", "taught"))
        self.assertIn("opposite", turns[2].answer["text"])
        self.assertNotEqual(turns[3].answer["outcome"], "verified")
        self.assertIn("shrink in cold temperatures", turns[3].answer["text"])

    def test_an_individual_told_in_the_same_breath(self):
        _, turns, _ = talk("there is a beagle and it can't swim",
                           "can it swim",
                           outcomes={"can a beagle swim": "verified"})
        self.assertEqual((turns[1].answer["outcome"],
                          turns[1].answer["source"]), ("denied", "told"))


class DefinitionTests(unittest.TestCase):
    """WordNet glosses read into definitions memory, as v688 retrieves them."""

    @classmethod
    def setUpClass(cls):
        cls.asker = TinyAsker()
        cls.reader = GlossReader(cls.asker)

    def facts(self, found) -> set:
        return {(fact.relation, fact.object) for fact in found.facts}

    def test_asides_and_examples_are_not_the_definition(self):
        self.assertEqual(
            pieces("a member of the genus Canis (probably descended from the "
                   "wolf) that barks; occurs in many breeds: collies"),
            ["a member of the genus Canis that barks",
             "occurs in many breeds"])

    def test_a_genus_and_its_adjectives(self):
        found = self.reader.read("kitten.n.01", "young domestic cat")
        self.assertEqual((found.genus, found.agrees), ("cat", True))
        self.assertEqual(self.facts(found), {("has_property", "young"),
                                             ("has_property", "domestic")})

    def test_through_one_of_to_the_glands_and_what_they_do(self):
        found = self.reader.read("testis.n.01", DEFINITIONS["testis.n.01"])
        self.assertEqual((found.genus, found.agrees), ("gland", True))
        self.assertLessEqual({("capable_of", "produce spermatozoa"),
                              ("capable_of", "secrete androgens"),
                              ("has_property", "male")}, self.facts(found))

    def test_alternatives_are_not_properties(self):
        found = self.reader.read("cat.n.01",
                                 "fruit with red or yellow or green skin")
        self.assertIn(("has_a", "skin"), self.facts(found))
        self.assertFalse([fact for fact in found.facts
                          if "red" in fact.object])

    def test_the_fragments_after_semicolons(self):
        found = self.reader.read(
            "dog.n.01", "small and light boat; pointed at both ends; "
                        "propelled with a paddle; used to stir or serve food")
        facts = self.facts(found)
        self.assertIn(("has_property", "pointed at both ends"), facts)
        self.assertIn(("receives_action", "propelled with a paddle"), facts)
        self.assertIn(("used_for", "stir food"), facts)

    def test_what_a_split_leaves_dangling_is_trimmed(self):
        found = self.reader.read("dog.n.01",
                                 "an iron bucket used for hoisting in wells "
                                 "or mining")
        self.assertFalse([fact for fact in found.facts
                          if fact.object.split()[-1] in ("or", "and")])

    def test_verbs_joined_by_or_share_their_object(self):
        found = self.reader.read("dog.n.01",
                                 "a worker who produces or sells petroleum")
        self.assertIn(("capable_of", "produce petroleum"), self.facts(found))

    def test_a_list_is_not_something_it_does(self):
        found = self.reader.read("dog.n.01",
                                 "mosquitoes; fungus gnats; crane flies")
        self.assertFalse([fact for fact in found.facts
                          if fact.relation == "capable_of"])

    def test_having_is_had_not_done(self):
        found = self.reader.read(
            "cat.n.01", "any of several plants of the genus Arctotis having "
                        "daisylike flowers")
        self.assertIn(("has_a", "daisylike flowers"), self.facts(found))
        self.assertFalse([fact for fact in found.facts
                          if fact.object.startswith("have")])

    def test_two_clauses_run_together_are_not_one_fact(self):
        found = self.reader.read("dog.n.01", "a document listing the "
                                             "alternatives that is used in "
                                             "voting")
        self.assertFalse([fact for fact in found.facts
                          if " is " in f" {fact.object} "])

    def test_a_person_is_not_a_kind(self):
        found = self.reader.read(
            "disraeli.n.01", "British statesman who as Prime Minister bought "
            "controlling interest in the Suez Canal (1804-1881)")
        self.assertEqual(found.facts, [])

    def test_a_name_is_not_a_property(self):
        found = self.reader.read("dog.n.01", "United States photographer")
        self.assertFalse([fact for fact in found.facts
                          if "states" in fact.object])

    def test_a_fact_as_the_question_the_teacher_is_asked(self):
        self.assertEqual(question_for("hammer", "used_for", "deliver force",
                                      "used to"),
                         "is a hammer used to deliver force")
        self.assertEqual(question_for("oak", "has_a", "acorns"),
                         "does an oak have acorns")

    def test_a_retrieved_definition_is_learned_and_answers_from_memory(self):
        run = {"summary": {"outcome": "retrieved", "trust": "",
                           "lines": ["DEFINED — what is a kitten"]},
               "cycles": [{"answers": [{"verdict": "DEFINED",
                                        "concept": "kitten.n.01",
                                        "question": "what is a kitten"}]}]}
        asker = TinyAsker({"what is a kitten": run})
        session = Session(asker, definitions=DefinitionMemory())
        first = session.say("what is a kitten")
        self.assertEqual((first.act, first.answer["source"]),
                         ("define", "definition"))
        self.assertIn("young domestic cat", first.answer["text"])
        self.assertEqual([one["concept"] for one in first.learned],
                         ["kitten.n.01"])
        session.say("what is a kitten")
        self.assertEqual(asker.asked, ["what is a kitten"])
        young = session.say("is a kitten young")
        self.assertEqual((young.answer["outcome"], young.answer["source"]),
                         ("verified", "definition"))

    def test_a_disputed_fact_is_kept_and_never_read(self):
        memory = DefinitionMemory()
        memory.keep(self.reader.read("kitten.n.01", "young domestic cat"),
                    "retrieved",
                    {("has_property", "domestic"): ("disputed", 0.995)})
        self.assertEqual({fact.object for fact in
                          memory.facts("kitten.n.01")}, {"young"})
        self.assertEqual(len(memory.entry("kitten.n.01")["facts"]), 2)

    def test_told_against_its_definition_is_said_out_loud(self):
        memory = DefinitionMemory()
        memory.keep(self.reader.read("kitten.n.01", "young domestic cat"),
                    "offline")
        session = Session(TinyAsker(), definitions=memory)
        session.say("there is a kitten")
        told = session.say("it is old")
        self.assertIn("against the definition of kitten",
                      told.answer["text"])

    def test_an_offline_reading_survives_the_trip_to_disk(self):
        from research.v689.learn_definitions import reading_of

        found = self.reader.read("kitten.n.01", "young domestic cat")
        self.assertEqual(reading_of(found.as_dict()).as_dict(),
                         found.as_dict())


class ArticleTests(unittest.TestCase):
    """Wikipedia lead paragraphs: only what is said of the kind itself."""

    @classmethod
    def setUpClass(cls):
        from research.v689.articles import ArticleReader

        cls.reader = ArticleReader(TinyAsker())

    def test_only_sentences_about_the_kind_in_the_present(self):
        found = self.reader.read(
            "beagle.n.01", "Beagle",
            "The beagle is a small hound. Foxes are clever. It can secrete "
            "oil. Some beagles can produce milk. It was bred for hunting.")
        facts = {(fact.relation, fact.object) for fact in found.facts}
        self.assertEqual(found.genus, "hound")
        self.assertIn(("has_property", "small"), facts)
        self.assertIn(("capable_of", "secrete oil"), facts)
        self.assertFalse([fact for fact in found.facts
                          if fact.object.split()[0] in ("clever", "produce",
                                                        "bred")])

    def test_a_noun_is_not_a_property(self):
        from research.v689.definitions import Defined

        self.assertFalse(self.reader.properly(
            Defined("has_property", "gland", "sentence", "")))
        self.assertFalse(self.reader.properly(
            Defined("has_property", "often much larger than cats",
                    "sentence", "")))
        self.assertTrue(self.reader.properly(
            Defined("has_property", "small", "sentence", "")))


class WiktionaryTests(unittest.TestCase):
    """Wiktionary senses: kept only where the taxonomy picks one synset."""

    @classmethod
    def setUpClass(cls):
        from research.v689.definitions import GlossReader

        cls.reader = GlossReader(TinyAsker())

    def test_a_sense_goes_to_the_one_synset_under_its_broader_kind(self):
        from research.v689.learn_wiktionary import align, cleaned

        found = align(self.reader, "kitten", cleaned("A young cat."),
                      ["beagle.n.01", "kitten.n.01"])
        self.assertEqual(found["status"], "aligned")
        self.assertEqual(found["reading"]["concept"], "kitten.n.01")
        self.assertIn(("has_property", "young"),
                      {(fact["relation"], fact["object"])
                       for fact in found["reading"]["facts"]})

    def test_two_synsets_under_it_or_none_and_the_sense_is_left(self):
        from research.v689.learn_wiktionary import align

        self.assertEqual(align(self.reader, "pet", "an animal kept at home",
                               ["dog.n.01", "cat.n.01"])["status"],
                         "ambiguous")
        self.assertEqual(align(self.reader, "kitten",
                               "a coquettish young woman",
                               ["kitten.n.01"])["status"], "no sense agrees")

    def test_uses_of_the_word_and_pointers_are_left(self):
        from research.v689.learn_wiktionary import cleaned, usable

        self.assertEqual(cleaned("A dog (noun sense 1)."), "a dog")
        self.assertEqual(cleaned("NATO member."), "NATO member")
        self.assertIsNone(usable({"word": "dog", "tags": [],
                                  "gloss": "A domestic animal."}))
        self.assertTrue(usable({"word": "dog", "tags": ["figuratively"],
                                "gloss": "A contemptible man."}))
        self.assertTrue(usable({"word": "dog", "tags": [],
                                "gloss": "Alternative form of dogge."}))
        self.assertTrue(usable({"word": "Dog", "tags": [],
                                "gloss": "A constellation."}))

    def test_what_the_wordnet_gloss_gave_is_not_kept_twice(self):
        from research.v689.definitions import Defined, Reading
        from research.v689.learn_wiktionary import merge

        young = Defined("has_property", "young", "adjective", "")
        small = Defined("has_property", "small", "adjective", "")
        merged = merge(
            [("kitten", Reading("kitten.n.01", "a young cat", "cat", True,
                                [young], [])),
             ("kitty", Reading("kitten.n.01", "a small cat", "cat", True,
                               [small, young], []))],
            {("kitten.n.01", "has_property", "young")})
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].gloss, "a young cat | a small cat")
        self.assertEqual([(fact.relation, fact.object)
                          for fact in merged[0].facts],
                         [("has_property", "small")])

    def test_two_senses_of_a_word_on_one_synset_are_both_left(self):
        from research.v689.definitions import Defined, Reading
        from research.v689.learn_wiktionary import merge

        fears = Defined("capable_of", "fear men", "clause", "")
        prejudiced = Defined("has_property", "prejudiced", "participle", "")
        self.assertEqual(merge(
            [("homophobe", Reading("person.n.01", "a person who fears men",
                                   "person", True, [fears], [])),
             ("homophobe", Reading("person.n.01", "a prejudiced person",
                                   "person", True, [prejudiced], []))],
            set()), [])

    def test_history_species_lists_and_naming_are_not_properties(self):
        from research.v689.definitions import Defined
        from research.v689.learn_wiktionary import kept

        def fact(relation, obj):
            return kept(self.reader, Defined(relation, obj, "clause", ""))

        self.assertFalse(fact("receives_action",
                              "proposed by clark kerr in the 1960s"))
        self.assertFalse(fact("has_a", "p. dominica"))
        self.assertFalse(fact("receives_action", "abbreviated as sebs"))
        self.assertFalse(fact("has_property", "able"))
        self.assertTrue(fact("has_property", "young"))
        # A lead's filing verbs are what a defined thing does.
        self.assertTrue(fact("used_for", "treat mental illness"))


class CarriedTests(unittest.TestCase):
    """E2: an action done while carried belongs to what carries it."""

    PIGS = {"can a pig fly": "denied", "does a pig fly": "denied"}

    def test_the_airplane_was_doing_the_flying(self):
        """Reported from the page: `it was in an airplane` left the pig's
        flying standing as an exception."""
        session, turns, _ = talk("there was a pig", "he was flying",
                                 "it was in an airplane", "can the pig fly",
                                 outcomes=self.PIGS)
        self.assertIn("E2", turns[2].answer["text"])
        self.assertEqual((turns[3].answer["outcome"],
                          turns[3].answer["source"]), ("denied", "kind"))
        self.assertIn("carried", [fact.relation for fact in
                                  session.memory.facts["r1"]])

    def test_either_order(self):
        _, turns, _ = talk("there was a pig", "it was in an airplane",
                           "it was flying", "can it fly", outcomes=self.PIGS)
        self.assertEqual(turns[3].answer["source"], "kind")

    def test_a_carrier_that_does_not_do_it_explains_nothing(self):
        _, turns, _ = talk("there was a pig", "it was on a cat",
                           "it was flying", "can it fly", outcomes=self.PIGS)
        self.assertEqual((turns[3].answer["outcome"],
                          turns[3].answer["source"]), ("verified", "told"))

    def test_a_claim_is_not_explained_away(self):
        _, turns, _ = talk("there was a pig", "it was in an airplane",
                           "it can fly", "can it fly", outcomes=self.PIGS)
        self.assertEqual((turns[3].answer["outcome"],
                          turns[3].answer["source"]), ("verified", "told"))

    def test_in_an_airplane_puts_one_airplane_on_the_table(self):
        session, turns, _ = talk("there was a pig", "he was flying",
                                 "it was in an airplane", outcomes=self.PIGS)
        self.assertTrue(turns[2].binding.introduced)
        self.assertEqual(session.memory.withdrawn[0].carrier_id, "r2")

    def test_what_the_carrier_was_told_comes_before_its_kind(self):
        """`the airplane couldn't fly`: then the pig was flying by itself,
        and E2 gives it back."""
        session, turns, _ = talk("there was a pig", "he was flying",
                                 "it was in an airplane",
                                 "the airplane couldn't fly",
                                 "can the pig fly", outcomes=self.PIGS)
        self.assertIn("E2 undone", turns[3].answer["text"])
        self.assertFalse(session.memory.withdrawn)
        self.assertEqual((turns[4].answer["outcome"],
                          turns[4].answer["source"]), ("verified", "told"))

    def test_the_plane_that_came_up_before(self):
        _, turns, _ = talk("there was a plane", "there was a pig",
                           "it was flying", "it was in the plane",
                           "can the pig fly", outcomes=self.PIGS)
        self.assertEqual(turns[3].binding.referent.id, "r1")
        self.assertEqual((turns[4].answer["outcome"],
                          turns[4].answer["source"]), ("denied", "kind"))


class ObjectTests(unittest.TestCase):
    """The individual after the verb, resolved as a subject is."""

    lexicon = FakeLexicon()

    def test_an_object_is_read_after_a_verb(self):
        found = reading.read("the dog chased the cat", self.lexicon)
        self.assertEqual((found.obj.form, found.obj.kind, found.obj_at),
                         ("definite", "cat", 1))

    def test_an_indefinite_object_stays_a_kind(self):
        self.assertIsNone(reading.read("it chased a cat", self.lexicon).obj)
        self.assertIsNone(reading.read("does it have a tail",
                                       self.lexicon).obj)

    def test_after_a_copula_only_what_carries_it_is_an_object(self):
        self.assertIsNone(reading.read("is it a dog", self.lexicon).obj)
        found = reading.read("it was on the cat", self.lexicon)
        self.assertEqual((found.obj.form, found.obj_at), ("definite", 1))

    def test_an_object_is_never_the_subject(self):
        _, turns, _ = talk("there is a dog", "there is a cat",
                           "the dog chased it")
        self.assertEqual((who(turns[2]), turns[2].binding.referent.id),
                         ("r1", "r2"))

    def test_an_object_with_no_one_else_to_be_stores_nothing(self):
        session, turns, _ = talk("there is a dog", "the dog chased it")
        self.assertEqual(turns[1].answer["outcome"], "which")
        self.assertEqual(session.memory.facts["r1"], [])

    def test_it_after_an_object_is_still_the_subject(self):
        _, turns, _ = talk("there was a pig", "it was in an airplane",
                           "can it fly")
        self.assertEqual(who(turns[2]), "r1")

    def test_the_kind_is_stored_and_which_one_beside_it(self):
        session, _, asker = talk("there is a dog", "there is a cat",
                                 "the dog chased it")
        self.assertIn("does a dog chase a cat", asker.asked)
        self.assertEqual(
            session.memory.bound[("r1", "capable_of", "chase a cat")],
            {"r2"})

    def test_a_fact_about_one_cat_is_not_an_answer_about_another(self):
        _, turns, _ = talk("there is a dog", "there is a cat",
                           "the dog chased it", "there is another cat",
                           "did the dog chase the first cat",
                           "did the dog chase the second cat")
        self.assertEqual((turns[4].answer["outcome"],
                          turns[4].answer["source"]), ("verified", "told"))
        self.assertEqual(turns[5].answer["outcome"], "unknown")
        self.assertIn("the first cat", turns[5].answer["text"])

    def test_told_of_both_cats_it_answers_for_both(self):
        _, turns, _ = talk("there is a dog", "there is a cat",
                           "there is another cat",
                           "the dog chased the first cat",
                           "the dog chased the second cat",
                           "did the dog chase the first cat")
        self.assertEqual(turns[5].answer["outcome"], "verified")

    def test_the_plane_finds_an_airplane_through_its_sense(self):
        _, turns, _ = talk("there was an airplane", "is the plane fast")
        self.assertEqual(who(turns[1]), "r1")


class LongTermTests(unittest.TestCase):
    """What was taught, and each conversation, after the server restarts."""

    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / "memory.sqlite"
        self.archives: list = []

    def tearDown(self) -> None:
        for archive in self.archives:
            archive.close()
        self.folder.cleanup()

    def restart(self, outcomes: dict | None = None) -> Keeper:
        """A new keeper over the same file: what a restart leaves behind."""
        archive = Archive(self.path)
        self.archives.append(archive)
        return Keeper(TinyAsker(outcomes), archive)

    def test_what_was_taught_answers_in_a_later_conversation(self):
        self.restart().say("a", "wembles can fly")
        keeper = self.restart()
        keeper.say("b", "there is a wemble")
        turn = keeper.say("b", "can it fly")
        self.assertEqual((turn["answer"]["outcome"],
                          turn["answer"]["source"]), ("verified", "learned"))
        self.assertIn("earlier conversation", turn["answer"]["text"])

    def test_a_taught_edge_is_kept(self):
        self.restart().say("a", "a wemble is a kind of animal")
        turn = self.restart().say("b", "is a wemble an animal")
        self.assertEqual(turn["answer"]["outcome"], "verified")

    def test_a_conversation_carries_on_after_a_restart(self):
        keeper = self.restart()
        for line in ("there is a beagle", "its name is Rex", "it can't swim"):
            keeper.say("a", line)
        keeper = self.restart({"can a beagle swim": "verified"})
        by_name = keeper.say("a", "can Rex swim")
        self.assertEqual((by_name["answer"]["outcome"],
                          by_name["answer"]["source"]), ("denied", "told"))
        again = keeper.say("a", "can it swim")
        self.assertEqual(again["resolution"]["referent"], "r1")
        self.assertEqual([turn["said"] for turn in
                          keeper.history("a")["turns"]][:3],
                         ["there is a beagle", "its name is Rex",
                          "it can't swim"])

    def test_e2_is_recomputed_after_a_restart(self):
        pigs = {"can a pig fly": "denied", "does a pig fly": "denied"}
        keeper = self.restart(pigs)
        for line in ("there was a pig", "he was flying",
                     "it was in an airplane"):
            keeper.say("a", line)
        keeper = self.restart(pigs)
        undone = keeper.say("a", "the airplane couldn't fly")
        self.assertIn("E2 undone", undone["answer"]["text"])
        flying = keeper.say("a", "can the pig fly")
        self.assertEqual((flying["answer"]["outcome"],
                          flying["answer"]["source"]), ("verified", "told"))

    def test_start_over_forgets_the_conversation_not_the_knowledge(self):
        keeper = self.restart()
        keeper.say("a", "wembles can fly")
        keeper.say("a", "there is a wemble")
        keeper.forget("a")
        keeper = self.restart()
        self.assertEqual(keeper.history("a")["turns"], [])
        keeper.say("a", "there is a wemble")
        self.assertEqual(keeper.say("a", "can it fly")["answer"]["outcome"],
                         "verified")

    def test_unlearn_forgets_the_knowledge(self):
        keeper = self.restart()
        keeper.say("a", "wembles can fly")
        keeper.unlearn()
        keeper = self.restart()
        keeper.say("b", "there is a wemble")
        self.assertNotEqual(
            keeper.say("b", "can it fly")["answer"]["outcome"], "verified")

    def test_an_example_keeps_what_it_teaches_to_itself(self):
        outcomes = {"can a beagle swim": "verified"}
        keeper = self.restart(outcomes)
        keeper.say("x", "beagles can't swim", example=True)
        keeper.say("y", "there is a beagle")
        self.assertEqual(keeper.say("y", "can it swim")["answer"]["source"],
                         "kind")
        keeper = self.restart(outcomes)
        keeper.say("x", "there is a beagle")
        own = keeper.say("x", "can it swim")
        self.assertEqual((own["answer"]["outcome"], own["answer"]["source"]),
                         ("denied", "taught"))
        keeper.say("z", "there is a beagle")
        self.assertEqual(keeper.say("z", "can it swim")["answer"]["source"],
                         "kind")


if __name__ == "__main__":
    unittest.main()
