"""Time and events in v689: episodes, occurrences, their order, what holds.

Run: python -m unittest research.v689.test_time -v

Like `test_v689`, everything after reading runs on v687 itself -- the real
`Reasoner` and `Parser` -- over a small store built here, which has what a
story needs and `test_v689`'s store does not: verbs, with WordNet's troponymy
between them, and the things that get broken, killed and put in drawers.
VerbNet is read from `data/verbnet3.3`; T4's tests are skipped without it.
"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from research.v687.language import Parser
from research.v687.reason import Reasoner
from research.v689 import change, reader, reading, tense
from research.v689.asker import Asker
from research.v689.session import Session
from research.v689.test_v689 import SCHEMA

#: (id, lemmas, parent). A verb's lemmas are all of its synset's, because
#: `did the dog move` finds chasing through `move`, a lemma of travel.v.01.
CONCEPTS = (
    ("entity.n.01", ("entity",), None),
    ("organism.n.01", ("organism",), "entity.n.01"),
    ("animal.n.01", ("animal",), "organism.n.01"),
    ("person.n.01", ("person",), "organism.n.01"),
    ("dog.n.01", ("dog",), "animal.n.01"),
    ("cat.n.01", ("cat",), "animal.n.01"),
    ("hog.n.03", ("hog", "pig"), "animal.n.01"),
    ("mouse.n.01", ("mouse",), "animal.n.01"),
    ("artifact.n.01", ("artifact",), "entity.n.01"),
    ("vase.n.01", ("vase",), "artifact.n.01"),
    ("key.n.01", ("key",), "artifact.n.01"),
    ("drawer.n.01", ("drawer",), "artifact.n.01"),
    ("table.n.02", ("table",), "artifact.n.01"),
    ("door.n.01", ("door",), "artifact.n.01"),
    ("cup.n.01", ("cup",), "artifact.n.01"),
    ("field.n.01", ("field",), "entity.n.01"),
    ("airplane.n.01", ("airplane", "plane"), "artifact.n.01"),
    ("travel.v.01", ("travel", "go", "move"), None),
    ("pursue.v.02", ("pursue", "follow"), "travel.v.01"),
    ("chase.v.01", ("chase",), "pursue.v.02"),
    ("run.v.01", ("run",), "travel.v.01"),
    ("fly.v.01", ("fly",), "travel.v.01"),
    ("jump.v.01", ("jump",), "travel.v.01"),
    ("bark.v.04", ("bark",), None),
    ("sleep.v.01", ("sleep",), None),
    ("eat.v.01", ("eat",), None),
    ("swim.v.01", ("swim",), "travel.v.01"),
    ("change.v.02", ("change",), None),
    ("change integrity.v.01", ("change integrity",), "change.v.02"),
    ("break.v.02", ("break",), "change integrity.v.01"),
    ("kill.v.01", ("kill",), None),
    ("put.v.01", ("put",), None),
    ("begin.v.03", ("begin", "start"), None),
    ("stop.v.01", ("stop",), None),
    ("close.v.01", ("close",), "change.v.02"),
    ("open.v.01", ("open",), "change.v.02"))

FACTS = (("dog.n.01", "capable_of", "bark", "ascentpp", 0.7, 1),
         ("airplane.n.01", "capable_of", "fly", "ascentpp", 0.6, 1))

STORE: dict = {}


def setUpModule() -> None:                      # noqa: N802
    folder = tempfile.TemporaryDirectory()
    path = Path(folder.name) / "story.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    connection.executemany(
        "INSERT INTO concepts VALUES (?, ?, ?, 1, ?, 1)",
        [(concept, lemmas[0], concept.split(".")[-2], f"a {lemmas[0]}")
         for concept, lemmas, _ in CONCEPTS])
    connection.executemany(
        "INSERT INTO taxonomy VALUES (?, ?)",
        [(concept, parent) for concept, _, parent in CONCEPTS if parent])
    connection.executemany(
        "INSERT INTO lemmas VALUES (?, ?, 1)",
        [(lemma, concept) for concept, lemmas, _ in CONCEPTS
         for lemma in lemmas])
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
        try:
            STORE["folder"].cleanup()
        except OSError:
            pass


class StoryAsker(Asker):
    """v687 over the story store; v688 answers what the test says."""

    def __init__(self, outcomes: dict | None = None) -> None:
        super().__init__(STORE["reasoner"], STORE["parser"])
        self.outcomes = outcomes or {}
        self.asked: list[str] = []

    def run(self, question: str) -> dict:
        self.asked.append(question)
        outcome = self.outcomes.get(question, "unknown")
        return {"summary": {"outcome": outcome, "trust": "",
                            "lines": [f"{outcome} — {question}"]}}


KINDS = {"does a dog bark": "verified", "can a pig fly": "denied",
         "does a pig fly": "denied"}


def talk(*lines: str, outcomes: dict | None = None):
    asker = StoryAsker(KINDS if outcomes is None else outcomes)
    session = Session(asker)
    return session, [session.say(line) for line in lines], asker


def answer(turn) -> tuple[str, str]:
    return turn.answer.get("outcome"), turn.answer.get("text", "")


# -- reading when ------------------------------------------------------------

@unittest.skipUnless(reader.enabled(), "needs llm/reader")
class TenseTests(unittest.TestCase):
    """What an utterance says about time, taken off before it is read
    (`reader.place`)."""

    def placed(self, text: str):
        return reader.place(text, StoryAsker(), frozenset())

    def taken(self, text: str):
        found = self.placed(text)
        return found.tokens, found.typed, found.when

    def test_a_frame_comes_off_either_end(self):
        tokens, _, when = self.taken("yesterday there was a dog")
        self.assertEqual((tokens, when.frame.key, when.frame.rank),
                         (["there", "was", "a", "dog"], "yesterday", -2.0))
        tokens, _, when = self.taken("it is empty now")
        self.assertEqual((tokens, when.frame.key), (["it", "is", "empty"],
                                                    "now"))
        self.assertEqual(self.taken("today the pig is in a field")[2]
                         .frame.label, "today")

    def test_the_longest_frame_is_taken(self):
        self.assertEqual(self.taken("the day before yesterday it rained")[2]
                         .frame.key, "the day before yesterday")

    def test_a_link_only_before_something_that_can_start_a_clause(self):
        for text, link in (("then it slept", "after"),
                           ("before that it had barked", "before"),
                           ("meanwhile the cat played", "during"),
                           ("finally it slept", "last")):
            self.assertEqual(self.taken(text)[2].link, link, text)
        self.assertEqual(self.taken("what happened then")[2].link, "")
        self.assertEqual(self.taken("the first beagle is black")[0][:2],
                         ["the", "first"])

    def test_again_is_one_more_of_the_same(self):
        tokens, _, when = self.taken("it barked once more")
        self.assertEqual((tokens, when.again), (["it", "barked"], True))

    def test_a_word_alone_is_not_a_time(self):
        self.assertEqual(self.taken("now what")[0], ["now", "what"])

    def test_an_anchor_needs_a_clause_either_side(self):
        found = self.placed("after the dog chased the cat, it slept")
        self.assertEqual((found.tokens, found.relation, found.anchor),
                         (["it", "slept"], "after", "the dog chased the cat"))
        found = self.placed("was the vase broken before the cat broke it")
        self.assertEqual((found.tokens, found.relation, found.anchor),
                         (["was", "the", "vase", "broken"], "before",
                          "the cat broke it"))
        self.assertEqual(self.placed("while the dog slept, the cat ate")
                         .relation, "during")
        for text in ("the dog slept after dinner",
                     "when did the dog chase the cat"):
            self.assertEqual(self.placed(text).relation, "", text)

    def test_reichenbach_from_the_auxiliary_and_the_form(self):
        for aux, rest, tags, expected in (
                (None, ["barked"], ["VBD"], ("past", "simple")),
                ("was", ["barking"], ["VBG"], ("past", "progressive")),
                ("had", ["eaten"], ["VBN"], ("past", "perfect")),
                ("will", ["swim"], ["VB"], ("future", "simple")),
                (None, ["barks"], ["VBZ"], ("present", "habit")),
                ("was", ["black"], ["JJ"], ("past", "state")),
                ("can", ["swim"], ["VB"], ("present", "ability"))):
            self.assertEqual(tense.tense_of(aux, rest, tags), expected, rest)
        self.assertFalse(tense.occurs("present", "habit"))
        self.assertFalse(tense.occurs("present", "ability"))
        self.assertTrue(tense.occurs("past", "progressive"))


class ReadingTimeTests(unittest.TestCase):
    """The shapes of question and statement time adds to the reader."""

    def read(self, text):
        return reading.read(text, StoryAsker())

    def test_time_is_kept_on_the_reading(self):
        found = self.read("yesterday there was a dog")
        self.assertEqual((found.act, found.when.frame.key),
                         ("introduce", "yesterday"))
        found = self.read("then it slept")
        self.assertEqual((found.act, found.when.link, found.when.main),
                         ("tell", "after", "it slept"))

    def test_an_anchor_that_is_not_a_clause_leaves_the_utterance_whole(self):
        found = self.read("what happened after the dog chased the cat")
        self.assertEqual((found.cells, found.when.relation, found.when.anchor),
                         ([("events", "story")], "after",
                          "the dog chased the cat"))
        self.assertEqual(self.read("the dog slept after dinner").when.anchor,
                         "")

    def test_the_questions_about_occurrences(self):
        for text, cell in (
                ("when did the dog chase the cat", ("time", "occurrence")),
                ("how many times did the dog bark", ("times", "occurrence")),
                ("what was the dog doing", ("verb", "occurrence")),
                ("what happened to the vase", ("events", "story")),
                ("what did the dog do", ("events", "story")),
                ("what did the cat do second", ("events", "story")),
                ("what did i tell you first", ("events", "told"))):
            self.assertIn(cell, self.read(text).cells, text)
        self.assertEqual(self.read("what did i tell you first").rest,
                         ["told", "first"])

    def test_had_is_an_auxiliary(self):
        found = self.read("it had eaten")
        self.assertEqual((found.act, found.aux, found.rest),
                         ("tell", "had", ["eaten"]))


# -- T1 and T2 ---------------------------------------------------------------

class OrderTests(unittest.TestCase):
    """T1: told in order, happened in order. T2: before, walked."""

    def test_narrative_progression(self):
        session, turns, _ = talk("there was a dog", "it barked",
                                 "then it slept",
                                 "did the dog bark before it slept",
                                 "did the dog sleep before it barked",
                                 "what happened first", "what happened last")
        story = session.timeline.story()
        self.assertEqual([one.verb for one in story], ["bark", "sleep"])
        self.assertEqual(answer(turns[3])[0], "verified")
        self.assertIn("T2", answer(turns[3])[1])
        self.assertEqual(answer(turns[4])[0], "denied")
        self.assertIn("it barked", answer(turns[5])[1])
        self.assertIn("it slept", answer(turns[6])[1])

    def test_a_past_perfect_is_before_the_time_the_story_is_at(self):
        session, turns, _ = talk("there was a dog", "it slept",
                                 "before that, it had barked",
                                 "did it bark before it slept",
                                 "what happened first")
        self.assertEqual([one.verb for one in session.timeline.story()],
                         ["bark", "sleep"])
        self.assertEqual(answer(turns[3])[0], "verified")
        self.assertIn("barked", answer(turns[4])[1])

    def test_an_anchor_nothing_was_told_of_is_told_first(self):
        session, turns, _ = talk("there is a dog", "there is a cat",
                                 "the dog barked after the cat ran",
                                 "did the cat run before the dog barked",
                                 "what happened first")
        self.assertEqual([one.verb for one in session.timeline.story()],
                         ["run", "bark"])
        self.assertEqual(answer(turns[3])[0], "verified")
        self.assertIn("the cat ran", answer(turns[4])[1])

    def test_after_an_anchor_at_the_start(self):
        _, turns, _ = talk("there was a dog", "there was a cat",
                           "after the dog chased the cat, it slept",
                           "what did the dog do after it chased the cat",
                           "what happened before the dog slept")
        self.assertIn("it slept", answer(turns[3])[1])
        self.assertIn("chased the cat", answer(turns[4])[1])

    def test_while_is_during(self):
        _, turns, _ = talk("there was a dog", "there was a cat",
                           "while the dog slept, the cat ate",
                           "did the cat eat while the dog slept")
        self.assertEqual(answer(turns[3])[0], "verified")

    def test_what_nothing_orders_is_not_told(self):
        _, turns, _ = talk("there was a dog", "there was a cat",
                           "the dog barked", "yesterday the cat ran",
                           "did the cat run before the dog barked")
        self.assertEqual(answer(turns[4])[0], "unknown")
        self.assertIn("absent, not false", answer(turns[4])[1])

    def test_days_are_in_order_without_being_told(self):
        _, turns, _ = talk("there is a dog", "there is a cat",
                           "yesterday the dog barked", "today the cat ran",
                           "did the dog bark before the cat ran")
        self.assertEqual(answer(turns[4])[0], "verified")
        self.assertIn("yesterday is before today", answer(turns[4])[1])


# -- episodes and T3 ---------------------------------------------------------

class EpisodeTests(unittest.TestCase):
    """Several episodes, and T3: a state belongs to the time it was told of."""

    def test_what_happened_on_a_day(self):
        session, turns, _ = talk("yesterday there was a dog", "it chased a cat",
                                 "today there is a cat", "it slept",
                                 "what happened yesterday",
                                 "what did the dog do yesterday")
        self.assertEqual([one.key for one in session.timeline.episodes],
                         ["yesterday", "now"])
        self.assertIn("chased a cat", answer(turns[4])[1])
        self.assertNotIn("slept", answer(turns[4])[1])
        self.assertIn("chased a cat", answer(turns[5])[1])

    def test_where_it_was_does_not_carry_to_another_day(self):
        _, turns, _ = talk("there is a pig",
                           "yesterday the pig was in an airplane",
                           "today the pig is in a field",
                           "is the pig in an airplane",
                           "where was the pig yesterday", "where is the pig")
        # T3, one place at a time: in a field today is not in an airplane
        # today. What it was yesterday answers nothing about now.
        outcome, text = answer(turns[3])
        self.assertEqual(outcome, "denied")
        self.assertIn("field", text)
        self.assertIn("T3", text)
        self.assertIn("airplane", answer(turns[4])[1])
        self.assertIn("field", answer(turns[5])[1])
        self.assertNotIn("airplane", answer(turns[5])[1])

    def test_a_quality_in_the_present_and_in_the_morning(self):
        _, turns, _ = talk("there is a cup", "it is empty now", "is it full",
                           "it was full this morning",
                           "was it full this morning", "is it empty",
                           "is it full")
        self.assertEqual(answer(turns[2])[0], "denied")
        self.assertEqual(answer(turns[4])[0], "verified")
        self.assertEqual(answer(turns[5])[0], "verified")
        self.assertEqual(answer(turns[6])[0], "denied")

    def test_now_when_only_yesterday_was_told(self):
        _, turns, _ = talk("there was a dog", "yesterday it was hungry",
                           "is it hungry now", "was it hungry yesterday")
        outcome, text = answer(turns[2])
        self.assertEqual(outcome, "unknown")
        self.assertIn("yesterday it was hungry", text)
        self.assertEqual(answer(turns[3])[0], "verified")

    def test_when_did_it_happen(self):
        _, turns, _ = talk("there is a dog", "there is a cat", "it barked",
                           "yesterday the dog chased the cat",
                           "when did the dog chase the cat",
                           "did the dog chase the cat today")
        self.assertIn("yesterday", answer(turns[4])[1])
        outcome, text = answer(turns[5])
        self.assertEqual(outcome, "unknown")
        self.assertIn("yesterday", text)

    def test_who_did_it_on_a_day(self):
        _, turns, _ = talk("there is a dog", "there is a cat",
                           "yesterday the dog chased the cat",
                           "today the cat chased the dog",
                           "who chased the cat yesterday",
                           "who chased the dog today")
        self.assertTrue(answer(turns[4])[1].startswith("the dog"))
        self.assertTrue(answer(turns[5])[1].startswith("the cat"))

    def test_e2_is_within_one_time(self):
        """`yesterday it was in an airplane` explains nothing about flying
        today."""
        session, turns, _ = talk("there was a pig",
                                 "yesterday it was in an airplane",
                                 "today it was flying", "can the pig fly")
        self.assertFalse(session.memory.withdrawn)
        self.assertEqual(turns[3].answer["outcome"], "verified")
        session, turns, _ = talk("there was a pig", "he was flying",
                                 "it was in an airplane", "can the pig fly")
        self.assertTrue(session.memory.withdrawn)


# -- T5 and T6 ---------------------------------------------------------------

class OccurrenceTests(unittest.TestCase):
    """T5: an occurrence is an individual of its verb. T6: nothing happens
    by inheritance."""

    def test_an_occurrence_answers_did(self):
        session, turns, _ = talk("there was a dog", "there was a cat",
                                 "the dog chased the cat",
                                 "did the dog chase the cat",
                                 "did the dog move")
        self.assertEqual(answer(turns[3])[0], "verified")
        outcome, text = answer(turns[4])
        self.assertEqual(outcome, "verified")
        self.assertIn("chase is a kind of move", text)
        self.assertEqual(session.timeline.occurrences[0].kinds[:3],
                         ["chase", "pursue", "follow"])

    def test_what_the_kind_does_is_not_what_this_one_did(self):
        _, turns, asker = talk("there is a dog", "did the dog bark",
                               "does it bark")
        outcome, text = answer(turns[1])
        self.assertEqual((outcome, turns[1].answer["source"]),
                         ("unknown", "tendency"))
        self.assertIn("T6", text)
        self.assertIn("does a dog bark", asker.asked)
        self.assertEqual(answer(turns[2])[0], "verified")

    def test_counting_occurrences(self):
        _, turns, _ = talk("there is a dog", "it barked", "it barked again",
                           "then it barked once more",
                           "how many times did the dog bark")
        self.assertTrue(answer(turns[4])[1].startswith("3 times"))
        self.assertIn("again", answer(turns[4])[1])

    def test_in_progress_then_and_now(self):
        _, turns, _ = talk("there was a pig", "he was flying",
                           "is the pig flying", "did the pig fly",
                           "what was the pig doing")
        self.assertEqual(answer(turns[2])[0], "verified")
        self.assertEqual(answer(turns[3])[0], "verified")
        self.assertIn("flying", answer(turns[4])[1])

    def test_did_is_never_what_is_to_come(self):
        """Reported by the probe: `did the dog swim` was yes from `tomorrow
        the dog will swim`."""
        _, turns, _ = talk("there is a dog", "tomorrow the dog will swim",
                           "will the dog swim tomorrow", "did the dog swim",
                           "what will happen tomorrow")
        self.assertEqual(answer(turns[2])[0], "verified")
        self.assertNotEqual(answer(turns[3])[0], "verified")
        self.assertIn("will swim", answer(turns[4])[1])

    def test_a_verb_that_is_its_own_participle(self):
        """`earlier it had run` was `has_part "run"`."""
        session, turns, _ = talk("there was a dog", "it slept",
                                 "earlier it had run", "what happened first")
        self.assertEqual([one.verb for one in session.timeline.story()],
                         ["run", "sleep"])
        self.assertIn("run", answer(turns[3])[1])

    def test_a_named_time_nothing_was_told_of(self):
        _, turns, _ = talk("there is a dog", "was it barking yesterday")
        outcome, text = answer(turns[1])
        self.assertEqual(outcome, "unknown")
        self.assertIn("yesterday", text)


# -- T4 ----------------------------------------------------------------------

@unittest.skipUnless(change.frames(), "VerbNet 3.3 is not in data/")
class ChangeTests(unittest.TestCase):
    """T4: what an occurrence changes, from VerbNet's event structure."""

    def test_only_the_classes_of_the_sense_meant(self):
        """`kill` is in amuse-31.1 and pain-40.8.1 too; only murder-42.1
        has the sense of killing a mouse. The state keyed by the verb is
        dropped later, when `killed` turns out not to be an adjective."""
        found = change.effects("kill", True, senses=["kill.v.01"])
        self.assertEqual({one.source.split("-")[0] for one in found},
                         {"murder"})
        self.assertIn(("alive", False, True),
                      [(one.word, one.after, one.before) for one in found])
        self.assertIn("amuse", {one.source.split("-")[0] for one in
                                change.effects("kill", True)})
        self.assertEqual(change.effects("break", False)[0].position,
                         "subject")
        self.assertEqual(change.effects("chase", True), [])

    def test_a_frame_of_two_objects_is_only_a_sentence_of_two(self):
        """A second bare `NP` after the verb is a second object only where
        VerbNet's own description counts two (`NP V NP NP`); otherwise it is
        an adverb, adjective or unprepositioned place spelled as a noun
        phrase (`NP V NP ADVP`), and the sentence leaves it unsaid. Neither
        may be the object: `took the football` must not put the taker at
        the football through bring-11.3's destination."""
        by_klass = {frame.klass: frame for frame, _ in change.frames()["take"]
                    if frame.klass == "bring-11.3"
                    and dict(frame.positions).get("Instrument")}
        advp = by_klass["bring-11.3"]
        self.assertNotIn("Destination", dict(advp.positions))
        self.assertEqual(advp.shape, frozenset({"object"}))
        moved = [one for one in change.effects("take", True)
                 if one.kind == "location" and one.position == "subject"
                 and one.at == "object"]
        self.assertEqual(moved, [], moved)
        give = [frame for frame, _ in change.frames()["give"]
                if frame.klass == "give-13.1"
                and "object2" in dict(frame.positions).values()]
        self.assertTrue(give, "give-13.1's `NP V NP NP` is two objects")

    def test_a_broken_vase(self):
        _, turns, _ = talk("there was a cat", "there was a vase",
                           "the cat broke the vase", "is the vase broken",
                           "was the vase broken before the cat broke it")
        self.assertIn("T4", answer(turns[2])[1])
        self.assertEqual(answer(turns[3])[0], "verified")
        self.assertEqual(answer(turns[4])[0], "denied")

    def test_a_killed_mouse(self):
        _, turns, _ = talk("there was a cat", "there was a mouse",
                           "the cat killed the mouse", "is the mouse alive",
                           "is the mouse dead",
                           "was the mouse alive before the cat killed it")
        self.assertEqual(answer(turns[3])[0], "denied")
        self.assertEqual(answer(turns[4])[0], "verified")
        self.assertEqual(answer(turns[5])[0], "verified")

    def test_a_key_put_in_a_drawer(self):
        _, turns, _ = talk("there is a key", "there is a drawer",
                           "i put the key in the drawer", "where is the key",
                           "was the key in the drawer before i put it in "
                           "the drawer")
        self.assertIn("drawer", answer(turns[3])[1])
        self.assertEqual(answer(turns[4])[0], "denied")

    def test_a_door_closed(self):
        _, turns, _ = talk("there is a door", "the door was open",
                           "i closed the door", "is the door open",
                           "was the door open before i closed it")
        self.assertEqual(answer(turns[3])[0], "denied")
        self.assertEqual(answer(turns[4])[0], "verified")

    def test_started_and_stopped(self):
        _, turns, _ = talk("there is a dog", "the dog started barking",
                           "then it stopped barking", "is the dog barking")
        self.assertEqual(answer(turns[3])[0], "denied")


# -- event sourcing ----------------------------------------------------------

class ReplayTests(unittest.TestCase):
    """Story time is a projection of the stream like every other table."""

    def test_the_timeline_replays(self):
        lines = ("there is a dog", "there is a cat",
                 "yesterday the dog chased the cat", "then it slept",
                 "today the cat is in the drawer", "the cat ran")
        session, _, _ = talk(*lines)
        again = Session.rebuild(StoryAsker(), session.conversation,
                                list(session.memory.log.conversation),
                                session.memory.knowledge)

        def without_growth(state):
            return {key: value for key, value in state.items()
                    if key != "growth"}
        self.assertEqual(without_growth(again.timeline.as_dict()),
                         without_growth(session.timeline.as_dict()))
        self.assertEqual(again.snapshot(), session.snapshot())
        kinds = {event.type for event in session.memory.log.conversation}
        self.assertTrue({"episode_opened", "occurred", "ordered"} <= kinds)


if __name__ == "__main__":
    unittest.main()
