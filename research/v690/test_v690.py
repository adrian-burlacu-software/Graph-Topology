"""v690: messages, reading replies back, and a turn's steps.

No model is needed: a message is taken apart from turns as the page gets
them, a reply is labelled by its message as the teacher labels it
(`roundtrip.label`) and traced against it, and the steps are built from a
turn. spaCy's small English model reads the replies' words; without it the
round-trip tests are skipped. The decoder's and the reply heads' own tests
need `llm/decoder` and a reader taught to read replies.
"""
from __future__ import annotations

import unittest

from research.v689 import social
from research.v690 import message as messages
from research.v690 import roundtrip, steps

BEAGLE = {"id": "r1", "description": "the beagle", "kind": "beagle"}
CAT = {"id": "r2", "description": "the cat", "kind": "cat"}


def turn(said: str, act: str, outcome: str, text: str, source: str = "told",
         mention: str = "it", aux: str | None = "can", rest: str = "swim",
         holds: bool = True, referent: str | None = "r1", run=None,
         obj=None) -> dict:
    return {"said": said, "act": act,
            "reading": {"act": act, "mention": {"text": mention},
                        "aux": aux, "rest": rest, "holds": holds},
            "resolution": ({"expression": mention, "referent": referent}
                           if referent else None),
            "object": obj,
            "answer": {"outcome": outcome, "source": source, "text": text},
            "run": run,
            "discourse": {"referents": [BEAGLE, CAT]}}


def _words():
    try:
        return roundtrip.Words()
    except Exception:                               # noqa: BLE001
        return None


WORDS = _words()


class MessageTests(unittest.TestCase):
    def test_a_yes_or_no_is_a_claim_about_who_it_meant(self):
        found = messages.of_turn(turn(
            "can it swim", "ask", "denied",
            "no — you told me so: “it can't swim”. R3 at distance 0, before "
            "anything is inherited"))
        self.assertEqual((found.stance, found.subject, found.claim),
                         ("no", "the beagle", "the beagle can swim"))
        self.assertEqual(found.quotes, ["it can't swim"])
        self.assertEqual(found.rules, ["R3"])

    def test_a_denial_noted_is_claimed_denied(self):
        found = messages.of_turn(turn("it can't swim", "tell", "noted",
                                      "noted — stored not_capable_of “swim”",
                                      holds=False))
        self.assertEqual(found.claim, "the beagle can't swim")

    def test_an_object_is_said_as_the_conversation_describes_it(self):
        found = messages.of_turn(turn(
            "the dog chased it", "tell", "noted", "noted", mention="the dog",
            aux=None, rest="chased it", referent=None,
            obj={"expression": "it", "referent": "r2"}))
        self.assertEqual(found.claim, "the dog chased the cat")

    def test_values_are_what_comes_before_the_dash(self):
        found = messages.of_turn(turn(
            "where is Mary", "question", "retrieved",
            "Mary: the kitchen — “Mary went to the kitchen” changed it (T4, "
            "escape-51.1)", mention="mary", aux=None, rest="",
            referent=None))
        self.assertEqual(found.values, ["the kitchen"])
        self.assertEqual(found.found, "Mary: the kitchen — “Mary went to the "
                                      "kitchen” changed it")

    def test_what_is_quoted_answers_when_nothing_comes_before(self):
        found = messages.of_turn(turn(
            "what is a kitten", "define", "retrieved",
            "a kitten: “young domestic cat” — a kind of cat",
            mention="a kitten", aux=None, rest="", referent=None,
            source="definition"))
        self.assertEqual(found.values, ["young domestic cat"])

    def test_a_label_followed_only_by_what_was_said_is_not_a_value(self):
        found = messages.of_turn(turn(
            "what happened", "question", "retrieved",
            "in the order you told me: “Verney moved to the office”; “the "
            "cinema is north of the office”", aux=None, rest="",
            referent=None, mention=""))
        self.assertEqual(found.values, ["Verney moved to the office",
                                        "the cinema is north of the office"])

    def test_a_listing_is_its_first_items(self):
        found = messages.of_turn(turn(
            "what can a bird do", "generic", "retrieved",
            "bird can build nest, fly and 30 more", aux=None, rest="",
            referent=None, mention="", source="kind",
            run={"trust": "content, not a verdict",
                 "content": {"items": ["build nest", "fly", "eat seed",
                                       "lay egg", "sing", "nest"],
                             "more": 30}}))
        self.assertEqual(found.values[:2], ["build nest", "fly"])
        self.assertEqual(found.more, 31)
        self.assertEqual(messages.required(found)["phrases"],
                         ["build nest", "fly", "eat seed"])

    def test_a_refusal_by_name_is_not_ignorance(self):
        found = messages.of_turn(turn(
            "what is the capital of France", "generic", "unknown",
            "not answerable here: this asks for a fact about a named "
            "individual", source="kind", referent=None))
        self.assertEqual(found.stance, "refused")

    def test_senses_and_rules_are_left_for_the_page(self):
        self.assertEqual(messages.plain(
            "noted: the beagle, placed under beagle.n.01 (T4, escape-51.1)"),
            "noted: the beagle")
        self.assertEqual(messages.plain("a kind of hunting_dog.n.01"),
                         "a kind of hunting dog")

    def test_the_prompt_says_every_field_it_has(self):
        found = messages.of_turn(turn("can it swim", "ask", "verified",
                                      "yes — " + "long " * 200))
        shown = messages.prompt(found)
        self.assertIn("stance: yes", shown)
        self.assertIn("claim: the beagle can swim", shown)
        self.assertLess(len(shown), 600)


@unittest.skipUnless(WORDS is not None, "needs spaCy's small English model")
class RoundTripTests(unittest.TestCase):
    def traced(self, found: messages.Message, reply: str,
               framing=frozenset({"tell", "know"})) -> roundtrip.Trace:
        read = roundtrip.label(found, reply, WORDS)
        return roundtrip.trace(found, read, WORDS, framing, reply)

    def no(self) -> messages.Message:
        return messages.of_turn(turn(
            "can it swim", "ask", "denied",
            "no — you told me so: “it can't swim”"))

    def test_a_faithful_reply_reads_back(self):
        found = self.traced(self.no(),
                            "No, the beagle can't swim. You told me so.")
        self.assertTrue(found.traced, found.why())

    def test_a_no_that_does_not_deny_is_caught(self):
        found = self.traced(self.no(), "No, the beagle can swim.")
        self.assertFalse(found.traced)
        self.assertEqual(found.denial, "does not deny the claim")

    def test_a_reason_nothing_said_is_caught(self):
        found = self.traced(self.no(),
                            "No, the beagle can't swim because it hates "
                            "water.")
        self.assertIn("water", found.added)

    def test_leaving_out_the_claim_is_caught(self):
        found = self.traced(self.no(), "No, it can't.")
        self.assertIn("swim", found.missing)

    def test_an_internal_word_is_caught(self):
        found = self.traced(self.no(), "No — R3: the beagle can't swim.")
        self.assertEqual(found.internal, ["R3"])

    def test_a_wrong_value_is_caught(self):
        where = messages.of_turn(turn(
            "where is Mary", "question", "retrieved",
            "Mary: the kitchen — “Mary went to the kitchen”", mention="mary",
            aux=None, rest="", referent=None))
        self.assertTrue(self.traced(where, "Mary is in the kitchen.").traced)
        wrong = self.traced(where, "Mary is in the garden.")
        self.assertEqual((wrong.missing, wrong.added), (["kitchen"],
                                                        ["garden"]))

    def test_a_denial_said_after_the_claim_is_caught(self):
        went = messages.of_turn(turn(
            "Fred went to the bar", "tell", "noted", "noted", mention="Fred",
            aux=None, rest="went to the bar", referent=None))
        kept = self.traced(went, "Fred went to the bar. You told me so.")
        self.assertTrue(kept.traced, kept.why())
        self.assertEqual(self.traced(
            went, "Fred went to the bar. You told me he did not.").denial,
            "denies the claim")

    def test_a_value_that_is_none_is_said_as_none(self):
        carried = messages.of_turn(turn(
            "what is Orrin carrying", "question", "retrieved",
            "nothing — nothing was said to be with Orrin now",
            mention="orrin", aux=None, rest="", referent=None))
        self.assertEqual(carried.values, ["nothing"])
        kept = self.traced(carried, "Orrin isn't carrying anything.")
        self.assertTrue(kept.traced, kept.why())
        self.assertEqual(self.traced(
            carried, "Orrin is carrying anything.").denial,
            "does not say there is none")

    def test_an_unknown_says_it_is_not_known(self):
        hungry = messages.of_turn(turn(
            "is the dog hungry", "ask", "unknown",
            "not told of today — yesterday, you told me “it was hungry”",
            mention="the dog", aux="is", rest="hungry", referent=None))
        self.assertTrue(self.traced(
            hungry, "I don't know if the dog is hungry today. You only told "
                    "me it was hungry yesterday.").traced)
        self.assertEqual(self.traced(
            hungry, "I do know if the dog is hungry today. I do not know "
                    "what it ate.").denial, "does not say it is not known")

    def test_saying_you_told_me_what_the_store_said_is_caught(self):
        kind = messages.of_turn(turn(
            "can a dog swim", "generic", "verified",
            "VERIFIED — can a dog swim (corroborated)", mention="a dog",
            source="kind", referent=None))
        self.assertTrue(self.traced(kind, "Yes, a dog can swim.").traced)
        wrong = self.traced(kind, "Yes, a dog can swim. You told me so.")
        self.assertEqual(wrong.source, "says you told it, and you did not")
        self.assertTrue(self.traced(
            kind, "Yes, a dog can swim, though you haven't told me about "
                  "this one.").traced)
        told = self.no()
        self.assertTrue(self.traced(
            told, "No, the beagle can't swim. You told me so.").traced)

    def test_the_answer_s_own_denial_does_not_excuse_you_said_so(self):
        """`No, a penguin can't fly. You said so.` of what the norms state:
        the `can't` is the answer's, in the sentence before."""
        kind = messages.of_turn(turn(
            "can a penguin fly", "generic", "denied",
            "CONTRADICTED — can a penguin fly (denied, unchallenged)",
            mention="a penguin", aux="can", rest="fly", source="kind",
            referent=None))
        self.assertEqual(self.traced(
            kind, "No, a penguin can't fly. You said so.").source,
            "says you told it, and you did not")
        self.assertTrue(self.traced(kind, "No, a penguin can't fly.").traced)

    def test_narrating_an_exchange_that_did_not_happen_is_caught(self):
        """`What about whales?` is the reader putting the last question to a
        new kind. The reply said `You asked if they can fly, and I said no`
        -- nobody asked that, and nothing was said before it. Framing does
        not stop it: `say` and `tell` are licensed for yes and no, and `ask`
        comes free with the message's own `asked as “Do whales fly”`."""
        again = messages.of_turn(turn(
            "What about whales?", "again question", "denied",
            "asked as “Do whales fly”: CONTRADICTED — Do whales fly "
            "(denied, unchallenged)", mention="whales", aux="do", rest="fly",
            source="kind", referent=None))
        self.assertTrue(self.traced(again, "No, whales don't fly.").traced)
        for reply in ("No, whales don't fly. You asked if they can fly, and "
                      "I said no.",
                      "No, whales don't fly. You asked if they could.",
                      "No, whales don't fly. I said so earlier."):
            with self.subTest(reply=reply):
                self.assertEqual(
                    self.traced(again, reply).source,
                    "says what was asked and answered, and it was not")

    def test_a_contraction_does_not_hide_who_told_it(self):
        """spaCy reads `You've` as one token, so a check looking for `you`
        beside the verb never saw it and `You've told me so` read back
        clean."""
        kind = messages.of_turn(turn(
            "can a robin fly", "generic", "verified",
            "VERIFIED — can a robin fly (corroborated)", mention="a robin",
            aux="can", rest="fly", source="kind", referent=None))
        self.assertEqual(
            self.traced(kind, "Yes, a robin can fly. You've told me so.")
            .source, "says you told it, and you did not")
        self.assertTrue(self.traced(kind, "Yes, a robin can fly.").traced)

    def test_a_reply_cut_off_is_caught(self):
        """The decoder stops at its longest reply; the words it wrote by then
        can all read back and still not be a sentence."""
        found = self.traced(self.no(), "No, the beagle can't swim. You told me")
        self.assertEqual(found.unfinished, "stops before its sentence ends")
        self.assertFalse(found.traced)

    def test_a_sentence_left_out_is_read_back_again(self):
        """What the speaker tries when nothing written reads back: the reply
        with one sentence left out, the last first, each read back as a
        reply of its own."""
        from research.v690.speaking import shortened

        self.assertEqual(shortened("No, a penguin can't fly. You said so."),
                         ["No, a penguin can't fly.", "You said so."])
        self.assertEqual(shortened("Got it."), [])
        cut = shortened("No, the beagle can't swim. You told me so. I can")
        self.assertEqual(cut[0], "No, the beagle can't swim. You told me so.")
        self.assertTrue(self.traced(self.no(), cut[0]).traced)
        self.assertFalse(self.traced(self.no(), "You told me so.").traced)

    def test_a_verb_is_said_by_one_it_is_a_kind_of(self):
        went = messages.of_turn(turn(
            "Morissa travelled to the cinema", "tell", "noted", "noted",
            mention="Morissa", aux=None, rest="travelled to the cinema",
            referent=None))
        found = self.traced(went, "Morissa went to the cinema.")
        self.assertTrue(found.traced, found.why())

    def test_a_denial_of_a_word_spacy_calls_a_stop_word(self):
        whole = messages.of_turn(turn("it isn't whole", "tell", "noted",
                                      "noted", aux="is", rest="whole",
                                      holds=False))
        self.assertTrue(self.traced(whole, "The beagle isn't whole.").traced)
        self.assertEqual(self.traced(whole, "The beagle is whole.").denial,
                         "does not deny the claim")

    def test_the_teacher_labels_each_part(self):
        read = roundtrip.label(self.no(), "No, the beagle can't swim.",
                               WORDS)
        parts = dict(zip(read.words, read.parts))
        self.assertEqual((parts["beagle"], parts["not"], parts["swim"]),
                         ("SUBJ", "NEG", "CLAIM"))
        self.assertEqual(parts["No"], "O")


class CorruptionTests(unittest.TestCase):
    """The evaluation's wrong replies keep the sentences they were written
    in, which the denial check reads."""

    def test_a_denial_is_taken_out_as_written(self):
        from research.v690.evaluate import undenied

        self.assertEqual(undenied("No, the beagle can't swim. You told me."),
                         "No, the beagle can swim. You told me.")
        self.assertEqual(undenied("I do not know if it is. Sure."),
                         "I do know if it is. Sure.")
        self.assertIsNone(undenied("Yes, it can."))

    def test_a_denial_is_added_after_an_auxiliary(self):
        from research.v690.evaluate import corruptions

        found = corruptions(messages.of_turn(turn(
            "can it swim", "ask", "verified", "yes")),
            "Yes, the beagle can swim.", [], __import__("random").Random(1))
        self.assertEqual(found["denial added"],
                         "Yes, the beagle can not swim.")


class StepTests(unittest.TestCase):
    def test_a_turn_reads_as_an_account(self):
        found = turn("can it swim", "ask", "denied",
                     "no — you told me so: “it can't swim”")
        found["heard"] = {"said": "can it swim", "acts": [["ask", 0.98]],
                          "roles": [["can", "AUX"], ["it", "B-SUBJ"],
                                    ["swim", "REST"]]}
        found["trace"] = [{"claim": "can it swim", "act": "ask", "fired": [
            {"operator": "ask", "rule": "", "outcome": "answered"}]}]
        said = {"text": "No, the beagle can't swim.", "traced": True,
                "message": messages.of_turn(found).as_dict(),
                "candidates": [{"text": "No, the beagle can't swim.",
                                "trace": {"traced": True}, "read": {}}]}
        names = [one["step"] for one in steps.steps_of(found, said)]
        self.assertEqual(names, ["heard", "read", "resolved", "reasoned",
                                 "answered", "said"])
        lines = {one["step"]: one["line"] for one in
                 steps.steps_of(found, said)}
        self.assertIn("98% sure", lines["read"])
        self.assertIn("“it” is the beagle", lines["resolved"])
        self.assertIn("the ask operator answered", lines["reasoned"].lower())


class PlannedStepTests(unittest.TestCase):
    """v691's planning, as a step of its own on the page."""

    def acted(self, planning=None):
        found = turn("get the book to the kitchen", "generic", "acted",
                     "I took the book to the kitchen")
        found["answer"]["source"] = "world"
        found["answer"]["planning"] = planning or {}
        found["trace"] = [{"claim": "get the book to the kitchen",
                           "act": "generic", "fired": [
                               {"operator": "noting", "rule": "",
                                "outcome": "continue"},
                               {"operator": "want", "rule": "",
                                "outcome": "answered"}]}]
        return found

    def test_an_order_gets_a_planned_step(self):
        found = self.acted({
            "goal": ["the book is in the kitchen"], "offered": 215,
            "verbs": 8, "plan": ["took the book to the kitchen"],
            "actions": ["take john book kitchen"], "solved": True,
            "fired": 3, "subgoals": 1, "deep": 2, "plans": 1,
            "stack": {"goal": "make at book kitchen true", "fired": [],
                      "subgoals": [], "more": 0}, "surprises": []})
        said = {"text": "I took the book to the kitchen", "traced": True,
                "candidates": [], "source": "the agent's own words"}
        found_steps = steps.steps_of(found, said)
        names = [one["step"] for one in found_steps]
        self.assertIn("planned", names)
        self.assertLess(names.index("reasoned"), names.index("planned"))
        lines = {one["step"]: one["line"] for one in found_steps}
        self.assertIn("215 actions were possible", lines["planned"])
        self.assertIn("goal stack 2 deep", lines["planned"])
        self.assertIn("own words", lines["said"])
        self.assertIn("Not a verdict", lines["answered"])

    def test_an_operator_that_went_on_did_not_have_nothing(self):
        """`noting` writes down what was said and returns CONTINUE; the
        account used to say it had nothing."""
        lines = {one["step"]: one["line"]
                 for one in steps.steps_of(self.acted())}
        self.assertIn("noting went first", lines["reasoned"])
        self.assertNotIn("noting had nothing", lines["reasoned"])

    def test_a_turn_that_asked_for_nothing_has_no_planned_step(self):
        found = turn("can it swim", "ask", "denied", "no")
        self.assertNotIn("planned",
                         [one["step"] for one in steps.steps_of(found)])


class SocialTests(unittest.TestCase):
    def test_a_social_phrase_with_fillers_around_it(self):
        for said, act in (("hello", "greet"), ("oh hello there", "greet"),
                          ("thanks again", "thank"), ("ok", "affirm"),
                          ("well done", "praise"), ("hi how are you",
                                                    "wellbeing"),
                          ("talk to you later", "farewell"),
                          ("what can you do", "abilities")):
            self.assertEqual(social.act_of(said.split()), act, said)

    def test_what_is_not_sociable_is_nothing(self):
        for said in ("hello is a word", "can a dog swim", "where is Mary",
                     "thanks to the dog"):
            self.assertEqual(social.act_of(said.split()), "", said)

    def test_what_it_can_do_is_what_its_operators_answer(self):
        said = social.answer("abilities", ["located", "holding"])
        self.assertIn("where someone or something is", said)
        self.assertIn("who has what", said)


class ServerTests(unittest.TestCase):
    """The page's senses, without a store: which words a turn offers, and a
    pinned sense used where a kind is placed."""

    def test_a_word_after_can_is_offered_as_a_verb(self):
        from research.v690.server import Conversations

        conversations = Conversations.__new__(Conversations)
        found = conversations._words({
            "reading": {"mention": {"kind": "beagle"}, "aux": "can"},
            "heard": {"roles": [["can", "AUX"], ["the", "B-SUBJ"],
                                ["beagle", "I-SUBJ"], ["swim", "REST"]]},
            "asked": "can a beagle swim"})
        self.assertEqual(found, [("beagle", "n"), ("swim", "v")])

    def test_a_pinned_sense_is_the_one_placed(self):
        from research.v690.server import PinnedAsker

        class Reasoner:
            def senses_of(self, word, pos):
                return [{"id": f"{word}.n.01"}, {"id": f"{word}.n.02"}]

        asker = PinnedAsker.__new__(PinnedAsker)
        asker.reasoner, asker.pins = Reasoner(), {}
        self.assertEqual(asker.sense("pig"), "pig.n.01")
        asker.pins = {"pig": "pig.n.02"}
        self.assertEqual(asker.sense("pig"), "pig.n.02")
        self.assertEqual(asker.sense("dog"), "dog.n.01")


if __name__ == "__main__":
    unittest.main()
