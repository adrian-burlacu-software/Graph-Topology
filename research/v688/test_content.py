"""`content.digest` over payloads in the shapes v687 returns them."""
from __future__ import annotations

import unittest

from research.v688.content import digest, listed, name_of


def row(concept, relation, obj):
    return {"concept": concept, "relation": relation, "object": obj,
            "source": "ascentpp", "confidence": 0.5, "distance": 0}


class DigestTests(unittest.TestCase):

    def test_a_yes_or_no_holds_no_content(self):
        self.assertIsNone(digest({"verdict": "VERIFIED", "evidence": [
            row("dog.n.01", "capable_of", "swim")]}))
        self.assertIsNone(digest(None))

    def test_a_listing_says_what_it_listed(self):
        found = digest({"verdict": "LISTING", "concept": "dog.n.01",
                        "parse": {"relation": "capable_of"},
                        "evidence": [row("dog.n.01", "capable_of", "bark"),
                                     row("dog.n.01", "capable_of", "swim"),
                                     row("dog.n.01", "capable_of", "bark")]})
        self.assertEqual(found["kind"], "listing")
        self.assertEqual(found["text"], "dog can bark, swim")
        self.assertTrue(found["answers"])

    def test_backwards_names_what_stands_in_front(self):
        found = digest({"verdict": "LISTING", "backwards": {
            "relation": "has_part", "phrase": "wings",
            "subjects": [{"concept": "bat"}, {"concept": "bird"},
                         {"concept": "rib.n.01"}]},
            "identification": {"candidates": [{"name": "bat"}]}})
        self.assertEqual((found["kind"], found["text"]),
                         ("backwards", "have wings: bat, bird, rib"))

    def test_an_identification_names_the_one_that_fits(self):
        found = digest({"verdict": "IDENTIFIED", "identification": {
            "candidates": [{"name": "dalmatian",
                            "matched": {"spots": "spots (stated)"}}]}})
        self.assertEqual(found["text"], "dalmatian — spots (stated)")
        several = digest({"verdict": "AMBIGUOUS", "identification": {
            "candidates": [{"name": "platypus"}, {"name": "squid"}]}})
        self.assertEqual(several["text"], "several fit: platypus, squid")

    def test_a_profile_is_its_trie_path(self):
        found = digest({"verdict": "PROFILE", "profile": {
            "name": "dog", "asked": None,
            "path": [{"predicate": "can run fast"}, {"predicate": "can bark"}]}})
        self.assertEqual(found["text"], "dog: can run fast, can bark")

    def test_the_members_sit_beside_a_quantified_verdict(self):
        found = digest({"verdict": "CONTRADICTED", "profile": {"asked": {
            "members": [{"name": "chicken", "verdict": "DENIED"},
                        {"name": "dove", "verdict": "HELD"},
                        {"name": "magpie", "verdict": "INHERITED"}]}}})
        self.assertEqual(found["text"], "1 do not: chicken; 2 do: dove, magpie")
        self.assertFalse(found["answers"])

    def test_contrast_by_mode(self):
        common = digest({"contrast": {"mode": "common", "left": "dog",
                                      "right": "cat", "shared_total": 2,
                                      "shared": ["can be a pet", "can chase"]}})
        self.assertEqual(common["text"],
                         "dog and cat share 2: can be a pet, can chase")
        apart = digest({"contrast": {"mode": "difference", "left": "frog",
                                     "right": "toad", "shared_total": 17,
                                     "only_left": ["has big eyes"],
                                     "only_right": ["has lumpy skin"]}})
        self.assertEqual(apart["text"], "only frog: has big eyes; only toad: "
                                        "has lumpy skin (17 shared)")
        near = digest({"contrast": {"mode": "nearest", "name": "dog",
                                    "nearest": [{"name": "cat",
                                                 "jaccard": 0.148}]}})
        self.assertEqual(near["text"], "nearest to dog: cat (15%)")
        typical = digest({"contrast": {"mode": "typicality",
                                       "member": "penguin", "klass": "bird",
                                       "score": 0.2, "rank": 24, "of": 29,
                                       "sound": False}})
        self.assertIn("not a sound measure", typical["text"])
        self.assertFalse(typical["answers"])

    def test_a_script_by_phase_and_an_explanation_by_rank(self):
        script = digest({"causal": {"mode": "then", "concept": "rain", "steps": [
            {"relation": "causes", "object": "flooding", "phase": "after"},
            {"relation": "has_subevent", "object": "dip", "phase": "during"},
            {"relation": "causes", "object": "mold", "phase": "after"}]}})
        self.assertEqual(script["text"],
                         "rain — after: flooding, mold; during: dip")
        explained = digest({"causal": {"mode": "abduction",
                                       "observation": "fire", "hypotheses": [
                                           {"cause": "wiring.n.02"},
                                           {"cause": "space heater.n.01"}]}})
        self.assertEqual(explained["text"],
                         "best explanations of fire: wiring, space heater")

    def test_kinds_definition_and_bridge(self):
        kinds = digest({"kinds": {"word": "dog", "total": 189, "direct": 2,
                                  "direct_kinds": ["corgi.n.01", "pug.n.01"],
                                  "described_kinds": ["collie"]}})
        self.assertEqual(kinds["text"], "189 kinds of dog, 2 directly beneath "
                                        "it: corgi, pug; the norms describe collie")
        defined = digest({"definition": {"word": "kitten",
                                         "gloss": "young domestic cat",
                                         "genus": "young mammal.n.01",
                                         "above": ["young mammal.n.01"]}})
        self.assertEqual(defined["text"],
                         "kitten: “young domestic cat” — a kind of young mammal")
        bridged = digest({"bridge": {"verdict": "BRIDGED", "anchor_word": "dog",
                                     "role_word": "owner",
                                     "relation": "has_prerequisite",
                                     "route": {"hops": [
                                         {"source": "dog.n.01",
                                          "relation": "has_a",
                                          "text": "owner"}]}},
                          "evidence": [row("owner.n.01", "has_prerequisite",
                                           "a leash")]})
        self.assertEqual(bridged["text"], "dog's owner needs: a leash "
                                          "(through dog has owner)")

    def test_why_says_what_a_yes_rests_on_and_what_the_doers_share(self):
        from research.v688.content import because

        payload = {"verdict": "VERIFIED", "concept": "bird.n.01",
                   "parse": {"subject": "bird", "target": "fly"},
                   "evidence": [{"concept": "animal.n.01",
                                 "relation": "capable_of", "object": "fly",
                                 "source": "ascentpp", "distance": 2}]}
        found = because(payload, requirement={
            "part": "wing", "action": "fly", "holders": 31, "doers": 87,
            "has": True, "decisive": True})
        self.assertEqual((found["kind"], found["answers"]), ("why", True))
        self.assertTrue(found["text"].startswith(
            "because bird is a kind of animal, 2 levels up, and animal can "
            "“fly” (ascentpp)"))
        self.assertIn("bird has wings", found["text"])
        self.assertIn("31 of the 87", found["text"])

    def test_a_requirement_that_does_not_lead_explains_nothing(self):
        from research.v688.content import because

        payload = {"verdict": "VERIFIED", "concept": "dog.n.01",
                   "parse": {"subject": "dog", "target": "swim"},
                   "note": "Not in the norms for dog, but animal.n.01 is."}
        found = because(payload, requirement={
            "part": "tooth", "action": "swim", "holders": 15, "doers": 54,
            "has": True, "decisive": False})
        self.assertNotIn("tooth", found["text"])
        self.assertTrue(found["text"].startswith("because not in the norms"))

    def test_why_of_a_false_premise_says_it_does_not_hold(self):
        from research.v688.content import because

        denied = {"verdict": "CONTRADICTED", "concept": "penguin",
                  "parse": {"subject": "penguin"},
                  "note": "The norms record “cannot fly” of penguin."}
        self.assertTrue(because(denied)["text"].startswith(
            "the premise does not hold — v687 answers CONTRADICTED: the "
            "norms record"))
        self.assertTrue(because(denied, negative=True)["text"].startswith(
            "because the norms record"))
        found = because(denied, negative=True, requirement={
            "part": "wing", "action": "fly", "holders": 31, "doers": 87,
            "has": True, "decisive": True})
        self.assertIn("not a matter of anatomy", found["text"])
        self.assertIn("nothing recorded settles",
                      because({"verdict": "UNKNOWN"})["text"])

    def test_a_refusal_says_why(self):
        found = digest({"verdict": "UNSUPPORTED",
                        "note": "This asks for a comparative. No scale."})
        self.assertEqual((found["kind"], found["text"], found["answers"]),
                         ("refused", "not answerable here: this asks for a "
                                     "comparative. No scale.", True))
        self.assertIsNone(digest({"verdict": "UNSUPPORTED", "note": ""}))

    def test_names_and_lists(self):
        self.assertEqual(name_of("hot dog.n.01"), "hot dog")
        self.assertEqual(name_of("bat"), "bat")
        self.assertEqual(listed(list("abcdefghij"), 3), "a, b, c and 7 more")


if __name__ == "__main__":
    unittest.main()
