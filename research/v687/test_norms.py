"""Regression suite. Run: python -m unittest research.v687.test_v686 -v

Data-dependent tests skip themselves when the norms have not been pulled in.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from research.v687.measure import measure
from research.v687.ordering import ORDERINGS, coverage, optimal
from research.v687.substrate import Corpus
from research.v687 import build
from research.v687 import corpora
from research.v687.identifiability import cue_validity, depths
from research.v687.identifiability import measure as identifiability
from research.v687.identify import Identifier
from research.v687.profile import Profiles
from research.v687.server import IdentifyingEngine

STORE = build.DEFAULT_STORE.with_name("v684_reasoning_compressed.sqlite")
HAVE_NORMS = (corpora.XCSLB_DIR / "comps_base.jsonl").exists() and \
             (corpora.AWA2_DIR / "classes.txt").exists()
requires_norms = unittest.skipUnless(HAVE_NORMS, "feature norms not downloaded")
requires_store = unittest.skipUnless(STORE.exists(), f"no store at {STORE}")


@requires_norms
class CorpusTests(unittest.TestCase):
    """The norms, read as Appendix 3 corpora."""

    def test_awa2_is_dense_and_closed(self):
        awa = corpora.load_awa2()
        self.assertEqual(len(awa), 50)
        self.assertEqual(awa.predicates, 85)

    def test_xcslb_scale(self):
        xcslb = corpora.load_xcslb()
        self.assertEqual(len(xcslb), 521)
        self.assertGreater(xcslb.predicates, 3500)

    def test_the_matrix_misalignment_that_made_this_read_the_pairs(self):
        """`budgie` must carry bird properties, not `can be covered in lip balm`."""
        birds = dict(corpora.load_xcslb(category="bird").items)
        self.assertIn("budgie", birds)
        self.assertIn("can chirp", birds["budgie"])
        self.assertNotIn("can be covered in lip balm", birds["budgie"])
        self.assertIn("has a red breast", dict(corpora.load_xcslb().items)["robin"])

    def test_slicing_by_feature_type(self):
        visual = corpora.load_xcslb(kinds=("visual perceptual",))
        self.assertLess(visual.cells, corpora.load_xcslb().cells)
        types = corpora.feature_types()
        for _, properties in visual.items:
            for prop in properties:
                self.assertEqual(types.get(prop), "visual perceptual")

    def test_every_feature_ships_a_negation(self):
        """What v684's R3 never had: the denial of each property."""
        negated = corpora.negations()
        self.assertIn("has keys", negated)
        self.assertEqual(negated["has keys"], "does not have keys")

    def test_the_wordnet_join_exists(self):
        keys = corpora.senses()
        self.assertGreater(len(keys), 500)
        self.assertTrue(keys["robin"].startswith("robin%"))


@requires_norms
class CompressionTests(unittest.TestCase):
    """Appendix 3's trie, unmodified, on both corpora."""

    def nodes(self, corpus, ordering):
        return measure(corpus, ordering, ORDERINGS[ordering](corpus)).nodes

    def test_the_headline_numbers_are_reproducible(self):
        """Pinned because an earlier ad-hoc slice was not.

        `Counter.most_common` broke ties by insertion order, which varies with
        the hash seed, so a top-8 slice differed between runs. The full corpora
        do not: these numbers held across four hash seeds.
        """
        awa = corpora.load_awa2()
        self.assertEqual(self.nodes(awa, "adaptive_coverage"), 883)
        self.assertEqual(self.nodes(awa, "global_coverage"), 1177)
        self.assertEqual(self.nodes(awa, "lexical"), 1405)
        self.assertEqual(self.nodes(awa, "anti_coverage"), 1521)
        self.assertEqual(self.nodes(corpora.load_xcslb(), "adaptive_coverage"),
                         10794)

    def test_the_figure_beats_the_text_on_every_corpus(self):
        """Appendix 3's branch-local figure against its one-global-order text."""
        for corpus in (corpora.load_awa2(), corpora.load_xcslb(),
                       corpora.load_xcslb(category="bird"),
                       corpora.load_xcslb(kinds=("visual perceptual",))):
            adaptive = self.nodes(corpus, "adaptive_coverage")
            global_ = self.nodes(corpus, "global_coverage")
            self.assertLess(adaptive, global_, corpus.name)

    def test_coverage_ordering_beats_its_own_reversal(self):
        for corpus in (corpora.load_awa2(), corpora.load_xcslb()):
            self.assertLess(self.nodes(corpus, "global_coverage"),
                            self.nodes(corpus, "anti_coverage"), corpus.name)

    def test_dense_and_closed_compresses_far_better_than_sparse(self):
        awa = measure(corpora.load_awa2(), "a",
                      ORDERINGS["adaptive_coverage"](corpora.load_awa2()))
        xcslb = measure(corpora.load_xcslb(), "a",
                        ORDERINGS["adaptive_coverage"](corpora.load_xcslb()))
        self.assertGreater(awa.reuse_rate, 0.40)
        self.assertLess(xcslb.reuse_rate, 0.20)

    def test_branch_local_ordering_beats_the_best_global_one(self):
        """`optimal` is exhaustive over global orders; adaptive is not global.

        Tie-broken by name so the slice is the same on every run.
        """
        awa = corpora.load_awa2()
        counts = coverage(awa)
        top = {p for p, _ in sorted(counts.items(),
                                    key=lambda kv: (-kv[1], kv[0]))[:8]}
        trimmed = Corpus("awa2/top8", tuple(
            (i, frozenset(p for p in s if p in top)) for i, s in awa.items))
        trimmed = Corpus("awa2/top8", tuple(x for x in trimmed.items if x[1]))
        best_global = measure(trimmed, "optimal", optimal(trimmed)).nodes
        adaptive = self.nodes(trimmed, "adaptive_coverage")
        self.assertEqual(best_global, 42)
        self.assertEqual(adaptive, 34)
        self.assertLess(adaptive, best_global)

    def test_compression_improves_with_more_individuals(self):
        awa = corpora.load_awa2()
        small = Corpus("awa2[:12]", awa.items[:12])
        self.assertLess(
            measure(small, "a", ORDERINGS["adaptive_coverage"](small)).reuse_rate,
            measure(awa, "a", ORDERINGS["adaptive_coverage"](awa)).reuse_rate)


@requires_norms
class TradeoffTests(unittest.TestCase):
    """Storing and asking pull in opposite directions on the same trie."""

    def rows(self, corpus):
        plans = dict(ORDERINGS)
        plans["cue_validity"] = cue_validity
        return [identifiability(corpus, name, build(corpus))
                for name, build in plans.items()]

    def test_compression_and_questions_are_perfectly_rank_inverted(self):
        """rho = +1.000 on every corpus: an ordering, not a tendency."""
        for corpus in (corpora.load_awa2(), corpora.load_xcslb(),
                       corpora.load_buchanan()):
            rows = sorted(self.rows(corpus), key=lambda r: r.reuse)
            questions = [r.mean_depth for r in rows]
            self.assertEqual(questions, sorted(questions), corpus.name)

    def test_the_best_compressor_asks_the_most_questions(self):
        rows = {r.ordering: r for r in self.rows(corpora.load_awa2())}
        best = max(rows.values(), key=lambda r: r.reuse)
        fewest = min(rows.values(), key=lambda r: r.mean_depth)
        self.assertEqual(best.ordering, "adaptive_coverage")
        self.assertEqual(fewest.ordering, "anti_coverage")
        self.assertGreater(best.mean_depth, fewest.mean_depth * 3)

    def test_distinctiveness_ordering_is_anti_coverage(self):
        """McRae's distinctiveness is 1/(concepts carrying it), so ranking by
        it is ranking by ascending coverage -- which is already implemented."""
        awa = corpora.load_awa2()
        counts = coverage(awa)
        by_distinctiveness = sorted(counts, key=lambda p: (1 / counts[p], str(p)))
        by_anti = sorted(counts, key=lambda p: (-counts[p], str(p)))
        self.assertEqual(by_distinctiveness, by_anti)

    def test_identification_depth_is_the_unique_prefix(self):
        corpus = Corpus("tiny", (
            ("a", frozenset({"x", "y"})),
            ("b", frozenset({"x", "z"})),
            ("c", frozenset({"w"}))))
        found, never = depths([("a", ("x", "y")), ("b", ("x", "z")),
                               ("c", ("w",))])
        self.assertEqual(never, 0)
        self.assertEqual(sorted(found), [1, 2, 2])

    def test_identical_predicate_sets_are_reported_not_scored(self):
        found, never = depths([("a", ("x",)), ("b", ("x",))])
        self.assertEqual(never, 2)
        self.assertEqual(found, [])

    def test_the_only_thing_buchanan_cannot_tell_apart_is_a_synonym_pair(self):
        rows = {r.ordering: r for r in self.rows(corpora.load_buchanan())}
        self.assertEqual(rows["anti_coverage"].never_unique, 2)


@requires_norms
@requires_store
class RoutingTests(unittest.TestCase):
    """Which questions identification takes, and which it leaves alone.

    Routing is grammatical, not a pattern list: a description leaves the thing
    unnamed and says what it is like -- a relative clause, an adjectival
    complement, or `what` used as a determiner. A naming question has a
    subject and asks what it does, and belongs to v684.
    """

    @classmethod
    def setUpClass(cls):
        cls.identifier = Identifier(STORE)

    @classmethod
    def tearDownClass(cls):
        cls.identifier.close()

    def test_descriptions_are_taken(self):
        for question in ("what kind of dog has spots",
                         "what kind of cat has stripes",
                         "what is an object that is round with spots",
                         "what is round with spots",
                         "what animal has stripes",
                         "which animal is big and furry",
                         "what is a bird that is red"):
            self.assertTrue(self.identifier.describes(question), question)

    def test_naming_questions_are_left_to_v684_and_v685(self):
        for question in ("what can a violin player do",
                         "what does a dog owner need",
                         "what does a dog's owner need",
                         "what can a violin do", "is a dog an animal",
                         "can a dog fall into a hole", "what is a dog",
                         "where do you find a hammer",
                         "what is a hammer used for"):
            self.assertFalse(self.identifier.describes(question), question)

    def test_the_class_is_read_from_three_shapes(self):
        for question, among, terms in (
                ("what kind of dog has spots", "dog", ["spots"]),
                ("what is an object that is round with spots", "object",
                 ["round", "spots"]),
                ("what animal has stripes", "animal", ["stripes"])):
            got_terms, got_among = self.identifier.terms_of(question)
            self.assertEqual(got_among, among, question)
            self.assertEqual(sorted(got_terms), sorted(terms), question)

    def test_an_unknown_class_is_dropped_rather_than_searched_for(self):
        self.assertIsNone(self.identifier.terms_of("what zzzqqq has stripes")[1])

@requires_norms
@requires_store
class IdentificationTests(unittest.TestCase):
    """Describe a thing; be told what it is."""

    @classmethod
    def setUpClass(cls):
        cls.identifier = Identifier(STORE)

    @classmethod
    def tearDownClass(cls):
        cls.identifier.close()

    def test_the_reported_question(self):
        found = self.identifier.identify("what kind of dog has spots")
        self.assertEqual(found.verdict, "IDENTIFIED")
        self.assertEqual([c.name for c in found.candidates], ["dalmatian"])
        self.assertIn("stated", found.candidates[0].matched["spots"])

    def test_it_generalises_to_other_classes(self):
        found = self.identifier.identify("what kind of cat has stripes")
        self.assertEqual([c.name for c in found.candidates], ["tiger"])

    def test_a_correct_ambiguity_is_reported_as_one(self):
        """Two birds are red, and both really are."""
        found = self.identifier.identify("what kind of bird is red")
        names = {c.name for c in found.candidates}
        self.assertEqual(found.verdict, "AMBIGUOUS")
        self.assertEqual(names, {"robin", "cockerel"})

    def test_stemming_matches_without_matching_too_much(self):
        """Prefix matching put cheetahs under stripes and desks under trunks."""
        self.assertEqual(Identifier.stem("flies"), "fly")
        self.assertEqual(Identifier.stem("spots"), "spot")
        self.assertNotEqual(Identifier.stem("striven"), Identifier.stem("stripes"))
        self.assertNotEqual(Identifier.stem("truncated"), Identifier.stem("trunk"))

    def test_the_class_constrains_but_does_not_describe(self):
        terms, among = self.identifier.terms_of("what kind of dog has spots")
        self.assertEqual(among, "dog")
        self.assertEqual(terms, ["spots"])

    def test_the_narrowing_is_recorded_step_by_step(self):
        found = self.identifier.identify("what kind of dog has spots")
        self.assertGreaterEqual(len(found.steps), 2)
        counts = [step["remaining"] for step in found.steps]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertEqual(counts[-1], 1)

    def test_awa2_classes_reach_the_taxonomy(self):
        """Without this join `dalmatian` is not a kind of dog and is invisible."""
        self.assertIn("dalmatian", self.identifier.synset)
        self.assertIn("tiger", self.identifier.synset)

    def test_the_narrowing_has_depth_when_several_properties_are_asked(self):
        """One property is one level; the funnel needs more than that.

        The page drew a flat two-row picture for "dog has spots" because
        every candidate was placed at depth 1 whatever eliminated it. Depth
        is now *when* a candidate fell out.
        """
        found = self.identifier.identify(
            "what kind of animal has stripes and eats meat and is fast")
        self.assertEqual(found.verdict, "IDENTIFIED")
        self.assertEqual([c.name for c in found.candidates], ["tiger"])
        depths = {entry["depth"] for entry in found.considered}
        self.assertGreaterEqual(len(depths), 3)
        survivor = next(e for e in found.considered if e["survived"])
        self.assertEqual(survivor["depth"], max(depths))

    def test_buchanan_scales_the_norms_up(self):
        buchanan = corpora.load_buchanan()
        self.assertGreater(len(buchanan), 3500)
        self.assertGreater(buchanan.cells, 24000)

    def test_root_forms_are_not_two_predicates(self):
        """`leaving` and `leave` are one property, not two."""
        rooted = corpora.load_buchanan(root_forms=True)
        surface = corpora.load_buchanan(root_forms=False)
        self.assertLess(rooted.predicates, surface.predicates / 2)

    def test_each_attribute_owns_the_rivals_it_removed(self):
        """`depth` is the index of the step that removed the candidate.

        Counting from the terms alone is off by one exactly when a class was
        named, which drew the rivals of `stripes` hanging off `animal`.
        """
        found = self.identifier.identify(
            "what kind of animal has stripes and eats meat and is fast")
        terms = [step["term"] for step in found.steps]
        self.assertEqual(terms[0], "animal")           # the class step
        by_step = {}
        for entry in found.considered:
            if not entry["survived"]:
                by_step.setdefault(entry["depth"], []).append(entry["name"])
        self.assertNotIn(0, by_step)                   # the class removed none
        removed_by_stripes = by_step.get(terms.index("stripes"), [])
        self.assertTrue(removed_by_stripes)
        for name in removed_by_stripes:
            self.assertNotIn("stripes", self.identifier.stated.get(name, ()))

    def test_the_first_attribute_owns_rivals_at_depth_zero(self):
        """Depth is a step index, so the first attribute's rivals are depth 0.

        The page read that as `entry.depth || 1`, and zero is falsy, so every
        rival of the first attribute was drawn on the second: `round` appeared
        to rule out nothing and `hexagons` carried ten.
        """
        found = self.identifier.identify("what is round with hexagons")
        self.assertEqual([step["term"] for step in found.steps],
                         ["round", "hexagons"])
        depths = {e["depth"] for e in found.considered if not e["survived"]}
        self.assertIn(0, depths)
        self.assertIn(1, depths)

    def test_properties_are_taken_general_before_specific(self):
        """Broadest first, so each step narrows visibly instead of the first
        one answering the whole question."""
        found = self.identifier.identify("what is round with hexagons")
        remaining = [step["remaining"] for step in found.steps]
        self.assertEqual(remaining, sorted(remaining, reverse=True))
        self.assertGreater(remaining[0], remaining[-1])

    def test_rivals_are_the_nearest_misses_not_the_alphabet(self):
        """Sorting by name gave `ambulance, accordion, antelope`."""
        found = self.identifier.identify("what is round with hexagons")
        self.assertEqual([c.name for c in found.candidates], ["football"])
        rivals = {e["name"] for e in found.considered if not e["survived"]}
        self.assertTrue(rivals & {"ball", "balloon", "frisbee"}, rivals)
        self.assertNotIn("ambulance", rivals)
        self.assertNotIn("accordion", rivals)

    def test_nothing_matching_is_reported_as_absent(self):
        found = self.identifier.identify("what kind of dog has feathers")
        self.assertEqual(found.verdict, "NO_MATCH")
        self.assertIn("Absent", found.note)


@requires_norms
@requires_store
class ProfileTests(unittest.TestCase):
    """R17: the same trie walked up instead of down."""

    @classmethod
    def setUpClass(cls):
        cls.identifier = Identifier(STORE)
        cls.profiles = Profiles(cls.identifier)

    @classmethod
    def tearDownClass(cls):
        cls.identifier.close()

    def test_the_walk_recovers_exactly_what_was_stored(self):
        """The retrieval claim, on every individual in the corpus.

        If walking a leaf back to the origin returned anything but the
        predicate set the individual was stored with, the trie would be a
        lossy index rather than the storage itself, and every compression
        figure in this experiment would be measuring the wrong thing.
        """
        for name, stored in self.profiles.stated.items():
            if not stored:
                continue
            walked = {held.predicate for held in self.profiles.walk_up(name)}
            self.assertEqual(walked, set(stored), name)

    def test_sharing_climbs_towards_the_origin(self):
        """The compression, seen from inside one branch: the predicates
        nearest the leaf are shared with nobody, the ones nearest the origin
        with most of the corpus."""
        climb = [held.shared for held in self.profiles.walk_up("blue whale")]
        self.assertEqual(climb, sorted(climb))
        self.assertEqual(climb[0], 1)
        self.assertGreater(climb[-1], 20)

    def test_segments_collapse_the_nodes_where_nothing_branched(self):
        found = self.profiles.describe("blue whale")
        self.assertLess(len(found.segments), len(found.path))
        # every predicate survives the collapse, in order
        flat = [p for segment in found.segments for p in segment.predicates]
        self.assertEqual(flat, [held.predicate for held in found.path])
        # and every segment but the last ends where something left
        self.assertTrue(all(s.dropped for s in found.segments[:-1]))

    def test_a_stated_attribute_is_held(self):
        answer = self.profiles.verify("killer whale", ["flippers"])
        self.assertEqual(answer.verdict, "HELD")
        self.assertEqual(answer.predicate, "flippers")

    def test_a_scored_zero_is_a_denial_and_not_a_silence(self):
        """AwA2 scored every class on every attribute, so "is a blue whale
        furry" has an answer and it is no."""
        answer = self.profiles.verify("blue whale", ["furry"])
        self.assertEqual(answer.verdict, "DENIED")
        self.assertEqual(self.profiles.verify("lion", ["stripes"]).verdict,
                         "DENIED")

    def test_a_property_phrased_as_a_denial_is_read_as_one(self):
        """The norms state `cannot fly` of a penguin. Matching "fly" against
        it and reporting a yes is the one way this can be confidently wrong."""
        answer = self.profiles.verify("penguin", ["fly"])
        self.assertEqual(answer.verdict, "DENIED")
        self.assertIn("cannot fly", answer.predicate)

    def test_the_taxonomy_answers_when_the_norms_are_silent(self):
        answer = self.profiles.verify("robin", ["fly"])
        self.assertEqual(answer.verdict, "INHERITED")
        self.assertTrue(answer.source.startswith("bird"), answer.source)
        self.assertGreater(answer.distance, 0)

    def test_a_class_is_answered_by_the_kinds_beneath_it(self):
        """"Is a whale furry" was silence: `whale` carries 27 properties and
        none of them mention fur, while four kinds of whale sit under it with
        the attribute scored and denied."""
        answer = self.profiles.verify("whale", ["furry"])
        self.assertEqual(answer.verdict, "DENIED")
        self.assertEqual(answer.source, "kinds")
        said = {member["name"] for member in answer.members}
        self.assertIn("blue whale", said)
        self.assertIn("killer whale", said)

    def test_a_foil_is_not_a_denial_however_true_it_sounds(self):
        """This asserted DENIED for `dog / spots` on the grounds that "the
        norms scored that". They did not. `dog` is COMPS' *taxonomic foil* for
        `hyena has spots on its body` — a near miss sampled out of a
        1.58%-dense free listing, never a judgement about dogs. That it
        happens to be true of the typical dog is exactly what made it
        convincing, and why the test encoded the bug rather than catching it.

        The positive half is untouched: a feature somebody listed really is an
        assertion, and `dalmatian has spots` is one.
        """
        self.assertEqual(self.profiles.verify("dog", ["spots"]).verdict,
                         "UNRECORDED")
        self.assertEqual(self.profiles.verify("dalmatian", ["spots"]).verdict,
                         "HELD")

    def test_a_class_that_does_not_agree_with_itself_says_so(self):
        answer = self.profiles.verify("whale", ["sing"])
        self.assertEqual(answer.verdict, "HELD")     # stated of whale itself
        self.assertEqual(answer.source, "stated")

    def test_subtypes_come_from_the_taxonomy_not_the_name(self):
        kinds = self.profiles.subtypes("whale")
        self.assertIn("blue whale", kinds)
        self.assertIn("dolphin", kinds)              # not called a whale
        self.assertNotIn("whale", kinds)
        self.assertEqual(self.profiles.subtypes("dalmatian"), [])

    def test_what_is_neither_stated_nor_denied_is_absent(self):
        answer = self.profiles.verify("blue whale", ["telephone"])
        self.assertEqual(answer.verdict, "UNRECORDED")
        self.assertIn("Absent", answer.detail)

    def test_the_participle_a_question_asks_in_reaches_the_stored_form(self):
        """The norms say `spots`; a person asks `spotted`."""
        self.assertEqual(Identifier.stem("spotted"), Identifier.stem("spots"))
        self.assertEqual(Identifier.stem("striped"), Identifier.stem("stripes"))
        self.assertEqual(self.profiles.verify("dalmatian", ["spotted"]).verdict,
                         "HELD")

    def test_stemming_did_not_reopen_the_traps_it_was_narrowed_for(self):
        self.assertNotEqual(Identifier.stem("striven"), Identifier.stem("stripes"))
        self.assertNotEqual(Identifier.stem("truncated"), Identifier.stem("trunk"))

    def test_ancestors_are_read_nearest_first_without_repeating_a_fact(self):
        levels = self.profiles.ancestry("blue whale")
        self.assertEqual([level.distance for level in levels],
                         sorted(level.distance for level in levels))
        seen = [(f["relation"], f["object"])
                for level in levels for f in level.all_facts]
        self.assertEqual(len(seen), len(set(seen)))

    def test_the_replay_shows_what_answered_and_not_only_the_verdict(self):
        """The page promises every answer is replayable, and this one was not:
        the steps walked whale's own branch and then announced a denial, with
        the four kinds that actually denied it appearing nowhere."""
        found = self.profiles.describe("whale")
        found.asked = self.profiles.verify("whale", ["furry"])
        steps = IdentifyingEngine._walk_steps(found)
        walked = [step["concept"] for step in steps]
        self.assertIn("furry?", walked)
        self.assertIn("blue whale", walked)
        self.assertIn("killer whale", walked)
        self.assertEqual(sum(1 for s in steps if s["kind"] == "block"), 4)
        self.assertEqual(walked[-1], "whale")

    def test_a_witness_is_drawn_once_where_it_answered(self):
        """`dolphin` is both a trie neighbour of whale and one of the kinds
        that denied `furry`. The tree keys nodes by name, so drawing it twice
        left the replay lighting up the wrong one."""
        found = self.profiles.describe("whale")
        found.asked = self.profiles.verify("whale", ["furry"])
        tree = IdentifyingEngine._as_identification(
            "is a whale furry", "verify", found)
        drawn = [entry["name"] for entry in tree["considered"]]
        self.assertEqual(len(drawn), len(set(drawn)))
        at_the_question = {entry["name"] for entry in tree["considered"]
                           if entry["depth"] == len(found.segments)}
        self.assertIn("dolphin", at_the_question)

    def test_a_profile_asks_nothing_so_it_gets_no_question_node(self):
        found = self.profiles.describe("robin")
        walked = [step["concept"]
                  for step in IdentifyingEngine._walk_steps(found)]
        self.assertEqual(walked[-1], "robin")
        self.assertFalse([node for node in walked if node.endswith("?")])

    def test_the_neighbours_at_a_branch_point_are_the_nearest_misses(self):
        found = self.profiles.describe("blue whale")
        left = {name for segment in found.segments for name in segment.dropped}
        self.assertTrue(left & {"dolphin", "humpback whale", "killer whale"},
                        left)


@requires_norms
@requires_store
class ProfileRoutingTests(unittest.TestCase):
    """Which questions the walk takes, and which it hands back."""

    @classmethod
    def setUpClass(cls):
        cls.identifier = Identifier(STORE)
        cls.profiles = Profiles(cls.identifier)

    @classmethod
    def tearDownClass(cls):
        cls.identifier.close()

    def test_questions_about_a_named_thing_are_taken(self):
        for question, mode, name in (
                ("what attributes does a blue whale have", "profile",
                 "blue whale"),
                ("what properties does a dalmatian have", "profile",
                 "dalmatian"),
                ("what is a robin like", "profile", "robin"),
                ("describe a penguin", "profile", "penguin"),
                ("is a blue whale furry", "verify", "blue whale"),
                ("does a killer whale have flippers", "verify",
                 "killer whale")):
            routed = self.profiles.route(question)
            self.assertIsNotNone(routed, question)
            self.assertEqual(routed[0], mode, question)
            self.assertEqual(routed[1], name, question)

    def test_the_longest_name_wins(self):
        """`blue whale` and `whale` are both concepts; the question asked
        about the first, and v684's parser reading `whale` is what sent this
        answer to the wrong animal."""
        self.assertEqual(
            self.profiles.route("is a blue whale furry")[1], "blue whale")
        self.assertEqual(
            self.profiles.route("is a killer whale fierce")[1], "killer whale")

    def test_everything_else_is_handed_back(self):
        for question in ("what can a violin do",
                         "what does a dog's owner need",
                         "what kind of dog has spots",
                         "what is round with hexagons",
                         "is a zzzqqq furry",
                         "where do you find a hammer"):
            self.assertIsNone(self.profiles.route(question), question)


if __name__ == "__main__":
    unittest.main()
