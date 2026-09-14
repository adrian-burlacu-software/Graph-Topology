"""The link-type table and the two walks.

No store and no engine. What is pinned is that every list the rules used to
keep for themselves comes out of the table exactly as it was, and that the
walks keep the order and bounds the searches they replaced had.
"""
from __future__ import annotations

import unittest

from research.v687 import links, rules, walks
from research.v687.links import LINKS, link, named


class TableTests(unittest.TestCase):
    def test_r2_inheritable_is_what_it_was(self):
        self.assertEqual(rules.INHERITABLE, frozenset({
            "capable_of", "has_property", "has_a", "has_part",
            "receives_action", "used_for", "desires", "not_desires",
            "not_capable_of", "not_has_property", "has_prerequisite",
            "has_subevent", "motivated_by_goal", "causes", "at_location",
            "part_of", "has_attribute", "entails", "located_near"}))
        self.assertEqual(set(rules.NOT_INHERITABLE), {
            "made_of", "similar_to", "instance_of", "created_by", "symbol_of",
            "defined_as", "manner_of"})

    def test_r7_gated_is_what_it_was(self):
        self.assertEqual(rules.GATED, frozenset({
            "related_to", "has_context", "form_of", "derived_from",
            "etymologically_related_to", "synonym", "antonym", "has_sense",
            "definition", "usage_count", "distinct_from", "verb_group"}))

    def test_r3_negations_and_r9_families(self):
        self.assertEqual(rules.NEGATIONS, {
            "not_capable_of": "capable_of", "not_has_property": "has_property",
            "not_desires": "desires"})
        self.assertEqual(set(rules.FAMILIES), {
            frozenset({"has_a", "has_part"}),
            frozenset({"at_location", "located_near"}),
            frozenset({"has_property", "has_attribute"})})
        self.assertEqual(rules.RANGES, {
            "at_location": "physical entity.n.01",
            "located_near": "physical entity.n.01"})

    def test_r29_stored_directions_and_r22_converses(self):
        from research.v687.inverse import PAIRS
        from research.v687.reason import Reasoner
        self.assertEqual(Reasoner.SENSE_TAGGED, {
            "has_part": ("part_of", False), "has_a": ("part_of", False),
            "part_of": ("part_of", True), "similar_to": ("similar_to", None),
            "causes": ("causes", False), "entails": ("entails", False)})
        self.assertEqual(PAIRS, {"has_part": "part_of", "part_of": "has_part",
                                 "has_a": "part_of"})

    def test_r27_partitions(self):
        from research.v687.reason import Reasoner
        self.assertEqual(Reasoner.PARTITIONS, (
            "plant.n.02", "animal.n.01", "person.n.01", "artifact.n.01",
            "abstraction.n.06"))
        self.assertEqual(Reasoner.NOT_REALLY_DISJOINT,
                         (("person.n.01", "animal.n.01"),))

    def test_e1_qualities_and_t3_time(self):
        self.assertEqual(named(lambda one: one.quality), frozenset({
            "has_property", "has_attribute", "not_has_property"}))
        self.assertEqual(named(lambda one: one.in_time), frozenset({
            "has_property", "not_has_property", "has_attribute",
            "at_location", "did_not"}))
        self.assertEqual(named(lambda one: one.exclusive),
                         frozenset({"at_location"}))

    def test_a_relation_with_no_row_is_a_plain_link(self):
        plain = link("chases")
        self.assertFalse(plain.inherited or plain.gated or plain.converse)
        self.assertNotIn("chases", LINKS)

    def test_inherited_and_gated_never_meet(self):
        for name, one in LINKS.items():
            self.assertFalse(one.inherited and (one.gated or one.why_not),
                             name)

    def test_the_store_never_writes_what_a_conversation_does(self):
        self.assertTrue(link("did_not").episodic)
        self.assertTrue(link("before").episodic)
        self.assertNotIn("before", rules.INHERITABLE)
        self.assertEqual(links.link("before").converse, "after")

    def test_dimensions_are_rows(self):
        axes = [one.axis for one in LINKS.values() if one.axis]
        self.assertEqual(axes, ["north-south", "east-west", "vertical",
                                "lateral", "size"])
        north = link("north")
        self.assertEqual((north.converse, north.across, north.compass),
                         ("south", "east-west", True))
        self.assertTrue(north.transitive and north.antisymmetric)


class WalkTests(unittest.TestCase):
    PARENTS = {"beagle": ["hound"], "hound": ["dog"], "dog": ["canine",
               "pet"], "canine": ["animal"], "pet": ["animal"], "animal": []}

    def test_levels_nearest_first_each_once(self):
        found = list(walks.levels("beagle", self.PARENTS.get))
        self.assertEqual(found, [(0, ["beagle"]), (1, ["hound"]),
                                 (2, ["dog"]), (3, ["canine", "pet"]),
                                 (4, ["animal"])])

    def test_levels_bounds(self):
        self.assertEqual(len(list(walks.levels("beagle", self.PARENTS.get,
                                                max_depth=2))), 3)
        # more than four seen: the level that crossed the bound is not given
        self.assertEqual([distance for distance, _ in walks.levels(
            "beagle", self.PARENTS.get, max_nodes=4)], [0, 1, 2])

    def test_levels_is_lazy(self):
        asked = []

        def parents(node):
            asked.append(node)
            return self.PARENTS[node]

        walk = walks.levels("beagle", parents)
        next(walk)
        self.assertEqual(asked, [])
        next(walk)
        self.assertEqual(asked, ["beagle"])

    def test_path_keeps_what_each_step_went_along(self):
        edges = {"a": [("n", "b")], "b": [("e", "c"), ("s", "a")],
                 "c": []}
        self.assertEqual(walks.path("a", "c", lambda node: edges[node]),
                         [("n", "b"), ("e", "c")])
        self.assertIsNone(walks.path("c", "a", lambda node: edges[node]))
        self.assertIsNone(walks.path("a", "a", lambda node: edges[node]))


if __name__ == "__main__":
    unittest.main()
