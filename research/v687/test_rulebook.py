"""The rulebook: every named rule is one row, and the code agrees with it.

Run: python -m unittest research.v687.test_rulebook -v
"""
from __future__ import annotations

import dataclasses
import re
import unittest
from pathlib import Path

from research.v687 import rulebook, truth
from research.v687.links import LINKS, LinkType

RESEARCH = Path(__file__).resolve().parents[1]

#: A rule's name, quoted in code.
CITED = re.compile(r"[\"'](R\d{1,2}|T[1-6]|S[1-4]|E[12]|I1)[\"']")


class RowTests(unittest.TestCase):
    def test_every_link_a_rule_reads_is_a_link_type(self):
        for one in rulebook.RULES.values():
            for name in one.links:
                self.assertIn(name, LINKS, one.name)

    def test_every_field_is_a_link_type_field(self):
        fields = {one.name for one in dataclasses.fields(LinkType)}
        for one in rulebook.RULES.values():
            if one.field:
                self.assertIn(one.field, fields, one.name)

    def test_every_truth_function_is_in_truth(self):
        for one in rulebook.RULES.values():
            if one.truth:
                self.assertTrue(callable(getattr(truth, one.truth, None)),
                                one.name)

    def test_every_text_formats(self):
        for one in rulebook.RULES.values():
            self.assertTrue(one.text(), one.name)

    def test_only_s3_and_motives_are_flagged(self):
        self.assertEqual({name for name, one in rulebook.RULES.items()
                          if one.flagged}, {"S3", "motives"})


class CodeAgreesTests(unittest.TestCase):
    def test_every_rule_the_code_cites_has_a_row(self):
        for version in ("v687", "v688", "v689"):
            for path in sorted((RESEARCH / version).glob("*.py")):
                if path.name.startswith("test_"):
                    continue
                for name in CITED.findall(path.read_text(encoding="utf-8")):
                    self.assertIn(name, rulebook.RULES, f"{path.name}: {name}")

    def test_the_pages_text_is_the_rows_text(self):
        from research.v687 import relevance, rules, server
        for table in (rules.RULE_TEXT, relevance.RULE_TEXT,
                      server.V686_RULES, server.V687_RULES):
            for name, text in table.items():
                self.assertEqual(text, rulebook.rule(name).text(), name)

    def test_the_parameters_are_the_code_s(self):
        from research.v687 import profile, relevance, rules
        self.assertEqual(rules.DECAY, rulebook.rule("R5").parameters["decay"])
        self.assertEqual(rules.FLOOR, rulebook.rule("R5").parameters["floor"])
        self.assertEqual(rules.BREADTH_LIMIT,
                         rulebook.rule("R12").parameters["breadth_limit"])
        self.assertEqual(relevance.SIBLING_LIMIT,
                         rulebook.rule("R14").parameters["sibling_limit"])
        self.assertEqual(profile.CORROBORATION_FLOOR,
                         rulebook.rule("R19").parameters["floor"])
        self.assertEqual(profile.CORROBORATION_MIN_KINDS,
                         rulebook.rule("R19").parameters["min_kinds"])


if __name__ == "__main__":
    unittest.main()
