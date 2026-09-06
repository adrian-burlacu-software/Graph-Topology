"""Corpora of (individual, predicate set) drawn from the V633 semantic graph.

An individual is a subject concept. A predicate is a `(relation, object)` pair,
matching Appendix 3, where `Big` and `Spots` are the alphabet and the dogs are
the goal nodes.

Which relations count as predicates is a judgement call, so it is made once,
here, by name, and every slice is reported -- including the ones that weaken
the result. `LEXICAL_RELATIONS` are excluded from `attributes` because they
describe a word's spelling history rather than a thing's properties: knowing
that "dogged" is `derived_from` "dog" classifies the string, not the animal.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .normalize import RAW, Normalization

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = REPOSITORY_ROOT / "data" / "v633_full_semantic.sqlite"

#: Relations asserting something about the individual itself.
ATTRIBUTE_RELATIONS = (
    "is_a", "instance_of", "has_property", "has_attribute", "capable_of",
    "used_for", "at_location", "part_of", "has_a", "has_part", "made_of",
    "desires", "not_desires", "not_capable_of", "not_has_property",
    "has_prerequisite", "receives_action", "has_subevent", "motivated_by_goal",
    "causes", "causes_desire", "similar_to", "created_by", "located_near",
    "symbol_of", "entails", "manner_of", "defined_as",
)

#: Strict subtype/composition backbone.
TAXONOMY_RELATIONS = (
    "is_a", "instance_of", "has_subtype", "part_of", "has_part", "has_a",
)

#: Word-form relations: about orthography, not ontology.
LEXICAL_RELATIONS = (
    "form_of", "derived_from", "etymologically_related_to",
    "etymologically_derived_from", "has_sense", "synonym", "antonym",
    "verb_group", "has_context", "distinct_from",
)

#: Objects that are free text or bookkeeping rather than concepts.
NON_CONCEPT_RELATIONS = ("definition", "usage_count")

SLICES: dict[str, tuple[str, ...]] = {
    "attributes": ATTRIBUTE_RELATIONS,
    "taxonomy": TAXONOMY_RELATIONS,
    "attributes_plus_related_to": ATTRIBUTE_RELATIONS + ("related_to",),
}


@dataclass(frozen=True)
class Corpus:
    """An ontology as Appendix 3 sees it: individuals carrying predicate sets."""

    name: str
    items: tuple[tuple[str, frozenset], ...]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index):
        return self.items[index]

    @property
    def cells(self) -> int:
        """Flat storage cost: one cell per individual-predicate pair."""
        return sum(len(predicates) for _, predicates in self.items)

    @property
    def predicates(self) -> int:
        return len({p for _, predicates in self.items for p in predicates})

    def filter_min_predicates(self, minimum: int) -> "Corpus":
        return Corpus(
            f"{self.name}>={minimum}",
            tuple(item for item in self.items if len(item[1]) >= minimum),
        )

    def head(self, count: int) -> "Corpus":
        return Corpus(self.name, self.items[:count])


def load(
    database: Path = DEFAULT_DATABASE,
    relations: Sequence[str] = ATTRIBUTE_RELATIONS,
    name: str = "attributes",
    min_predicates: int = 1,
    normalization: Normalization = RAW,
) -> Corpus:
    """Read one relation slice out of the read-only semantic database.

    Relations are selected before normalization, so a slice names the relations
    as they are spelled in the database. `drop_redundant_inverses` may rewrite
    a selected relation into another one -- `has_subtype` becomes `is_a` -- so
    a slice should name both members of an inverse pair or neither.
    """
    if not database.is_file():
        raise FileNotFoundError(f"semantic database not found: {database}")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        placeholders = ",".join("?" * len(relations))
        rows = connection.execute(
            f"SELECT subject, relation, object FROM edges WHERE relation IN ({placeholders})",
            tuple(relations),
        )
        grouped: dict[str, set] = {}
        for subject, relation, obj in normalization.apply(rows):
            grouped.setdefault(subject, set()).add((relation, obj))
    finally:
        connection.close()
    items = tuple(
        (subject, frozenset(predicates))
        for subject, predicates in sorted(grouped.items())
        if len(predicates) >= min_predicates
    )
    return Corpus(f"{name}/{normalization.name}", items)


#: Table 1 of the paper, verbatim. The regression anchor for Figure 20.
PAPER_TABLE_1 = Corpus(
    "paper_table_1",
    (
        ("Border Collie", frozenset({"Big"})),
        ("Briard", frozenset({"Long Hair", "Big"})),
        ("Dalmatian", frozenset({"Big", "Spots"})),
        ("Jack Russell Terrier", frozenset({"Spots"})),
        ("Saint Bernard", frozenset({"Long Hair", "Big", "Loud"})),
    ),
)
