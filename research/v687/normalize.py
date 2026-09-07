"""Relation and node normalization for the V633 semantic graph.

Every rule here was derived by querying the database rather than assumed, and
the query that justifies each one is quoted next to it. Nothing is normalized
"just in case": two of the transforms below turn out to be pure deduplication
and one of them is genuinely lossy, and they are kept apart for that reason.

What the data actually looks like
---------------------------------
54 relations, all snake_case. There is no `en:is_a` or `type` variant to fold
in -- the importer already emitted one spelling per relation. The only family
is `dbpedia/*`.

Two node namespaces carry every edge:

    en:<lemma>                       1,174,169 nodes, spaces, no POS suffix
    wn:synset:<lemma>.<pos>.<NN>       117,659 nodes, underscores, pos in nvsar

Both are already lowercase with no stray whitespace, so case folding and
trimming are no-ops and are not implemented.

The overlap is total: all 86,571 distinct WordNet lemmas also exist as `en:`
concepts. The two graphs describe the same vocabulary and share no node ids,
so an unnormalized run scores them as two disjoint ontologies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator

Triple = tuple[str, str, str]

#: Relations that are the stored inverse of another, carrying no new fact.
#:
#:     is_a=319,232  has_subtype=97,666
#:     has_subtype inverted and already present as is_a: 97,665  (of 97,666)
#:     part_of=34,696  has_part=22,187
#:     has_part inverted and already present as part_of: 22,187  (of 22,187)
#:
#: Rewriting `A has_subtype B` to `B is_a A` therefore deletes 119,852 edges
#: and adds one. Keeping both spellings splits the taxonomy across two
#: relations and puts each fact on both of its endpoints.
INVERSE_RELATIONS: dict[str, str] = {
    "has_subtype": "is_a",
    "has_part": "part_of",
}

#: Relations whose direction carries no meaning. ConceptNet stores them
#: inconsistently -- similar_to 89.8% both ways, synonym 34.3%, related_to
#: 7.6% -- so the pair is sorted to a single canonical direction.
SYMMETRIC_RELATIONS: frozenset[str] = frozenset({
    "synonym", "similar_to", "related_to", "antonym", "distinct_from",
    "verb_group", "etymologically_related_to",
})

#: Spellings of a relation that mean the same thing. `dbpedia/genus` is a
#: biological type assertion; `defined_as` is a definitional is_a. Kept
#: deliberately small -- `instance_of` is NOT folded into `is_a`, because
#: instance-of and subclass-of are a real ontological distinction and the
#: paper's trie is built over subclass structure.
RELATION_ALIASES: dict[str, str] = {
    "dbpedia/genus": "is_a",
    "defined_as": "is_a",
}

#: Wiktionary and Wikipedia meta-pages. These are documents about the
#: dictionary, not concepts in it: `en:appendix:animals` is a word list.
#: 631 edges across all of them.
META_NAMESPACES: tuple[str, ...] = (
    "en:appendix:", "en:w:", "en:thesaurus:", "en:wiktionary:", "en:index:",
    "en:wikipedia:", "en:special:", "en:mlp:", "en:rs:", "en:wikibooks:",
)

_SYNSET = re.compile(r"^wn:synset:(?P<lemma>.+)\.(?P<pos>[nvsar])\.(?P<sense>\d+)$")

#: WordNet splits adjectives into head (`a`) and satellite (`s`) senses. The
#: distinction is internal bookkeeping about adjective clusters, not a
#: different part of speech. 10,693 satellites fold into `a`.
_SATELLITE = "s"


@dataclass(frozen=True)
class Normalization:
    """A named, explicit set of transforms.

    Each flag is separately reportable so the experiment can show what every
    rule is worth rather than presenting one opaque "cleaned" number.
    """

    name: str
    drop_meta_namespaces: bool = False
    drop_redundant_inverses: bool = False
    canonicalize_symmetric: bool = False
    apply_relation_aliases: bool = False
    underscore_to_space: bool = False
    satellite_to_adjective: bool = False
    strip_sense_number: bool = False
    strip_part_of_speech: bool = False
    bridge_senses_to_lemmas: bool = False

    def node(self, value: str) -> str:
        """Normalize one node identifier."""
        match = _SYNSET.match(value)
        if match is None:
            return value
        lemma, pos, sense = match["lemma"], match["pos"], match["sense"]
        if self.underscore_to_space:
            lemma = lemma.replace("_", " ")
        if self.bridge_senses_to_lemmas:
            return f"en:{lemma}"
        if self.strip_part_of_speech:
            return f"wn:synset:{lemma}"
        if self.satellite_to_adjective and pos == _SATELLITE:
            pos = "a"
        if self.strip_sense_number:
            return f"wn:synset:{lemma}.{pos}"
        return f"wn:synset:{lemma}.{pos}.{sense}"

    def triple(self, subject: str, relation: str, obj: str) -> Triple | None:
        """Normalize one edge, or drop it by returning None."""
        if self.drop_meta_namespaces and (
            subject.startswith(META_NAMESPACES) or obj.startswith(META_NAMESPACES)
        ):
            return None
        if self.apply_relation_aliases:
            relation = RELATION_ALIASES.get(relation, relation)
        if self.drop_redundant_inverses and relation in INVERSE_RELATIONS:
            subject, relation, obj = obj, INVERSE_RELATIONS[relation], subject
        subject, obj = self.node(subject), self.node(obj)
        if subject == obj:
            # Merging namespaces can make an edge point at itself, most often
            # `en:dog has_sense wn:synset:dog.n.01`. A self-predicate is not a
            # fact about the individual and would allocate a node for nothing.
            return None
        if self.canonicalize_symmetric and relation in SYMMETRIC_RELATIONS:
            subject, obj = min(subject, obj), max(subject, obj)
        return subject, relation, obj

    def apply(self, triples: Iterable[Triple]) -> Iterator[Triple]:
        """Normalize a stream of edges. Deduplication is the caller's job:
        `substrate.load` groups predicates into sets, which absorbs it."""
        for subject, relation, obj in triples:
            result = self.triple(subject, relation, obj)
            if result is not None:
                yield result


#: What the first V683 run used: raw strings, straight from SQLite.
RAW = Normalization("raw")

#: Everything provably redundant, and nothing that loses a distinction.
#: Inverse relations are duplicates here by measurement, not by assumption;
#: symmetric canonicalization only merges an edge with its own reverse;
#: meta-namespaces are not concepts; satellite adjectives are adjectives;
#: underscores are WordNet's multiword separator and spaces are ConceptNet's.
SAFE = Normalization(
    "safe",
    drop_meta_namespaces=True,
    drop_redundant_inverses=True,
    canonicalize_symmetric=True,
    apply_relation_aliases=True,
    underscore_to_space=True,
    satellite_to_adjective=True,
)

#: SAFE plus collapsing WordNet's sense numbering. Lossy: `dog.n.01` (the
#: animal) and `dog.n.03` (a chap) become one concept.
SENSE_MERGED = Normalization(
    "sense_merged",
    drop_meta_namespaces=True,
    drop_redundant_inverses=True,
    canonicalize_symmetric=True,
    apply_relation_aliases=True,
    underscore_to_space=True,
    satellite_to_adjective=True,
    strip_sense_number=True,
)

#: SAFE plus fusing the two namespaces into one lexical vocabulary. This is
#: what joins WordNet's taxonomy to ConceptNet's assertions about the same
#: word. It is the most aggressive and the most lossy: `en:break` carries 75
#: WordNet senses, and all 75 become one individual.
LEMMA_BRIDGED = Normalization(
    "lemma_bridged",
    drop_meta_namespaces=True,
    drop_redundant_inverses=True,
    canonicalize_symmetric=True,
    apply_relation_aliases=True,
    underscore_to_space=True,
    satellite_to_adjective=True,
    bridge_senses_to_lemmas=True,
)

NORMALIZATIONS: dict[str, Normalization] = {
    normalization.name: normalization
    for normalization in (RAW, SAFE, SENSE_MERGED, LEMMA_BRIDGED)
}
