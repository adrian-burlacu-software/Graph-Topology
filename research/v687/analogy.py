"""Structure mapping, and only where the data can carry it.

    fins are to a fish as what is to a bird

The v686 audit put analogy on the skip list on the grounds that the data could
not support it. That was asserted without testing and the reason was wrong.
Testing it separates two cases cleanly:

    the scraped graph   cannot. `part_of` there is largely taxonomy misfiled
                        as mereology -- `acanthisitta.n.01 part_of rifleman
                        bird.n.01` -- and `bird has_part` returns `band`,
                        `broken wing`, `enough room`, `feast`. Mapping
                        relational structure across two concepts whose
                        relations are that noisy is the plausible-nonsense
                        machine the audit warned about.

    the norms           can. The vocabulary is closed, and XCSLB ships a
                        *feature type* for every property -- visual
                        perceptual, functional, encyclopedic, taxonomic.

That type is what makes this possible, because it is the missing half of a
role. Gentner's point about analogy is that it maps relations rather than
attributes, and a bare feature norm is all attributes. But `has fins` and
`has wings` are both visual-perceptual properties carried by a comparable
share of their own class, and that pairing -- same kind of property, same
standing within the concept -- is a role in everything but name.

So a mapping is scored on three things:

    kind        the two properties must be the same type of property. A
                functional property does not answer a visual one.
    standing    how distinctive each is within its own concept, by the same
                carrier count `anti_coverage` uses. `has fins` is to a fish
                roughly what `has wings` is to a bird: not universal, not
                unique, and the same order of rare.
    exclusive   the answer must not be something the source has too, or the
                analogy is a restatement rather than a mapping.

This is deliberately not proportional analogy over free text, which is the
thing that does not work. It is one bounded claim: within a closed elicited
vocabulary, a role can be approximated by type plus standing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import corpora
from .identify import Identifier

#: How many mappings to offer. More than a few and it stops being an answer.
OFFERED = 4

#: A candidate must be at least this different in standing before it is
#: preferred on standing alone -- below it, two properties are equally good
#: and the tie is broken by how well known each is.
CLOSE = 0.05


@dataclass
class Mapping:
    """One proposed correspondence, and the grounds for it."""
    property: str
    kind: str | None
    standing: float
    distance: float
    carriers: int

    def as_dict(self) -> dict[str, Any]:
        return {"property": self.property, "kind": self.kind,
                "standing": round(self.standing, 3),
                "distance": round(self.distance, 3), "carriers": self.carriers}


@dataclass
class Analogy:
    """A : B :: ? : D, with the reasoning kept."""
    source: str
    source_property: str | None
    target: str
    kind: str | None = None
    standing: float = 0.0
    mappings: list[Mapping] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"source": self.source, "source_property": self.source_property,
                "target": self.target, "kind": self.kind,
                "standing": round(self.standing, 3), "note": self.note,
                "mappings": [m.as_dict() for m in self.mappings]}


class Analogies:
    """Role mapping over the elicited norms."""

    def __init__(self, profiles) -> None:
        self.profiles = profiles
        self.stated = profiles.stated
        self.kinds = corpora.feature_types()
        self._carriers: dict[str, int] = {}
        for predicates in self.stated.values():
            for predicate in predicates:
                self._carriers[predicate] = self._carriers.get(predicate, 0) + 1

    def carriers(self, predicate: str) -> int:
        return self._carriers.get(predicate, 0)

    def standing(self, predicate: str, name: str) -> float:
        """How distinctive a property is *within its own concept*, 0 to 1.

        Not global rarity: the comparison is between one concept's properties
        and another's, so each is ranked against its own. A property no other
        concept shares stands at 1; one everything shares stands near 0.
        """
        held = self.stated.get(name, frozenset())
        if not held:
            return 0.0
        counts = sorted(self.carriers(p) for p in held)
        mine = self.carriers(predicate)
        rarer = sum(1 for count in counts if count > mine)
        return rarer / len(counts)

    def properties(self, name: str) -> frozenset[str]:
        """What a concept carries -- or, for a class, what its kinds share.

        `fins : fish :: ? : bird` needs `fish` and `bird` to have properties,
        and neither is one of XCSLB's 521 concepts: the norms cover cod and
        robin, not the categories above them. A class stands for what a
        quarter of its kinds agree on, which is the same core `contrast`
        measures typicality against.
        """
        if self.profiles.knows(name):
            return self.stated[name]
        kinds = self.profiles.subtypes(name)
        if not kinds:
            return frozenset()
        counts: dict[str, int] = {}
        for kind in kinds:
            for predicate in self.stated.get(kind, ()):
                counts[predicate] = counts.get(predicate, 0) + 1
        floor = max(2, len(kinds) // 4)
        return frozenset(p for p, n in counts.items() if n >= floor)

    def standing_in(self, predicate: str, held: frozenset[str]) -> float:
        if not held:
            return 0.0
        mine = self.carriers(predicate)
        return sum(1 for p in held if self.carriers(p) > mine) / len(held)

    def solve(self, source_property: str, source: str, target: str) -> Analogy:
        """`fins` : fish :: ? : bird."""
        found = Analogy(source=source, target=target,
                        source_property=None)
        mine, theirs_all = self.properties(source), self.properties(target)
        if not mine or not theirs_all:
            missing = source if not mine else target
            found.note = (f"Analogy is only attempted over the norms, where "
                          f"the property vocabulary is closed, and “{missing}” "
                          f"is neither one of their concepts nor a class with "
                          f"kinds among them.")
            return found
        anchor = Identifier._hit(source_property, mine)
        if anchor is None:
            found.note = (f"The norms record nothing like “{source_property}” "
                          f"of {source}, so there is no role to map.")
            return found
        found.source_property = anchor
        found.kind = self.kinds.get(anchor)
        found.standing = self.standing_in(anchor, mine)

        theirs = theirs_all - mine
        weighed: list[Mapping] = []
        for candidate in theirs:
            kind = self.kinds.get(candidate)
            if found.kind and kind != found.kind:
                continue                  # a role is a kind of property first
            standing = self.standing_in(candidate, theirs_all)
            weighed.append(Mapping(
                property=candidate, kind=kind, standing=standing,
                distance=abs(standing - found.standing),
                carriers=self.carriers(candidate)))
        # Closest standing wins. Ties are then broken by predicate frame --
        # `has fins` is answered by `has big wings`, not by `can stand on one
        # leg` -- and only then by how well known the property is. The frame
        # matters most where standing goes flat, which is exactly the
        # class-core case: every property of a class core is carried by a
        # similar share of its kinds, so the standings collapse together and
        # something else has to decide.
        frame = anchor.split()[0]
        weighed.sort(key=lambda m: (round(m.distance / CLOSE),
                                    0 if m.property.startswith(frame) else 1,
                                    -m.carriers, m.property))
        found.mappings = weighed[:OFFERED]
        if not weighed:
            found.note = (f"{target} has no {found.kind or 'comparable'} "
                          f"property the norms record that {source} lacks.")
        return found
