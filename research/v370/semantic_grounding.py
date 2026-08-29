
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from conceptnet import (
    is_reasonable_anchor,
    normalize_concept_text,
)


@dataclass(frozen=True)
class GroundingMatch:
    construction: str
    anchor: str
    relation: str
    source: str
    target: str


class SemanticGrounder:
    """
    Grammar -> ConceptNet grounding with explicit anchor quality controls.

    The old implementation grounded arbitrary punctuation fragments such as
    "'" and "''", which caused unrelated matches like "beer o'clock". This
    layer:
      1. rejects invalid anchors,
      2. normalizes labels,
      3. requires exact ConceptNet endpoint equality,
      4. records rejected anchors for diagnostics.
    """

    def __init__(self,conceptnet):
        self.conceptnet=conceptnet
        self.rejected_anchors=[]

    def ground(self,grammar):
        grounded=[]
        by_construction={}

        for construction in grammar.constructions:
            matches=[]

            anchors=[
                a for a in construction.semantic_anchors
                if is_reasonable_anchor(a)
            ]

            rejected=[
                a for a in construction.semantic_anchors
                if not is_reasonable_anchor(a)
            ]
            for anchor in rejected:
                self.rejected_anchors.append(
                    {
                        "construction":construction,
                        "anchor":anchor,
                        "reason":"invalid_anchor",
                    }
                )

            for anchor in anchors:
                edges=self.conceptnet.edges_for_label(
                    anchor,
                    limit=12,
                )

                for edge in edges:
                    # Final exact check here too, so this layer is safe even
                    # if a different ConceptNet adapter is supplied.
                    needle=normalize_concept_text(anchor)
                    if (
                        normalize_concept_text(edge.source)!=needle
                        and normalize_concept_text(edge.target)!=needle
                    ):
                        continue

                    match=GroundingMatch(
                        construction=(
                            f"{construction.lhs} -> "
                            + " ".join(construction.rhs)
                        ),
                        anchor=anchor,
                        relation=edge.relation,
                        source=edge.source,
                        target=edge.target,
                    )
                    matches.append(match)
                    grounded.append(match)

                    if len(matches)>=6:
                        break

                if len(matches)>=6:
                    break

            by_construction[
                f"{construction.lhs} -> "
                + " ".join(construction.rhs)
            ]=matches

        covered=sum(
            int(bool(v))
            for v in by_construction.values()
        )
        coverage=covered/max(
            1,
            len(by_construction),
        )

        return by_construction,{
            "grounded_constructions":covered,
            "total_constructions":len(by_construction),
            "semantic_coverage":coverage,
            "matches":len(grounded),
            "rejected_anchors":len(
                self.rejected_anchors
            ),
        }
