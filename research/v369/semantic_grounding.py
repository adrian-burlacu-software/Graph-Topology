
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class GroundingMatch:
    construction: str
    anchor: str
    relation: str
    source: str
    target: str


class SemanticGrounder:
    """
    Grounds induced grammar constructions against the user's ConceptNet
    semantic graph by following lexical anchors from the construction.
    """

    def __init__(self, conceptnet):
        self.conceptnet=conceptnet

    def ground(self, grammar):
        grounded=[]
        by_construction={}

        for c in grammar.constructions:
            matches=[]
            for anchor in c.semantic_anchors:
                edges=self.conceptnet.edges_for_label(
                    anchor,
                    limit=12,
                )
                for edge in edges:
                    match=GroundingMatch(
                        construction=(
                            f"{c.lhs} -> "
                            + " ".join(c.rhs)
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
                f"{c.lhs} -> {' '.join(c.rhs)}"
            ]=matches

        coverage=sum(
            int(bool(x))
            for x in by_construction.values()
        )/max(1,len(by_construction))

        return by_construction,{
            "grounded_constructions":sum(
                int(bool(x))
                for x in by_construction.values()
            ),
            "total_constructions":len(
                by_construction
            ),
            "semantic_coverage":coverage,
            "matches":len(grounded),
        }
