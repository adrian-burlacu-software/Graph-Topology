
from __future__ import annotations

import re


class KnowledgeStore:
    """
    Read-only semantic graph view.

    Metadata/classification edges are never answerable content.
    """

    NON_ANSWERABLE={
        "in_domain","domain","has_domain","belongs_to","member_of",
        "source","from_source","dataset","provenance","type","node_type",
        "category","subcategory","class","class_of","instance_of",
        "source_domain","mapped_to","mapping","label","tag",
    }

    GENERIC={
        "add","get","make","do","be","have","go","come","use",
        "say","want","need","tell","show","give","find","check",
        "put","take","keep","know","think","look",
    }

    def __init__(self,con):
        self.con=con

    def _answerable(self,relation: str) -> bool:
        rel=relation.lower().replace(":","_").replace("-","_")
        return rel not in {
            x.lower().replace(":","_").replace("-","_")
            for x in self.NON_ANSWERABLE
        }

    def retrieve(self,perception):
        if perception.speech_act in {
            "greeting","thanks","goodbye","affection",
        }:
            return []

        terms=list(dict.fromkeys(
            perception.predicates
            +perception.nouns
            +[
                x["lemma"]
                for x in perception.entities
            ],
        ))

        hits=[]

        for term in terms[:10]:
            rows=self.con.execute(
                """
                SELECT node_id,node_type,label,source,count
                FROM nodes
                WHERE lower(label)=lower(?)
                ORDER BY count DESC
                LIMIT 8
                """,
                (term,),
            ).fetchall()

            for node_id,node_type,label,source,count in rows:
                try:
                    edges=self.con.execute(
                        """
                        SELECT relation,target_node,weight,source_dataset
                        FROM edges
                        WHERE source_node=?
                        ORDER BY weight DESC
                        LIMIT 40
                        """,
                        (node_id,),
                    ).fetchall()
                except Exception:
                    edges=[]

                answerable=[]

                for relation,target,weight,edge_source in edges:
                    if not self._answerable(relation):
                        continue

                    row=self.con.execute(
                        "SELECT node_type,label FROM nodes WHERE node_id=?",
                        (target,),
                    ).fetchone()

                    if not row:
                        continue

                    answerable.append({
                        "relation":relation,
                        "target_type":row[0],
                        "target":row[1],
                        "weight":weight,
                        "source":edge_source,
                    })

                # A node with only classification metadata must not enter the
                # answerable set.
                if not answerable:
                    continue

                relevance=0
                if label.lower() in {
                    x.lower() for x in perception.nouns
                }:
                    relevance+=8
                elif label.lower() in {
                    x.lower() for x in perception.entities
                }:
                    relevance+=7
                elif label.lower() in self.GENERIC:
                    relevance-=5
                else:
                    relevance+=4

                hits.append({
                    "node_id":node_id,
                    "node_type":node_type,
                    "label":label,
                    "source":source,
                    "count":count,
                    "goal_relevance":relevance,
                    "outgoing":answerable,
                })

        hits.sort(
            key=lambda x:(
                x["goal_relevance"],
                x["count"],
            ),
            reverse=True,
        )

        return hits[:8]

    def relations(self,hits):
        result=[]
        for hit in hits:
            for edge in hit["outgoing"]:
                result.append(
                    (
                        hit["label"],
                        edge["relation"],
                        edge["target"],
                    )
                )
        return result
