
from __future__ import annotations


class KnowledgeStore:
    """
    Read-only view over the semantic net.

    Only relations that can express an actual proposition are exposed as
    answerable knowledge. UD/spaCy/grammar artifacts and domain/classification
    metadata are never treated as world knowledge.
    """

    BLOCKED_RELATIONS={
        # Grammar / parser structure
        "subject","object","iobj","indirect_object",
        "nsubj","nsubjpass","obj","dobj","ccomp","xcomp",
        "complement","modifier","amod","advmod","nmod",
        "obl","oblique","obl_arg","obl_mod","obl_tmod",
        "obl_agent","conj","cc","det","aux","auxpass","cop",
        "mark","punct","root","dep","appos","compound","case",
        "expl","acl","advcl","parataxis","flat","fixed",
        "goeswith","list","orphan","vocative","dislocated",
        "reparandum",

        # Classification / provenance
        "in_domain","domain","has_domain","belongs_to","member_of",
        "source","from_source","dataset","provenance","type","node_type",
        "category","subcategory","class","class_of","instance_of",
        "source_domain","mapped_to","mapping","label","tag",
    }

    ALLOWED_SEMANTIC_PREFIXES={
        "is","has","likes","lives","located","causes","used_for",
        "used_for","capable_of","made_of","part_of","related_to",
        "synonym","antonym","hypernym","hyponym","entails",
        "similar_to","desires","receives","produces","contains",
        "created_by","created_for","defined_as","means",
    }

    def __init__(self,con):
        self.con=con

    def _key(self,relation):
        return str(relation or "").strip().lower().replace(":","_").replace("-","_")

    def _answerable(self,relation):
        rel=self._key(relation)
        if rel in self.BLOCKED_RELATIONS:
            return False
        return (
            rel in self.ALLOWED_SEMANTIC_PREFIXES
            or any(
                rel.startswith(prefix+"_")
                for prefix in self.ALLOWED_SEMANTIC_PREFIXES
            )
        )

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
                        LIMIT 50
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
                        """
                        SELECT node_type,label
                        FROM nodes
                        WHERE node_id=?
                        """,
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

                if not answerable:
                    continue

                hits.append({
                    "node_id":node_id,
                    "node_type":node_type,
                    "label":label,
                    "source":source,
                    "count":count,
                    "goal_relevance":self._lexical_relevance(
                        label,
                        perception,
                    ),
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

    def _lexical_relevance(self,label,perception):
        label=str(label).lower()
        nouns={x.lower() for x in perception.nouns}
        entities={
            x["lemma"].lower()
            for x in perception.entities
        }

        if label in nouns:
            return 8
        if label in entities:
            return 7
        return 3

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
