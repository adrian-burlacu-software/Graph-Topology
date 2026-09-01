from __future__ import annotations

import heapq
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Edge:
    subject: str
    relation: str
    object: str


@dataclass(frozen=True)
class Parse:
    text: str
    tokens: list[dict]
    entities: list[dict]
    noun_chunks: list[str]
    root: str
    root_lemma: str
    question: str
    subjects: list[str]
    objects: list[str]


@dataclass(frozen=True)
class Hypothesis:
    subject: str | None
    relation: str
    intent: str
    lexical_score: float
    evidence: dict


class Graph:
    def __init__(self, db, cache_entries=12000):
        self.db = Path(db)
        self.cache_entries = int(cache_entries)
        self.conn = sqlite3.connect(
            str(self.db),
            timeout=60.0,
            check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            "PRAGMA query_only=ON"
        )
        self.conn.execute(
            "PRAGMA busy_timeout=60000"
        )
        self.sc = "subject"
        self.rc = "relation"
        self.oc = "object"

        self.cache = {}
        self.order = []

    def _put(self, key, value):
        if key in self.cache:
            return
        self.cache[key] = value
        self.order.append(key)
        while len(self.order) > self.cache_entries:
            old = self.order.pop(0)
            self.cache.pop(old, None)

    def outgoing(self, subject, limit=60):
        key = (
            "out",
            subject,
            int(limit),
        )
        if key in self.cache:
            return self.cache[key]

        rows = self.conn.execute(
            """
            SELECT subject,relation,object
            FROM edges
            WHERE subject=?
            LIMIT ?
            """,
            (
                subject,
                int(limit),
            ),
        ).fetchall()

        value = tuple(
            Edge(
                str(row["subject"]),
                str(row["relation"]),
                str(row["object"]),
            )
            for row in rows
        )
        self._put(
            key,
            value,
        )
        return value

    def relation_vocab(self, limit=200):
        rows = self.conn.execute(
            """
            SELECT relation,COUNT(*) AS n
            FROM edges
            GROUP BY relation
            ORDER BY n DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [
            str(row["relation"])
            for row in rows
        ]

    def resolve_exact(self, mention):
        mention = str(
            mention or ""
        ).strip()

        if not mention:
            return {
                "status": "unresolved",
                "mention": mention,
                "canonical": None,
                "confidence": 0.0,
                "candidates": [],
            }

        normalized = " ".join(
            mention.lower().split()
        )

        row = self.conn.execute(
            """
            SELECT node,label
            FROM nodes
            WHERE normalized=?
            ORDER BY
                CASE
                    WHEN node_type='concept'
                    THEN 0
                    ELSE 1
                END,
                node
            LIMIT 1
            """,
            (normalized,),
        ).fetchone()

        if row:
            return {
                "status": "resolved",
                "mention": mention,
                "canonical": str(
                    row["node"]
                ),
                "confidence": 1.0,
                "candidates": [
                    {
                        "node": str(
                            row["node"]
                        ),
                        "kind": "exact",
                        "score": 1.0,
                        "accepted": True,
                        "label": row["label"],
                    }
                ],
            }

        return {
            "status": "unresolved",
            "mention": mention,
            "canonical": None,
            "confidence": 0.0,
            "candidates": [],
        }

    def resolve_alias(self, mention, limit=12):
        exact = self.resolve_exact(
            mention
        )

        if exact["status"] == "resolved":
            return exact

        tokens = [
            token
            for token in re.findall(
                r"[a-z]+",
                str(mention).lower(),
            )
            if len(token) >= 3
        ]

        if not tokens:
            return exact

        # Candidate lookup only against indexed node strings, never edges.
        probe = max(
            tokens,
            key=len,
        )

        rows = self.conn.execute(
            """
            SELECT node,label,normalized
            FROM nodes
            WHERE normalized LIKE ?
              AND node_type='concept'
            LIMIT ?
            """,
            (
                f"%{probe}%",
                max(
                    32,
                    int(limit) * 4,
                ),
            ),
        ).fetchall()

        scored = []

        for row in rows:
            candidate_tokens = set(
                str(
                    row["normalized"]
                ).split()
            )

            overlap = sum(
                token in candidate_tokens
                for token in tokens
            )

            coverage = overlap / max(
                len(tokens),
                1,
            )
            precision = overlap / max(
                len(candidate_tokens),
                1,
            )

            score = (
                0.75 * coverage
                + 0.25 * precision
            )

            if all(
                token in candidate_tokens
                for token in tokens
            ):
                score += 0.10

            scored.append(
                {
                    "node": str(
                        row["node"]
                    ),
                    "kind": "alias",
                    "score": round(
                        min(
                            score,
                            1.0,
                        ),
                        6,
                    ),
                    "accepted": False,
                    "label": row["label"],
                }
            )

        scored.sort(
            key=lambda item: (
                -item["score"],
                len(item["node"]),
                item["node"],
            )
        )

        candidates = scored[:limit]

        if not candidates:
            return exact

        top = candidates[0]
        second = (
            candidates[1]
            if len(candidates) > 1
            else None
        )

        margin = (
            top["score"]
            - second["score"]
            if second
            else top["score"]
        )

        if (
            top["score"] >= 0.82
            and margin >= 0.10
        ):
            top["accepted"] = True
            return {
                "status": "resolved",
                "mention": str(
                    mention
                ),
                "canonical": top["node"],
                "confidence": top["score"],
                "candidates": candidates,
            }

        return {
            "status": "ambiguous",
            "mention": str(
                mention
            ),
            "canonical": None,
            "confidence": 0.0,
            "candidates": candidates,
        }

    def definition(self, node):
        row = self.conn.execute(
            """
            SELECT definition
            FROM nodes
            WHERE node=?
            LIMIT 1
            """,
            (node,),
        ).fetchone()

        if row and row["definition"]:
            return str(
                row["definition"]
            )

        return None


class Context:
    def __init__(self):
        self.active_subject = None
        self.turns = []
        self.entities = {}
        self.relation_memory = defaultdict(float)
        self.path_memory = defaultdict(float)


class Attention:
    def __init__(self, decay=0.65):
        self.decay = float(decay)
        self.values = defaultdict(float)

    def rank(
        self,
        goal,
        prefix,
        relations,
    ):
        rows = []

        for relation in relations:
            rows.append(
                (
                    self.values.get(
                        (
                            goal,
                            tuple(prefix),
                            relation,
                        ),
                        0.0,
                    ),
                    relation,
                )
            )

        rows.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )
        return rows


def entity_mention(parse):
    wh = {
        "who",
        "what",
        "where",
        "when",
        "which",
        "why",
        "how",
        "whom",
        "whose",
    }

    refs = {
        "it",
        "he",
        "she",
        "they",
        "him",
        "her",
        "them",
        "this",
        "that",
        "these",
        "those",
    }

    for entity in parse.entities:
        value = str(
            entity["text"]
        ).strip()
        if (
            value
            and value.lower() not in wh
        ):
            return value

    for value in (
        parse.subjects
        + parse.noun_chunks
    ):
        parts = [
            part
            for part in str(value).split()
            if part.lower()
            not in wh | refs
        ]
        if parts:
            return " ".join(
                parts
            )

    return None


def concept_mention(parse):
    stop = {
        "what",
        "is",
        "are",
        "was",
        "were",
        "be",
        "a",
        "an",
        "the",
        "of",
    }

    for value in (
        parse.noun_chunks
        + parse.subjects
        + parse.objects
    ):
        parts = [
            part
            for part in str(value).split()
            if part.lower()
            not in stop
        ]
        if parts:
            return " ".join(
                parts
            )

    return None


def structural_concept_question(parse):
    return (
        parse.question == "WH_WHAT"
        and parse.root_lemma in {
            "be",
            "mean",
            "refer",
        }
        and not parse.entities
    )


def relation_hypotheses(
    parse,
    graph,
    context,
    max_n=12,
):
    if structural_concept_question(
        parse
    ):
        concept = concept_mention(
            parse
        )

        if concept:
            resolution = graph.resolve_alias(
                concept
            )

            if resolution["status"] == "resolved":
                subject = resolution[
                    "canonical"
                ]

                definition = Hypothesis(
                    subject,
                    "definition",
                    "concept_lookup",
                    1.5,
                    {
                        "concept": concept,
                        "entity_resolution": resolution,
                    },
                )

                semantic_relations = [
                    "is_a",
                    "has_part",
                    "part_of",
                    "capable_of",
                    "used_for",
                    "has_property",
                    "related_to",
                ]

                candidates = [
                    Hypothesis(
                        subject,
                        relation,
                        "concept_lookup",
                        1.0,
                        {
                            "concept": concept,
                            "entity_resolution": resolution,
                        },
                    )
                    for relation in semantic_relations
                ]

                return (
                    [definition]
                    + candidates
                )[:max_n]

    mention = entity_mention(
        parse
    )

    resolution = (
        graph.resolve_alias(
            mention
        )
        if mention
        else {
            "status": "none",
            "mention": None,
            "canonical": None,
            "confidence": 0.0,
            "candidates": [],
        }
    )

    if (
        mention
        and resolution["status"]
        != "resolved"
    ):
        return [
            Hypothesis(
                None,
                "",
                "entity_unresolved",
                0.0,
                {
                    "entity_resolution": resolution,
                },
            )
        ]

    subject = (
        resolution["canonical"]
        if resolution.get(
            "status"
        ) == "resolved"
        else context.active_subject
    )

    if not subject:
        return [
            Hypothesis(
                None,
                "",
                "conversation",
                0.5,
                {},
            )
        ]

    relations = graph.relation_vocab(
        max_n
    )

    output = []

    for index, relation in enumerate(
        relations
    ):
        output.append(
            Hypothesis(
                subject,
                relation,
                "relation_lookup",
                1.0
                / (
                    1 + index
                ),
                {
                    "entity_resolution": resolution,
                },
            )
        )

    return output


def search(
    graph,
    attention,
    hypothesis,
    budget=40,
    per_node=60,
    max_depth=3,
):
    base = {
        "success": False,
        "intent_only": hypothesis.intent
        in {
            "conversation",
        },
        "steps": 0,
        "path": [],
        "target": None,
        "attention": 0,
        "exploration": 0,
        "direct_proof": False,
    }

    if (
        not hypothesis.subject
        or not hypothesis.relation
        or hypothesis.intent
        in {
            "conversation",
            "entity_unresolved",
        }
    ):
        return base

    for edge in graph.outgoing(
        hypothesis.subject,
        per_node,
    ):
        if edge.relation == hypothesis.relation:
            return {
                **base,
                "success": True,
                "steps": 1,
                "path": [
                    edge.relation
                ],
                "target": edge.object,
                "direct_proof": True,
            }

    queue = [
        (
            0.0,
            0,
            hypothesis.subject,
            (),
        )
    ]

    seen = {
        (
            hypothesis.subject,
            (),
        )
    }

    expansions = 0
    attention_hits = 0
    exploration = 0

    while (
        queue
        and expansions < budget
    ):
        _, depth, node, prefix = (
            heapq.heappop(queue)
        )

        if depth >= max_depth - 1:
            continue

        expansions += 1

        edges = list(
            graph.outgoing(
                node,
                per_node,
            )
        )

        ranked = attention.rank(
            hypothesis.relation,
            prefix,
            [
                edge.relation
                for edge in edges
            ],
        )
        score_by_relation = dict(
            ranked
        )

        edges.sort(
            key=lambda edge: (
                -score_by_relation.get(
                    edge.relation,
                    0.0,
                ),
                edge.relation,
            )
        )

        for edge in edges:
            next_prefix = (
                prefix
                + (
                    edge.relation,
                )
            )

            state = (
                edge.object,
                next_prefix,
            )

            if state in seen:
                continue

            seen.add(state)

            if score_by_relation.get(
                edge.relation,
                0.0,
            ) > 0:
                attention_hits += 1
            else:
                exploration += 1

            for goal in graph.outgoing(
                edge.object,
                per_node,
            ):
                if goal.relation == hypothesis.relation:
                    return {
                        **base,
                        "success": True,
                        "steps": expansions,
                        "path": list(
                            next_prefix
                        ) + [
                            goal.relation
                        ],
                        "target": goal.object,
                        "attention": attention_hits,
                        "exploration": exploration,
                        "direct_proof": False,
                    }

            if len(
                next_prefix
            ) < max_depth - 1:
                heapq.heappush(
                    queue,
                    (
                        -score_by_relation.get(
                            edge.relation,
                            0.0,
                        ),
                        len(
                            next_prefix
                        ),
                        edge.object,
                        next_prefix,
                    )
                )

    return {
        **base,
        "steps": expansions,
        "attention": attention_hits,
        "exploration": exploration,
    }


class SpaCyParser:
    def __init__(self, model):
        import spacy
        self.nlp = spacy.load(
            model
        )

    def parse(self, text):
        doc = self.nlp(
            text
        )

        tokens = [
            {
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "dep": token.dep_,
            }
            for token in doc
        ]

        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
            }
            for ent in doc.ents
        ]

        chunks = [
            chunk.text
            for chunk in doc.noun_chunks
        ]

        root = next(
            (
                token
                for token in doc
                if token.dep_ == "ROOT"
            ),
            doc[0]
            if len(doc)
            else None,
        )

        lower = text.lower().strip()

        if lower.startswith("who "):
            question = "WH_WHO"
        elif lower.startswith("what "):
            question = "WH_WHAT"
        elif lower.startswith("where "):
            question = "WH_WHERE"
        elif lower.startswith("when "):
            question = "WH_WHEN"
        elif lower.startswith("why "):
            question = "WH_WHY"
        elif lower.startswith("how "):
            question = "WH_HOW"
        elif lower.startswith("which "):
            question = "WH_WHICH"
        else:
            question = (
                "QUESTION"
                if "?" in text
                else "DECLARATIVE"
            )

        subjects = [
            token.text
            for token in doc
            if token.dep_ in {
                "nsubj",
                "nsubjpass",
            }
        ]

        objects = [
            token.text
            for token in doc
            if token.dep_ in {
                "dobj",
                "obj",
                "attr",
                "pobj",
            }
        ]

        return Parse(
            text=text,
            tokens=tokens,
            entities=entities,
            noun_chunks=chunks,
            root=root.text
            if root else "",
            root_lemma=root.lemma_
            if root else "",
            question=question,
            subjects=subjects,
            objects=objects,
        )
