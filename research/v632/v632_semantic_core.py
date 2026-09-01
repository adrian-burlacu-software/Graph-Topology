from __future__ import annotations

import heapq
import json
import math
import random
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path


TARGET_RELATIONS = (
    "definition",
    "is_a",
    "has_part",
    "part_of",
    "capable_of",
    "used_for",
    "has_property",
    "at_location",
    "related_to",
    "causes",
    "made_of",
    "has_a",
)

RELATION_PHRASES = {
    "definition": "definition meaning what is explain describe",
    "is_a": "is a type kind category class",
    "has_part": "has part contains includes made of",
    "part_of": "part of belongs component",
    "capable_of": "can do capable of able to",
    "used_for": "used for purpose function",
    "has_property": "has property characteristic quality",
    "at_location": "located at in place location",
    "related_to": "related to associated with connected to",
    "causes": "causes leads to produces makes",
    "made_of": "made of material substance",
    "has_a": "has contains possesses",
}


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
    def __init__(
        self,
        db: Path,
        cache_entries: int = 12000,
    ):
        self.db = Path(db)
        self.cache_entries = int(cache_entries)
        self.conn = sqlite3.connect(
            str(self.db),
            timeout=30.0,
            check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            "PRAGMA query_only=ON"
        )
        self.conn.execute(
            "PRAGMA busy_timeout=30000"
        )
        cols = {
            str(row["name"])
            for row in self.conn.execute(
                "PRAGMA table_info(edges)"
            )
        }
        if not cols:
            raise RuntimeError(
                "semantic database has no edges table"
            )

        self.sc = "subject" if "subject" in cols else "source"
        self.rc = "relation" if "relation" in cols else "predicate"
        self.oc = "object" if "object" in cols else "target"

        self.has_nodes = bool(
            self.conn.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type='table'
                  AND name='nodes'
                LIMIT 1
                """
            ).fetchone()
        )

        self.cache = {}
        self.order = []

    def close(self):
        self.conn.close()

    def _put(
        self,
        key,
        value,
    ):
        if self.cache_entries <= 0:
            return
        if key not in self.cache:
            self.order.append(key)
        self.cache[key] = value
        while len(self.order) > self.cache_entries:
            self.cache.pop(
                self.order.pop(0),
                None,
            )

    def outgoing(
        self,
        subject: str,
        limit: int = 60,
    ) -> tuple[Edge, ...]:
        key = ("o", subject, int(limit))
        if key in self.cache:
            return self.cache[key]

        rows = self.conn.execute(
            f"""
            SELECT
                {self.sc} AS subject,
                {self.rc} AS relation,
                {self.oc} AS object
            FROM edges
            WHERE {self.sc}=?
            LIMIT ?
            """,
            (subject, int(limit)),
        ).fetchall()

        value = tuple(
            Edge(
                str(row["subject"]),
                str(row["relation"]),
                str(row["object"]),
            )
            for row in rows
        )
        self._put(key, value)
        return value

    def relation_vocab(
        self,
        limit=200,
    ):
        rows = self.conn.execute(
            f"""
            SELECT
                {self.rc} AS relation,
                COUNT(*) AS n
            FROM edges
            GROUP BY {self.rc}
            ORDER BY n DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()

        out = [
            str(row["relation"])
            for row in rows
        ]

        for relation in TARGET_RELATIONS:
            if relation not in out:
                out.append(relation)

        return out[: max(
            int(limit),
            len(TARGET_RELATIONS),
        )]

    def relation_phrases(self, relation):
        if relation in RELATION_PHRASES:
            return RELATION_PHRASES[relation]

        row = self.conn.execute(
            """
            SELECT phrases
            FROM relations
            WHERE relation=?
            LIMIT 1
            """,
            (relation,),
        ).fetchone()

        if row and row["phrases"]:
            return str(row["phrases"])

        return str(relation).replace(
            "_",
            " ",
        )

    @staticmethod
    def _norm(value):
        return " ".join(
            str(value)
            .strip()
            .lower()
            .split()
        )

    def resolve_entity_strict(
        self,
        mention,
        limit=16,
    ):
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

        if self.has_nodes:
            row = self.conn.execute(
                """
                SELECT node, label
                FROM nodes
                WHERE normalized=?
                LIMIT 1
                """,
                (self._norm(mention),),
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
                            "kind": "dictionary_exact",
                            "score": 1.0,
                            "accepted": True,
                            "label": row["label"],
                        }
                    ],
                }

        row = self.conn.execute(
            f"""
            SELECT {self.sc} AS node
            FROM edges
            WHERE lower({self.sc})=?
            LIMIT ?
            """,
            (
                self._norm(mention),
                int(limit),
            ),
        ).fetchone()

        if row:
            return {
                "status": "resolved",
                "mention": mention,
                "canonical": str(row["node"]),
                "confidence": 1.0,
                "candidates": [
                    {
                        "node": str(row["node"]),
                        "kind": "edge_exact",
                        "score": 1.0,
                        "accepted": True,
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

    def resolve_entity_alias(
        self,
        mention,
        limit=8,
    ):
        """
        Dictionary-first resolution.

        Since the compact network contains a controlled vocabulary, fuzzy
        matching is intentionally conservative and uses the normalized nodes
        table. There is no whole-edge-table wildcard scan.
        """
        exact = self.resolve_entity_strict(
            mention,
            limit,
        )
        if exact["status"] == "resolved":
            return exact

        if not self.has_nodes:
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

        # One selective token, bounded candidate retrieval.
        probe = max(
            tokens,
            key=len,
        )

        rows = self.conn.execute(
            """
            SELECT node, normalized, label
            FROM nodes
            WHERE normalized LIKE ?
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
                    "node": str(row["node"]),
                    "kind": "dictionary_alias",
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
            top["score"] - second["score"]
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
                "mention": str(mention),
                "canonical": top["node"],
                "confidence": top["score"],
                "candidates": candidates,
            }

        return {
            "status": "ambiguous",
            "mention": str(mention),
            "canonical": None,
            "confidence": 0.0,
            "candidates": candidates,
        }

    def definition(
        self,
        node,
    ):
        if not self.has_nodes:
            return None

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
            return str(row["definition"])

        return None


class SpaCyParser:
    def __init__(self, model_name):
        import spacy
        self.nlp = spacy.load(
            model_name
        )

    def parse(self, text):
        doc = self.nlp(text)

        tokens = [
            {
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
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

        noun_chunks = [
            chunk.text
            for chunk in doc.noun_chunks
        ]

        root = next(
            (
                token
                for token in doc
                if token.dep_ == "ROOT"
            ),
            doc[0] if doc else None,
        )

        question = "DECLARATIVE"
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
        elif "?" in text:
            question = "QUESTION"

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
            noun_chunks=noun_chunks,
            root=root.text if root else "",
            root_lemma=root.lemma_ if root else "",
            question=question,
            subjects=subjects,
            objects=objects,
        )


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
        ranked = []

        for relation in relations:
            key = (
                str(goal),
                tuple(prefix),
                str(relation),
            )
            ranked.append(
                (
                    self.values.get(
                        key,
                        0.0,
                    ),
                    relation,
                )
            )

        ranked.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )
        return ranked

    def update(
        self,
        goal,
        prefix,
        next_relation,
        strength=1.0,
    ):
        key = (
            str(goal),
            tuple(prefix),
            str(next_relation),
        )
        self.values[key] = (
            self.decay
            * self.values.get(
                key,
                0.0,
            )
            + float(strength)
        )

    def export(self):
        return {
            "|".join(
                (
                    goal,
                    *prefix,
                    next_relation,
                )
            ): value
            for (
                goal,
                prefix,
                next_relation,
            ), value in self.values.items()
        }


class ContextRelationAttention:
    def __init__(self, decay=0.65):
        self.decay = float(decay)
        self.values = defaultdict(float)
        self.updates = 0

    @staticmethod
    def features(
        parse,
    ):
        return (
            "question:" + parse.question,
            "root:" + parse.root_lemma,
            *[
                "pos:" + token["pos"]
                for token in parse.tokens
            ],
        )

    @staticmethod
    def lexical_score(
        parse,
        relation,
        phrases,
    ):
        qwords = {
            token
            for token in re.findall(
                r"[a-z]+",
                parse.text.lower(),
            )
            if len(token) > 1
        }
        rwords = {
            token
            for token in re.findall(
                r"[a-z]+",
                phrases.lower(),
            )
            if len(token) > 1
        }

        overlap = len(
            qwords & rwords
        )

        return min(
            1.0,
            overlap / 3.0,
        )

    def rank(
        self,
        parse,
        vocab,
        graph,
    ):
        features = self.features(
            parse
        )

        ranked = []
        for relation in vocab:
            lexical = self.lexical_score(
                parse,
                relation,
                graph.relation_phrases(
                    relation
                ),
            )

            learned = sum(
                self.values.get(
                    (
                        feature,
                        relation,
                    ),
                    0.0,
                )
                for feature in features
            )

            ranked.append(
                (
                    0.70 * lexical
                    + 0.30 * learned,
                    relation,
                )
            )

        ranked.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )
        return ranked

    def update(
        self,
        parse,
        relation,
        strength=1.0,
    ):
        for feature in self.features(
            parse
        ):
            key = (
                feature,
                relation,
            )
            self.values[key] = (
                self.decay
                * self.values.get(
                    key,
                    0.0,
                )
                + float(strength)
            )

        self.updates += 1

    def export(self):
        return {
            "|".join(key): value
            for key, value
            in self.values.items()
        }


def entity_mention_from_parse(
    parse,
):
    reference = {
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
    wh = {
        "who",
        "what",
        "where",
        "when",
        "which",
        "whom",
        "whose",
        "why",
        "how",
    }

    if any(
        token["text"].lower()
        in reference
        for token in parse.tokens
    ):
        return None

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
        + parse.objects
        + parse.noun_chunks
    ):
        words = [
            word
            for word in str(value).split()
            if word.lower() not in wh
        ]
        if words:
            candidate = " ".join(
                words
            ).strip()
            if candidate:
                return candidate

    return None


def concept_mention_from_parse(
    parse,
):
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
        parse.objects
        + parse.subjects
        + parse.noun_chunks
    ):
        words = [
            word
            for word in str(value).split()
            if word.lower() not in stop
        ]
        if words:
            return " ".join(words)

    return None


def is_general_concept_question(
    parse,
):
    return (
        parse.question == "WH_WHAT"
        and parse.root_lemma in {
            "be",
            "mean",
            "refer",
        }
        and not parse.entities
        and bool(
            parse.noun_chunks
            or parse.subjects
            or parse.objects
        )
    )


class Memory:
    def __init__(
        self,
        path: Path,
    ):
        self.path = Path(path)
        self.active_subject = None
        self.entities = {}
        self.turns = []
        self.intent_values = defaultdict(
            Counter
        )
        self.relation_outcomes = defaultdict(
            Counter
        )
        self.path_values = defaultdict(
            float
        )
        self.load()

    def load(self):
        if not self.path.exists():
            return

        try:
            payload = json.loads(
                self.path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            return

        self.active_subject = payload.get(
            "active_subject"
        )
        self.entities = dict(
            payload.get(
                "entities",
                {}
            )
        )
        self.turns = list(
            payload.get(
                "turns",
                []
            )
        )

        for key, value in payload.get(
            "intent_values",
            {}
        ).items():
            self.intent_values[
                key
            ].update(value)

        for key, value in payload.get(
            "relation_outcomes",
            {}
        ).items():
            self.relation_outcomes[
                key
            ].update(value)

        self.path_values.update(
            payload.get(
                "path_values",
                {}
            )
        )

    def save(self):
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "active_subject": self.active_subject,
            "entities": self.entities,
            "turns": self.turns[-256:],
            "intent_values": {
                key: dict(value)
                for key, value
                in self.intent_values.items()
            },
            "relation_outcomes": {
                key: dict(value)
                for key, value
                in self.relation_outcomes.items()
            },
            "path_values": dict(
                self.path_values
            ),
        }

        self.path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def relation_hypotheses(
    parse,
    graph,
    memory,
    max_n=12,
):
    mention = entity_mention_from_parse(
        parse
    )

    resolution = (
        graph.resolve_entity_alias(
            mention,
            12,
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
        not in {
            "resolved",
            "ambiguous",
        }
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
        resolution.get(
            "canonical"
        )
        if resolution.get(
            "status"
        ) == "resolved"
        else memory.active_subject
    )

    concepts = is_general_concept_question(
        parse
    )

    if concepts:
        concept = concept_mention_from_parse(
            parse
        )
        concept_resolution = graph.resolve_entity_alias(
            concept,
            12,
        ) if concept else {
            "status": "unresolved",
            "mention": concept,
            "canonical": None,
            "confidence": 0.0,
            "candidates": [],
        }

        if (
            concept_resolution.get(
                "status"
            ) == "resolved"
        ):
            subject = concept_resolution[
                "canonical"
            ]
        else:
            # General concept questions may be answered conversationally when
            # this compact semantic network does not contain the concept.
            return [
                Hypothesis(
                    None,
                    "",
                    "conversation",
                    0.90,
                    {
                        "concept_question": True,
                        "concept_resolution": concept_resolution,
                    },
                )
            ]

    if not subject:
        return [
            Hypothesis(
                None,
                "",
                "conversation",
                0.50,
                {
                    "entity_resolution": resolution,
                },
            )
        ]

    vocab = graph.relation_vocab(
        max(
            32,
            max_n * 4,
        )
    )

    attention = ContextRelationAttention(
        0.65
    )
    ranked = attention.rank(
        parse,
        vocab,
        graph,
    )

    # Memory supplies a global conditional prior over intent/relation context.
    scored = []
    for attention_score, relation in ranked:
        history_score = float(
            memory.relation_outcomes[
                relation
            ].get(
                "success",
                0.0,
            )
        )

        score = (
            attention_score
            + 0.03
            * min(
                history_score,
                10.0,
            )
        )

        # Concept definitions are especially well aligned with "what is".
        if (
            concepts
            and parse.question
            == "WH_WHAT"
            and relation
            == "definition"
        ):
            score += 0.35

        scored.append(
            (
                score,
                relation,
            )
        )

    scored.sort(
        key=lambda item: (
            -item[0],
            item[1],
        )
    )

    output = []
    for score, relation in scored[
        :max_n
    ]:
        output.append(
            Hypothesis(
                subject,
                relation,
                "relation_lookup",
                float(score),
                {
                    "relation_score": score,
                    "entity_resolution": resolution,
                    "concept_question": concepts,
                    "active_subject": memory.active_subject,
                },
            )
        )

    return output


def search(
    graph,
    path_attention,
    hypothesis,
    budget=40,
    per_node=60,
    max_depth=3,
    seed=0,
):
    empty = {
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
        "goal_relation": hypothesis.relation,
    }

    if (
        hypothesis.intent
        in {
            "conversation",
            "entity_unresolved",
        }
        or not hypothesis.subject
        or not hypothesis.relation
    ):
        return empty

    # Direct proof.
    for edge in graph.outgoing(
        hypothesis.subject,
        per_node,
    ):
        if edge.relation == hypothesis.relation:
            return {
                **empty,
                "success": True,
                "intent_only": False,
                "steps": 1,
                "path": [
                    hypothesis.relation
                ],
                "target": edge.object,
                "direct_proof": True,
            }

    if max_depth <= 1:
        return empty

    rng = random.Random(
        seed
    )

    queue = []
    tie = 0
    heapq.heappush(
        queue,
        (
            -path_attention.values.get(
                (
                    hypothesis.relation,
                    (),
                    "is_a",
                ),
                0.0,
            ),
            0,
            tie,
            hypothesis.subject,
            (),
        ),
    )

    seen = {
        (
            hypothesis.subject,
            (),
        )
    }

    expanded = 0
    attention_hits = 0
    exploration = 0

    while queue and expanded < budget:
        _, depth, _, node, prefix = (
            heapq.heappop(queue)
        )

        if depth >= max_depth - 1:
            continue

        expanded += 1

        edges = list(
            graph.outgoing(
                node,
                per_node,
            )
        )

        ranked = path_attention.rank(
            hypothesis.relation,
            prefix,
            [
                edge.relation
                for edge in edges
            ],
        )
        scores = dict(ranked)

        edges.sort(
            key=lambda edge: (
                -scores.get(
                    edge.relation,
                    0.0,
                ),
                rng.random(),
            )
        )

        for edge in edges:
            next_prefix = (
                prefix
                + (edge.relation,)
            )

            state = (
                edge.object,
                next_prefix,
            )
            if state in seen:
                continue

            seen.add(state)

            if scores.get(
                edge.relation,
                0.0,
            ) > 0:
                attention_hits += 1
            else:
                exploration += 1

            for goal_edge in graph.outgoing(
                edge.object,
                per_node,
            ):
                if (
                    goal_edge.relation
                    == hypothesis.relation
                ):
                    return {
                        **empty,
                        "success": True,
                        "steps": expanded,
                        "path": list(
                            next_prefix
                        ) + [
                            hypothesis.relation
                        ],
                        "target": goal_edge.object,
                        "attention": attention_hits,
                        "exploration": exploration,
                        "direct_proof": False,
                    }

            if len(
                next_prefix
            ) < max_depth - 1:
                tie += 1
                heapq.heappush(
                    queue,
                    (
                        -scores.get(
                            edge.relation,
                            0.0,
                        ),
                        len(
                            next_prefix
                        ),
                        tie,
                        edge.object,
                        next_prefix,
                    )
                )

    return {
        **empty,
        "steps": expanded,
        "attention": attention_hits,
        "exploration": exploration,
    }


def remember_success(
    memory,
    hypothesis,
    result,
    parse,
):
    if not (
        hypothesis
        and result.get(
            "success",
            False,
        )
    ):
        return

    relation = hypothesis.relation
    path = tuple(
        result.get(
            "path",
            [],
        )
    )

    if relation:
        memory.relation_outcomes[
            relation
        ]["success"] += 1

    if path:
        prefix = ()
        for next_relation in path:
            key = "|".join(
                (
                    relation,
                    *prefix,
                    next_relation,
                )
            )
            memory.path_values[
                key
            ] += 1.0
            prefix += (
                next_relation,
            )

    memory.save()


class LLMRealizer:
    def __init__(
        self,
        model_path,
        temperature=0.15,
        max_new_tokens=96,
    ):
        self.model_path = str(
            model_path
        )
        self.temperature = float(
            temperature
        )
        self.max_new_tokens = int(
            max_new_tokens
        )
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return

        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
        )

        self._tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
        )

        if (
            self._tokenizer.pad_token_id
            is None
        ):
            self._tokenizer.pad_token = (
                self._tokenizer.eos_token
            )

        self._model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_path,
                local_files_only=True,
                device_map="auto",
            )
        )

    def generate(
        self,
        prompt,
        temperature=None,
    ):
        self._load()

        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
        )

        device = getattr(
            self._model,
            "device",
            None,
        )
        if device is not None:
            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

        temp = (
            self.temperature
            if temperature is None
            else float(temperature)
        )

        import torch

        with torch.no_grad():
            generated = (
                self._model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=temp > 0.0,
                    temperature=(
                        temp
                        if temp > 0.0
                        else 1.0
                    ),
                    pad_token_id=(
                        self._tokenizer.pad_token_id
                    ),
                    eos_token_id=(
                        self._tokenizer.eos_token_id
                    ),
                )
            )

        prompt_tokens = encoded[
            "input_ids"
        ].shape[1]

        return self._tokenizer.decode(
            generated[
                0,
                prompt_tokens:,
            ],
            skip_special_tokens=True,
        ).strip().split(
            "\n",
            1,
        )[0].strip()

    def grounded_prompt(
        self,
        question,
        hypothesis,
        result,
        definition=None,
    ):
        evidence = ", ".join(
            result.get(
                "path",
                [],
            )[:8]
        )

        return (
            "You are a surface-language realizer.\n"
            "The semantic system is authoritative.\n"
            "Use ONLY the verified result below.\n"
            "Do not add outside knowledge.\n"
            "Do not invent facts.\n"
            "Return one concise natural sentence.\n\n"
            "QUESTION: "
            + str(question)
            + "\nSUBJECT: "
            + str(
                hypothesis.subject
            )
            + "\nRELATION: "
            + str(
                hypothesis.relation
            )
            + "\nVERIFIED RESULT: "
            + str(
                result.get(
                    "target",
                    "",
                )
            )
            + "\nEVIDENCE PATH: "
            + evidence
            + (
                "\nDICTIONARY DEFINITION: "
                + str(definition)
                if definition
                else ""
            )
            + "\nANSWER:"
        )

    def conversation_prompt(
        self,
        question,
        history,
    ):
        recent = []
        for turn in history[-6:]:
            if not isinstance(
                turn,
                dict,
            ):
                continue
            recent.append(
                "USER: "
                + str(
                    turn.get(
                        "text",
                        "",
                    )
                )[:240]
            )
            recent.append(
                "ASSISTANT: "
                + str(
                    turn.get(
                        "answer",
                        "",
                    )
                )[:280]
            )

        return (
            "You are the conversational assistant.\n"
            "Respond naturally.\n"
            "You may greet the user, tell jokes, "
            "explain concepts, and chat.\n"
            "Do not claim the semantic graph verified "
            "a fact when it did not.\n"
            "Be concise.\n\n"
            "RECENT:\n"
            + (
                "\n".join(recent)
                if recent
                else "none"
            )
            + "\nUSER: "
            + str(question)
            + "\nASSISTANT:"
        )


