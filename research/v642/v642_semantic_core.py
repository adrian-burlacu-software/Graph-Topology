from __future__ import annotations

import heapq
import re
import sqlite3
import time
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
        # The semantic facts remain immutable by convention. V642 writes only
        # its distilled-memory tables, created below.
        self.conn.execute(
            "PRAGMA query_only=OFF"
        )
        self.conn.execute(
            "PRAGMA busy_timeout=60000"
        )
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS distilled_decisions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_type TEXT NOT NULL,
                context_key TEXT NOT NULL,
                surface TEXT NOT NULL,
                candidate_set TEXT NOT NULL,
                selected TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 1,
                confidence REAL NOT NULL DEFAULT 0.0,
                created_unix REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_distilled_context
                ON distilled_decisions(
                    decision_type,
                    context_key
                );

            CREATE INDEX IF NOT EXISTS idx_distilled_surface
                ON distilled_decisions(
                    decision_type,
                    surface
                );
            """
        )
        self.conn.commit()
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

    def definition(
        self,
        node,
        sense_node=None,
    ):
        # If the cognitive/teacher layer explicitly selected a WordNet sense,
        # that selection is authoritative for definition retrieval.
        if sense_node:
            row = self.conn.execute(
                """
                SELECT definition
                FROM nodes
                WHERE node=?
                  AND node_type='synset'
                LIMIT 1
                """,
                (sense_node,),
            ).fetchone()

            if row and row["definition"]:
                return str(
                    row["definition"]
                )

            row = self.conn.execute(
                """
                SELECT n.definition
                FROM edges e
                JOIN nodes n
                  ON n.node=e.object
                WHERE e.subject=?
                  AND e.relation='has_sense'
                  AND e.object=?
                  AND n.definition IS NOT NULL
                LIMIT 1
                """,
                (
                    node,
                    sense_node,
                ),
            ).fetchone()

            if row and row["definition"]:
                return str(
                    row["definition"]
                )

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

        row = self.conn.execute(
            """
            SELECT n.definition
            FROM edges e
            JOIN nodes n
              ON n.node=e.object
            WHERE e.subject=?
              AND e.relation='has_sense'
              AND n.definition IS NOT NULL
            ORDER BY n.node
            LIMIT 1
            """,
            (node,),
        ).fetchone()

        if row and row["definition"]:
            return str(
                row["definition"]
            )

        return None


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

    def resolve_exact(
        self,
        mention,
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

        normalized = " ".join(
            mention.lower().split()
        )

        # Prefer lexical concept nodes over synsets and other auxiliary nodes.
        row = self.conn.execute(
            """
            SELECT node,label,node_type
            FROM nodes
            WHERE normalized=?
            ORDER BY
                CASE
                    WHEN node_type='concept'
                    THEN 0
                    WHEN node_type='synset'
                    THEN 1
                    ELSE 2
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
                        "kind": "normalized_exact",
                        "score": 1.0,
                        "accepted": True,
                        "label": row["label"],
                    }
                ],
            }

        # Full graph lexical concepts are stored canonically as en:<term>.
        # Check that representation explicitly. This is the critical V635 fix.
        concept_node = (
            "en:"
            + normalized
        )

        row = self.conn.execute(
            """
            SELECT node,label,node_type
            FROM nodes
            WHERE node=?
            LIMIT 1
            """,
            (concept_node,),
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
                        "kind": "canonical_en_exact",
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


    def resolve_alias(
        self,
        mention,
        limit=12,
    ):
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
                    int(limit) * 8,
                ),
            ),
        ).fetchall()

        scored = []

        for row in rows:
            normalized = str(
                row["normalized"]
            ).strip()

            candidate_tokens = set(
                normalized.split()
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

            # A one-word exact lexical concept is preferred to compounds
            # whenever it is encountered.
            if normalized == " ".join(tokens):
                score = 1.0

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


class DistilledMemory:
    """
    Candidate-constrained online semantic distillation.

    The LLM never writes a new semantic fact. It can only select one candidate
    supplied by the graph. The selected candidate is stored as a durable
    decision in the semantic database.
    """

    def __init__(self, graph):
        self.graph = graph

    @staticmethod
    def _key(
        decision_type,
        surface,
        context_text,
    ):
        import hashlib

        raw = "|".join(
            (
                decision_type,
                str(surface).strip().lower(),
                " ".join(
                    str(context_text)
                    .strip()
                    .lower()
                    .split()
                ),
            )
        )

        return hashlib.sha256(
            raw.encode(
                "utf-8"
            )
        ).hexdigest()

    def lookup(
        self,
        decision_type,
        surface,
        context_text,
        candidates,
    ):
        if not candidates:
            return None

        key = self._key(
            decision_type,
            surface,
            context_text,
        )

        row = self.graph.conn.execute(
            """
            SELECT
                selected,
                count,
                confidence,
                candidate_set
            FROM distilled_decisions
            WHERE decision_type=?
              AND context_key=?
            ORDER BY
                count DESC,
                confidence DESC,
                id DESC
            LIMIT 1
            """,
            (
                decision_type,
                key,
            ),
        ).fetchone()

        if not row:
            return None

        selected = str(
            row["selected"]
        )

        if selected not in candidates:
            return None

        return {
            "selected": selected,
            "count": int(
                row["count"]
            ),
            "confidence": float(
                row["confidence"]
            ),
            "source": "distilled_memory",
        }

    def learn(
        self,
        decision_type,
        surface,
        context_text,
        candidates,
        selected,
        confidence,
    ):
        if (
            selected not in candidates
            or not candidates
        ):
            raise ValueError(
                "distilled decision must select "
                "one of the supplied candidates"
            )

        key = self._key(
            decision_type,
            surface,
            context_text,
        )

        candidate_set = ", ".join(
            candidates
        )

        existing = self.graph.conn.execute(
            """
            SELECT id,count
            FROM distilled_decisions
            WHERE decision_type=?
              AND context_key=?
              AND selected=?
            ORDER BY id DESC
            LIMIT 1
            """,
            (
                decision_type,
                key,
                selected,
            ),
        ).fetchone()

        if existing:
            self.graph.conn.execute(
                """
                UPDATE distilled_decisions
                SET
                    count=?,
                    confidence=?,
                    candidate_set=?
                WHERE id=?
                """,
                (
                    int(
                        existing["count"]
                    ) + 1,
                    max(
                        0.0,
                        min(
                            1.0,
                            float(
                                confidence
                            ),
                        ),
                    ),
                    candidate_set,
                    int(
                        existing["id"]
                    ),
                ),
            )
        else:
            self.graph.conn.execute(
                """
                INSERT INTO distilled_decisions(
                    decision_type,
                    context_key,
                    surface,
                    candidate_set,
                    selected,
                    count,
                    confidence,
                    created_unix
                )
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    decision_type,
                    key,
                    str(surface),
                    candidate_set,
                    selected,
                    1,
                    max(
                        0.0,
                        min(
                            1.0,
                            float(
                                confidence
                            ),
                        ),
                    ),
                    time.time(),
                ),
            )

        self.graph.conn.commit()

    def counts(self):
        row = self.graph.conn.execute(
            """
            SELECT
                COUNT(*) AS decisions,
                COALESCE(
                    SUM(count),
                    0
                ) AS observations
            FROM distilled_decisions
            """
        ).fetchone()

        return {
            "decisions": int(
                row["decisions"]
            ),
            "observations": int(
                row["observations"]
            ),
        }


def candidate_senses(
    graph,
    subject,
    limit=12,
):
    rows = graph.conn.execute(
        """
        SELECT
            n.node,
            n.label,
            n.definition
        FROM edges e
        JOIN nodes n
          ON n.node=e.object
        WHERE e.subject=?
          AND e.relation='has_sense'
        ORDER BY
            CASE
                WHEN n.definition IS NULL THEN 1
                ELSE 0
            END,
            n.node
        LIMIT ?
        """,
        (
            subject,
            int(limit),
        ),
    ).fetchall()

    return [
        {
            "node": str(
                row["node"]
            ),
            "label": str(
                row["label"]
            ),
            "definition": (
                str(
                    row["definition"]
                )
                if row["definition"]
                else ""
            ),
        }
        for row in rows
    ]


def relation_candidates(
    graph,
    subject,
    parse,
    limit=12,
):
    relations = graph.relation_vocab(
        max(
            64,
            int(limit) * 8,
        )
    )

    scored = []

    for relation in relations:
        score = relation_lexical_score(
            parse,
            relation,
        )

        # Add observed first-hop availability as a graph-derived prior.
        direct = any(
            edge.relation == relation
            for edge in graph.outgoing(
                subject,
                120,
            )
        )

        if direct:
            score += 0.15

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

    return [
        relation
        for _, relation
        in scored[:limit]
    ]




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

def relation_lexical_score(
    parse,
    relation,
):
    relation_text = str(
        relation
    ).replace(
        "_",
        " ",
    ).lower()

    aliases = {
        "definition": "definition meaning define explain",
        "is_a": "is a type kind category",
        "has_part": "has part parts contains includes",
        "part_of": "part of component belongs",
        "capable_of": "can capable able do action",
        "used_for": "used use purpose function",
        "has_property": "property characteristic trait",
        "at_location": "where located location place",
        "related_to": "related relation associated connected",
        "causes": "cause causes leads produces makes",
        "made_of": "made material made of consists",
        "has_a": "has owns contains",
    }

    phrase = relation_text
    phrase += " "
    phrase += aliases.get(
        relation,
        "",
    )

    qwords = {
        str(token["lemma"]).lower()
        for token in parse.tokens
        if token["lemma"]
    }

    qwords.update(
        token
        for token in re.findall(
            r"[a-z]+",
            parse.text.lower(),
        )
        if len(token) > 2
    )

    rwords = {
        token
        for token in re.findall(
            r"[a-z]+",
            phrase,
        )
        if len(token) > 2
    }

    overlap = len(
        qwords & rwords
    )

    # Structural compatibility supplies general signals independent of a
    # particular entity or dataset.
    structural = 0.0

    if (
        parse.question == "WH_WHAT"
        and parse.root_lemma == "be"
        and relation == "definition"
    ):
        structural += 0.60

    if (
        parse.question == "WH_WHAT"
        and any(
            word in qwords
            for word in {
                "part",
                "parts",
            }
        )
        and relation in {
            "has_part",
            "part_of",
        }
    ):
        structural += 0.50

    if (
        parse.root_lemma == "do"
        and any(
            token["text"].lower()
            in {"can", "could"}
            for token in parse.tokens
        )
        and relation == "capable_of"
    ):
        structural += 0.60

    if (
        parse.question == "WH_WHERE"
        and relation == "at_location"
    ):
        structural += 0.60

    return min(
        2.0,
        overlap / 3.0
        + structural,
    )




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
    # Concept questions are resolved against the lexical graph first.
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

                candidates = relation_candidates(
                    graph,
                    subject,
                    parse,
                    max_n,
                )

                # Keep definition among candidates for "what is" so the
                # runtime can ask the distillation teacher to discriminate
                # semantic possibilities rather than hard-coding one.
                if "definition" not in candidates:
                    candidates.insert(
                        0,
                        "definition",
                    )

                return [
                    Hypothesis(
                        subject,
                        relation,
                        "concept_lookup",
                        relation_lexical_score(
                            parse,
                            relation,
                        ),
                        {
                            "concept": concept,
                            "entity_resolution": resolution,
                            "candidate_relation_set": candidates,
                            "relation_lexical_score":
                                relation_lexical_score(
                                    parse,
                                    relation,
                                ),
                        },
                    )
                    for relation in candidates[:max_n]
                ]

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
                {
                    "entity_resolution": resolution,
                },
            )
        ]

    candidates = relation_candidates(
        graph,
        subject,
        parse,
        max_n,
    )

    return [
        Hypothesis(
            subject,
            relation,
            "relation_lookup",
            relation_lexical_score(
                parse,
                relation,
            ),
            {
                "entity_resolution": resolution,
                "candidate_relation_set":
                    candidates,
            },
        )
        for relation in candidates
    ]



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
        "intent_only": hypothesis.intent in {"conversation"},
        "steps": 0,
        "path": [],
        "target": None,
        "attention": 0,
        "exploration": 0,
        "direct_proof": False,
        "proof_kind": None,
    }

    if (
        not hypothesis.subject
        or not hypothesis.relation
        or hypothesis.intent in {
            "conversation",
            "entity_unresolved",
        }
    ):
        return base

    # Definitions are semantic terminal facts, not graph edges to traverse.
    if hypothesis.relation == "definition":
        definition = graph.definition(
            hypothesis.subject
        )
        if definition:
            return {
                **base,
                "success": True,
                "steps": 1,
                "path": ["definition"],
                "target": definition,
                "direct_proof": True,
                "proof_kind": "definition_lookup",
            }
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
                "path": [edge.relation],
                "target": edge.object,
                "direct_proof": True,
                "proof_kind": "direct_edge",
            }

    if max_depth <= 1:
        return base

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

    while queue and expansions < budget:
        _, depth, node, prefix = heapq.heappop(queue)

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
        score_by_relation = dict(ranked)

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
            next_prefix = prefix + (
                edge.relation,
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
                        "path": list(next_prefix) + [
                            goal.relation
                        ],
                        "target": goal.object,
                        "attention": attention_hits,
                        "exploration": exploration,
                        "direct_proof": False,
                        "proof_kind": "path",
                    }

            if len(next_prefix) < max_depth - 1:
                heapq.heappush(
                    queue,
                    (
                        -score_by_relation.get(
                            edge.relation,
                            0.0,
                        ),
                        len(next_prefix),
                        edge.object,
                        next_prefix,
                    )
                )

    return {
        **base,
        "steps": expansions,
        "attention": attention_hits,
        "exploration": exploration,
        "proof_kind": "search_exhausted",
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
