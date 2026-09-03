from __future__ import annotations

import heapq
import json
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


def normalize_surface(value):
    """Normalize presentation-only variation without adding semantic evidence."""
    text = str(value or "").lower().replace("’", "'")
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"(^|\s)([a-z]+)['’]s(?=\s|$)", r"\1\2", text)
    text = text.replace("'", " ")
    text = re.sub(r"\b(?:a|an|the)\s+", "", text)
    text = " ".join(text.split())
    return text


def normalize_question_text(value):
    """Expand standard interrogative contractions before structural parsing."""
    text = str(value or "").lower().replace("’", "'")
    return re.sub(
        r"\b(what|who|where|when|why|how|which)'(s|re)\b",
        lambda match: (
            f"{match.group(1)} is"
            if match.group(2) == "s"
            else f"{match.group(1)} are"
        ),
        text,
    )


def lexical_forms(value):
    """Return deterministic spelling and regular-inflection equivalents."""
    base = normalize_surface(value)
    if not base:
        return []
    forms = [base]
    if not re.fullmatch(r"[a-z]+", base):
        return forms
    if base.endswith("ies") and len(base) > 3:
        forms.append(base[:-3] + "y")
    elif base.endswith("es") and len(base) > 3:
        forms.append(base[:-2])
    elif base.endswith("s") and len(base) > 3:
        forms.append(base[:-1])
    if base.endswith("e") and len(base) > 2:
        forms.append(base[:-1] + "ing")
    forms.extend([base + "s", base + "ed", base + "ing"])
    if base.endswith("y") and len(base) > 2:
        forms.extend([base[:-1] + "ies", base[:-1] + "ied"])
    return list(dict.fromkeys(forms))


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
        required_tables = {"nodes", "edges", "relations", "metadata"}
        tables = {
            str(row["name"])
            for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        missing = required_tables - tables
        if missing:
            self.conn.close()
            raise ValueError(
                f"{self.db} is not a V678 semantic graph; missing tables: "
                f"{', '.join(sorted(missing))}. Build it with "
                "v678_semantic_network_builder.py or pass the focused graph."
            )
        # The semantic facts remain immutable by convention. V662 writes only
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

            CREATE TABLE IF NOT EXISTS realized_answers(
                cache_key TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                semantic_key TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_unix REAL NOT NULL,
                hits INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_realized_question
                ON realized_answers(question);
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
        # A distilled sense is authoritative. Never fall back to the first
        # lexical sense when an explicit sense was selected.
        if sense_node:
            row = self.conn.execute(
                """
                SELECT definition
                FROM nodes
                WHERE node=?
                  AND node_type='synset'
                  AND definition IS NOT NULL
                LIMIT 1
                """,
                (sense_node,),
            ).fetchone()

            if row and row["definition"]:
                return str(
                    row["definition"]
                )

            # Defensive check: the selected sense must really belong to the
            # lexical concept.
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

            return None

        row = self.conn.execute(
            """
            SELECT definition
            FROM nodes
            WHERE node=?
              AND definition IS NOT NULL
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
            ORDER BY
              CASE WHEN n.node LIKE 'wn:synset:' || substr(?,4) || '.%' THEN 0 ELSE 1 END,
              n.node
            LIMIT 1
            """,
            (node, node),
        ).fetchone()

        if row and row["definition"]:
            return str(
                row["definition"]
            )

        return None


    @staticmethod
    def _realization_key(
        question,
        subject,
        relation,
        target,
        path,
    ):
        import hashlib
        raw = json.dumps(
            {
                "schema": "v665",
                "question": str(question).strip().lower(),
                "subject": subject,
                "relation": relation,
                "target": target,
                "path": list(path or []),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()

    def get_realized_answer(
        self,
        question,
        subject,
        relation,
        target,
        path,
    ):
        key = self._realization_key(
            question,
            subject,
            relation,
            target,
            path,
        )
        row = self.conn.execute(
            """
            SELECT answer,hits
            FROM realized_answers
            WHERE cache_key=?
            LIMIT 1
            """,
            (key,),
        ).fetchone()

        if not row:
            return None

        self.conn.execute(
            """
            UPDATE realized_answers
            SET hits=?
            WHERE cache_key=?
            """,
            (
                int(row["hits"]) + 1,
                key,
            ),
        )
        self.conn.commit()

        return {
            "answer": str(row["answer"]),
            "source": "semantic_answer_cache",
            "hits": int(row["hits"]) + 1,
        }

    def save_realized_answer(
        self,
        question,
        subject,
        relation,
        target,
        path,
        answer,
    ):
        key = self._realization_key(
            question,
            subject,
            relation,
            target,
            path,
        )

        self.conn.execute(
            """
            INSERT OR IGNORE INTO realized_answers(
                cache_key,
                question,
                semantic_key,
                answer,
                created_unix,
                hits
            )
            VALUES(?,?,?,?,?,0)
            """,
            (
                key,
                str(question),
                key,
                str(answer),
                time.time(),
            ),
        )
        self.conn.commit()

    def realized_answer_stats(self):
        row = self.conn.execute(
            """
            SELECT
                COUNT(*) AS entries,
                COALESCE(SUM(hits),0) AS hits
            FROM realized_answers
            """
        ).fetchone()

        return {
            "entries": int(row["entries"]),
            "hits": int(row["hits"]),
        }


    def target_matches_terms(self, target, target_terms):
        """Return True when a graph node is the requested semantic argument.

        This is a generic argument-grounding primitive. It does not know
        anything about properties, adjectives, types, or specific relations.
        """
        term_groups=[]
        for value in (target_terms or []):
            text=normalize_surface(value)
            if text and all(text not in group for group in term_groups):
                term_groups.append(lexical_forms(text))

        terms = [
            term
            for group in term_groups
            for term in group
        ]

        if not terms:
            return True

        target_text=str(target or "").lower()
        label=self.node_label(target).lower()
        normalized=target_text

        # Exact canonical/label matches are preferred; token-boundary
        # containment handles multi-word labels and canonical node IDs.
        candidates={
            target_text,
            label,
            normalized,
        }
        for term in terms:
            if term in candidates or ("en:" + term) == target_text:
                return True
            if re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", label):
                return True
        return False

    def prove_edge(
        self,
        subject,
        relation,
        target,
    ):
        row = self.conn.execute(
            """
            SELECT 1
            FROM edges
            WHERE subject=?
              AND relation=?
              AND object=?
            LIMIT 1
            """,
            (
                subject,
                relation,
                target,
            ),
        ).fetchone()

        if row:
            return {
                "success": True,
                "steps": 1,
                "path": [relation],
                "target": target,
                "attention": 0,
                "exploration": 0,
                "direct_proof": True,
                "proof_kind": "selected_graph_fact",
            }

        return {
            "success": False,
            "steps": 0,
            "path": [],
            "target": None,
            "attention": 0,
            "exploration": 0,
            "direct_proof": False,
            "proof_kind": "selected_graph_fact_missing",
        }



    def outgoing_candidates(
        self,
        subject,
        query_text="",
        limit=8,
        question_frame="general",
    ):
        limit = max(
            1,
            int(limit),
        )

        qtokens = [
            token
            for token in re.findall(
                r"[a-z]+",
                str(query_text).lower(),
            )
            if len(token) >= 3
        ]

        meanings = {
            "is_a": "type or category",
            "has_property": "property or characteristic",
            "related_to": "general association",
            "antonym": "opposite or contrast",
            "capable_of": "ability to perform an action",
            "has_part": "part or component",
            "part_of": "component of something",
            "used_for": "purpose or use",
            "at_location": "location or place",
            "causes": "cause or effect",
            "made_of": "material or substance",
            "has_a": "possession or containment",
        }

        # First retrieve exact lexical target matches. This prevents a large
        # adjacency list from hiding the actual answer behind SQL LIMIT.
        lexical_rows = []

        for token in qtokens[:12]:
            token_rows = self.conn.execute(
                """
                SELECT DISTINCT
                    e.relation,
                    e.object,
                    COALESCE(n.label, e.object) AS label,
                    COALESCE(n.normalized, e.object) AS normalized
                FROM edges e
                LEFT JOIN nodes n
                  ON n.node=e.object
                WHERE e.subject=?
                  AND (
                    lower(COALESCE(n.label,'')) LIKE ?
                    OR lower(COALESCE(n.normalized,'')) LIKE ?
                    OR lower(e.object) LIKE ?
                  )
                LIMIT 128
                """,
                (
                    subject,
                    f"%{token}%",
                    f"%{token}%",
                    f"%{token}%",
                ),
            ).fetchall()

            lexical_rows.extend(
                token_rows
            )

        # Then add a bounded deterministic adjacency sample for alternatives.
        sample_rows = self.conn.execute(
            """
            SELECT DISTINCT
                e.relation,
                e.object,
                COALESCE(n.label, e.object) AS label,
                COALESCE(n.normalized, e.object) AS normalized
            FROM edges e
            LEFT JOIN nodes n
              ON n.node=e.object
            WHERE e.subject=?
            ORDER BY
                e.relation,
                e.object
            LIMIT 192
            """,
            (
                subject,
            ),
        ).fetchall()

        rows = list(
            lexical_rows
        ) + list(
            sample_rows
        )

        unique_rows = {}
        for row in rows:
            key = (
                str(row["relation"]),
                str(row["object"]),
            )
            unique_rows[key] = row

        frame_bias = {}

        facts = []

        for row in unique_rows.values():
            relation = str(
                row["relation"]
            )
            target = str(
                row["object"]
            )
            label = str(
                row["label"]
            )
            normalized = str(
                row["normalized"]
            )

            target_tokens = set(
                re.findall(
                    r"[a-z]+",
                    (
                        label
                        + " "
                        + normalized
                    ).lower(),
                )
            )

            overlap = len(
                set(qtokens)
                & target_tokens
            )

            facts.append(
                {
                    "id": (
                        f"{relation}|{target}"
                    ),
                    "subject": str(
                        subject
                    ),
                    "relation": relation,
                    "relation_meaning": meanings.get(
                        relation,
                        relation.replace("_"," "),
                    ),
                    "target": target,
                    "label": label,
                    "score": (
                        3.0 * overlap
                        + frame_bias.get(
                            relation,
                            0.0,
                        )
                        + 0.05
                    ),
                }
            )

        facts.sort(
            key=lambda item: (
                -item["score"],
                item["relation"],
                item["label"].lower(),
                item["target"],
            )
        )

        return facts[:limit]


    def node_label(
        self,
        node,
    ):
        row = self.conn.execute(
            """
            SELECT COALESCE(label,normalized,node) AS value
            FROM nodes
            WHERE node=?
            LIMIT 1
            """,
            (node,),
        ).fetchone()

        if row:
            return str(
                row["value"]
            )

        return str(
            node
        )



    def relation_frame_key(self, frame_signature):
        import hashlib
        import json

        raw = json.dumps(
            frame_signature,
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()

    def relation_frame_lookup(
        self,
        frame_key,
        candidates,
        min_confidence=0.85,
        min_count=2,
    ):
        if not candidates:
            return None

        candidate_set = sorted(
            str(item)
            for item in candidates
        )
        candidate_json = json.dumps(
            candidate_set,
            ensure_ascii=False,
            sort_keys=True,
        )

        rows = self.conn.execute(
            """
            SELECT selected, count, confidence
            FROM distilled_decisions
            WHERE decision_type='relation_from_frame_v662'
              AND context_key=?
              AND candidate_set=?
            ORDER BY count DESC, confidence DESC, id DESC
            """,
            (
                frame_key,
                candidate_json,
            ),
        ).fetchall()

        for row in rows:
            selected = str(row["selected"])
            count = int(row["count"])
            confidence = float(row["confidence"])

            if (
                selected in candidate_set
                and count >= min_count
                and confidence >= min_confidence
            ):
                return {
                    "selected": selected,
                    "count": count,
                    "confidence": confidence,
                    "source": "learned_frame_memory",
                }

        return None

    def relation_frame_learn(
        self,
        frame_key,
        candidates,
        selected,
        confidence,
    ):
        if selected not in candidates:
            return False

        candidate_set = sorted(
            str(item)
            for item in candidates
        )
        candidate_json = json.dumps(
            candidate_set,
            ensure_ascii=False,
            sort_keys=True,
        )

        confidence = max(
            0.0,
            min(
                1.0,
                float(confidence),
            ),
        )

        row = self.conn.execute(
            """
            SELECT id, count, confidence
            FROM distilled_decisions
            WHERE decision_type='relation_from_frame_v662'
              AND context_key=?
              AND candidate_set=?
              AND selected=?
            LIMIT 1
            """,
            (
                frame_key,
                candidate_json,
                str(selected),
            ),
        ).fetchone()

        if row:
            count = int(row["count"]) + 1
            average = (
                (
                    float(row["confidence"])
                    * (count - 1)
                )
                + confidence
            ) / count

            self.conn.execute(
                """
                UPDATE distilled_decisions
                SET count=?, confidence=?
                WHERE id=?
                """,
                (
                    count,
                    average,
                    int(row["id"]),
                ),
            )
        else:
            self.conn.execute(
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
                VALUES(?,?,?,?,?,?,?,strftime('%s','now'))
                """,
                (
                    "relation_from_frame_v662",
                    frame_key,
                    frame_key,
                    candidate_json,
                    str(selected),
                    1,
                    confidence,
                ),
            )

        self.conn.commit()
        return True



    def semantic_relation_candidates(
        self,
        subject,
        query_text="",
        local_relations=None,
        limit=32,
        examples_per_relation=4,
    ):
        subject=str(subject or "")
        query_text=str(query_text or "")

        qtokens=[
            token
            for token in re.findall(
                r"[a-z]+",
                query_text.lower(),
            )
            if len(token)>=3
        ]

        relations=[]
        for relation in (local_relations or []):
            relation=str(relation)
            if relation and relation not in relations:
                relations.append(relation)

        subject_rows=self.conn.execute(
            """
            SELECT DISTINCT relation
            FROM edges
            WHERE subject=?
            ORDER BY relation
            LIMIT ?
            """,
            (
                subject,
                max(int(limit)*4,64),
            ),
        ).fetchall()

        for row in subject_rows:
            relation=str(row["relation"])
            if relation and relation not in relations:
                relations.append(relation)

        # Ask the graph for relations that connect this subject to nodes
        # matching words requested by the question. This returns actual
        # subject-specific evidence, not global relation samples.
        requested_hits=[]
        for token in qtokens:
            rows=self.conn.execute(
                """
                SELECT
                    e.relation,
                    e.subject,
                    e.object,
                    COALESCE(s.label,e.subject) AS subject_label,
                    COALESCE(o.label,e.object) AS object_label,
                    COALESCE(o.normalized,e.object) AS object_normalized
                FROM edges e
                LEFT JOIN nodes s ON s.node=e.subject
                LEFT JOIN nodes o ON o.node=e.object
                WHERE e.subject=?
                  AND (
                    lower(COALESCE(o.normalized,''))=?
                    OR lower(COALESCE(o.label,''))=?
                    OR lower(e.object)=?
                    OR lower(o.normalized) LIKE ?
                    OR lower(o.label) LIKE ?
                  )
                ORDER BY e.relation,e.object
                LIMIT 128
                """,
                (
                    subject,
                    token,
                    token,
                    "en:"+token,
                    "%"+token+"%",
                    "%"+token+"%",
                ),
            ).fetchall()

            requested_hits.extend(rows)

        for row in requested_hits:
            relation=str(row["relation"])
            if relation and relation not in relations:
                relations.append(relation)

        relations=relations[:int(limit)]
        if not relations:
            return []

        # Build evidence only from:
        #   A) this subject's edges
        #   B) subject->requested-object matches.
        rows=self.conn.execute(
            """
            SELECT
                e.relation,
                e.subject,
                e.object,
                COALESCE(s.label,e.subject) AS subject_label,
                COALESCE(o.label,e.object) AS object_label
            FROM edges e
            LEFT JOIN nodes s ON s.node=e.subject
            LEFT JOIN nodes o ON o.node=e.object
            WHERE e.subject=?
            ORDER BY e.relation,e.object
            LIMIT ?
            """,
            (
                subject,
                max(
                    int(limit)
                    * int(examples_per_relation)
                    * 8,
                    256,
                ),
            ),
        ).fetchall()

        grouped={
            relation:[]
            for relation in relations
        }

        def relevant_object(label):
            text=str(label).lower()
            return any(
                token in text
                for token in qtokens
            )

        for row in rows:
            relation=str(row["relation"])
            if relation not in grouped:
                continue

            bucket=grouped[relation]
            item={
                "subject":str(row["subject_label"]),
                "object":str(row["object_label"]),
            }

            if (
                item not in bucket
                and (
                    relevant_object(
                        row["object_label"]
                    )
                    or len(bucket)<1
                )
                and len(bucket)<int(
                    examples_per_relation
                )
            ):
                bucket.append(item)

        # Guarantee direct requested-object evidence is visible first.
        direct_by_relation={}
        for row in requested_hits:
            relation=str(row["relation"])
            if relation not in grouped:
                continue
            direct_by_relation.setdefault(
                relation,
                [],
            ).append(
                {
                    "subject":str(
                        row["subject_label"]
                    ),
                    "object":str(
                        row["object_label"]
                    ),
                    "direct_question_match":True,
                }
            )

        result=[]
        for relation in relations:
            direct=direct_by_relation.get(
                relation,
                [],
            )
            base=grouped.get(
                relation,
                [],
            )

            merged=[]
            for item in direct+base:
                key=(
                    item.get("subject"),
                    item.get("object"),
                )
                if key not in {
                    (
                        x.get("subject"),
                        x.get("object"),
                    )
                    for x in merged
                }:
                    merged.append(item)

            result.append(
                {
                    "relation":relation,
                    "examples":merged[
                        :int(examples_per_relation)
                    ],
                    "direct_question_matches":len(
                        direct
                    ),
                }
            )

        return result









    def semantic_goal_schema(self):
        """
        Stable public semantic vocabulary. Raw graph relations are an
        implementation detail and are never exposed through this interface.
        """
        return {
            "definition": {
                "meaning":
                    "what the subject is; its meaning or definition",
                "relations": (
                    "definition",
                    "has_sense",
                ),
            },
            "type": {
                "meaning":
                    "what category or kind the subject belongs to",
                "relations": (
                    "is_a",
                ),
            },
            "property": {
                "meaning":
                    "whether the subject has a characteristic or property",
                "relations": (
                    "has_property",
                    "has_attribute",
                ),
            },
            "part": {
                "meaning":
                    "what physical or conceptual parts the subject has",
                "relations": (
                    "has_part",
                    "has_a",
                ),
            },
            "capability": {
                "meaning":
                    "what the subject can do or is able to do",
                "relations": (
                    "capable_of",
                ),
            },
            "location": {
                "meaning":
                    "where the subject is located or found",
                "relations": (
                    "at_location",
                ),
            },
            "purpose": {
                "meaning":
                    "what the subject is used for or intended for",
                "relations": (
                    "used_for",
                ),
            },
            "cause": {
                "meaning":
                    "what the subject causes or what causes it",
                "relations": (
                    "causes",
                ),
            },
            "association": {
                "meaning":
                    "a general semantic association",
                "relations": (
                    "related_to",
                ),
            },
            "contrast": {
                "meaning":
                    "an opposite or contrasting concept",
                "relations": (
                    "antonym",
                ),
            },
        }

    def semantic_goal_candidates(self):
        schema=self.semantic_goal_schema()
        result=[]
        for goal,info in schema.items():
            available=[]
            for relation in info["relations"]:
                row=self.conn.execute(
                    """
                    SELECT 1
                    FROM edges
                    WHERE relation=?
                    LIMIT 1
                    """,
                    (relation,),
                ).fetchone()
                if row:
                    available.append(relation)

            if available:
                result.append({
                    "goal":goal,
                    "meaning":info["meaning"],
                    "available_relations":available,
                })

        return result

    def semantic_relations_for_goal(
        self,
        goal,
    ):
        info=self.semantic_goal_schema().get(
            str(goal)
        )
        if not info:
            return []
        return list(info["relations"])

    def find_goal_facts(
        self,
        subject,
        goal,
        query_text="",
        target_terms=None,
        limit=24,
    ):
        """
        Return graph facts that implement a clean semantic goal.

        V665 adds argument grounding: when the question contains an explicit
        target term (for example ``brown`` in ``is it brown?``), the adapter
        first restricts facts to graph objects matching that term. This is not
        a relation-specific rule; it is a generic subject + goal + argument
        constraint.

        Raw graph relations remain internal to this adapter.
        """
        relations=set(
            self.semantic_relations_for_goal(
                goal
            )
        )
        if not relations:
            return []

        term_groups=[]
        for value in (target_terms or []):
            text=normalize_surface(value)
            if text and all(text not in group for group in term_groups):
                term_groups.append(lexical_forms(text))
        terms = [term for group in term_groups for term in group]

        placeholders=",".join(
            "?" for _ in relations
        )

        params=[str(subject), *sorted(relations)]
        target_clause=""
        if terms:
            # Every explicit argument term constrains the same graph object.
            # OR would let a query such as "big heart" select an unrelated
            # edge to "big", producing a false positive.
            clauses = []
            for group in term_groups:
                alternatives = []
                for _term in group:
                    alternatives.append(
                        "(instr(lower(COALESCE(n.normalized,'')),?) > 0 "
                        "OR instr(lower(COALESCE(n.label,'')),?) > 0 "
                        "OR instr(lower(e.object),?) > 0)"
                    )
                    params.extend([_term, _term, _term])
                clauses.append(
                    "(" + " OR ".join(alternatives) + ")"
                )
            target_clause=" AND " + " AND ".join(clauses)

        params.append(int(limit))

        rows=self.conn.execute(
            f"""
            SELECT
                e.subject,
                e.relation,
                e.object,
                COALESCE(
                    n.label,
                    e.object
                ) AS label,
                COALESCE(
                    n.normalized,
                    e.object
                ) AS normalized
            FROM edges e
            LEFT JOIN nodes n
              ON n.node=e.object
            WHERE e.subject=?
              AND e.relation IN ({placeholders})
              {target_clause}
            ORDER BY e.relation,e.object
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()

        facts=[
            {
                "subject":str(row["subject"]),
                "relation":str(row["relation"]),
                "object":str(row["object"]),
                "label":str(row["label"]),
                "normalized":str(row["normalized"]),
                "goal":str(goal),
            }
            for row in rows
        ]

        # If the question supplies an exact graph label, retain exact target
        # evidence instead of a longer phrase that merely contains that word
        # (for example, ``black`` versus ``see in black and white``).
        if terms:
            exact_facts = [
                item for item in facts
                if any(
                    term in {
                        str(item.get("normalized", "")).lower(),
                        str(item.get("label", "")).lower(),
                        str(item.get("object", "")).lower(),
                    }
                    for term in terms
                )
            ]
            facts = exact_facts

        # Deterministic lexical evidence score for downstream diagnostics.
        # The teacher never sees these raw relation names.
        if terms:
            def fact_score(item):
                value=(
                    str(item.get("normalized", ""))
                    + " "
                    + str(item.get("label", ""))
                ).lower()
                return sum(1 for term in terms if term in value)

            facts.sort(
                key=lambda item:(
                    -fact_score(item),
                    item["relation"],
                    item["label"].lower(),
                    item["object"],
                )
            )

        return facts[:int(limit)]

    def has_goal_path(self, subject, goal, target_terms, max_depth=2):
        """Check bounded same-goal reachability without lexical goal heuristics."""
        relations = set(self.semantic_relations_for_goal(goal))
        if not relations or not target_terms:
            return False
        frontier = {str(subject)}
        visited = set(frontier)
        for _ in range(max(1, int(max_depth))):
            next_frontier = set()
            for node in frontier:
                for edge in self.outgoing(node, 256):
                    if edge.relation not in relations:
                        continue
                    if self.target_matches_terms(edge.object, target_terms):
                        return True
                    if edge.object not in visited:
                        visited.add(edge.object)
                        next_frontier.add(edge.object)
            frontier = next_frontier
            if not frontier:
                break
        return False

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

        normalized = normalize_surface(mention)

        if normalized.endswith("s") and len(normalized) > 3:
            singular = normalized[:-1]
            row = self.conn.execute(
                """
                SELECT node,label,node_type FROM nodes
                WHERE normalized=? AND node_type='concept'
                ORDER BY node LIMIT 1
                """,
                (singular,),
            ).fetchone()
            if row:
                return {
                    "status": "resolved",
                    "mention": mention,
                    "canonical": str(row["node"]),
                    "confidence": 0.95,
                    "candidates": [{
                        "node": str(row["node"]), "kind": "plural_singular",
                        "score": 0.95, "accepted": True, "label": row["label"],
                    }],
                }

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

        if normalized.endswith("s") and len(normalized) > 3:
            singular = normalized[:-1]
            row = self.conn.execute(
                """
                SELECT node,label,node_type FROM nodes
                WHERE normalized=? AND node_type='concept'
                ORDER BY node LIMIT 1
                """,
                (singular,),
            ).fetchone()
            if row:
                return {
                    "status": "resolved",
                    "mention": mention,
                    "canonical": str(row["node"]),
                    "confidence": 0.95,
                    "candidates": [{
                        "node": str(row["node"]), "kind": "plural_singular",
                        "score": 0.95, "accepted": True, "label": row["label"],
                    }],
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




PRONOUNS = {
    "it", "he", "she", "they", "them", "him", "her",
    "this", "that", "these", "those",
}


def contextual_entity_reference(parse, context):
    tokens = set(
        re.findall(
            r"[a-z]+",
            parse.text.lower(),
        )
    )
    if tokens.intersection(PRONOUNS):
        return context.active_subject
    return None



def structural_question_frame(parse):
    """
    Stable structural signature from frozen spaCy output.
    Lexical subject/object values are intentionally excluded so a learned
    relation mapping can transfer across concepts.
    """
    tokens = [
        item
        for item in (parse.tokens or [])
        if isinstance(item, dict)
    ]

    return {
        "question": str(
            parse.question or ""
        ),
        "root": str(
            parse.root_lemma or ""
        ).lower(),
        "pos": sorted({
            str(item.get("pos", ""))
            for item in tokens
        }),
        "dependencies": sorted({
            str(item.get("dep", ""))
            for item in tokens
        }),
        "has_pronoun": any(
            item.get("pos") == "PRON"
            for item in tokens
        ),
        "has_auxiliary": any(
            item.get("pos") == "AUX"
            for item in tokens
        ),
        "has_adjective": any(
            item.get("pos") == "ADJ"
            for item in tokens
        ),
        "has_noun": any(
            item.get("pos") == "NOUN"
            for item in tokens
        ),
        "has_verb": any(
            item.get("pos") == "VERB"
            for item in tokens
        ),
    }




def structural_frame_key(parse):
    import hashlib
    import json

    raw = json.dumps(
        structural_question_frame(parse),
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()




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
    excluded = wh | refs | {"a", "an", "the"}

    tokens = [
        item for item in (parse.tokens or [])
        if isinstance(item, dict)
    ]
    question = normalize_question_text(parse.text)
    # The semantic subject may occur outside spaCy's nominal-subject position
    # in possessive and prepositional part questions.
    for pattern in (
        r"\b([a-z0-9_-]+)'s\s+(?:part|parts|component|components)\b",
        r"\b(?:part|parts|component|components)\s+of\s+(?:(?:a|an|the)\s+)?([a-z0-9_-]+)\b",
        r"\b(?:what|which)\s+(?:part|parts|component|components)\s+(?:do|does|did)\s+(?:(?:a|an|the)\s+)?([a-z0-9_-]+)\s+(?:have|has|contain|contains)\b",
        r"^(?:do|does|did|can|could|is|are|was|were)\s+(?:(?:a|an|the)\s+)?([a-z0-9_-]+)\b",
    ):
        match = re.search(pattern, question)
        if match:
            return normalize_surface(match.group(1))

    for item in tokens:
        if str(item.get("dep", "")).lower() in {"nsubj", "nsubjpass"}:
            value = str(item.get("text", "")).strip()
            if value and value.lower() not in excluded:
                return value

    # spaCy can label the lexical subject of short copular questions (for
    # example, "Is dog fun?") as a compound. In a polar copular form the
    # first noun after the auxiliary is still the subject.
    if str(parse.question or "") == "QUESTION":
        for item in tokens[1:]:
            if str(item.get("pos", "")).upper() in {"NOUN", "PROPN"}:
                value = str(item.get("text", "")).strip()
                if value and value.lower() not in refs:
                    return value

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
            if part.lower() not in excluded
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
    question_text = normalize_question_text(parse.text)
    words = re.findall(
        r"[a-z]+",
        question_text,
    )
    definition_frame = (
        len(words) >= 2
        and (
            (
                words[0] in {"what", "who"}
                and words[1] in {"is", "are", "was", "were"}
            )
            or (
                words[0] == "what"
                and len(words) >= 3
                and words[1] in {"does", "do", "did"}
                and words[-1] == "mean"
            )
        )
    )
    return (
        definition_frame
        and not re.search(
            r"\b(?:part|parts|component|components)\b",
            question_text,
        )
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

    contextual_subject = contextual_entity_reference(
        parse,
        context,
    )

    mention = entity_mention(
        parse
    )

    if contextual_subject:
        resolution = {
            "status": "context_resolved",
            "mention": mention,
            "canonical": contextual_subject,
            "confidence": 0.85,
            "candidates": [],
        }
    else:
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
        not in {"resolved", "context_resolved"}
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
        sense = (
            hypothesis.evidence.get(
                "distilled_sense"
            )
            if isinstance(
                hypothesis.evidence,
                dict,
            )
            else None
        )

        sense_node = (
            sense.get("node")
            if isinstance(
                sense,
                dict,
            )
            else None
        )

        definition = graph.definition(
            hypothesis.subject,
            sense_node=sense_node,
        )
        if definition:
            return {
                **base,
                "success": True,
                "steps": 1,
                "path": ["definition"],
                "target": definition,
                "sense_node": sense_node,
                "direct_proof": True,
                "proof_kind": "definition_lookup",
            }
        return base

    target_terms=(
        hypothesis.evidence.get("target_terms", [])
        if isinstance(hypothesis.evidence, dict)
        else []
    )

    for edge in graph.outgoing(
        hypothesis.subject,
        per_node,
    ):
        if (
            edge.relation == hypothesis.relation
            and graph.target_matches_terms(
                edge.object,
                target_terms,
            )
        ):
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
                if (
                    goal.relation == hypothesis.relation
                    and graph.target_matches_terms(
                        goal.object,
                        target_terms,
                    )
                ):
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
