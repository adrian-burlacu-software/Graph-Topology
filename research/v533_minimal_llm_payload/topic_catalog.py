from __future__ import annotations

from collections import defaultdict
import re
import sqlite3

BLOCKED_PREDICATES = {
    'in_domain','domain','source','provenance','dataset','node_type','type','label',
    'nsubj','nsubjpass','obj','dobj','iobj','ccomp','xcomp','amod','advmod','nmod',
    'obl','oblique','root','dep','aux','auxpass','cop','det','case','mark','punct',
    'conj','cc','compound','appos','acl','advcl',
}

GENERIC_SUBJECTS = {
    'entity','thing','object','concept','someone','something','word','term',
    'question','answer','user','assistant','i','you','we','they','it',
}


def topic_rows(con: sqlite3.Connection, limit: int = 40) -> list[dict]:
    """Return the strongest semantic topics in the local knowledge graph.

    Score rewards facts, predicate diversity, confidence, source diversity,
    and answerable coverage. It intentionally ignores parser artifacts and
    generic concepts so this is useful as a conversation guide.
    """
    rows = con.execute(
        """
        SELECT
            c.canonical AS topic,
            COUNT(*) AS fact_count,
            COUNT(DISTINCT f.predicate) AS predicate_count,
            COUNT(DISTINCT f.source_id) AS source_count,
            COUNT(DISTINCT f.fact_type) AS type_count,
            AVG(f.confidence) AS avg_confidence,
            SUM(CASE WHEN f.answerable=1 THEN 1 ELSE 0 END) AS answerable_count
        FROM facts f
        JOIN concepts c ON c.concept_id=f.subject_id
        WHERE f.answerable=1
          AND f.predicate NOT IN ({})
        GROUP BY c.canonical
        HAVING COUNT(*) >= 3
        ORDER BY
            (
                log(1 + COUNT(*)) * 4.0
                + MIN(COUNT(DISTINCT f.predicate), 8) * 2.5
                + MIN(COUNT(DISTINCT f.source_id), 5) * 1.5
                + MIN(COUNT(DISTINCT f.fact_type), 4) * 1.0
                + AVG(f.confidence) * 2.0
            ) DESC,
            COUNT(*) DESC,
            c.canonical ASC
        LIMIT ?
        """.format(",".join("?" for _ in BLOCKED_PREDICATES)),
        [*sorted(BLOCKED_PREDICATES), int(limit)],
    ).fetchall()

    result = []
    for row in rows:
        topic = str(row["topic"] or "").strip().lower()
        if not topic or topic in GENERIC_SUBJECTS or len(topic) > 60:
            continue
        if not re.search(r"[a-z]", topic):
            continue
        result.append(dict(row))
    return result[:limit]


def format_topics(rows: list[dict], max_items: int = 25) -> str:
    lines = ["=== TOPICS THE ARCHITECTURE KNOWS WELL ==="]
    if not rows:
        lines.append("No sufficiently populated topics found in the current knowledge graph.")
        return "\n".join(lines)
    for idx, row in enumerate(rows[:max_items], 1):
        topic = row["topic"]
        facts = int(row["fact_count"])
        predicates = int(row["predicate_count"])
        sources = int(row["source_count"])
        lines.append(
            f"{idx:2d}. {topic:<28} facts={facts:<5d} "
            f"relations={predicates:<2d} sources={sources:<2d}"
        )
    return "\n".join(lines)


def inspect_topic(con: sqlite3.Connection, topic: str, depth: int = 2, limit: int = 12) -> dict:
    """Inspect a semantic topic without touching live conversation state.

    Returns direct facts plus recursively discovered neighboring concepts.
    Only indexed long-term `facts` are traversed.
    """
    topic = str(topic or "").strip().lower()
    depth = max(0, min(int(depth), 3))
    if not topic:
        return {"topic": "", "depth": depth, "levels": []}

    levels = []
    frontier = [topic]
    visited = {topic}

    for level in range(depth + 1):
        nodes = []
        next_frontier = []
        for node in frontier:
            rows = con.execute(
                """
                SELECT c.canonical AS subject, f.predicate,
                       COALESCE(o.canonical,f.object_text) AS object_text,
                       f.fact_type, f.domain, f.confidence, f.answerable
                FROM facts f
                JOIN concepts c ON c.concept_id=f.subject_id
                LEFT JOIN concepts o ON o.concept_id=f.object_id
                WHERE lower(c.canonical)=lower(?)
                  AND f.answerable=1
                  AND f.predicate NOT IN ({})
                ORDER BY f.confidence DESC, f.frequency DESC
                LIMIT ?
                """.format(",".join("?" for _ in BLOCKED_PREDICATES)),
                [node, *sorted(BLOCKED_PREDICATES), int(limit)],
            ).fetchall()
            for r in rows:
                item = dict(r)
                item["level"] = level
                nodes.append(item)
                obj = str(item.get("object_text") or "").strip().lower()
                if obj and re.fullmatch(r"[a-z][a-z0-9 _-]{1,60}", obj):
                    if obj not in visited and len(obj.split()) <= 4:
                        visited.add(obj)
                        next_frontier.append(obj)
        levels.append({"level": level, "nodes": nodes})
        frontier = next_frontier[: max(4, limit // 2)]
        if not frontier:
            break
    return {"topic": topic, "depth": depth, "levels": levels}


def format_topic_inspection(report: dict, max_per_level: int = 12) -> str:
    lines = [
        f"=== KNOWLEDGE INSPECTION: {report.get('topic','')} ===",
        f"depth={report.get('depth',0)}",
        "(conversation state is not consulted)",
    ]
    for level in report.get("levels", []):
        lines.append(f"-- LEVEL {level['level']} --")
        nodes = level.get("nodes", [])
        if not nodes:
            lines.append("  <no direct facts>")
            continue
        for fact in nodes[:max_per_level]:
            lines.append(
                f"  {fact['subject']} --{fact['predicate']}--> {fact['object_text']} "
                f"[{fact['fact_type']}, conf={float(fact['confidence']):.2f}]"
            )
    return "\n".join(lines)
