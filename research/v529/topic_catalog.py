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
