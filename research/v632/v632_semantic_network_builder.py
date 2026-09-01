from __future__ import annotations

import argparse
import csv
import gzip
import math
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote

from v632_semantic_core import (
    RELATION_PHRASES,
)

NS = "http://wordnet-rdf.princeton.edu/wn#"

CN_RELATIONS = {
    "/r/IsA": "is_a",
    "/r/PartOf": "part_of",
    "/r/HasA": "has_a",
    "/r/UsedFor": "used_for",
    "/r/CapableOf": "capable_of",
    "/r/AtLocation": "at_location",
    "/r/HasProperty": "has_property",
    "/r/RelatedTo": "related_to",
    "/r/Causes": "causes",
    "/r/MadeOf": "made_of",
    "/r/Synonym": "synonym",
    "/r/Antonym": "antonym",
}

WN_RELATIONS = {
    "hypernym": "is_a",
    "instance_hypernym": "is_a",
    "part_holonym": "has_part",
    "member_holonym": "has_part",
    "substance_holonym": "has_part",
    "part_meronym": "part_of",
    "member_meronym": "part_of",
    "substance_meronym": "part_of",
    "entailment": "causes",
    "causes": "causes",
    "similar_tos": "related_to",
}


def normalize_word(value):
    value = unquote(
        str(value).strip()
    ).lower()

    value = value.replace(
        "_",
        " ",
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    if not re.fullmatch(
        r"[a-z]+",
        value,
    ):
        return None

    return value


def conceptnet_word(value):
    value = unquote(
        str(value).strip()
    )

    if not value.startswith(
        "/c/en/"
    ):
        return None

    parts = value.split("/")
    if len(parts) < 4:
        return None

    return normalize_word(
        parts[3]
    )


def get_wordnet():
    try:
        from nltk.corpus import wordnet as wn
        # Force corpus access now for a useful error.
        next(
            wn.all_lemma_names(
                lang="eng"
            )
        )
        return wn
    except Exception as exc:
        raise RuntimeError(
            "NLTK WordNet is unavailable. "
            "Install/download it with:\n"
            "  python -m nltk.downloader wordnet omw-1.4\n"
            f"Original error: {exc}"
        ) from exc


def common_word_scores(wn):
    scores = Counter()

    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            word = normalize_word(
                lemma.name()
            )
            if not word:
                continue

            count = (
                int(
                    lemma.count()
                )
                if hasattr(
                    lemma,
                    "count",
                )
                else 0
            )

            scores[word] += (
                count
                if count > 0
                else 0.25
            )

    return scores


def choose_vocabulary(
    wn,
    size,
    min_wordnet_score=0.25,
):
    scores = common_word_scores(
        wn
    )

    # Conservative beginner-word shape:
    # single alphabetic lowercase tokens only.
    candidates = [
        (
            score,
            word,
        )
        for word, score in scores.items()
        if score >= min_wordnet_score
        and len(word) >= 2
        and len(word) <= 18
    ]

    candidates.sort(
        key=lambda item: (
            -item[0],
            len(item[1]),
            item[1],
        )
    )

    selected = [
        word
        for _, word
        in candidates[
            : int(size)
        ]
    ]

    if len(selected) < int(size):
        raise RuntimeError(
            f"WordNet produced only {len(selected)} suitable words; "
            f"requested {size}."
        )

    return selected


def insert_wordnet(
    conn,
    wn,
    vocabulary,
):
    vocab = set(vocabulary)
    edge_buffer = set()
    definition_by_word = {}

    for word in vocabulary:
        synsets = wn.synsets(
            word,
            lang="eng",
        )

        if not synsets:
            continue

        # Highest empirical WordNet lemma count / then earliest synset.
        best = sorted(
            synsets,
            key=lambda synset: (
                -sum(
                    max(
                        0,
                        lemma.count(),
                    )
                    for lemma in synset.lemmas()
                ),
                synset.name(),
            ),
        )[0]

        gloss = best.definition()
        if gloss:
            definition_by_word[word] = gloss

        for relation_name, canonical in WN_RELATIONS.items():
            if not hasattr(
                best,
                relation_name,
            ):
                continue

            targets = getattr(
                best,
                relation_name,
            )()

            for target_synset in targets:
                target_words = set()

                for target_lemma in target_synset.lemmas():
                    target_word = normalize_word(
                        target_lemma.name()
                    )
                    if (
                        target_word
                        and target_word in vocab
                    ):
                        target_words.add(
                            target_word
                        )

                for target_word in target_words:
                    edge_buffer.add(
                        (
                            word,
                            canonical,
                            target_word,
                            "wordnet",
                        )
                    )

        # Synonym edges within the selected vocabulary.
        for other_lemma in best.lemmas():
            other = normalize_word(
                other_lemma.name()
            )
            if (
                other
                and other in vocab
                and other != word
            ):
                edge_buffer.add(
                    (
                        word,
                        "synonym",
                        other,
                        "wordnet",
                    )
                )

    conn.executemany(
        """
        INSERT OR IGNORE INTO edges(
            subject,
            relation,
            object,
            source
        )
        VALUES (?,?,?,?)
        """,
        list(
            edge_buffer
        ),
    )

    return definition_by_word


def insert_conceptnet(
    conn,
    conceptnet_path,
    vocabulary,
):
    vocab = set(
        vocabulary
    )
    edge_count = 0
    rejected = 0

    with gzip.open(
        conceptnet_path,
        "rt",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as handle:
        reader = csv.reader(
            handle
        )

        for row in reader:
            if len(row) < 4:
                continue

            relation = CN_RELATIONS.get(
                row[1]
            )
            if not relation:
                continue

            start = conceptnet_word(
                row[2]
            )
            end = conceptnet_word(
                row[3]
            )

            if (
                not start
                or not end
                or start not in vocab
                or end not in vocab
            ):
                rejected += 1
                continue

            conn.execute(
                """
                INSERT OR IGNORE INTO edges(
                    subject,
                    relation,
                    object,
                    source
                )
                VALUES (?,?,?,?)
                """,
                (
                    start,
                    relation,
                    end,
                    "conceptnet",
                ),
            )
            edge_count += 1

            if edge_count % 50000 == 0:
                conn.commit()
                print(
                    f"    ConceptNet edges retained: "
                    f"{edge_count:,}",
                    flush=True,
                )

    conn.commit()
    return edge_count


def create_database(
    output,
    vocabulary,
    definitions,
):
    if output.exists():
        output.unlink()

    conn = sqlite3.connect(
        str(output)
    )
    conn.execute(
        "PRAGMA journal_mode=WAL"
    )
    conn.execute(
        "PRAGMA synchronous=NORMAL"
    )
    conn.execute(
        """
        CREATE TABLE nodes(
            node TEXT PRIMARY KEY,
            normalized TEXT UNIQUE NOT NULL,
            label TEXT NOT NULL,
            definition TEXT,
            is_common INTEGER NOT NULL DEFAULT 1
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE edges(
            subject TEXT NOT NULL,
            relation TEXT NOT NULL,
            object TEXT NOT NULL,
            source TEXT NOT NULL,
            UNIQUE(subject,relation,object)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE relations(
            relation TEXT PRIMARY KEY,
            phrases TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE metadata(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )

    conn.executemany(
        """
        INSERT INTO nodes(
            node,
            normalized,
            label,
            definition,
            is_common
        )
        VALUES (?,?,?,?,1)
        """,
        [
            (
                word,
                word,
                word,
                definitions.get(
                    word
                ),
            )
            for word in vocabulary
        ],
    )

    conn.executemany(
        """
        INSERT INTO relations(
            relation,
            phrases
        )
        VALUES (?,?)
        """,
        list(
            RELATION_PHRASES.items()
        ),
    )

    return conn


def add_definition_edges(
    conn,
    definitions,
):
    conn.executemany(
        """
        INSERT OR IGNORE INTO edges(
            subject,
            relation,
            object,
            source
        )
        VALUES (?,?,?,?)
        """,
        [
            (
                word,
                "definition",
                definition,
                "wordnet",
            )
            for word, definition
            in definitions.items()
            if definition
        ],
    )


def indexes(conn):
    conn.executescript(
        """
        CREATE INDEX idx_edges_subject
            ON edges(subject);

        CREATE INDEX idx_edges_relation
            ON edges(relation);

        CREATE INDEX idx_edges_object
            ON edges(object);

        CREATE INDEX idx_edges_subject_relation
            ON edges(subject,relation);

        CREATE INDEX idx_nodes_normalized
            ON nodes(normalized);

        ANALYZE;
        """
    )


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Build a compact beginner semantic graph from "
            "NLTK WordNet + ConceptNet 5.7."
        )
    )
    ap.add_argument(
        "--conceptnet",
        required=True,
    )
    ap.add_argument(
        "--output",
        required=True,
    )
    ap.add_argument(
        "--vocab-size",
        type=int,
        default=4000,
    )
    ap.add_argument(
        "--min-degree",
        type=int,
        default=1,
    )
    args = ap.parse_args()

    conceptnet = Path(
        args.conceptnet
    ).resolve()
    output = Path(
        args.output
    ).resolve()

    if not conceptnet.exists():
        raise FileNotFoundError(
            f"ConceptNet file not found: {conceptnet}"
        )

    print(
        "=== V632 SEMANTIC NETWORK BUILDER ===",
        flush=True,
    )
    print(
        f"ConceptNet source : {conceptnet}",
        flush=True,
    )
    print(
        f"Output            : {output}",
        flush=True,
    )
    print(
        f"Target vocabulary : {args.vocab_size:,}",
        flush=True,
    )

    started = time.perf_counter()

    print(
        "[1/5] loading NLTK WordNet...",
        flush=True,
    )
    wn = get_wordnet()

    print(
        "[2/5] selecting beginner vocabulary...",
        flush=True,
    )
    vocabulary = choose_vocabulary(
        wn,
        args.vocab_size,
    )
    print(
        f"    words={len(vocabulary):,}",
        flush=True,
    )

    conn = create_database(
        output,
        vocabulary,
        {},
    )

    print(
        "[3/5] ingesting WordNet relations...",
        flush=True,
    )
    definitions = insert_wordnet(
        conn,
        wn,
        vocabulary,
    )
    add_definition_edges(
        conn,
        definitions,
    )
    conn.commit()

    print(
        f"    definitions={len(definitions):,}",
        flush=True,
    )
    print(
        f"    WordNet edges retained="
        f"{conn.execute('SELECT COUNT(*) FROM edges').fetchone()[0]:,}",
        flush=True,
    )

    print(
        "[4/5] streaming ConceptNet 5.7...",
        flush=True,
    )
    cn_added = insert_conceptnet(
        conn,
        conceptnet,
        vocabulary,
    )
    print(
        f"    ConceptNet rows retained="
        f"{cn_added:,}",
        flush=True,
    )

    print(
        "[5/5] indexing compact network...",
        flush=True,
    )
    indexes(conn)

    conn.execute(
        """
        INSERT INTO metadata(key,value)
        VALUES
          ('version','V632'),
          ('vocabulary_size',?),
          ('conceptnet_source',?),
          ('wordnet_source','NLTK WordNet'),
          ('created_unix',?),
          ('target_design','beginner semantic dictionary'),
          ('edge_count',?)
        """,
        (
            str(len(vocabulary)),
            str(conceptnet),
            str(time.time()),
            str(
                conn.execute(
                    "SELECT COUNT(*) FROM edges"
                ).fetchone()[0]
            ),
        ),
    )

    conn.commit()

    edge_count = conn.execute(
        "SELECT COUNT(*) FROM edges"
    ).fetchone()[0]

    relation_counts = conn.execute(
        """
        SELECT relation,COUNT(*)
        FROM edges
        GROUP BY relation
        ORDER BY COUNT(*) DESC
        """
    ).fetchall()

    degree = conn.execute(
        """
        SELECT AVG(n), MIN(n), MAX(n)
        FROM (
            SELECT subject,COUNT(*) AS n
            FROM edges
            GROUP BY subject
        )
        """
    ).fetchone()

    conn.close()

    elapsed = (
        time.perf_counter()
        - started
    )

    print(
        "\n=== V632 BUILD COMPLETE ===",
        flush=True,
    )
    print(
        f"vocabulary           : {len(vocabulary):,}",
        flush=True,
    )
    print(
        f"edges                : {edge_count:,}",
        flush=True,
    )
    print(
        f"mean outgoing degree : "
        f"{float(degree[0] or 0):.2f}",
        flush=True,
    )
    print(
        f"min outgoing degree  : "
        f"{int(degree[1] or 0)}",
        flush=True,
    )
    print(
        f"max outgoing degree  : "
        f"{int(degree[2] or 0)}",
        flush=True,
    )
    print(
        "top relations:",
        flush=True,
    )

    for row in relation_counts[:12]:
        print(
            f"  {row[0]:<16} {row[1]:,}",
            flush=True,
        )

    print(
        f"elapsed              : {elapsed:.2f}s",
        flush=True,
    )
    print(
        f"DATABASE             : {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
