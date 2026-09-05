from __future__ import annotations

import argparse
import json
import csv
import gzip
import re
import sqlite3
import time
from pathlib import Path
from urllib.parse import unquote


CN_PREFIX = "/c/en/"
CN_REL_PREFIX = "/r/"


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

    return value or None


def conceptnet_node(uri):
    """
    Accept every English ConceptNet concept URI, including sense-qualified
    forms such as /c/en/dog/n. The canonical graph node is the base English
    concept (/c/en/dog -> en:dog), while all English assertions are retained.
    """
    uri = unquote(
        str(uri).strip()
    )

    if not uri.startswith(
        CN_PREFIX
    ):
        return None

    parts = uri.split("/")

    if len(parts) < 4:
        return None

    if parts[2] != "en":
        return None

    term = normalize_word(
        parts[3]
    )

    if not term:
        return None

    return (
        "en:"
        + term
    )


def conceptnet_relation(uri):
    uri = unquote(
        str(uri).strip()
    )

    if not uri.startswith(
        CN_REL_PREFIX
    ):
        return None

    value = uri[
        len(CN_REL_PREFIX):
    ].strip()

    if not value:
        return None

    # Keep every relation, but give the cognitive layer stable readable names.
    value = re.sub(
        r"(?<!^)([A-Z])",
        r"_\1",
        value,
    ).lower()

    return value


def wordnet_word_node(lemma):
    value = normalize_word(
        lemma.name()
    )
    if not value:
        return None
    return (
        "en:"
        + value
    )




def get_wordnet():
    try:
        from nltk.corpus import wordnet as wn
        # Force corpus access now so missing corpora fail immediately.
        next(wn.all_synsets())
        return wn
    except Exception as exc:
        raise RuntimeError(
            "NLTK WordNet is unavailable. "
            "Run:\n"
            "  python -m nltk.downloader wordnet omw-1.4\n"
            f"Original error: {exc}"
        ) from exc



def create_db(path):
    path = Path(path)
    if path.exists():
        path.unlink()

    conn = sqlite3.connect(
        str(path),
        timeout=60.0,
    )
    conn.execute(
        "PRAGMA journal_mode=WAL"
    )
    conn.execute(
        "PRAGMA synchronous=NORMAL"
    )
    conn.execute(
        "PRAGMA temp_store=MEMORY"
    )
    conn.execute(
        "PRAGMA cache_size=-500000"
    )

    conn.executescript(
        """
        CREATE TABLE nodes(
            node TEXT PRIMARY KEY,
            normalized TEXT,
            label TEXT,
            definition TEXT,
            source_mask INTEGER NOT NULL DEFAULT 0,
            node_type TEXT NOT NULL DEFAULT 'concept'
        );

        CREATE TABLE edges(
            subject TEXT NOT NULL,
            relation TEXT NOT NULL,
            object TEXT NOT NULL,
            source TEXT NOT NULL,
            UNIQUE(subject, relation, object)
        );

        CREATE TABLE relations(
            relation TEXT PRIMARY KEY,
            phrases TEXT NOT NULL
        );

        CREATE TABLE metadata(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    return conn


def upsert_node(
    conn,
    node,
    label,
    definition=None,
    source_mask=0,
    node_type="concept",
):
    conn.execute(
        """
        INSERT INTO nodes(
            node,
            normalized,
            label,
            definition,
            source_mask,
            node_type
        )
        VALUES(?,?,?,?,?,?)
        ON CONFLICT(node) DO UPDATE SET
            normalized=COALESCE(
                nodes.normalized,
                excluded.normalized
            ),
            label=COALESCE(
                nodes.label,
                excluded.label
            ),
            definition=COALESCE(
                nodes.definition,
                excluded.definition
            ),
            source_mask=nodes.source_mask | excluded.source_mask,
            node_type=COALESCE(
                nodes.node_type,
                excluded.node_type
            )
        """,
        (
            node,
            normalize_word(label),
            str(label),
            definition,
            int(source_mask),
            node_type,
        ),
    )


def ingest_wordnet(
    conn,
    wn,
    batch_size=10000,
):
    edge_buffer = []
    node_buffer = {}

    synsets = 0
    lemmas = 0
    edges = 0
    definitions = 0

    def flush_nodes():
        nonlocal node_buffer
        if not node_buffer:
            return

        conn.executemany(
            """
            INSERT INTO nodes(
                node,
                normalized,
                label,
                definition,
                source_mask,
                node_type
            )
            VALUES(?,?,?,?,?,?)
            ON CONFLICT(node) DO UPDATE SET
                normalized=COALESCE(
                    nodes.normalized,
                    excluded.normalized
                ),
                label=COALESCE(
                    nodes.label,
                    excluded.label
                ),
                definition=COALESCE(
                    nodes.definition,
                    excluded.definition
                ),
                source_mask=nodes.source_mask | excluded.source_mask
            """,
            list(
                node_buffer.values()
            ),
        )
        node_buffer = {}

    def flush_edges():
        nonlocal edge_buffer, edges
        if not edge_buffer:
            return

        conn.executemany(
            """
            INSERT OR IGNORE INTO edges(
                subject,
                relation,
                object,
                source
            )
            VALUES(?,?,?,?)
            """,
            edge_buffer,
        )
        edges += len(edge_buffer)
        edge_buffer = []

    relation_map = {
        "hypernyms": "is_a",
        "instance_hypernyms": "is_a",
        "hyponyms": "has_subtype",
        "instance_hyponyms": "has_subtype",
        "part_holonyms": "has_part",
        "member_holonyms": "has_part",
        "substance_holonyms": "has_part",
        "part_meronyms": "part_of",
        "member_meronyms": "part_of",
        "substance_meronyms": "part_of",
        "entailments": "entails",
        "causes": "causes",
        "also_sees": "related_to",
        "similar_tos": "similar_to",
        "verb_groups": "verb_group",
        "attributes": "has_attribute",
    }

    for synset in wn.all_synsets():
        synsets += 1

        sid = f"wn:synset:{synset.name()}"
        gloss = synset.definition() or ""

        node_buffer[sid] = (
            sid,
            sid,
            sid,
            gloss or None,
            1,
            "synset",
        )

        if gloss:
            edge_buffer.append(
                (
                    sid,
                    "definition",
                    gloss,
                    "wordnet",
                )
            )
            definitions += 1

        for lemma in synset.lemmas():
            word_node = wordnet_word_node(
                lemma
            )
            if not word_node:
                continue

            lemmas += 1

            word_label = normalize_word(
                lemma.name()
            )

            existing = node_buffer.get(
                word_node
            )
            if existing:
                source_mask = (
                    existing[4] | 1
                )
                node_buffer[word_node] = (
                    word_node,
                    word_node,
                    word_label,
                    existing[3],
                    source_mask,
                    "concept",
                )
            else:
                node_buffer[word_node] = (
                    word_node,
                    word_node,
                    word_label,
                    None,
                    1,
                    "concept",
                )

            edge_buffer.append(
                (
                    word_node,
                    "has_sense",
                    sid,
                    "wordnet",
                )
            )

            if lemma.count() > 0:
                edge_buffer.append(
                    (
                        word_node,
                        "usage_count",
                        str(
                            lemma.count()
                        ),
                        "wordnet",
                    )
                )

        for attr, relation in relation_map.items():
            method = getattr(
                synset,
                attr,
                None,
            )
            if not method:
                continue

            for target in method():
                target_sid = (
                    f"wn:synset:{target.name()}"
                )
                edge_buffer.append(
                    (
                        sid,
                        relation,
                        target_sid,
                        "wordnet",
                    )
                )

        if synsets % batch_size == 0:
            flush_nodes()
            flush_edges()
            conn.commit()
            print(
                f"    WordNet synsets={synsets:,} "
                f"lemmas={lemmas:,} "
                f"buffered_edges={edges:,}",
                flush=True,
            )

    flush_nodes()
    flush_edges()
    conn.commit()

    return {
        "synsets": synsets,
        "lemmas": lemmas,
        "edges": edges,
        "definitions": definitions,
    }


def ingest_conceptnet(
    conn,
    path,
    batch_size=25000,
):
    """
    ConceptNet 5.7 assertions are TSV, despite the filename ending in
    assertions-5.7.0.csv.gz. The documented layout is:

        assertion_uri<TAB>relation<TAB>start<TAB>end<TAB>json_metadata

    See ConceptNet's download documentation. This reader deliberately ignores
    metadata for the graph itself while retaining every English -> English
    assertion.
    """
    path = Path(path)

    rows_seen = 0
    english_rows = 0
    retained_edges = 0

    node_buffer = {}
    edge_buffer = []

    def flush():
        nonlocal node_buffer
        nonlocal edge_buffer
        nonlocal retained_edges

        if node_buffer:
            conn.executemany(
                """
                INSERT INTO nodes(
                    node,
                    normalized,
                    label,
                    definition,
                    source_mask,
                    node_type
                )
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(node) DO UPDATE SET
                    source_mask =
                        nodes.source_mask
                        | excluded.source_mask
                """,
                list(
                    node_buffer.values()
                ),
            )
            node_buffer = {}

        if edge_buffer:
            conn.executemany(
                """
                INSERT OR IGNORE INTO edges(
                    subject,
                    relation,
                    object,
                    source
                )
                VALUES(?,?,?,?)
                """,
                edge_buffer,
            )
            retained_edges += len(
                edge_buffer
            )
            edge_buffer = []

        conn.commit()

    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as handle:
        reader = csv.reader(
            handle,
            delimiter="\t",
        )

        for row in reader:
            rows_seen += 1

            if len(row) < 4:
                continue

            relation = conceptnet_relation(
                row[1]
            )

            start = conceptnet_node(
                row[2]
            )
            end = conceptnet_node(
                row[3]
            )

            if (
                not relation
                or not start
                or not end
            ):
                continue

            english_rows += 1

            node_buffer.setdefault(
                start,
                (
                    start,
                    start[3:],
                    start[3:],
                    None,
                    2,
                    "concept",
                ),
            )

            node_buffer.setdefault(
                end,
                (
                    end,
                    end[3:],
                    end[3:],
                    None,
                    2,
                    "concept",
                ),
            )

            edge_buffer.append(
                (
                    start,
                    relation,
                    end,
                    "conceptnet",
                )
            )

            if (
                len(edge_buffer)
                >= batch_size
            ):
                flush()

            if rows_seen % 1000000 == 0:
                print(
                    f"    ConceptNet rows scanned="
                    f"{rows_seen:,} "
                    f"english assertions="
                    f"{english_rows:,} "
                    f"retained inserts="
                    f"{retained_edges:,}",
                    flush=True,
                )

    flush()

    # INSERT OR IGNORE means retained_edges above is an insertion attempt
    # count, not necessarily unique rows. Report the actual database count.
    actual = conn.execute(
        """
        SELECT COUNT(*)
        FROM edges
        WHERE source='conceptnet'
        """
    ).fetchone()[0]

    return {
        "rows_seen": rows_seen,
        "english_rows": english_rows,
        "edges": actual,
    }


UPWARD_TYPE_RELATIONS = frozenset({"is_a"})


def ingest_wordnet_focused(conn, wn, focus_terms, focus_depth):
    """Ingest seed concepts, their direct evidence, and bounded type closure."""
    synsets = list(wn.all_synsets())
    seed_synsets = {
        synset
        for synset in synsets
        if any(
            normalize_word(lemma.name()) in focus_terms
            for lemma in synset.lemmas()
        )
    }
    selected = set(seed_synsets)
    frontier = set(seed_synsets)

    def related(synset):
        for attribute, relation in {
            "hypernyms": "is_a",
            "instance_hypernyms": "is_a",
            "hyponyms": "has_subtype",
            "instance_hyponyms": "has_subtype",
            "part_holonyms": "has_part",
            "member_holonyms": "has_part",
            "substance_holonyms": "has_part",
            "part_meronyms": "part_of",
            "member_meronyms": "part_of",
            "substance_meronyms": "part_of",
            "entailments": "entails",
            "causes": "causes",
            "also_sees": "related_to",
            "similar_tos": "similar_to",
            "verb_groups": "verb_group",
            "attributes": "has_attribute",
        }.items():
            for target in getattr(synset, attribute)():
                yield relation, target

    for depth in range(focus_depth + 1):
        next_frontier = set()
        for synset in frontier:
            for relation, target in related(synset):
                if relation in UPWARD_TYPE_RELATIONS:
                    if target not in selected:
                        selected.add(target)
                        next_frontier.add(target)
                elif depth == 0 and relation != "has_subtype":
                    selected.add(target)
        frontier = next_frontier

    edge_count = 0
    for synset in selected:
        sid = f"wn:synset:{synset.name()}"
        upsert_node(conn, sid, sid, synset.definition() or None, 1, "synset")
        if synset.definition():
            conn.execute(
                "INSERT OR IGNORE INTO edges VALUES(?,?,?,?)",
                (sid, "definition", synset.definition(), "wordnet"),
            )
            edge_count += 1
        for lemma in synset.lemmas():
            word_node = wordnet_word_node(lemma)
            if word_node:
                upsert_node(conn, word_node, normalize_word(lemma.name()), None, 1)
                conn.execute(
                    "INSERT OR IGNORE INTO edges VALUES(?,?,?,?)",
                    (word_node, "has_sense", sid, "wordnet"),
                )
                edge_count += 1
        for relation, target in related(synset):
            if target in selected:
                conn.execute(
                    "INSERT OR IGNORE INTO edges VALUES(?,?,?,?)",
                    (sid, relation, f"wn:synset:{target.name()}", "wordnet"),
                )
                edge_count += 1
    conn.commit()
    return {"seed_synsets": len(seed_synsets), "synsets": len(selected), "edges": edge_count}


def ingest_conceptnet_focused(
    conn,
    path,
    focus_terms,
    focus_depth,
    progress_every,
):
    """Keep direct seed evidence, then only follow bounded type relationships."""
    included = set(focus_terms)
    frontier = set(focus_terms)
    rows_seen = 0
    retained = 0
    started = time.perf_counter()

    for depth in range(focus_depth + 1):
        pass_started = time.perf_counter()
        pass_rows = 0
        pass_retained = 0
        next_frontier = set()
        print(
            f"    ConceptNet pass {depth + 1}/{focus_depth + 1}: "
            f"{'direct seed evidence' if depth == 0 else 'is_a closure'}; "
            f"frontier={len(frontier):,}",
            flush=True,
        )
        with gzip.open(
            path,
            "rt",
            encoding="utf-8",
            errors="replace",
            newline="",
        ) as handle:
            for row in csv.reader(handle, delimiter="\t"):
                pass_rows += 1
                if progress_every and pass_rows % progress_every == 0:
                    elapsed = time.perf_counter() - pass_started
                    print(
                        f"      rows={pass_rows:,} "
                        f"kept_this_pass={pass_retained:,} "
                        f"frontier_next={len(next_frontier):,} "
                        f"rows/s={pass_rows / elapsed:,.0f}",
                        flush=True,
                    )
                if len(row) < 4:
                    continue
                if depth == 0:
                    rows_seen += 1
                relation = conceptnet_relation(row[1])
                start = conceptnet_node(row[2])
                end = conceptnet_node(row[3])
                if not relation or not start or not end:
                    continue
                start_term, end_term = start[3:], end[3:]
                follows_type = (
                    depth < focus_depth
                    and relation in UPWARD_TYPE_RELATIONS
                    and start_term in frontier
                )
                direct_evidence = (
                    depth == 0
                    and relation not in UPWARD_TYPE_RELATIONS
                    and relation != "has_subtype"
                    and (start_term in frontier or end_term in frontier)
                )
                if not (follows_type or direct_evidence):
                    continue

                upsert_node(conn, start, start_term, None, 2)
                upsert_node(conn, end, end_term, None, 2)
                conn.execute(
                    "INSERT OR IGNORE INTO edges VALUES(?,?,?,?)",
                    (start, relation, end, "conceptnet"),
                )
                retained += 1
                pass_retained += 1
                included.update((start_term, end_term))
                if follows_type and end_term not in frontier:
                    next_frontier.add(end_term)
        frontier = next_frontier
        conn.commit()
        elapsed = time.perf_counter() - pass_started
        print(
            f"    ConceptNet pass {depth + 1} complete: "
            f"rows={pass_rows:,} kept={pass_retained:,} "
            f"next_frontier={len(frontier):,} elapsed={elapsed:.1f}s",
            flush=True,
        )
        if not frontier and depth < focus_depth:
            print("    Type closure complete: no further ancestors.", flush=True)
            break
    return {
        "rows_seen": rows_seen,
        "concepts": len(included),
        "edges": retained,
        "elapsed_seconds": time.perf_counter() - started,
    }



def add_relation_dictionary(conn):
    phrases = {
        "definition": "definition meaning define what is explain",
        "is_a": "is a type kind category class",
        "has_subtype": "has subtype kind category type",
        "has_part": "has part contains includes",
        "part_of": "part of belongs component",
        "capable_of": "can do capable able",
        "used_for": "used for purpose function",
        "at_location": "located at place location",
        "related_to": "related associated connected",
        "causes": "causes leads produces",
        "made_of": "made of material substance",
        "similar_to": "similar alike resembles",
        "entails": "entails implies requires",
        "has_attribute": "has attribute property",
        "verb_group": "verb group related action",
        "has_sense": "sense meaning word sense",
        "usage_count": "usage frequency count",
    }

    conn.executemany(
        """
        INSERT OR IGNORE INTO relations(
            relation,
            phrases
        )
        VALUES(?,?)
        """,
        list(
            phrases.items()
        ),
    )


def add_metadata(
    conn,
    conceptnet_path,
    wn_stats,
    cn_stats,
    focus_terms=None,
    focus_depth=None,
):
    metadata = {
        "version": "V679",
        "wordnet": json.dumps(
            wn_stats
        ),
        "conceptnet": json.dumps(
            cn_stats
        ),
        "conceptnet_path": str(
            conceptnet_path
        ),
        "scope": (
            f"focused semantic closure: {', '.join(sorted(focus_terms))}; "
            f"structural depth={focus_depth}"
            if focus_terms
            else "ALL WordNet + ALL English ConceptNet"
        ),
        "created_unix": str(
            time.time()
        ),
    }

    conn.executemany(
        """
        INSERT INTO metadata(
            key,
            value
        )
        VALUES(?,?)
        """,
        list(
            metadata.items()
        ),
    )


def build_indexes(conn):
    print(
        "[index] creating indexes...",
        flush=True,
    )

    conn.executescript(
        """
        CREATE INDEX idx_nodes_normalized
            ON nodes(normalized);

        CREATE INDEX idx_nodes_type
            ON nodes(node_type);

        CREATE INDEX idx_nodes_source
            ON nodes(source_mask);

        CREATE INDEX idx_edges_subject
            ON edges(subject);

        CREATE INDEX idx_edges_relation
            ON edges(relation);

        CREATE INDEX idx_edges_object
            ON edges(object);

        CREATE INDEX idx_edges_subject_relation
            ON edges(subject, relation);

        CREATE INDEX idx_edges_object_relation
            ON edges(object, relation);

        ANALYZE;
        """
    )
    conn.commit()


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Build a WordNet + English ConceptNet semantic network."
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
        "--wordnet-only",
        action="store_true",
    )
    ap.add_argument(
        "--conceptnet-only",
        action="store_true",
    )
    ap.add_argument(
        "--focus-concepts",
        nargs="+",
        metavar="CONCEPT",
        help=(
            "Build a small graph centered on these concepts. Direct semantic "
            "evidence is retained; type relations are followed to --focus-depth."
        ),
    )
    ap.add_argument(
        "--focus-depth",
        type=int,
        default=2,
        help="Number of structural type-closure hops after direct seed evidence.",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=250000,
        help="Print focused ConceptNet scan progress every N rows; use 0 to disable.",
    )
    args = ap.parse_args()

    if (
        args.wordnet_only
        and args.conceptnet_only
    ):
        raise SystemExit(
            "Choose at most one of --wordnet-only/--conceptnet-only"
        )

    if args.focus_depth < 0:
        raise SystemExit("--focus-depth must be non-negative")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be non-negative")

    focus_terms = (
        {normalize_word(term) for term in args.focus_concepts}
        if args.focus_concepts
        else None
    )
    if focus_terms and None in focus_terms:
        raise SystemExit("--focus-concepts must contain non-empty concepts")

    conceptnet_path = Path(
        args.conceptnet
    ).resolve()
    output = Path(
        args.output
    ).resolve()

    if not args.wordnet_only and not conceptnet_path.exists():
        raise FileNotFoundError(
            conceptnet_path
        )

    started = time.perf_counter()

    print(
        "=== V679 FULL SEMANTIC NETWORK BUILD ===",
        flush=True,
    )
    print(
        (
            "scope      : focused semantic closure "
            + ", ".join(sorted(focus_terms))
            if focus_terms
            else "scope      : ALL WordNet + ALL English ConceptNet"
        ),
        flush=True,
    )
    print(
        f"ConceptNet : {conceptnet_path}",
        flush=True,
    )
    print(
        f"output     : {output}",
        flush=True,
    )

    conn = create_db(
        output
    )

    add_relation_dictionary(
        conn
    )

    wn_stats = {}
    cn_stats = {}

    if not args.conceptnet_only:
        print(
            (
                "[1/4] WordNet: focused closure..."
                if focus_terms
                else "[1/4] WordNet: ALL synsets and lemmas..."
            ),
            flush=True,
        )
        wn = get_wordnet()
        wn_stats = (
            ingest_wordnet_focused(conn, wn, focus_terms, args.focus_depth)
            if focus_terms
            else ingest_wordnet(conn, wn)
        )

        print(
            "    WordNet complete:",
            wn_stats,
            flush=True,
        )

    if not args.wordnet_only:
        print(
            (
                "[2/4] ConceptNet: focused closure..."
                if focus_terms
                else "[2/4] ConceptNet: ALL English assertions..."
            ),
            flush=True,
        )
        cn_stats = (
            ingest_conceptnet_focused(
                conn,
                conceptnet_path,
                focus_terms,
                args.focus_depth,
                args.progress_every,
            )
            if focus_terms
            else ingest_conceptnet(conn, conceptnet_path)
        )

        print(
            "    ConceptNet complete:",
            cn_stats,
            flush=True,
        )

    print(
        "[3/4] indexing...",
        flush=True,
    )
    build_indexes(
        conn
    )

    add_metadata(
        conn,
        conceptnet_path,
        wn_stats,
        cn_stats,
        focus_terms,
        args.focus_depth if focus_terms else None,
    )
    conn.commit()

    nodes = conn.execute(
        "SELECT COUNT(*) FROM nodes"
    ).fetchone()[0]
    edges = conn.execute(
        "SELECT COUNT(*) FROM edges"
    ).fetchone()[0]

    print(
        "[4/4] final stats",
        flush=True,
    )
    print(
        f"    nodes={nodes:,}",
        flush=True,
    )
    print(
        f"    edges={edges:,}",
        flush=True,
    )

    rel_count = conn.execute(
        "SELECT COUNT(*) FROM relations"
    ).fetchone()[0]

    print(
        f"    relation dictionary={rel_count:,}",
        flush=True,
    )

    db_bytes = output.stat().st_size

    elapsed = (
        time.perf_counter()
        - started
    )

    conn.close()

    print(
        "\n=== V679 BUILD COMPLETE ===",
        flush=True,
    )
    print(
        f"nodes       : {nodes:,}",
        flush=True,
    )
    print(
        f"edges       : {edges:,}",
        flush=True,
    )
    print(
        f"db size     : {db_bytes / (1024**3):.2f} GiB",
        flush=True,
    )
    print(
        f"elapsed     : {elapsed:.1f}s",
        flush=True,
    )
    print(
        f"DATABASE    : {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
