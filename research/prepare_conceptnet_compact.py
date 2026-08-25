from __future__ import annotations

"""
prepare_conceptnet_compact.py

Turn ConceptNet 5 assertions.csv into a small, local, query-friendly SQLite
database for Graph-Topology experiments.

ConceptNet assertions contain:
    assertion_id, relation, start, end, metadata JSON

ConceptNet's downloadable assertions are tab-separated and the core semantic
fields are:
    whole assertion URI
    relation
    start concept
    end concept
    JSON metadata

See:
    https://github.com/commonsense/conceptnet5/wiki/Downloads
    https://github.com/commonsense/conceptnet5/wiki/Edges

This script intentionally keeps the representation simple:

    edge(
        start,
        relation,
        end,
        weight,
        dataset
    )

Default filtering:
    * English only
    * selected high-value semantic relations
    * optional dictionary-centered filtering
    * one- to three-word concepts
    * minimum weight
    * duplicate assertions collapsed into one weighted edge

Outputs:
    conceptnet_compact.db
    conceptnet_compact_stats.json

Why SQLite?
    The raw ConceptNet CSV is enormous and expensive to scan repeatedly.
    SQLite gives us indexed:
        start -> relations/endpoints
        end   -> relations/startpoints
        relation -> edges

That makes it a much better substrate for the next Graph-Topology experiment.
"""

import argparse
import csv
import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path


DEFAULT_RELATIONS = (
    "IsA",
    "PartOf",
    "HasA",
    "UsedFor",
    "CapableOf",
    "HasProperty",
    "Causes",
    "CausesDesire",
    "AtLocation",
    "MadeOf",
    "ReceivesAction",
    "HasPrerequisite",
    "HasFirstSubevent",
    "HasLastSubevent",
    "MotivatedByGoal",
    "Synonym",
    "Antonym",
    "RelatedTo",
    "SimilarTo",
    "DefinedAs",
    "HasContext",
)

LANG_RE = re.compile(
    r"^/c/(?P<lang>[^/]+)/(?P<term>.+)$"
)

TERM_CLEAN_RE = re.compile(
    r"[^a-z0-9_ '\-]"
)

WHITESPACE_RE = re.compile(
    r"\s+"
)


# ---------------------------------------------------------------------------
# URI normalization
# ---------------------------------------------------------------------------

def parse_concept_uri(
    uri: str,
) -> tuple[str, str] | None:
    """
    Convert:
        /c/en/dog
        /c/en/dog/n
        /c/en/play_game
    into:
        ("en", "dog")
        ("en", "dog")
        ("en", "play game")
    """
    uri = uri.strip()

    match = LANG_RE.match(uri)

    if not match:
        return None

    lang = match.group(
        "lang"
    )

    term = match.group(
        "term"
    )

    # Remove optional POS / sense component.
    pieces = term.split("/")

    term = pieces[0]

    term = term.replace(
        "_",
        " ",
    )

    term = WHITESPACE_RE.sub(
        " ",
        term,
    ).strip().lower()

    if not term:
        return None

    return (
        lang,
        term,
    )


def is_clean_term(
    term: str,
    max_words: int,
) -> bool:
    if not term:
        return False

    if len(term) > 80:
        return False

    if len(term.split()) > max_words:
        return False

    # Keep normal lexical concepts, not URI/control garbage.
    cleaned = TERM_CLEAN_RE.sub(
        "",
        term,
    )

    if cleaned != term:
        return False

    return True


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def create_database(
    path: Path,
) -> sqlite3.Connection:
    if path.exists():
        path.unlink()

    conn = sqlite3.connect(
        str(path)
    )

    conn.execute(
        """
        PRAGMA journal_mode = WAL;
        """
    )

    conn.execute(
        """
        PRAGMA synchronous = NORMAL;
        """
    )

    conn.execute(
        """
        CREATE TABLE edge (
            id INTEGER PRIMARY KEY,
            start TEXT NOT NULL,
            relation TEXT NOT NULL,
            end TEXT NOT NULL,
            weight REAL NOT NULL,
            dataset TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE UNIQUE INDEX edge_unique
        ON edge(start, relation, end, dataset)
        """
    )

    return conn


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------

def detect_input_dialect(
    path: Path,
) -> str:
    """
    ConceptNet's official download is TSV, despite the historical
    assertions.csv naming.
    """
    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as handle:
        sample = handle.read(
            8192
        )

    if "\t" in sample:
        return "\t"

    return ","


# ---------------------------------------------------------------------------
# Dictionary
# ---------------------------------------------------------------------------

def load_dictionary(
    path: Path | None,
) -> set[str]:
    if path is None:
        return set()

    words = set()

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for raw in handle:
            word = raw.strip().lower()

            if word and word.isalpha():
                words.add(word)

    return words


# ---------------------------------------------------------------------------
# Main transformation
# ---------------------------------------------------------------------------

def build(
    input_path: Path,
    output_db: Path,
    stats_path: Path,
    dictionary_path: Path | None,
    allowed_relations: set[str],
    min_weight: float,
    max_words: int,
    dictionary_centered: bool,
) -> None:
    started = time.perf_counter()

    dialect = detect_input_dialect(
        input_path
    )

    dictionary = load_dictionary(
        dictionary_path
    )

    print(
        "input:",
        input_path,
        flush=True,
    )

    print(
        "output_db:",
        output_db,
        flush=True,
    )

    print(
        "dialect:",
        "TSV" if dialect == "\t" else "CSV",
        flush=True,
    )

    print(
        "allowed_relations:",
        ", ".join(
            sorted(
                allowed_relations
            )
        ),
        flush=True,
    )

    print(
        "dictionary_terms:",
        len(dictionary),
        flush=True,
    )

    try:
        input_size_mb = input_path.stat().st_size / (1024 * 1024)
        print(
            f"input_size_mb={input_size_mb:,.1f}",
            flush=True,
        )
    except OSError:
        pass

    print(
        "starting ConceptNet ingest...",
        flush=True,
    )

    conn = create_database(
        output_db
    )

    insert_sql = """
        INSERT INTO edge (
            start,
            relation,
            end,
            weight,
            dataset
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(
            start,
            relation,
            end,
            dataset
        )
        DO UPDATE SET
            weight = MAX(
                edge.weight,
                excluded.weight
            )
    """

    total_rows = 0
    kept_rows = 0
    skipped_relation = 0
    skipped_language = 0
    skipped_term = 0
    skipped_weight = 0
    skipped_dictionary = 0

    relation_counts = Counter()
    dataset_counts = Counter()

    transaction_rows = 0

    with input_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as handle:
        reader = csv.reader(
            handle,
            delimiter=dialect,
        )

        for row in reader:
            total_rows += 1

            if not row:
                continue

            # Official ConceptNet assertions download:
            # 0 = assertion id
            # 1 = relation
            # 2 = start
            # 3 = end
            # 4 = metadata JSON
            if len(row) < 5:
                continue

            relation_uri = row[1].strip()
            start_uri = row[2].strip()
            end_uri = row[3].strip()

            if not relation_uri.startswith(
                "/r/"
            ):
                continue

            relation = relation_uri[
                3:
            ]

            if relation not in allowed_relations:
                skipped_relation += 1
                continue

            start = parse_concept_uri(
                start_uri
            )

            end = parse_concept_uri(
                end_uri
            )

            if (
                start is None
                or end is None
            ):
                skipped_language += 1
                continue

            start_lang, start_term = start
            end_lang, end_term = end

            if (
                start_lang != "en"
                or end_lang != "en"
            ):
                skipped_language += 1
                continue

            if not is_clean_term(
                start_term,
                max_words,
            ) or not is_clean_term(
                end_term,
                max_words,
            ):
                skipped_term += 1
                continue

            try:
                metadata = json.loads(
                    row[4]
                )
            except (
                json.JSONDecodeError,
                TypeError,
            ):
                metadata = {}

            try:
                weight = float(
                    metadata.get(
                        "weight",
                        1.0,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                weight = 1.0

            if weight < min_weight:
                skipped_weight += 1
                continue

            if dictionary_centered:
                if (
                    start_term not in dictionary
                    and end_term not in dictionary
                ):
                    skipped_dictionary += 1
                    continue

            dataset = str(
                metadata.get(
                    "dataset",
                    "",
                )
            )

            conn.execute(
                insert_sql,
                (
                    start_term,
                    relation,
                    end_term,
                    weight,
                    dataset,
                ),
            )

            kept_rows += 1
            transaction_rows += 1

            relation_counts[
                relation
            ] += 1

            dataset_counts[
                dataset
            ] += 1

            if transaction_rows >= 10000:
                conn.commit()
                transaction_rows = 0

            if (
                total_rows <= 10
                or total_rows % 100000 == 0
            ):
                elapsed = time.perf_counter() - started
                rate = (
                    total_rows / elapsed
                    if elapsed > 0
                    else 0.0
                )
                keep_rate = (
                    100.0 * kept_rows / total_rows
                    if total_rows > 0
                    else 0.0
                )

                print(
                    f"PROGRESS "
                    f"rows={total_rows:,} "
                    f"kept={kept_rows:,} "
                    f"keep={keep_rate:.1f}% "
                    f"rows/s={rate:,.0f} "
                    f"elapsed={elapsed:.1f}s "
                    f"last_relation={relation or '-'}",
                    flush=True,
                )

                print(
                    "  relations:",
                    ", ".join(
                        f"{name}={count:,}"
                        for name, count
                        in relation_counts.most_common(8)
                    ),
                    flush=True,
                )

    conn.commit()

    # Query indexes for graph traversal.
    conn.execute(
        """
        CREATE INDEX idx_edge_start
        ON edge(start)
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_edge_end
        ON edge(end)
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_edge_relation
        ON edge(relation)
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_edge_start_relation
        ON edge(start, relation)
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_edge_end_relation
        ON edge(end, relation)
        """
    )

    conn.commit()

    # Remove WAL after successful build.
    conn.execute(
        "PRAGMA wal_checkpoint(TRUNCATE)"
    )

    conn.close()

    elapsed = (
        time.perf_counter()
        - started
    )

    stats = {
        "input": str(input_path),
        "output_db": str(output_db),
        "dialect": (
            "TSV"
            if dialect == "\t"
            else "CSV"
        ),
        "allowed_relations": sorted(
            allowed_relations
        ),
        "min_weight": min_weight,
        "max_words_per_term": max_words,
        "dictionary_centered": dictionary_centered,
        "dictionary_terms": len(
            dictionary
        ),
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "skipped_relation": skipped_relation,
        "skipped_language": skipped_language,
        "skipped_term": skipped_term,
        "skipped_weight": skipped_weight,
        "skipped_dictionary": skipped_dictionary,
        "relation_counts": dict(
            relation_counts
        ),
        "dataset_counts_top20": dict(
            dataset_counts.most_common(
                20
            )
        ),
        "elapsed_seconds": elapsed,
    }

    stats_path.write_text(
        json.dumps(
            stats,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=== COMPLETE ==="
    )

    print(
        "total_rows:",
        f"{total_rows:,}",
    )

    print(
        "kept_rows:",
        f"{kept_rows:,}",
    )

    print(
        "relations:",
        dict(
            relation_counts
        ),
    )

    print(
        "elapsed_seconds:",
        f"{elapsed:.2f}",
    )

    print(
        "stats:",
        stats_path,
    )

    print(
        "db:",
        output_db,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compress ConceptNet assertions.csv into an English SQLite graph."
        )
    )

    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data"

    # ConceptNet in this project is stored as:
    #     ../data/assertions
    # (or assertions.csv depending on how it was downloaded).
    # Prefer assertions exactly as supplied, then fall back to .csv.
    assertions_path = data_root / "assertions"

    if not assertions_path.exists():
        csv_path = data_root / "assertions.csv"
        if csv_path.exists():
            assertions_path = csv_path

    parser.add_argument(
        "--input",
        type=Path,
        default=assertions_path,
    )

    parser.add_argument(
        "--output-db",
        type=Path,
        default=data_root / "conceptnet_compact.db",
    )

    parser.add_argument(
        "--stats",
        type=Path,
        default=data_root / "conceptnet_compact_stats.json",
    )

    parser.add_argument(
        "--dictionary",
        type=Path,
        default=data_root / "dictionary.csv",
    )

    parser.add_argument(
        "--min-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max-words",
        type=int,
        default=3,
        help=(
            "Maximum words in a ConceptNet term. "
            "Use 1 for lexical units only."
        ),
    )

    parser.add_argument(
        "--all-english",
        action="store_true",
        help=(
            "Keep all selected English edges instead of requiring that at "
            "least one endpoint occurs in the local dictionary."
        ),
    )

    parser.add_argument(
        "--relations",
        nargs="*",
        default=list(
            DEFAULT_RELATIONS
        ),
        help=(
            "ConceptNet relations to retain. "
            "Defaults to a compact semantic subset."
        ),
    )

    args = parser.parse_args()

    args.output_db.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.stats.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    build(
        input_path=args.input,
        output_db=args.output_db,
        stats_path=args.stats,
        dictionary_path=(
            None
            if args.all_english
            else args.dictionary
        ),
        allowed_relations=set(
            args.relations
        ),
        min_weight=args.min_weight,
        max_words=args.max_words,
        dictionary_centered=(
            not args.all_english
        ),
    )


if __name__ == "__main__":
    main()
