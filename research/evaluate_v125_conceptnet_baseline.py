from __future__ import annotations

"""
V125 — CONCEPTNET BASELINE / SEMANTIC GRAPH STUDY

This is the first serious baseline on the compact ConceptNet SQLite graph.

Input:
    ../data/conceptnet_compact.db

Optional evaluation corpus:
    ../data/semantics-large.csv

Optional dictionary:
    ../data/dictionary.csv

The script intentionally does NOT use the LLM.

It establishes a clean baseline for what a real typed semantic graph gives us
before we teach/compress anything.

Experiments
-----------

1. GRAPH SIZE / COVERAGE
       nodes, edges, relations, degree, relation frequencies

2. DICTIONARY COVERAGE
       how many of the 4925 lexical units have ConceptNet edges

3. RELATION-SPECIFIC COVERAGE
       which dictionary words have IsA / UsedFor / HasProperty / etc.

4. FLAT SEMANTIC BASELINE
       For each dictionary word:
           collect direct ConceptNet endpoints across selected relations
       compare against semantics-large.csv gold features

5. RELATION-AWARE BASELINE
       evaluate each ConceptNet relation independently against the same gold
       vocabulary. This tells us which ConceptNet relations carry useful
       lexical-semantic signal.

6. TWO-HOP BASELINE
       word -> relation -> concept -> relation -> concept
       with controlled depth=2 expansion.
       This asks whether simple graph traversal already improves over direct
       neighbors.

7. RANDOM CONTROL
       same candidate counts, random concepts from the ConceptNet vocabulary.
       This prevents us from mistaking large candidate sets for semantic power.

8. WORD / RELATION TRACE
       detailed examples for:
           hello, greeting, dog, animal, ability, abandon, water, music,
           chair, car

9. GRAPH QUERY DEMO
       print the most useful typed edges for each trace word.

No graph mutation.
No neural model.
No training.

Output:
    results/v125_conceptnet_baseline.json
"""

import csv
import json
import math
import random
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DB_PATH = ROOT / "data" / "conceptnet_compact.db"

DICTIONARY_PATH = ROOT / "data" / "dictionary.csv"

SEMANTICS_PATH = ROOT / "data" / "semantics-large.csv"

OUTPUT_PATH = (
    ROOT
    / "results"
    / "v125_conceptnet_baseline.json"
)

SEED = 12501

# Use the full local dictionary.
MAX_WORDS = None

# Direct semantic baseline relation set.
SEMANTIC_RELATIONS = (
    "IsA",
    "PartOf",
    "HasA",
    "UsedFor",
    "CapableOf",
    "HasProperty",
    "Causes",
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

# Relations that are especially interesting as lexical/semantic structure.
RELATION_FOCUS = (
    "IsA",
    "PartOf",
    "HasA",
    "UsedFor",
    "CapableOf",
    "HasProperty",
    "Synonym",
    "Antonym",
    "RelatedTo",
    "SimilarTo",
)

MAX_DIRECT_CONCEPTS = 32
MAX_HOP2_CONCEPTS = 64

TRACE_WORDS = (
    "hello",
    "greeting",
    "dog",
    "animal",
    "ability",
    "abandon",
    "water",
    "music",
    "chair",
    "car",
)


# ---------------------------------------------------------------------------
# Dictionary
# ---------------------------------------------------------------------------

def load_dictionary(
    path: Path,
) -> list[str]:
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

    result = sorted(words)

    if MAX_WORDS is not None:
        result = result[:MAX_WORDS]

    return result


# ---------------------------------------------------------------------------
# Human semantic gold
# ---------------------------------------------------------------------------

class HumanGold:
    def __init__(self) -> None:
        self.cue_features: dict[str, Counter[str]] = defaultdict(Counter)

    def load(
        self,
        path: Path,
    ) -> None:
        if not path.exists():
            print(
                "WARNING: semantics-large.csv not found; "
                "gold evaluations will be skipped.",
                flush=True,
            )
            return

        with path.open(
            "r",
            encoding="utf-8",
            newline="",
            errors="replace",
        ) as handle:
            reader = csv.DictReader(handle)

            for row in reader:
                cue = row.get(
                    "cue",
                    "",
                ).strip().lower()

                feature = row.get(
                    "translated",
                    "",
                ).strip().lower()

                if not cue or not feature:
                    continue

                try:
                    weight = float(
                        row.get(
                            "normalized_translated",
                            0.0,
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    weight = 0.0

                if weight <= 0.0:
                    try:
                        frequency = float(
                            row.get(
                                "frequency_translated",
                                0.0,
                            )
                        )

                        n = float(
                            row.get(
                                "n",
                                0.0,
                            )
                        )

                        if n > 0.0:
                            weight = (
                                frequency
                                / n
                            )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        weight = 0.0

                if weight > 0.0:
                    self.cue_features[
                        cue
                    ][feature] += weight

    def gold(
        self,
        word: str,
        limit: int = 8,
    ) -> set[str]:
        return {
            feature
            for feature, _weight
            in self.cue_features.get(
                word,
                Counter(),
            ).most_common(limit)
        }


# ---------------------------------------------------------------------------
# ConceptNet DB
# ---------------------------------------------------------------------------

class ConceptNetDB:
    def __init__(
        self,
        path: Path,
    ) -> None:
        self.conn = sqlite3.connect(
            str(path)
        )

        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def scalar(
        self,
        sql: str,
        params: tuple = (),
    ):
        row = self.conn.execute(
            sql,
            params,
        ).fetchone()

        if row is None:
            return None

        return row[0]

    def relation_counts(
        self,
    ) -> dict[str, int]:
        rows = self.conn.execute(
            """
            SELECT relation, COUNT(*) AS n
            FROM edge
            GROUP BY relation
            ORDER BY n DESC
            """
        )

        return {
            row["relation"]: row["n"]
            for row in rows
        }

    def total_edges(
        self,
    ) -> int:
        return int(
            self.scalar(
                "SELECT COUNT(*) FROM edge"
            )
            or 0
        )

    def distinct_start_nodes(
        self,
    ) -> int:
        return int(
            self.scalar(
                "SELECT COUNT(DISTINCT start) FROM edge"
            )
            or 0
        )

    def distinct_end_nodes(
        self,
    ) -> int:
        return int(
            self.scalar(
                "SELECT COUNT(DISTINCT end) FROM edge"
            )
            or 0
        )

    def distinct_concepts(
        self,
    ) -> int:
        return int(
            self.scalar(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT start AS concept FROM edge
                    UNION
                    SELECT end AS concept FROM edge
                )
                """
            )
            or 0
        )

    def edges_for_start(
        self,
        start: str,
        relations: tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list[sqlite3.Row]:
        if relations:
            placeholders = ",".join(
                "?" for _ in relations
            )

            sql = f"""
                SELECT start, relation, end, weight, dataset
                FROM edge
                WHERE start = ?
                  AND relation IN ({placeholders})
                ORDER BY weight DESC, end ASC
            """

            params = (
                start,
                *relations,
            )
        else:
            sql = """
                SELECT start, relation, end, weight, dataset
                FROM edge
                WHERE start = ?
                ORDER BY weight DESC, relation ASC, end ASC
            """

            params = (start,)

        if limit is not None:
            sql += " LIMIT ?"
            params = (
                *params,
                limit,
            )

        return list(
            self.conn.execute(
                sql,
                params,
            )
        )

    def edges_for_end(
        self,
        end: str,
        relations: tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list[sqlite3.Row]:
        if relations:
            placeholders = ",".join(
                "?" for _ in relations
            )

            sql = f"""
                SELECT start, relation, end, weight, dataset
                FROM edge
                WHERE end = ?
                  AND relation IN ({placeholders})
                ORDER BY weight DESC, start ASC
            """

            params = (
                end,
                *relations,
            )
        else:
            sql = """
                SELECT start, relation, end, weight, dataset
                FROM edge
                WHERE end = ?
                ORDER BY weight DESC, relation ASC, start ASC
            """

            params = (end,)

        if limit is not None:
            sql += " LIMIT ?"
            params = (
                *params,
                limit,
            )

        return list(
            self.conn.execute(
                sql,
                params,
            )
        )

    def relation_count_for_word(
        self,
        word: str,
    ) -> dict[str, int]:
        rows = self.conn.execute(
            """
            SELECT relation, COUNT(*) AS n
            FROM edge
            WHERE start = ?
            GROUP BY relation
            ORDER BY n DESC
            """,
            (word,),
        )

        return {
            row["relation"]: row["n"]
            for row in rows
        }

    def direct_concepts(
        self,
        word: str,
        relations: tuple[str, ...] = SEMANTIC_RELATIONS,
        limit: int = MAX_DIRECT_CONCEPTS,
    ) -> list[str]:
        rows = self.edges_for_start(
            word,
            relations=relations,
            limit=None,
        )

        scores = Counter()

        for row in rows:
            scores[
                row["end"]
            ] += float(
                row["weight"]
            )

        return [
            concept
            for concept, _score
            in scores.most_common(
                limit
            )
        ]

    def relation_concepts(
        self,
        word: str,
        relation: str,
        limit: int = MAX_DIRECT_CONCEPTS,
    ) -> list[str]:
        rows = self.edges_for_start(
            word,
            relations=(relation,),
            limit=limit,
        )

        return [
            row["end"]
            for row in rows
        ]

    def direct_weighted(
        self,
        word: str,
        relations: tuple[str, ...],
        limit: int,
    ) -> list[tuple[str, float]]:
        rows = self.edges_for_start(
            word,
            relations=relations,
            limit=None,
        )

        scores = Counter()

        for row in rows:
            scores[
                row["end"]
            ] += float(
                row["weight"]
            )

        return scores.most_common(
            limit
        )

    def relation_degree_statistics(
        self,
        dictionary: set[str],
    ) -> dict[str, dict[str, float]]:
        result = {}

        for relation in RELATION_FOCUS:
            rows = self.conn.execute(
                """
                SELECT start, COUNT(*) AS n
                FROM edge
                WHERE relation = ?
                  AND start IN (
                      SELECT start
                      FROM edge
                      WHERE start IS NOT NULL
                  )
                GROUP BY start
                """,
                (relation,),
            )

            values = [
                row["n"]
                for row in rows
                if row["start"] in dictionary
            ]

            if values:
                result[
                    relation
                ] = {
                    "words_with_relation": len(values),
                    "mean_degree": statistics.mean(values),
                    "median_degree": statistics.median(values),
                    "max_degree": max(values),
                }
            else:
                result[
                    relation
                ] = {
                    "words_with_relation": 0,
                    "mean_degree": 0.0,
                    "median_degree": 0.0,
                    "max_degree": 0,
                }

        return result

    def hop2_concepts(
        self,
        word: str,
        relations: tuple[str, ...] = SEMANTIC_RELATIONS,
        limit: int = MAX_HOP2_CONCEPTS,
    ) -> list[tuple[str, float]]:
        """
        Word -> concept1 -> concept2

        Only follows outgoing ConceptNet edges from the first-hop concept.
        This is intentionally simple and reproducible.
        """
        first = self.edges_for_start(
            word,
            relations=relations,
            limit=MAX_DIRECT_CONCEPTS,
        )

        scores = Counter()

        for row in first:
            intermediate = row["end"]
            first_weight = float(
                row["weight"]
            )

            second = self.edges_for_start(
                intermediate,
                relations=relations,
                limit=32,
            )

            for row2 in second:
                target = row2["end"]

                if target == word:
                    continue

                second_weight = float(
                    row2["weight"]
                )

                # Damped two-hop score.
                scores[
                    target
                ] += (
                    math.sqrt(
                        max(
                            0.0,
                            first_weight,
                        )
                        * max(
                            0.0,
                            second_weight,
                        )
                    )
                    * 0.5
                )

        return scores.most_common(
            limit
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def prf(
    predicted: set[str],
    gold: set[str],
) -> tuple[float, float, float]:
    if not predicted or not gold:
        return 0.0, 0.0, 0.0

    hits = len(
        predicted & gold
    )

    precision = (
        hits
        / len(predicted)
    )

    recall = (
        hits
        / len(gold)
    )

    if precision + recall == 0:
        return precision, recall, 0.0

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision + recall
        )
    )

    return precision, recall, f1


def evaluate_predictions(
    predictions: dict[str, list[str]],
    gold: HumanGold,
) -> dict[str, float]:
    precisions = []
    recalls = []
    f1s = []

    for word, concepts in predictions.items():
        target = gold.gold(
            word
        )

        if not target:
            continue

        p, r, f = prf(
            set(concepts),
            target,
        )

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return {
        "evaluated": len(f1s),
        "precision": (
            sum(precisions)
            / max(
                1,
                len(precisions),
            )
        ),
        "recall": (
            sum(recalls)
            / max(
                1,
                len(recalls),
            )
        ),
        "f1": (
            sum(f1s)
            / max(
                1,
                len(f1s),
            )
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    started = time.perf_counter()

    print(
        "=== V125 CONCEPTNET BASELINE ==="
    )

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Missing ConceptNet DB: {DB_PATH}"
        )

    words = load_dictionary(
        DICTIONARY_PATH
    )

    dictionary = set(words)

    gold = HumanGold()
    gold.load(
        SEMANTICS_PATH
    )

    db = ConceptNetDB(
        DB_PATH
    )

    try:
        # ---------------------------------------------------------------
        # 1. Graph size
        # ---------------------------------------------------------------

        total_edges = db.total_edges()
        start_nodes = db.distinct_start_nodes()
        end_nodes = db.distinct_end_nodes()
        concepts = db.distinct_concepts()
        relation_counts = db.relation_counts()

        print()
        print(
            "=== GRAPH SIZE ==="
        )

        print(
            "edges:",
            f"{total_edges:,}",
        )

        print(
            "start_nodes:",
            f"{start_nodes:,}",
        )

        print(
            "end_nodes:",
            f"{end_nodes:,}",
        )

        print(
            "distinct_concepts:",
            f"{concepts:,}",
        )

        print(
            "relations:",
            relation_counts,
        )

        # ---------------------------------------------------------------
        # 2. Dictionary coverage
        # ---------------------------------------------------------------

        covered = 0
        degrees = []

        relation_presence = Counter()

        for word in words:
            counts = db.relation_count_for_word(
                word
            )

            degree = sum(
                counts.values()
            )

            if degree > 0:
                covered += 1
                degrees.append(
                    degree
                )

            for relation in counts:
                relation_presence[
                    relation
                ] += 1

        coverage = (
            covered
            / max(
                1,
                len(words),
            )
        )

        print()
        print(
            "=== DICTIONARY COVERAGE ==="
        )

        print(
            "dictionary_words:",
            len(words),
        )

        print(
            "covered_words:",
            covered,
        )

        print(
            "coverage:",
            f"{coverage:.4f}",
        )

        if degrees:
            print(
                "mean_edges_per_covered_word:",
                statistics.mean(degrees),
            )

            print(
                "median_edges_per_covered_word:",
                statistics.median(degrees),
            )

            print(
                "max_edges_for_word:",
                max(degrees),
            )

        print(
            "relation_presence:",
            dict(
                relation_presence.most_common()
            ),
        )

        # ---------------------------------------------------------------
        # 3. Relation-specific coverage
        # ---------------------------------------------------------------

        relation_stats = (
            db.relation_degree_statistics(
                dictionary
            )
        )

        print()
        print(
            "=== RELATION COVERAGE ==="
        )

        for relation in RELATION_FOCUS:
            print(
                relation,
                relation_stats.get(
                    relation,
                    {},
                ),
            )

        # ---------------------------------------------------------------
        # 4. Direct ConceptNet -> semantic gold baseline
        # ---------------------------------------------------------------

        direct_predictions = {
            word: db.direct_concepts(
                word,
                relations=SEMANTIC_RELATIONS,
                limit=MAX_DIRECT_CONCEPTS,
            )
            for word in words
        }

        direct_scores = (
            evaluate_predictions(
                direct_predictions,
                gold,
            )
        )

        print()
        print(
            "=== DIRECT CONCEPTNET -> HUMAN GOLD ==="
        )

        print(
            direct_scores
        )

        # ---------------------------------------------------------------
        # 5. Relation-specific baselines
        # ---------------------------------------------------------------

        relation_scores = {}

        print()
        print(
            "=== RELATION-SPECIFIC GOLD SCORES ==="
        )

        for relation in RELATION_FOCUS:
            predictions = {
                word: db.relation_concepts(
                    word,
                    relation,
                    limit=MAX_DIRECT_CONCEPTS,
                )
                for word in words
            }

            score = evaluate_predictions(
                predictions,
                gold,
            )

            relation_scores[
                relation
            ] = score

            print(
                f"{relation:18s}",
                score,
            )

        # ---------------------------------------------------------------
        # 6. Two-hop baseline
        # ---------------------------------------------------------------

        hop2_predictions = {}

        print()
        print(
            "=== TWO-HOP BASELINE ==="
        )

        for index, word in enumerate(
            words,
            start=1,
        ):
            hop2 = db.hop2_concepts(
                word,
                relations=SEMANTIC_RELATIONS,
                limit=MAX_HOP2_CONCEPTS,
            )

            hop2_predictions[
                word
            ] = [
                concept
                for concept, _score
                in hop2
            ]

            if (
                index <= 5
                or index % 500 == 0
                or index == len(words)
            ):
                print(
                    f"HOP2 "
                    f"{index:4d}/{len(words):4d}",
                    flush=True,
                )

        hop2_scores = evaluate_predictions(
            hop2_predictions,
            gold,
        )

        print(
            "hop2_scores:",
            hop2_scores,
        )

        # ---------------------------------------------------------------
        # 7. Random control
        # ---------------------------------------------------------------

        all_concepts = [
            row["concept"]
            for row in db.conn.execute(
                """
                SELECT concept
                FROM (
                    SELECT DISTINCT start AS concept
                    FROM edge
                    UNION
                    SELECT DISTINCT end AS concept
                    FROM edge
                )
                """
            )
        ]

        rng = random.Random(
            SEED
        )

        random_predictions = {}

        for word in words:
            k = len(
                direct_predictions[
                    word
                ]
            )

            if k <= 0:
                random_predictions[
                    word
                ] = []
                continue

            if k >= len(all_concepts):
                random_predictions[
                    word
                ] = all_concepts
            else:
                random_predictions[
                    word
                ] = rng.sample(
                    all_concepts,
                    k,
                )

        random_scores = (
            evaluate_predictions(
                random_predictions,
                gold,
            )
        )

        print()
        print(
            "=== RANDOM CONTROL ==="
        )

        print(
            random_scores
        )

        # ---------------------------------------------------------------
        # 8. Trace examples
        # ---------------------------------------------------------------

        traces = []

        print()
        print(
            "=== TRACE WORDS ==="
        )

        for word in TRACE_WORDS:
            direct = db.edges_for_start(
                word,
                relations=SEMANTIC_RELATIONS,
                limit=40,
            )

            print()
            print(
                f"[{word}]"
            )

            print(
                "relation_counts:",
                db.relation_count_for_word(
                    word
                ),
            )

            for row in direct[:20]:
                print(
                    f"  {row['relation']:20s} "
                    f"-> {row['end']:<30s} "
                    f"w={row['weight']}"
                )

            traces.append(
                {
                    "word": word,
                    "relation_counts": db.relation_count_for_word(
                        word
                    ),
                    "direct_edges": [
                        {
                            "relation": row["relation"],
                            "end": row["end"],
                            "weight": row["weight"],
                            "dataset": row["dataset"],
                        }
                        for row in direct[:40]
                    ],
                    "direct_concepts": direct_predictions.get(
                        word,
                        [],
                    ),
                    "hop2_concepts": hop2_predictions.get(
                        word,
                        [],
                    )[:32],
                    "human_gold": sorted(
                        gold.gold(word)
                    ),
                }
            )

        # ---------------------------------------------------------------
        # 9. Degree distribution snapshot
        # ---------------------------------------------------------------

        start_degree_rows = list(
            db.conn.execute(
                """
                SELECT start, COUNT(*) AS degree
                FROM edge
                GROUP BY start
                ORDER BY degree DESC
                LIMIT 100
                """
            )
        )

        top_degree = [
            {
                "concept": row["start"],
                "degree": row["degree"],
            }
            for row in start_degree_rows
        ]

        # ---------------------------------------------------------------
        # 10. Save everything
        # ---------------------------------------------------------------

        report = {
            "experiment": (
                "V125 ConceptNet baseline"
            ),
            "database": str(
                DB_PATH
            ),
            "dictionary_words": len(
                words
            ),
            "graph": {
                "edges": total_edges,
                "start_nodes": start_nodes,
                "end_nodes": end_nodes,
                "distinct_concepts": concepts,
                "relation_counts": relation_counts,
            },
            "dictionary_coverage": {
                "covered_words": covered,
                "coverage": coverage,
                "mean_edges_per_covered_word": (
                    statistics.mean(degrees)
                    if degrees
                    else 0.0
                ),
                "median_edges_per_covered_word": (
                    statistics.median(degrees)
                    if degrees
                    else 0.0
                ),
                "max_edges_for_word": (
                    max(degrees)
                    if degrees
                    else 0
                ),
                "relation_presence": dict(
                    relation_presence
                ),
            },
            "relation_degree_statistics": relation_stats,
            "direct_conceptnet_gold": direct_scores,
            "relation_specific_gold": relation_scores,
            "two_hop_gold": hop2_scores,
            "random_control": random_scores,
            "top_start_degree": top_degree,
            "traces": traces,
            "elapsed_seconds": (
                time.perf_counter()
                - started
            ),
        }

        OUTPUT_PATH.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print(
            "=== V125 SUMMARY ==="
        )

        print(
            "direct_f1:",
            direct_scores["f1"],
        )

        print(
            "hop2_f1:",
            hop2_scores["f1"],
        )

        print(
            "random_f1:",
            random_scores["f1"],
        )

        print(
            "relation_f1:"
        )

        for relation, score in relation_scores.items():
            print(
                f"  {relation:18s}",
                score["f1"],
            )

        print()
        print(
            "saved:",
            OUTPUT_PATH,
        )

        print(
            "elapsed_seconds:",
            f"{time.perf_counter() - started:.2f}",
        )

        print(
            "=== V125 COMPLETE ==="
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
