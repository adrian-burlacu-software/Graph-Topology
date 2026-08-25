from __future__ import annotations

"""
V126 — CONCEPTNET TEACHER -> COMPRESSED STUDENT GRAPH

This is the first real compression experiment after establishing that the
compact ConceptNet graph itself is a strong semantic baseline.

Teacher:
    data/conceptnet_compact.db
    ~2.3M retained typed edges

Student:
    a much smaller in-memory typed graph selected from the teacher

Goal:
    How much semantic performance survives when we keep only a small fraction
    of the teacher graph?

Budgets:
    1%
    2%
    5%
    10%

The student graph is built WITHOUT an LLM.

We use a relation-aware importance score built from:
    * ConceptNet assertion weight
    * relation prior
    * lexical-word coverage
    * endpoint degree / hub penalty
    * edge novelty for preserving graph connectivity

Selection is performed under a global edge budget while guaranteeing that
dictionary-centered concepts get representation.

Evaluation:
    * dictionary coverage
    * teacher edge recall for dictionary words
    * direct semantic F1 vs semantics-large.csv
    * relation-specific F1
    * random control
    * compression ratio
    * student-vs-teacher prediction overlap

The intent is to produce a strong, quantitative baseline for the eventual
LLM distillation experiment:

    ConceptNet teacher
          |
          v
    compressed graph
          |
          v
    frozen LLM / semantic memory

Important:
    * no mutation of ConceptNet DB
    * no LLM
    * no training
    * semantics-large.csv is evaluation-only
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

DB_PATH = (
    ROOT
    / "data"
    / "conceptnet_compact.db"
)

DICTIONARY_PATH = (
    ROOT
    / "data"
    / "dictionary.csv"
)

SEMANTICS_PATH = (
    ROOT
    / "data"
    / "semantics-large.csv"
)

OUTPUT_PATH = (
    ROOT
    / "results"
    / "v126_conceptnet_compression.json"
)

SEED = 12601

BUDGETS = (
    0.01,
    0.02,
    0.05,
    0.10,
)

# Relations with direct semantic value.
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

FOCUS_RELATION_WEIGHT = {
    "IsA": 2.8,
    "UsedFor": 2.6,
    "CapableOf": 2.5,
    "HasProperty": 2.5,
    "PartOf": 2.3,
    "HasA": 2.2,
    "Synonym": 2.3,
    "Antonym": 1.8,
    "RelatedTo": 1.4,
    "SimilarTo": 1.8,
    "AtLocation": 1.3,
    "Causes": 1.6,
    "HasPrerequisite": 1.5,
    "ReceivesAction": 1.5,
    "MadeOf": 1.3,
    "DefinedAs": 1.4,
    "HasContext": 0.9,
    "HasFirstSubevent": 1.0,
    "HasLastSubevent": 1.0,
    "MotivatedByGoal": 1.0,
}

MAX_DIRECT_CONCEPTS = 32

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
# Input helpers
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

    return sorted(words)


class HumanGold:
    def __init__(self) -> None:
        self.cue_features = defaultdict(Counter)

    def load(
        self,
        path: Path,
    ) -> None:
        if not path.exists():
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

                if weight <= 0:
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

                        if n > 0:
                            weight = (
                                frequency
                                / n
                            )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        weight = 0.0

                if weight > 0:
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


class TeacherDB:
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

    def edge_count(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM edge"
        ).fetchone()

        return int(row["n"])

    def word_edges(
        self,
        word: str,
    ) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                """
                SELECT start, relation, end, weight, dataset
                FROM edge
                WHERE start = ?
                ORDER BY weight DESC, relation ASC, end ASC
                """,
                (word,),
            )
        )

    def all_dictionary_edges(
        self,
        dictionary: set[str],
    ) -> list[sqlite3.Row]:
        placeholders = ",".join(
            "?"
            for _ in dictionary
        )

        if not placeholders:
            return []

        sql = f"""
            SELECT
                start,
                relation,
                end,
                weight,
                dataset
            FROM edge
            WHERE start IN ({placeholders})
        """

        return list(
            self.conn.execute(
                sql,
                tuple(dictionary),
            )
        )

    def concept_frequency(
        self,
    ) -> Counter[str]:
        result = Counter()

        rows = self.conn.execute(
            """
            SELECT start AS concept, COUNT(*) AS n
            FROM edge
            GROUP BY start
            UNION ALL
            SELECT end AS concept, COUNT(*) AS n
            FROM edge
            GROUP BY end
            """
        )

        for row in rows:
            result[
                row["concept"]
            ] += row["n"]

        return result


# ---------------------------------------------------------------------------
# Student graph
# ---------------------------------------------------------------------------

class StudentGraph:
    def __init__(
        self,
    ) -> None:
        self.edges_by_start: dict[
            str,
            list[tuple[str, str, float]],
        ] = defaultdict(list)

        self.edges_by_relation: dict[
            str,
            int,
        ] = Counter()

        self.edge_count = 0

    def add(
        self,
        row: sqlite3.Row,
    ) -> None:
        self.edges_by_start[
            row["start"]
        ].append(
            (
                row["relation"],
                row["end"],
                float(row["weight"]),
            )
        )

        self.edges_by_relation[
            row["relation"]
        ] += 1

        self.edge_count += 1

    def direct_concepts(
        self,
        word: str,
        limit: int = MAX_DIRECT_CONCEPTS,
    ) -> list[str]:
        rows = self.edges_by_start.get(
            word,
            [],
        )

        scores = Counter()

        for relation, end, weight in rows:
            relation_weight = (
                FOCUS_RELATION_WEIGHT.get(
                    relation,
                    1.0,
                )
            )

            scores[end] += (
                weight
                * relation_weight
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
        rows = self.edges_by_start.get(
            word,
            [],
        )

        scored = []

        for row_relation, end, weight in rows:
            if row_relation == relation:
                scored.append(
                    (
                        weight,
                        end,
                    )
                )

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        return [
            end
            for _weight, end
            in scored[:limit]
        ]


# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------

def edge_importance(
    row: sqlite3.Row,
    endpoint_degree: Counter[str],
) -> float:
    relation = row["relation"]
    start = row["start"]
    end = row["end"]

    weight = max(
        0.01,
        float(
            row["weight"]
        ),
    )

    relation_weight = (
        FOCUS_RELATION_WEIGHT.get(
            relation,
            1.0,
        )
    )

    # Very high-degree generic hubs are less informative. This is deliberately
    # only a soft penalty; we still retain hubs when needed for coverage.
    hub_penalty = 1.0 / math.sqrt(
        1.0
        + endpoint_degree[start]
        + endpoint_degree[end]
    )

    lexical_bonus = 1.0

    # Prefer short, lexical concepts to giant phrases in the student memory.
    for term in (
        start,
        end,
    ):
        words = term.split()

        if len(words) == 1:
            lexical_bonus += 0.25
        elif len(words) >= 4:
            lexical_bonus -= 0.10

    return (
        math.log1p(weight)
        * relation_weight
        * max(
            0.10,
            hub_penalty,
        )
        * max(
            0.25,
            lexical_bonus,
        )
    )


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

def build_student(
    teacher_rows: list[sqlite3.Row],
    dictionary: set[str],
    budget_fraction: float,
) -> tuple[
    StudentGraph,
    dict[str, float],
]:
    if not teacher_rows:
        return (
            StudentGraph(),
            {},
        )

    total_budget = max(
        1,
        int(
            len(teacher_rows)
            * budget_fraction
        ),
    )

    endpoint_degree = Counter()

    for row in teacher_rows:
        endpoint_degree[
            row["start"]
        ] += 1

        endpoint_degree[
            row["end"]
        ] += 1

    # ---------------------------------------------------------------
    # Score all candidate teacher edges.
    # ---------------------------------------------------------------

    scored = []

    for row in teacher_rows:
        score = edge_importance(
            row,
            endpoint_degree,
        )

        scored.append(
            (
                score,
                row,
            )
        )

    scored.sort(
        key=lambda item: (
            -item[0],
            item[1]["start"],
            item[1]["relation"],
            item[1]["end"],
        )
    )

    student = StudentGraph()

    selected_keys = set()

    # ---------------------------------------------------------------
    # Coverage pass:
    # Guarantee at least one strong edge for every dictionary word that
    # appears in the teacher graph, subject to the global budget.
    # ---------------------------------------------------------------

    best_for_word = {}

    for score, row in scored:
        word = row["start"]

        if word not in dictionary:
            continue

        if word not in best_for_word:
            best_for_word[
                word
            ] = (
                score,
                row,
            )

    coverage_rows = sorted(
        best_for_word.values(),
        key=lambda item: (
            -item[0],
            item[1]["start"],
        ),
    )

    for score, row in coverage_rows:
        if student.edge_count >= total_budget:
            break

        key = (
            row["start"],
            row["relation"],
            row["end"],
        )

        if key in selected_keys:
            continue

        selected_keys.add(
            key
        )

        student.add(
            row
        )

    # ---------------------------------------------------------------
    # Relation coverage:
    # prevent a small number of relations, especially RelatedTo, from
    # consuming the entire student budget.
    # ---------------------------------------------------------------

    relation_targets = {
        relation: max(
            1,
            int(
                total_budget
                * (
                    FOCUS_RELATION_WEIGHT.get(
                        relation,
                        1.0,
                    )
                    / max(
                        1.0,
                        sum(
                            FOCUS_RELATION_WEIGHT.get(
                                r,
                                1.0,
                            )
                            for r in SEMANTIC_RELATIONS
                        ),
                    )
                )
            ),
        )
        for relation in SEMANTIC_RELATIONS
    }

    relation_counts = Counter(
        row[1]["relation"]
        for row in selected_keys
        if False
    )

    # More directly derive counts from the student.
    relation_counts.clear()
    for start, rows in student.edges_by_start.items():
        for relation, _end, _weight in rows:
            relation_counts[relation] += 1

    relation_pass_candidates = defaultdict(list)

    for score, row in scored:
        relation_pass_candidates[
            row["relation"]
        ].append(
            (
                score,
                row,
            )
        )

    for relation in SEMANTIC_RELATIONS:
        target = min(
            relation_targets.get(
                relation,
                1,
            ),
            len(
                relation_pass_candidates[
                    relation
                ]
            ),
        )

        for score, row in relation_pass_candidates[
            relation
        ]:
            if student.edge_count >= total_budget:
                break

            if relation_counts[
                relation
            ] >= target:
                break

            key = (
                row["start"],
                row["relation"],
                row["end"],
            )

            if key in selected_keys:
                continue

            selected_keys.add(
                key
            )

            student.add(
                row
            )

            relation_counts[
                relation
            ] += 1

    # ---------------------------------------------------------------
    # Fill remaining budget with globally strongest edges.
    # ---------------------------------------------------------------

    for score, row in scored:
        if student.edge_count >= total_budget:
            break

        key = (
            row["start"],
            row["relation"],
            row["end"],
        )

        if key in selected_keys:
            continue

        selected_keys.add(
            key
        )

        student.add(
            row
        )

    summary = {
        "budget_fraction": budget_fraction,
        "teacher_edges": len(
            teacher_rows
        ),
        "student_edges": student.edge_count,
        "actual_fraction": (
            student.edge_count
            / max(
                1,
                len(teacher_rows),
            )
        ),
        "dictionary_words_with_edge": sum(
            1
            for word in dictionary
            if word in student.edges_by_start
        ),
    }

    return (
        student,
        summary,
    )


# ---------------------------------------------------------------------------
# Evaluation
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
        return (
            precision,
            recall,
            0.0,
        )

    return (
        precision,
        recall,
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        ),
    )


def evaluate_direct(
    graph: StudentGraph,
    words: list[str],
    gold: HumanGold,
) -> dict[str, float]:
    precisions = []
    recalls = []
    f1s = []

    covered = 0

    for word in words:
        predicted = set(
            graph.direct_concepts(
                word
            )
        )

        if predicted:
            covered += 1

        target = gold.gold(
            word
        )

        if not target:
            continue

        p, r, f = prf(
            predicted,
            target,
        )

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return {
        "covered_words": covered,
        "coverage": (
            covered
            / max(
                1,
                len(words),
            )
        ),
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


def evaluate_teacher_recall(
    teacher_rows: list[sqlite3.Row],
    student: StudentGraph,
    dictionary: set[str],
) -> dict[str, float]:
    teacher_keys = {
        (
            row["start"],
            row["relation"],
            row["end"],
        )
        for row in teacher_rows
        if row["start"] in dictionary
    }

    student_keys = set()

    for start, rows in student.edges_by_start.items():
        if start not in dictionary:
            continue

        for relation, end, _weight in rows:
            student_keys.add(
                (
                    start,
                    relation,
                    end,
                )
            )

    if not teacher_keys:
        return {
            "teacher_edges": 0,
            "student_edges": 0,
            "edge_recall": 0.0,
        }

    return {
        "teacher_edges": len(
            teacher_keys
        ),
        "student_edges": len(
            student_keys
        ),
        "edge_recall": (
            len(
                teacher_keys
                & student_keys
            )
            / len(
                teacher_keys
            )
        ),
    }


def evaluate_relation_scores(
    graph: StudentGraph,
    words: list[str],
    gold: HumanGold,
) -> dict[str, dict[str, float]]:
    result = {}

    for relation in SEMANTIC_RELATIONS:
        precisions = []
        recalls = []
        f1s = []

        covered = 0

        for word in words:
            predicted = set(
                graph.relation_concepts(
                    word,
                    relation,
                )
            )

            if predicted:
                covered += 1

            target = gold.gold(
                word
            )

            if not target:
                continue

            p, r, f = prf(
                predicted,
                target,
            )

            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        result[
            relation
        ] = {
            "covered_words": covered,
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

    return result


def random_student(
    teacher_rows: list[sqlite3.Row],
    dictionary: set[str],
    edge_count: int,
) -> StudentGraph:
    rng = random.Random(
        SEED
        + edge_count
    )

    candidates = [
        row
        for row in teacher_rows
        if row["start"] in dictionary
    ]

    if edge_count >= len(candidates):
        chosen = candidates
    else:
        chosen = rng.sample(
            candidates,
            edge_count,
        )

    graph = StudentGraph()

    for row in chosen:
        graph.add(
            row
        )

    return graph


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------

def trace_word(
    word: str,
    teacher: TeacherDB,
    student: StudentGraph,
    gold: HumanGold,
) -> dict:
    teacher_rows = teacher.word_edges(
        word
    )

    return {
        "word": word,
        "gold": sorted(
            gold.gold(word)
        ),
        "teacher_direct": [
            {
                "relation": row["relation"],
                "end": row["end"],
                "weight": row["weight"],
            }
            for row in teacher_rows[
                :32
            ]
        ],
        "teacher_concepts": [
            row["end"]
            for row in teacher_rows[
                :32
            ]
        ],
        "student_direct": student.direct_concepts(
            word
        ),
        "student_by_relation": {
            relation: student.relation_concepts(
                word,
                relation,
            )
            for relation in (
                "IsA",
                "UsedFor",
                "CapableOf",
                "HasProperty",
                "PartOf",
                "Synonym",
                "RelatedTo",
            )
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    started = time.perf_counter()

    print(
        "=== V126 CONCEPTNET TEACHER -> COMPRESSED STUDENT ==="
    )

    words = load_dictionary(
        DICTIONARY_PATH
    )

    dictionary = set(
        words
    )

    gold = HumanGold()
    gold.load(
        SEMANTICS_PATH
    )

    teacher = TeacherDB(
        DB_PATH
    )

    try:
        raw_edge_count = teacher.edge_count()

        print(
            "dictionary_words:",
            len(words),
        )

        print(
            "teacher_edges_total:",
            f"{raw_edge_count:,}",
        )

        # ---------------------------------------------------------------
        # Build the dictionary-centered teacher edge set.
        # ---------------------------------------------------------------

        extraction_started = time.perf_counter()

        print(
            "loading dictionary-centered teacher edges...",
            flush=True,
        )

        teacher_rows = teacher.all_dictionary_edges(
            dictionary
        )

        print(
            "teacher_rows_for_dictionary:",
            f"{len(teacher_rows):,}",
        )

        print(
            "teacher_extraction_seconds:",
            f"{time.perf_counter() - extraction_started:.2f}",
        )

        # ---------------------------------------------------------------
        # Teacher baseline.
        # ---------------------------------------------------------------

        teacher_graph = StudentGraph()

        for row in teacher_rows:
            teacher_graph.add(
                row
            )

        teacher_scores = evaluate_direct(
            teacher_graph,
            words,
            gold,
        )

        teacher_relation_scores = (
            evaluate_relation_scores(
                teacher_graph,
                words,
                gold,
            )
        )

        print()
        print(
            "=== TEACHER BASELINE ==="
        )

        print(
            "teacher_edges:",
            teacher_graph.edge_count,
        )

        print(
            "teacher_scores:",
            teacher_scores,
        )

        # ---------------------------------------------------------------
        # Compression budgets.
        # ---------------------------------------------------------------

        budget_results = {}

        for fraction in BUDGETS:
            print()
            print(
                f"=== STUDENT {fraction * 100:.1f}% ===",
                flush=True,
            )

            build_started = time.perf_counter()

            student, build_stats = build_student(
                teacher_rows,
                dictionary,
                fraction,
            )

            build_seconds = (
                time.perf_counter()
                - build_started
            )

            scores = evaluate_direct(
                student,
                words,
                gold,
            )

            relation_scores = (
                evaluate_relation_scores(
                    student,
                    words,
                    gold,
                )
            )

            teacher_recall = (
                evaluate_teacher_recall(
                    teacher_rows,
                    student,
                    dictionary,
                )
            )

            random_graph = random_student(
                teacher_rows,
                dictionary,
                student.edge_count,
            )

            random_scores = evaluate_direct(
                random_graph,
                words,
                gold,
            )

            print(
                "student_edges:",
                student.edge_count,
            )

            print(
                "actual_fraction:",
                build_stats[
                    "actual_fraction"
                ],
            )

            print(
                "coverage:",
                scores["coverage"],
            )

            print(
                "semantic_f1:",
                scores["f1"],
            )

            print(
                "teacher_edge_recall:",
                teacher_recall[
                    "edge_recall"
                ],
            )

            print(
                "random_f1:",
                random_scores["f1"],
            )

            print(
                "build_seconds:",
                f"{build_seconds:.3f}",
            )

            budget_results[
                str(fraction)
            ] = {
                "build": {
                    **build_stats,
                    "seconds": build_seconds,
                },
                "semantic": scores,
                "relation_scores": relation_scores,
                "teacher_edge_recall": teacher_recall,
                "random_control": random_scores,
                "student_edges_by_relation": dict(
                    student.edges_by_relation
                ),
            }

            # Trace from the first budget and the largest budget; those are
            # especially useful when reading the JSON.
            if fraction in (
                BUDGETS[0],
                BUDGETS[-1],
            ):
                budget_results[
                    str(fraction)
                ][
                    "traces"
                ] = [
                    trace_word(
                        word,
                        teacher,
                        student,
                        gold,
                    )
                    for word in TRACE_WORDS
                ]

        # ---------------------------------------------------------------
        # Compression curve.
        # ---------------------------------------------------------------

        curve = []

        for fraction in BUDGETS:
            item = budget_results[
                str(fraction)
            ]

            curve.append(
                {
                    "fraction": fraction,
                    "edges": item[
                        "build"
                    ]["student_edges"],
                    "semantic_f1": item[
                        "semantic"
                    ]["f1"],
                    "coverage": item[
                        "semantic"
                    ]["coverage"],
                    "teacher_edge_recall": item[
                        "teacher_edge_recall"
                    ]["edge_recall"],
                    "random_f1": item[
                        "random_control"
                    ]["f1"],
                }
            )

        print()
        print(
            "=== COMPRESSION CURVE ==="
        )

        print(
            "fraction | edges | coverage | F1 | teacher_recall | random_F1"
        )

        for row in curve:
            print(
                f"{row['fraction']:8.2%} | "
                f"{row['edges']:7d} | "
                f"{row['coverage']:.4f} | "
                f"{row['semantic_f1']:.4f} | "
                f"{row['teacher_edge_recall']:.4f} | "
                f"{row['random_f1']:.6f}"
            )

        # ---------------------------------------------------------------
        # Save.
        # ---------------------------------------------------------------

        report = {
            "experiment": (
                "V126 ConceptNet teacher -> compressed student"
            ),
            "seed": SEED,
            "dictionary_words": len(words),
            "teacher_total_edges": raw_edge_count,
            "teacher_dictionary_edges": len(
                teacher_rows
            ),
            "teacher_baseline": {
                "semantic": teacher_scores,
                "relation_scores": teacher_relation_scores,
                "edges": teacher_graph.edge_count,
            },
            "budgets": budget_results,
            "compression_curve": curve,
            "trace_words": TRACE_WORDS,
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
            "saved:",
            OUTPUT_PATH,
        )

        print(
            "elapsed_seconds:",
            f"{time.perf_counter() - started:.2f}",
        )

        print(
            "=== V126 COMPLETE ==="
        )

    finally:
        teacher.close()


if __name__ == "__main__":
    main()
