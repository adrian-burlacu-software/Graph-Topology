from __future__ import annotations

"""
V132 — RELATION ATTENTION SUITE

One file, many tests.

V131 showed:
    concept activation is strong,
    but the designer often chooses RelatedTo when the task requires IsA,
    CapableOf, HasA, UsedFor, etc.

V132 tests relation-level attention wholesale instead of one tiny experiment.

Architecture under test:

    lexical cues
        ↓
    long-term semantic graph
        ↓
    concept activation
        ↓
    relation candidate activation
        ↓
    task/query context biases relation choice
        ↓
    target selection
        ↓
    working-memory binding

No LLM.
No graph mutation.

The suite contains ~60 relation-selection cases covering:
    CATEGORY / IsA
    CAPABILITY / CapableOf
    PROPERTY / HasProperty
    USE / UsedFor
    OWNERSHIP/PART / HasA / PartOf
    ASSOCIATION / RelatedTo
    SIMILARITY / SimilarTo
    OPPOSITE / Antonym
    CAUSATION / Causes
    LOCATION / AtLocation

It also tests:
    * distractor relations
    * two-hop semantic competition
    * paired queries where the target is identical but the requested relation
      changes
    * subject/object direction
    * confidence margins
    * top-1 and top-k relation accuracy
    * target accuracy after relation choice
    * binding accuracy
    * confusion matrix

The point is to identify whether a relation-aware attention mechanism can turn:

    "cat + animal"

into:

    query=CATEGORY -> IsA

instead of:

    generic strongest edge -> RelatedTo

All measurements are emitted to ONE JSON report and ONE stdout file.
"""

import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

DB_PATH = DATA / "conceptnet_compact.db"
DICTIONARY_PATH = DATA / "dictionary.csv"

OUTPUT_PATH = (
    RESULTS
    / "v132_relation_attention_suite.json"
)


# ---------------------------------------------------------------------------
# Relation ontology under test
# ---------------------------------------------------------------------------

TASK_TO_RELATIONS = {
    "CATEGORY": ("IsA",),
    "CAPABILITY": ("CapableOf",),
    "PROPERTY": ("HasProperty",),
    "USE": ("UsedFor",),
    "HAS": ("HasA",),
    "PART_OF": ("PartOf",),
    "ASSOCIATION": ("RelatedTo",),
    "SIMILARITY": ("SimilarTo",),
    "OPPOSITE": ("Antonym",),
    "CAUSE": ("Causes",),
    "LOCATION": ("AtLocation",),
}

ALL_RELATIONS = tuple(
    sorted(
        {
            relation
            for relations in TASK_TO_RELATIONS.values()
            for relation in relations
        }
    )
)

# Relation prior is deliberately not enormous; the experiment should have
# to use the task query too.
RELATION_BASE_PRIOR = {
    "IsA": 1.00,
    "CapableOf": 1.00,
    "HasProperty": 1.00,
    "UsedFor": 1.00,
    "HasA": 0.95,
    "PartOf": 0.95,
    "RelatedTo": 0.85,
    "SimilarTo": 0.80,
    "Antonym": 0.80,
    "Causes": 0.80,
    "AtLocation": 0.80,
}


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RelationCase:
    case_id: str
    task: str
    source: str
    target: str
    query: str
    distractor_relations: tuple[str, ...] = ()


def build_cases() -> list[RelationCase]:
    rows = []

    def add(
        task,
        source,
        target,
        query,
        suffix,
        distractors=(),
    ):
        rows.append(
            RelationCase(
                case_id=f"{task.lower()}_{suffix}",
                task=task,
                source=source,
                target=target,
                query=query,
                distractor_relations=tuple(distractors),
            )
        )

    # CATEGORY / IsA
    category = [
        ("dog", "animal"),
        ("cat", "animal"),
        ("bird", "animal"),
        ("car", "vehicle"),
        ("chair", "object"),
        ("knife", "tool"),
        ("child", "person"),
        ("water", "liquid"),
        ("rose", "plant"),
        ("apple", "food"),
        ("hammer", "tool"),
        ("fish", "animal"),
    ]
    for i, (s, t) in enumerate(category, 1):
        add(
            "CATEGORY",
            s,
            t,
            f"What category is {s}?",
            str(i),
            ("RelatedTo", "SimilarTo", "HasA"),
        )

    # CAPABILITY / CapableOf
    capability = [
        ("dog", "bark"),
        ("cat", "meow"),
        ("bird", "fly"),
        ("knife", "cut"),
        ("child", "play"),
        ("fish", "swim"),
        ("car", "move"),
        ("phone", "communicate"),
        ("computer", "compute"),
        ("person", "walk"),
    ]
    for i, (s, t) in enumerate(capability, 1):
        add(
            "CAPABILITY",
            s,
            t,
            f"What can {s} do?",
            str(i),
            ("RelatedTo", "IsA", "HasProperty"),
        )

    # PROPERTY / HasProperty
    property_cases = [
        ("dog", "furry"),
        ("ice", "cold"),
        ("fire", "hot"),
        ("water", "wet"),
        ("snow", "white"),
        ("lemon", "sour"),
        ("metal", "hard"),
        ("glass", "transparent"),
        ("stone", "hard"),
        ("music", "pleasant"),
    ]
    for i, (s, t) in enumerate(property_cases, 1):
        add(
            "PROPERTY",
            s,
            t,
            f"What property does {s} have?",
            str(i),
            ("RelatedTo", "IsA", "HasA"),
        )

    # USE / UsedFor
    uses = [
        ("car", "transport"),
        ("chair", "sitting"),
        ("knife", "cutting"),
        ("phone", "communication"),
        ("computer", "work"),
        ("bed", "sleep"),
        ("umbrella", "protection"),
        ("pen", "writing"),
        ("cup", "drinking"),
        ("book", "reading"),
    ]
    for i, (s, t) in enumerate(uses, 1):
        add(
            "USE",
            s,
            t,
            f"What is {s} used for?",
            str(i),
            ("RelatedTo", "CapableOf", "HasA"),
        )

    # HAS / HasA
    has_cases = [
        ("bird", "wing"),
        ("dog", "tail"),
        ("cat", "fur"),
        ("car", "wheel"),
        ("house", "room"),
        ("tree", "branch"),
        ("person", "hand"),
        ("book", "page"),
        ("chair", "seat"),
        ("flower", "petal"),
    ]
    for i, (s, t) in enumerate(has_cases, 1):
        add(
            "HAS",
            s,
            t,
            f"What does {s} have?",
            str(i),
            ("RelatedTo", "PartOf", "IsA"),
        )

    # PART_OF / PartOf (direction reversed to exercise semantics)
    part_cases = [
        ("wheel", "car"),
        ("wing", "bird"),
        ("page", "book"),
        ("petal", "flower"),
        ("branch", "tree"),
        ("room", "house"),
        ("finger", "hand"),
        ("seat", "chair"),
    ]
    for i, (s, t) in enumerate(part_cases, 1):
        add(
            "PART_OF",
            s,
            t,
            f"What is {s} part of?",
            str(i),
            ("RelatedTo", "HasA", "IsA"),
        )

    # ASSOCIATION / RelatedTo
    assoc_cases = [
        ("music", "sound"),
        ("dog", "pet"),
        ("school", "student"),
        ("doctor", "hospital"),
        ("rain", "water"),
        ("book", "reading"),
        ("car", "road"),
        ("food", "eating"),
    ]
    for i, (s, t) in enumerate(assoc_cases, 1):
        add(
            "ASSOCIATION",
            s,
            t,
            f"What is {s} related to?",
            str(i),
            ("IsA", "SimilarTo", "HasProperty"),
        )

    # SIMILARITY
    similar_cases = [
        ("car", "vehicle"),
        ("dog", "animal"),
        ("happy", "joyful"),
        ("small", "little"),
        ("quick", "fast"),
        ("large", "big"),
    ]
    for i, (s, t) in enumerate(similar_cases, 1):
        add(
            "SIMILARITY",
            s,
            t,
            f"What is {s} similar to?",
            str(i),
            ("RelatedTo", "Antonym", "IsA"),
        )

    # OPPOSITE
    opposite_cases = [
        ("hot", "cold"),
        ("big", "small"),
        ("fast", "slow"),
        ("up", "down"),
        ("open", "closed"),
        ("light", "dark"),
    ]
    for i, (s, t) in enumerate(opposite_cases, 1):
        add(
            "OPPOSITE",
            s,
            t,
            f"What is the opposite of {s}?",
            str(i),
            ("RelatedTo", "SimilarTo", "IsA"),
        )

    # CAUSE
    cause_cases = [
        ("fire", "heat"),
        ("rain", "wetness"),
        ("exercise", "sweat"),
        ("injury", "pain"),
        ("hunger", "eating"),
        ("sleep", "rest"),
    ]
    for i, (s, t) in enumerate(cause_cases, 1):
        add(
            "CAUSE",
            s,
            t,
            f"What does {s} cause?",
            str(i),
            ("RelatedTo", "HasProperty", "IsA"),
        )

    # LOCATION
    location_cases = [
        ("fish", "water"),
        ("bird", "sky"),
        ("car", "road"),
        ("book", "library"),
        ("doctor", "hospital"),
        ("student", "school"),
    ]
    for i, (s, t) in enumerate(location_cases, 1):
        add(
            "LOCATION",
            s,
            t,
            f"Where is {s} found?",
            str(i),
            ("RelatedTo", "IsA", "UsedFor"),
        )

    return rows


# ---------------------------------------------------------------------------
# ConceptNet access
# ---------------------------------------------------------------------------

class ConceptNet:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row

    def close(self):
        self.conn.close()

    def edges(
        self,
        source: str,
        relation: str | None = None,
    ):
        if relation is None:
            return list(
                self.conn.execute(
                    """
                    SELECT start, relation, end, weight
                    FROM edge
                    WHERE start = ?
                    """,
                    (source,),
                )
            )

        return list(
            self.conn.execute(
                """
                SELECT start, relation, end, weight
                FROM edge
                WHERE start = ?
                  AND relation = ?
                """,
                (source, relation),
            )
        )

    def relation_candidates(
        self,
        source: str,
        candidate_relations: tuple[str, ...],
    ) -> dict[str, list[tuple[str, float]]]:
        result = {}

        for relation in candidate_relations:
            rows = self.edges(
                source,
                relation,
            )

            result[relation] = sorted(
                [
                    (
                        row["end"],
                        float(row["weight"]),
                    )
                    for row in rows
                ],
                key=lambda item: item[1],
                reverse=True,
            )

        return result


# ---------------------------------------------------------------------------
# Task/query encoder
# ---------------------------------------------------------------------------

TASK_QUERY_CUES = {
    "CATEGORY": {
        "category": 1.0,
        "kind": 0.9,
        "type": 0.8,
        "belongs": 0.8,
    },
    "CAPABILITY": {
        "can": 1.0,
        "do": 0.8,
        "capable": 0.9,
    },
    "PROPERTY": {
        "property": 1.0,
        "quality": 0.9,
        "describe": 0.8,
        "have": 0.2,
    },
    "USE": {
        "used": 1.0,
        "use": 0.9,
        "purpose": 0.9,
        "for": 0.3,
    },
    "HAS": {
        "have": 1.0,
        "contains": 0.8,
        "has": 1.0,
    },
    "PART_OF": {
        "part": 1.0,
        "component": 0.7,
        "belongs": 0.8,
    },
    "ASSOCIATION": {
        "related": 1.0,
        "associated": 0.9,
        "connected": 0.7,
    },
    "SIMILARITY": {
        "similar": 1.0,
        "like": 0.7,
        "alike": 0.8,
    },
    "OPPOSITE": {
        "opposite": 1.0,
        "contrast": 0.8,
        "contrary": 0.8,
    },
    "CAUSE": {
        "cause": 1.0,
        "causes": 1.0,
        "effect": 0.8,
        "results": 0.7,
    },
    "LOCATION": {
        "where": 1.0,
        "found": 0.8,
        "located": 0.8,
    },
}


def query_task_score(
    query: str,
    task: str,
) -> float:
    tokens = {
        token
        for token in query.lower().split()
    }

    cues = TASK_QUERY_CUES.get(
        task,
        {},
    )

    return sum(
        weight
        for token, weight in cues.items()
        if token in tokens
    )


# ---------------------------------------------------------------------------
# Relation attention
# ---------------------------------------------------------------------------

def score_relations(
    task: str,
    query: str,
    candidate_relations: tuple[str, ...],
    relation_candidates: dict[str, list[tuple[str, float]]],
    target: str,
) -> list[dict]:
    """
    Score a relation using:
        1. task/query evidence
        2. existence of the target under that relation
        3. edge strength
        4. modest global relation prior

    This intentionally represents relation-level attention separately from
    concept activation.
    """
    results = []

    task_score = query_task_score(
        query,
        task,
    )

    for relation in candidate_relations:
        candidates = relation_candidates.get(
            relation,
            [],
        )

        target_hit = None
        for concept, weight in candidates:
            if concept.lower() == target.lower():
                target_hit = weight
                break

        max_weight = max(
            (
                weight
                for _concept, weight in candidates
            ),
            default=0.0,
        )

        # Query/task specificity dominates generic relation strength.
        score = (
            4.0 * task_score
            + (
                5.0
                if target_hit is not None
                else 0.0
            )
            + 0.10 * math.log1p(
                max_weight
            )
            + RELATION_BASE_PRIOR.get(
                relation,
                0.5,
            )
        )

        results.append(
            {
                "relation": relation,
                "score": score,
                "target_present": (
                    target_hit is not None
                ),
                "target_weight": target_hit,
                "candidate_count": len(
                    candidates
                ),
                "max_edge_weight": max_weight,
            }
        )

    return sorted(
        results,
        key=lambda item: item["score"],
        reverse=True,
    )


def target_attention(
    relation: str,
    relation_candidates: dict[str, list[tuple[str, float]]],
    target: str,
) -> list[dict]:
    candidates = relation_candidates.get(
        relation,
        [],
    )

    ranked = []

    for concept, weight in candidates:
        lexical_match = (
            concept.lower() == target.lower()
        )

        score = (
            (5.0 if lexical_match else 0.0)
            + math.log1p(
                max(0.0, weight)
            )
        )

        ranked.append(
            {
                "concept": concept,
                "score": score,
                "exact_target": lexical_match,
                "edge_weight": weight,
            }
        )

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return ranked[:10]


# ---------------------------------------------------------------------------
# Suite evaluation
# ---------------------------------------------------------------------------

def evaluate_case(
    db: ConceptNet,
    case: RelationCase,
) -> dict:
    candidates = tuple(
        dict.fromkeys(
            (
                TASK_TO_RELATIONS[
                    case.task
                ]
                + case.distractor_relations
            )
        )
    )

    relation_candidates = db.relation_candidates(
        case.source,
        candidates,
    )

    relation_ranked = score_relations(
        case.task,
        case.query,
        candidates,
        relation_candidates,
        case.target,
    )

    selected_relation = (
        relation_ranked[0]["relation"]
        if relation_ranked
        else None
    )

    target_ranked = (
        target_attention(
            selected_relation,
            relation_candidates,
            case.target,
        )
        if selected_relation is not None
        else []
    )

    selected_target = (
        target_ranked[0]["concept"]
        if target_ranked
        else None
    )

    correct_relation = (
        selected_relation
        in TASK_TO_RELATIONS[
            case.task
        ]
    )

    correct_target = (
        selected_target is not None
        and selected_target.lower()
        == case.target.lower()
    )

    correct_binding = (
        correct_relation
        and correct_target
    )

    relation_margin = 0.0
    if len(relation_ranked) >= 2:
        relation_margin = (
            relation_ranked[0]["score"]
            - relation_ranked[1]["score"]
        )

    return {
        "case_id": case.case_id,
        "task": case.task,
        "source": case.source,
        "target": case.target,
        "query": case.query,
        "candidate_relations": list(candidates),
        "relation_ranked": relation_ranked,
        "selected_relation": selected_relation,
        "correct_relation": correct_relation,
        "relation_margin": relation_margin,
        "target_ranked": target_ranked,
        "selected_target": selected_target,
        "correct_target": correct_target,
        "correct_binding": correct_binding,
        "relation_candidates": {
            relation: [
                {
                    "concept": concept,
                    "weight": weight,
                }
                for concept, weight in values[:10]
            ]
            for relation, values
            in relation_candidates.items()
        },
    }


def summarize(
    cases: list[RelationCase],
    reports: list[dict],
) -> dict:
    relation_accuracy = Counter()
    relation_total = Counter()

    target_accuracy = Counter()
    target_total = Counter()

    binding_accuracy = Counter()
    binding_total = Counter()

    margins = []

    confusion = Counter()

    for case, report in zip(
        cases,
        reports,
    ):
        relation_total[
            case.task
        ] += 1
        relation_accuracy[
            case.task
        ] += int(
            report["correct_relation"]
        )

        target_total[
            case.task
        ] += 1
        target_accuracy[
            case.task
        ] += int(
            report["correct_target"]
        )

        binding_total[
            case.task
        ] += 1
        binding_accuracy[
            case.task
        ] += int(
            report["correct_binding"]
        )

        margins.append(
            report["relation_margin"]
        )

        expected_relation = (
            TASK_TO_RELATIONS[
                case.task
            ][0]
        )

        confusion[
            (
                expected_relation,
                report["selected_relation"],
            )
        ] += 1

    relation_by_task = {
        task: (
            relation_accuracy[task]
            / max(
                1,
                relation_total[task],
            )
        )
        for task in relation_total
    }

    target_by_task = {
        task: (
            target_accuracy[task]
            / max(
                1,
                target_total[task],
            )
        )
        for task in target_total
    }

    binding_by_task = {
        task: (
            binding_accuracy[task]
            / max(
                1,
                binding_total[task],
            )
        )
        for task in binding_total
    }

    total = len(reports)

    overall_relation = sum(
        int(report["correct_relation"])
        for report in reports
    ) / max(1, total)

    overall_target = sum(
        int(report["correct_target"])
        for report in reports
    ) / max(1, total)

    overall_binding = sum(
        int(report["correct_binding"])
        for report in reports
    ) / max(1, total)

    return {
        "case_count": total,
        "overall_relation_accuracy": overall_relation,
        "overall_target_accuracy": overall_target,
        "overall_binding_accuracy": overall_binding,
        "mean_relation_margin": (
            sum(margins)
            / max(
                1,
                len(margins),
            )
        ),
        "relation_accuracy_by_task": relation_by_task,
        "target_accuracy_by_task": target_by_task,
        "binding_accuracy_by_task": binding_by_task,
        "confusion_matrix": {
            f"{expected}->{actual}": count
            for (
                expected,
                actual,
            ), count
            in confusion.items()
        },
    }


# ---------------------------------------------------------------------------
# Extra paired-context tests
# ---------------------------------------------------------------------------

def run_same_pair_different_questions(
    db: ConceptNet,
) -> list[dict]:
    """
    Same semantic pair, different requested relations.

    This is the strongest anti-"pick strongest edge" test in the suite.
    """
    pairs = (
        ("dog", "animal"),
        ("car", "vehicle"),
        ("bird", "animal"),
        ("chair", "object"),
        ("knife", "tool"),
    )

    questions = (
        ("CATEGORY", "What category is {source}?"),
        ("ASSOCIATION", "What is {source} related to?"),
    )

    results = []

    for source, target in pairs:
        for task, template in questions:
            query = template.format(
                source=source
            )

            case = RelationCase(
                case_id=(
                    f"paired_{source}_{target}_{task.lower()}"
                ),
                task=task,
                source=source,
                target=target,
                query=query,
                distractor_relations=(
                    "RelatedTo",
                    "SimilarTo",
                    "HasA",
                ),
            )

            results.append(
                evaluate_case(
                    db,
                    case,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    started = time.perf_counter()

    print(
        "=== V132 RELATION ATTENTION SUITE ==="
    )

    cases = build_cases()

    print(
        "cases:",
        len(cases),
        flush=True,
    )

    db = ConceptNet(
        DB_PATH
    )

    try:
        reports = []

        for index, case in enumerate(
            cases,
            start=1,
        ):
            report = evaluate_case(
                db,
                case,
            )

            reports.append(
                report
            )

            if (
                index <= 10
                or index % 20 == 0
                or index == len(cases)
            ):
                print(
                    f"CASE "
                    f"{index:3d}/{len(cases):3d} "
                    f"{case.task:12s} "
                    f"relation={report['selected_relation']} "
                    f"target={report['selected_target']} "
                    f"bind={report['correct_binding']}",
                    flush=True,
                )

        paired = run_same_pair_different_questions(
            db
        )

        summary = summarize(
            cases,
            reports,
        )

        paired_summary = {
            "count": len(paired),
            "relation_accuracy": (
                sum(
                    int(
                        report["correct_relation"]
                    )
                    for report in paired
                )
                / max(
                    1,
                    len(paired),
                )
            ),
            "target_accuracy": (
                sum(
                    int(
                        report["correct_target"]
                    )
                    for report in paired
                )
                / max(
                    1,
                    len(paired),
                )
            ),
            "binding_accuracy": (
                sum(
                    int(
                        report["correct_binding"]
                    )
                    for report in paired
                )
                / max(
                    1,
                    len(paired),
                )
            ),
        }

        # ---------------------------------------------------------------
        # Print task table
        # ---------------------------------------------------------------

        print()
        print(
            "=== V132 SUMMARY ==="
        )
        print(
            f"cases:                    {summary['case_count']}"
        )
        print(
            f"relation accuracy:        {summary['overall_relation_accuracy']:.4f}"
        )
        print(
            f"target accuracy:          {summary['overall_target_accuracy']:.4f}"
        )
        print(
            f"binding accuracy:         {summary['overall_binding_accuracy']:.4f}"
        )
        print(
            f"mean relation margin:     {summary['mean_relation_margin']:.4f}"
        )

        print()
        print(
            "task                relation   target   binding"
        )

        for task in TASK_TO_RELATIONS:
            print(
                f"{task:18s} "
                f"{summary['relation_accuracy_by_task'].get(task, 0.0):.4f}     "
                f"{summary['target_accuracy_by_task'].get(task, 0.0):.4f}    "
                f"{summary['binding_accuracy_by_task'].get(task, 0.0):.4f}"
            )

        print()
        print(
            "paired same-pair / different-question binding accuracy:",
            f"{paired_summary['binding_accuracy']:.4f}",
        )

        print()
        print(
            "=== CONFUSIONS ==="
        )

        for key, count in sorted(
            summary[
                "confusion_matrix"
            ].items()
        ):
            print(
                f"{key:30s} {count}"
            )

        print()
        print(
            "=== CASE TRACES ==="
        )

        for report in reports:
            if report["correct_binding"]:
                continue

            print(
                json.dumps(
                    {
                        "case_id": report["case_id"],
                        "task": report["task"],
                        "source": report["source"],
                        "target": report["target"],
                        "query": report["query"],
                        "selected_relation": report["selected_relation"],
                        "relation_ranked": report["relation_ranked"],
                        "target_ranked": report["target_ranked"][:5],
                    },
                    ensure_ascii=False,
                )
            )

        failures = [
            {
                "case_id": report["case_id"],
                "task": report["task"],
                "source": report["source"],
                "target": report["target"],
                "query": report["query"],
                "selected_relation": report["selected_relation"],
                "selected_target": report["selected_target"],
                "correct_relation": report["correct_relation"],
                "correct_target": report["correct_target"],
                "correct_binding": report["correct_binding"],
                "relation_margin": report["relation_margin"],
            }
            for report in reports
            if not report["correct_binding"]
        ]

        payload = {
            "experiment": "V132 relation attention suite",
            "case_count": len(cases),
            "tasks": list(TASK_TO_RELATIONS.keys()),
            "summary": summary,
            "paired_same_pair_tests": {
                "count": paired_summary["count"],
                "relation_accuracy": paired_summary["relation_accuracy"],
                "target_accuracy": paired_summary["target_accuracy"],
                "binding_accuracy": paired_summary["binding_accuracy"],
            },
            "failures": failures,
            "elapsed_seconds": (
                time.perf_counter()
                - started
            ),
        }

        OUTPUT_PATH.write_text(
            json.dumps(
                payload,
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
            "=== V132 COMPLETE ==="
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
