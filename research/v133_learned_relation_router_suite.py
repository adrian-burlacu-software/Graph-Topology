from __future__ import annotations

"""
V133 — LEARNED TASK-CONDITIONED RELATION ROUTER

V132 showed:
    - target retrieval is often strong,
    - relation selection is the bottleneck,
    - the same semantic pair can support different answers depending on query.

V133 adds a persistent, learnable relation-routing module.

Architecture:

    QUERY
      |
      v
    task/context cue encoder
      |
      v
    relation router
      |
      +--> IsA
      +--> CapableOf
      +--> HasProperty
      +--> UsedFor
      +--> HasA
      +--> PartOf
      +--> RelatedTo
      +--> ...
      |
      v
    target attention within selected relation
      |
      v
    working-memory binding

The router is intentionally tiny and explicit.

It learns:
    task -> relation preference

using a perceptron-style online update:

    score(gold relation) += learning_rate
    score(chosen wrong relation) -= learning_rate

The graph itself is NOT changed during router training.

There are three evaluation modes:

1. ZERO-SHOT
       neutral task relation weights

2. TRAINED / HELD-OUT
       train on one deterministic subset per task
       evaluate on held-out cases

3. PAIRED QUERY
       same source/target pair, different questions
       verifies task-conditioned routing rather than raw edge strength

This is a baseline for learned modular attention, not a claim of unsupervised
cognition. The point is to see whether a small persistent routing controller
can solve the exact failure exposed by V132.

No LLM.
No graph mutation.
One file / one report.
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
OUTPUT_PATH = RESULTS / "v133_learned_relation_router.json"


# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------

TASK_TO_RELATION = {
    "CATEGORY": "IsA",
    "CAPABILITY": "CapableOf",
    "PROPERTY": "HasProperty",
    "USE": "UsedFor",
    "HAS": "HasA",
    "PART_OF": "PartOf",
    "ASSOCIATION": "RelatedTo",
    "SIMILARITY": "SimilarTo",
    "OPPOSITE": "Antonym",
    "CAUSE": "Causes",
    "LOCATION": "AtLocation",
}

RELATIONS = tuple(
    sorted(
        set(
            TASK_TO_RELATION.values()
        )
    )
)

BASE_PRIOR = {
    "IsA": 0.00,
    "CapableOf": 0.00,
    "HasProperty": 0.00,
    "UsedFor": 0.00,
    "HasA": -0.05,
    "PartOf": -0.05,
    "RelatedTo": -0.10,
    "SimilarTo": -0.15,
    "Antonym": -0.15,
    "Causes": -0.15,
    "AtLocation": -0.15,
}


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    case_id: str
    task: str
    source: str
    target: str
    query: str
    distractors: tuple[str, ...] = ()


def build_cases() -> list[Case]:
    result: list[Case] = []

    def add_many(
        task: str,
        pairs: list[tuple[str, str]],
        query_template: str,
        distractors: tuple[str, ...],
        prefix: str,
    ) -> None:
        for index, (source, target) in enumerate(
            pairs,
            1,
        ):
            result.append(
                Case(
                    case_id=f"{prefix}_{index:02d}",
                    task=task,
                    source=source,
                    target=target,
                    query=query_template.format(
                        source=source,
                    ),
                    distractors=distractors,
                )
            )

    add_many(
        "CATEGORY",
        [
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
        ],
        "What category is {source}?",
        ("RelatedTo", "SimilarTo", "HasA"),
        "category",
    )

    add_many(
        "CAPABILITY",
        [
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
        ],
        "What can {source} do?",
        ("RelatedTo", "IsA", "HasProperty"),
        "capability",
    )

    add_many(
        "PROPERTY",
        [
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
        ],
        "What property does {source} have?",
        ("RelatedTo", "IsA", "HasA"),
        "property",
    )

    add_many(
        "USE",
        [
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
        ],
        "What is {source} used for?",
        ("RelatedTo", "CapableOf", "HasA"),
        "use",
    )

    add_many(
        "HAS",
        [
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
        ],
        "What does {source} have?",
        ("RelatedTo", "PartOf", "IsA"),
        "has",
    )

    add_many(
        "PART_OF",
        [
            ("wheel", "car"),
            ("wing", "bird"),
            ("page", "book"),
            ("petal", "flower"),
            ("branch", "tree"),
            ("room", "house"),
            ("finger", "hand"),
            ("seat", "chair"),
        ],
        "What is {source} part of?",
        ("RelatedTo", "HasA", "IsA"),
        "part",
    )

    add_many(
        "ASSOCIATION",
        [
            ("music", "sound"),
            ("dog", "pet"),
            ("school", "student"),
            ("doctor", "hospital"),
            ("rain", "water"),
            ("book", "reading"),
            ("car", "road"),
            ("food", "eating"),
        ],
        "What is {source} related to?",
        ("IsA", "SimilarTo", "HasProperty"),
        "association",
    )

    add_many(
        "SIMILARITY",
        [
            ("car", "vehicle"),
            ("dog", "animal"),
            ("happy", "joyful"),
            ("small", "little"),
            ("quick", "fast"),
            ("large", "big"),
        ],
        "What is {source} similar to?",
        ("RelatedTo", "Antonym", "IsA"),
        "similarity",
    )

    add_many(
        "OPPOSITE",
        [
            ("hot", "cold"),
            ("big", "small"),
            ("fast", "slow"),
            ("up", "down"),
            ("open", "closed"),
            ("light", "dark"),
        ],
        "What is the opposite of {source}?",
        ("RelatedTo", "SimilarTo", "IsA"),
        "opposite",
    )

    add_many(
        "CAUSE",
        [
            ("fire", "heat"),
            ("rain", "wetness"),
            ("exercise", "sweat"),
            ("injury", "pain"),
            ("hunger", "eating"),
            ("sleep", "rest"),
        ],
        "What does {source} cause?",
        ("RelatedTo", "HasProperty", "IsA"),
        "cause",
    )

    add_many(
        "LOCATION",
        [
            ("fish", "water"),
            ("bird", "sky"),
            ("car", "road"),
            ("book", "library"),
            ("doctor", "hospital"),
            ("student", "school"),
        ],
        "Where is {source} found?",
        ("RelatedTo", "IsA", "UsedFor"),
        "location",
    )

    return result


# ---------------------------------------------------------------------------
# ConceptNet access
# ---------------------------------------------------------------------------

class ConceptNet:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row

    def close(self):
        self.conn.close()

    def candidates(
        self,
        source: str,
        relation: str,
    ) -> list[tuple[str, float]]:
        rows = self.conn.execute(
            """
            SELECT end, weight
            FROM edge
            WHERE start = ?
              AND relation = ?
            ORDER BY weight DESC
            """,
            (source, relation),
        )

        return [
            (
                row["end"],
                float(row["weight"]),
            )
            for row in rows
        ]

    def target_exists(
        self,
        source: str,
        relation: str,
        target: str,
    ) -> tuple[bool, float]:
        row = self.conn.execute(
            """
            SELECT weight
            FROM edge
            WHERE start = ?
              AND relation = ?
              AND end = ?
            LIMIT 1
            """,
            (source, relation, target),
        ).fetchone()

        if row is None:
            return False, 0.0

        return True, float(row["weight"])


# ---------------------------------------------------------------------------
# Query encoder
# ---------------------------------------------------------------------------

QUERY_FEATURES = {
    "CATEGORY": {
        "category": 1.0,
        "kind": 0.9,
        "type": 0.8,
        "belongs": 0.7,
    },
    "CAPABILITY": {
        "can": 1.0,
        "capable": 0.9,
        "do": 0.8,
    },
    "PROPERTY": {
        "property": 1.0,
        "quality": 0.9,
        "describe": 0.8,
    },
    "USE": {
        "used": 1.0,
        "use": 0.9,
        "purpose": 0.8,
    },
    "HAS": {
        "have": 1.0,
        "has": 1.0,
        "contains": 0.8,
    },
    "PART_OF": {
        "part": 1.0,
        "component": 0.8,
    },
    "ASSOCIATION": {
        "related": 1.0,
        "associated": 0.9,
    },
    "SIMILARITY": {
        "similar": 1.0,
        "alike": 0.8,
    },
    "OPPOSITE": {
        "opposite": 1.0,
        "contrast": 0.8,
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
        "located": 0.7,
    },
}


def query_features(
    query: str,
) -> set[str]:
    tokens = {
        token.strip("?!.,")
        for token in query.lower().split()
    }
    return tokens


# ---------------------------------------------------------------------------
# Learned relation router
# ---------------------------------------------------------------------------

class RelationRouter:
    def __init__(
        self,
        relations: tuple[str, ...],
        learning_rate: float = 0.60,
    ):
        self.relations = relations
        self.learning_rate = learning_rate

        # task -> relation -> learnable logit
        self.weights = {
            task: {
                relation: 0.0
                for relation in relations
            }
            for task in TASK_TO_RELATION
        }

        # Query token -> relation correction. This gives us a modular
        # context-to-relation pathway without a neural network.
        self.token_weights: dict[
            str,
            Counter[str],
        ] = defaultdict(Counter)

        self.updates = 0

    def score(
        self,
        task: str,
        query: str,
        relation: str,
    ) -> float:
        score = (
            self.weights[
                task
            ][
                relation
            ]
            + BASE_PRIOR.get(
                relation,
                -0.20,
            )
        )

        for token in query_features(query):
            score += self.token_weights[
                token
            ].get(
                relation,
                0.0,
            )

        return score

    def rank(
        self,
        task: str,
        query: str,
    ) -> list[tuple[str, float]]:
        return sorted(
            (
                (
                    relation,
                    self.score(
                        task,
                        query,
                        relation,
                    ),
                )
                for relation in self.relations
            ),
            key=lambda item: item[1],
            reverse=True,
        )

    def update(
        self,
        task: str,
        query: str,
        gold_relation: str,
        predicted_relation: str,
    ) -> bool:
        if gold_relation == predicted_relation:
            return False

        delta = self.learning_rate

        self.weights[
            task
        ][
            gold_relation
        ] += delta

        self.weights[
            task
        ][
            predicted_relation
        ] -= delta

        for token in query_features(query):
            self.token_weights[
                token
            ][
                gold_relation
            ] += delta * 0.15

            self.token_weights[
                token
            ][
                predicted_relation
            ] -= delta * 0.15

        self.updates += 1
        return True

    def snapshot(self) -> dict:
        return {
            "task_weights": self.weights,
            "token_weights": {
                token: dict(counter)
                for token, counter
                in self.token_weights.items()
            },
            "updates": self.updates,
        }


# ---------------------------------------------------------------------------
# Relation attention + target attention
# ---------------------------------------------------------------------------

def target_attention(
    db: ConceptNet,
    source: str,
    relation: str,
    target: str,
) -> dict:
    candidates = db.candidates(
        source,
        relation,
    )

    ranked = []

    for concept, weight in candidates[
        :20
    ]:
        exact = (
            concept.lower()
            == target.lower()
        )

        score = (
            (5.0 if exact else 0.0)
            + math.log1p(
                max(
                    0.0,
                    weight,
                )
            )
        )

        ranked.append(
            {
                "concept": concept,
                "score": score,
                "edge_weight": weight,
                "exact_target": exact,
            }
        )

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    selected = (
        ranked[0]["concept"]
        if ranked
        else None
    )

    return {
        "selected": selected,
        "correct": (
            selected is not None
            and selected.lower()
            == target.lower()
        ),
        "ranked": ranked[:10],
    }


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_by_task(
    cases: list[Case],
    train_fraction: float = 0.67,
) -> tuple[list[Case], list[Case]]:
    grouped = defaultdict(list)

    for case in cases:
        grouped[
            case.task
        ].append(case)

    train = []
    test = []

    for task, task_cases in grouped.items():
        cut = max(
            1,
            int(
                len(task_cases)
                * train_fraction
            ),
        )

        train.extend(
            task_cases[:cut]
        )

        test.extend(
            task_cases[cut:]
        )

    return train, test


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_router(
    router: RelationRouter,
    db: ConceptNet,
    cases: list[Case],
    update: bool = False,
) -> list[dict]:
    reports = []

    for case in cases:
        ranked = router.rank(
            case.task,
            case.query,
        )

        selected_relation = (
            ranked[0][0]
            if ranked
            else None
        )

        gold_relation = TASK_TO_RELATION[
            case.task
        ]

        target_result = target_attention(
            db,
            case.source,
            selected_relation,
            case.target,
        )

        correct_relation = (
            selected_relation
            == gold_relation
        )

        correct_binding = (
            correct_relation
            and target_result["correct"]
        )

        margin = 0.0

        if len(ranked) >= 2:
            margin = (
                ranked[0][1]
                - ranked[1][1]
            )

        report = {
            "case_id": case.case_id,
            "task": case.task,
            "source": case.source,
            "target": case.target,
            "query": case.query,
            "gold_relation": gold_relation,
            "ranked_relations": ranked,
            "selected_relation": selected_relation,
            "correct_relation": correct_relation,
            "relation_margin": margin,
            "selected_target": target_result[
                "selected"
            ],
            "correct_target": target_result[
                "correct"
            ],
            "correct_binding": correct_binding,
        }

        reports.append(
            report
        )

        if update:
            router.update(
                case.task,
                case.query,
                gold_relation,
                selected_relation,
            )

    return reports


def aggregate(
    reports: list[dict],
) -> dict:
    by_task = defaultdict(list)

    for report in reports:
        by_task[
            report["task"]
        ].append(
            report
        )

    relation_accuracy = {}
    target_accuracy = {}
    binding_accuracy = {}

    for task, task_reports in by_task.items():
        relation_accuracy[
            task
        ] = sum(
            int(r["correct_relation"])
            for r in task_reports
        ) / max(
            1,
            len(task_reports),
        )

        target_accuracy[
            task
        ] = sum(
            int(r["correct_target"])
            for r in task_reports
        ) / max(
            1,
            len(task_reports),
        )

        binding_accuracy[
            task
        ] = sum(
            int(r["correct_binding"])
            for r in task_reports
        ) / max(
            1,
            len(task_reports),
        )

    margins = [
        r["relation_margin"]
        for r in reports
    ]

    return {
        "count": len(reports),
        "relation_accuracy": (
            sum(
                int(r["correct_relation"])
                for r in reports
            )
            / max(
                1,
                len(reports),
            )
        ),
        "target_accuracy": (
            sum(
                int(r["correct_target"])
                for r in reports
            )
            / max(
                1,
                len(reports),
            )
        ),
        "binding_accuracy": (
            sum(
                int(r["correct_binding"])
                for r in reports
            )
            / max(
                1,
                len(reports),
            )
        ),
        "mean_relation_margin": (
            sum(margins)
            / max(
                1,
                len(margins),
            )
        ),
        "relation_accuracy_by_task": relation_accuracy,
        "target_accuracy_by_task": target_accuracy,
        "binding_accuracy_by_task": binding_accuracy,
    }


# ---------------------------------------------------------------------------
# Paired tests
# ---------------------------------------------------------------------------

def paired_cases() -> list[Case]:
    pairs = (
        ("dog", "animal"),
        ("cat", "animal"),
        ("car", "vehicle"),
        ("bird", "animal"),
        ("chair", "object"),
        ("knife", "tool"),
    )

    result = []

    for source, target in pairs:
        result.append(
            Case(
                case_id=f"paired_category_{source}",
                task="CATEGORY",
                source=source,
                target=target,
                query=f"What category is {source}?",
            )
        )

        result.append(
            Case(
                case_id=f"paired_association_{source}",
                task="ASSOCIATION",
                source=source,
                target=target,
                query=f"What is {source} related to?",
            )
        )

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    started = time.perf_counter()

    print(
        "=== V133 LEARNED RELATION ROUTER ==="
    )

    cases = build_cases()

    train_cases, test_cases = split_by_task(
        cases,
        train_fraction=0.67,
    )

    paired = paired_cases()

    print(
        "total_cases:",
        len(cases),
    )

    print(
        "train_cases:",
        len(train_cases),
    )

    print(
        "heldout_cases:",
        len(test_cases),
    )

    db = ConceptNet(
        DB_PATH
    )

    try:
        # ---------------------------------------------------------------
        # Zero-shot baseline
        # ---------------------------------------------------------------

        zero_router = RelationRouter(
            RELATIONS,
            learning_rate=0.60,
        )

        zero_reports = evaluate_router(
            zero_router,
            db,
            cases,
            update=False,
        )

        zero_summary = aggregate(
            zero_reports
        )

        print()
        print(
            "=== ZERO-SHOT ==="
        )
        print(
            "relation_accuracy:",
            zero_summary[
                "relation_accuracy"
            ],
        )
        print(
            "target_accuracy:",
            zero_summary[
                "target_accuracy"
            ],
        )
        print(
            "binding_accuracy:",
            zero_summary[
                "binding_accuracy"
            ],
        )

        # ---------------------------------------------------------------
        # Learned router
        # ---------------------------------------------------------------

        router = RelationRouter(
            RELATIONS,
            learning_rate=0.60,
        )

        train_history = []

        # A few passes are cheap and let us test whether the routing policy
        # converges rather than memorizing a single ordering artifact.
        for epoch in range(1, 6):
            before = evaluate_router(
                router,
                db,
                train_cases,
                update=False,
            )

            before_summary = aggregate(
                before
            )

            train_reports = evaluate_router(
                router,
                db,
                train_cases,
                update=True,
            )

            after = evaluate_router(
                router,
                db,
                train_cases,
                update=False,
            )

            after_summary = aggregate(
                after
            )

            train_history.append(
                {
                    "epoch": epoch,
                    "before": before_summary,
                    "after": after_summary,
                    "updates": router.updates,
                }
            )

            print(
                f"EPOCH {epoch} "
                f"train_before="
                f"{before_summary['binding_accuracy']:.4f} "
                f"train_after="
                f"{after_summary['binding_accuracy']:.4f} "
                f"updates={router.updates}",
                flush=True,
            )

        heldout_reports = evaluate_router(
            router,
            db,
            test_cases,
            update=False,
        )

        heldout_summary = aggregate(
            heldout_reports
        )

        print()
        print(
            "=== HELD-OUT ==="
        )
        print(
            "relation_accuracy:",
            heldout_summary[
                "relation_accuracy"
            ],
        )
        print(
            "target_accuracy:",
            heldout_summary[
                "target_accuracy"
            ],
        )
        print(
            "binding_accuracy:",
            heldout_summary[
                "binding_accuracy"
            ],
        )

        # ---------------------------------------------------------------
        # Paired same-pair / different-question test
        # ---------------------------------------------------------------

        paired_zero = evaluate_router(
            zero_router,
            db,
            paired,
            update=False,
        )

        paired_learned = evaluate_router(
            router,
            db,
            paired,
            update=False,
        )

        paired_zero_summary = aggregate(
            paired_zero
        )

        paired_learned_summary = aggregate(
            paired_learned
        )

        print()
        print(
            "=== PAIRED SAME-PAIR / DIFFERENT-QUESTION ==="
        )
        print(
            "zero_binding:",
            paired_zero_summary[
                "binding_accuracy"
            ],
        )
        print(
            "learned_binding:",
            paired_learned_summary[
                "binding_accuracy"
            ],
        )

        # ---------------------------------------------------------------
        # Confusion matrix
        # ---------------------------------------------------------------

        confusion = Counter()

        for report in heldout_reports:
            confusion[
                (
                    report["gold_relation"],
                    report["selected_relation"],
                )
            ] += 1

        confusion_payload = {
            f"{gold}->{predicted}": count
            for (
                gold,
                predicted,
            ), count in confusion.items()
        }

        # ---------------------------------------------------------------
        # Compact failures
        # ---------------------------------------------------------------

        failures = [
            {
                "case_id": report["case_id"],
                "task": report["task"],
                "source": report["source"],
                "target": report["target"],
                "gold_relation": report["gold_relation"],
                "selected_relation": report["selected_relation"],
                "selected_target": report["selected_target"],
                "correct_relation": report["correct_relation"],
                "correct_target": report["correct_target"],
                "correct_binding": report["correct_binding"],
                "margin": report["relation_margin"],
            }
            for report in heldout_reports
            if not report["correct_binding"]
        ]

        payload = {
            "experiment": (
                "V133 learned task-conditioned relation router"
            ),
            "cases": {
                "total": len(cases),
                "train": len(train_cases),
                "heldout": len(test_cases),
                "paired": len(paired),
            },
            "zero_shot": zero_summary,
            "train_history": train_history,
            "heldout": heldout_summary,
            "paired_zero_shot": paired_zero_summary,
            "paired_learned": paired_learned_summary,
            "router": router.snapshot(),
            "heldout_confusion": confusion_payload,
            "heldout_failures": failures,
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
            "=== V133 SUMMARY ==="
        )
        print(
            f"zero-shot binding:  "
            f"{zero_summary['binding_accuracy']:.4f}"
        )
        print(
            f"held-out binding:   "
            f"{heldout_summary['binding_accuracy']:.4f}"
        )
        print(
            f"paired zero-shot:   "
            f"{paired_zero_summary['binding_accuracy']:.4f}"
        )
        print(
            f"paired learned:     "
            f"{paired_learned_summary['binding_accuracy']:.4f}"
        )
        print(
            "router_updates:",
            router.updates,
        )
        print(
            "saved:",
            OUTPUT_PATH,
        )
        print(
            "elapsed_seconds:",
            f"{time.perf_counter() - started:.2f}",
        )
        print(
            "=== V133 COMPLETE ==="
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
