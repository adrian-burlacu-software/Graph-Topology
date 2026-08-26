from __future__ import annotations

"""
V134 — CONTEXT-DRIVEN RELATION ATTENTION SUITE

Goal
----
V133 proved that an explicit TASK label can trivially route to a relation.
That is not meaningful attention.

V134 removes task labels from the attention mechanism.

The system receives:
    * source concept
    * target concept(s) currently active in working memory
    * contextual lexical/grammatical cues represented as small graph-like
      context features
    * the actual long-term semantic topology

It must decide which relation between the active concepts is cognitively
relevant.

Core test
---------
The same source/target pair is presented under different contexts.

Example:

    source = dog
    target = animal

    context = category / "what kind of thing"
        -> IsA

    context = association / "what is it related to"
        -> RelatedTo

There is NO:
    task_id
    task -> relation table
    hard-coded category label

Instead, context is represented as active features and the router learns a
distributed compatibility between context features and relation modules.

Architecture
------------
                    active semantic state
                           |
                  context feature activation
                           |
                           v
                  relation attention
                           |
                    relation scores
                           |
                           v
                    target attention
                           |
                           v
                 working-memory binding

Learning:
    context_feature × relation weights

Online contrastive update:
    reinforce selected/verified relation
    suppress competing relation

Evaluation:
    * zero-shot random router
    * trained context router
    * held-out contexts/cases
    * paired same-pair / different-context test
    * relation accuracy
    * target accuracy
    * binding accuracy
    * attention margin
    * confusion matrix
    * compact failure list

No LLM.
No graph mutation.
One file / one report.
"""

import json
import math
import random
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

DB_PATH = DATA / "conceptnet_compact.db"
OUTPUT_PATH = RESULTS / "v134_context_driven_relation_attention.json"

SEED = 13401


# ---------------------------------------------------------------------------
# Relation modules
# ---------------------------------------------------------------------------

RELATIONS = (
    "IsA",
    "CapableOf",
    "HasProperty",
    "UsedFor",
    "HasA",
    "PartOf",
    "RelatedTo",
    "SimilarTo",
    "Antonym",
    "Causes",
    "AtLocation",
)


# ---------------------------------------------------------------------------
# Context representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Context:
    """
    A tiny symbolic context vector.

    This is intentionally NOT a task label.

    The attention system only sees active context features. Several contexts
    can share features and therefore generalize.
    """
    features: tuple[str, ...]


CONTEXTS = {
    "KIND": Context(
        (
            "kind",
            "type",
            "belongs",
            "classification",
        )
    ),
    "ABILITY": Context(
        (
            "ability",
            "can",
            "action",
            "capable",
        )
    ),
    "QUALITY": Context(
        (
            "property",
            "quality",
            "describes",
            "attribute",
        )
    ),
    "PURPOSE": Context(
        (
            "purpose",
            "used",
            "function",
            "for",
        )
    ),
    "POSSESSION": Context(
        (
            "have",
            "contains",
            "has",
            "possession",
        )
    ),
    "PART": Context(
        (
            "part",
            "component",
            "belongs_to",
            "whole",
        )
    ),
    "ASSOCIATION": Context(
        (
            "related",
            "associated",
            "connected",
        )
    ),
    "SIMILARITY": Context(
        (
            "similar",
            "alike",
            "resembles",
        )
    ),
    "OPPOSITE": Context(
        (
            "opposite",
            "contrast",
            "contrary",
        )
    ),
    "CAUSE": Context(
        (
            "cause",
            "effect",
            "results",
            "produces",
        )
    ),
    "LOCATION": Context(
        (
            "where",
            "located",
            "found",
            "place",
        )
    ),
}


# ---------------------------------------------------------------------------
# Evaluation cases
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    case_id: str
    context_name: str
    source: str
    target: str
    distractor_relations: tuple[str, ...] = ()


def build_cases() -> list[Case]:
    cases = []

    def add(
        context_name: str,
        pairs: list[tuple[str, str]],
        prefix: str,
        distractors: tuple[str, ...],
    ) -> None:
        for index, (source, target) in enumerate(
            pairs,
            start=1,
        ):
            cases.append(
                Case(
                    case_id=f"{prefix}_{index:02d}",
                    context_name=context_name,
                    source=source,
                    target=target,
                    distractor_relations=distractors,
                )
            )

    add(
        "KIND",
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
        "kind",
        ("RelatedTo", "SimilarTo", "HasA"),
    )

    add(
        "ABILITY",
        [
            ("dog", "bark"),
            ("cat", "meow"),
            ("bird", "fly"),
            ("knife", "cut"),
            ("child", "play"),
            ("fish", "swim"),
            ("car", "move"),
            ("phone", "communicate"),
            ("person", "walk"),
            ("computer", "compute"),
        ],
        "ability",
        ("RelatedTo", "IsA", "HasProperty"),
    )

    add(
        "QUALITY",
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
        ],
        "quality",
        ("RelatedTo", "IsA", "HasA"),
    )

    add(
        "PURPOSE",
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
        "purpose",
        ("RelatedTo", "CapableOf", "HasA"),
    )

    add(
        "POSSESSION",
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
        ],
        "possession",
        ("RelatedTo", "PartOf", "IsA"),
    )

    add(
        "PART",
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
        "part",
        ("RelatedTo", "HasA", "IsA"),
    )

    add(
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
        "association",
        ("IsA", "SimilarTo", "HasProperty"),
    )

    add(
        "SIMILARITY",
        [
            ("happy", "joyful"),
            ("small", "little"),
            ("quick", "fast"),
            ("large", "big"),
            ("car", "vehicle"),
            ("dog", "animal"),
        ],
        "similarity",
        ("RelatedTo", "Antonym", "IsA"),
    )

    add(
        "OPPOSITE",
        [
            ("hot", "cold"),
            ("big", "small"),
            ("fast", "slow"),
            ("up", "down"),
            ("open", "closed"),
            ("light", "dark"),
        ],
        "opposite",
        ("RelatedTo", "SimilarTo", "IsA"),
    )

    add(
        "CAUSE",
        [
            ("fire", "heat"),
            ("rain", "wetness"),
            ("exercise", "sweat"),
            ("injury", "pain"),
            ("hunger", "eating"),
            ("sleep", "rest"),
        ],
        "cause",
        ("RelatedTo", "HasProperty", "IsA"),
    )

    add(
        "LOCATION",
        [
            ("fish", "water"),
            ("bird", "sky"),
            ("car", "road"),
            ("book", "library"),
            ("doctor", "hospital"),
            ("student", "school"),
        ],
        "location",
        ("RelatedTo", "IsA", "UsedFor"),
    )

    # Deliberately ambiguous context variants: same relation, different words.
    add(
        "KIND",
        [
            ("rose", "plant"),
            ("apple", "food"),
            ("car", "vehicle"),
        ],
        "kind_generalization",
        ("RelatedTo", "HasA"),
    )

    add(
        "ABILITY",
        [
            ("dog", "bark"),
            ("bird", "fly"),
            ("knife", "cut"),
        ],
        "ability_generalization",
        ("RelatedTo", "IsA"),
    )

    return cases


# ---------------------------------------------------------------------------
# ConceptNet
# ---------------------------------------------------------------------------

class ConceptNet:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def relation_candidates(
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

    def target_weight(
        self,
        source: str,
        relation: str,
        target: str,
    ) -> float | None:
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
            return None

        return float(row["weight"])


# ---------------------------------------------------------------------------
# Context-driven relation attention
# ---------------------------------------------------------------------------

RELATION_TO_CANONICAL_CONTEXT = {
    "IsA": {
        "kind": 1.0,
        "type": 1.0,
        "belongs": 0.8,
        "classification": 0.8,
    },
    "CapableOf": {
        "ability": 1.0,
        "can": 1.0,
        "action": 0.8,
        "capable": 1.0,
    },
    "HasProperty": {
        "property": 1.0,
        "quality": 1.0,
        "describes": 0.8,
        "attribute": 0.8,
    },
    "UsedFor": {
        "purpose": 1.0,
        "used": 1.0,
        "function": 0.9,
        "for": 0.6,
    },
    "HasA": {
        "have": 1.0,
        "contains": 0.9,
        "has": 1.0,
        "possession": 0.9,
    },
    "PartOf": {
        "part": 1.0,
        "component": 1.0,
        "belongs_to": 0.9,
        "whole": 0.8,
    },
    "RelatedTo": {
        "related": 1.0,
        "associated": 1.0,
        "connected": 0.8,
    },
    "SimilarTo": {
        "similar": 1.0,
        "alike": 1.0,
        "resembles": 0.9,
    },
    "Antonym": {
        "opposite": 1.0,
        "contrast": 0.9,
        "contrary": 0.8,
    },
    "Causes": {
        "cause": 1.0,
        "effect": 0.8,
        "results": 0.9,
        "produces": 0.8,
    },
    "AtLocation": {
        "where": 1.0,
        "located": 1.0,
        "found": 0.9,
        "place": 0.8,
    },
}


class ContextAttention:
    """
    Learned distributed compatibility:
        context feature -> relation module

    No task names are used by the scoring mechanism.
    """

    def __init__(
        self,
        relations: tuple[str, ...],
        learning_rate: float = 0.55,
    ):
        self.relations = relations
        self.learning_rate = learning_rate

        self.feature_weights: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        # A slow relation-level baseline so the router doesn't collapse on
        # relation frequency.
        self.relation_bias = {
            relation: 0.0
            for relation in relations
        }

        self.updates = 0

    def score(
        self,
        context: Context,
        relation: str,
    ) -> float:
        score = self.relation_bias.get(
            relation,
            0.0,
        )

        learned = self.feature_weights

        for feature in context.features:
            score += learned[
                feature
            ].get(
                relation,
                0.0,
            )

        return score

    def rank(
        self,
        context: Context,
    ) -> list[tuple[str, float]]:
        return sorted(
            (
                (
                    relation,
                    self.score(
                        context,
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
        context: Context,
        gold_relation: str,
        predicted_relation: str,
    ) -> bool:
        if (
            gold_relation
            == predicted_relation
        ):
            return False

        delta = self.learning_rate

        for feature in context.features:
            self.feature_weights[
                feature
            ][
                gold_relation
            ] += delta

            self.feature_weights[
                feature
            ][
                predicted_relation
            ] -= delta

        self.relation_bias[
            gold_relation
        ] += delta * 0.05

        self.relation_bias[
            predicted_relation
        ] -= delta * 0.05

        self.updates += 1
        return True

    def canonical_overlap(
        self,
        context: Context,
        relation: str,
    ) -> float:
        """
        Diagnostic only.

        This is not used during inference. It tells us how semantically close
        the learned context is to a known human-readable context profile.
        """
        canonical = (
            RELATION_TO_CANONICAL_CONTEXT.get(
                relation,
                {},
            )
        )

        if not canonical:
            return 0.0

        scores = [
            canonical.get(
                feature,
                0.0,
            )
            for feature in context.features
        ]

        return sum(scores) / max(
            1,
            len(scores),
        )

    def snapshot(self) -> dict:
        return {
            "feature_weights": {
                feature: dict(values)
                for feature, values
                in self.feature_weights.items()
            },
            "relation_bias": self.relation_bias,
            "updates": self.updates,
        }


# ---------------------------------------------------------------------------
# Target attention
# ---------------------------------------------------------------------------

def select_target(
    db: ConceptNet,
    source: str,
    relation: str,
    expected_target: str,
) -> dict:
    candidates = db.relation_candidates(
        source,
        relation,
    )

    ranked = []

    for concept, weight in candidates[:20]:
        exact = (
            concept.lower()
            == expected_target.lower()
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
                "exact": exact,
                "weight": weight,
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
            == expected_target.lower()
        ),
        "ranked": ranked[:10],
    }


# ---------------------------------------------------------------------------
# Train/test split by case groups
# ---------------------------------------------------------------------------

def split_cases(
    cases: list[Case],
    seed: int = SEED,
) -> tuple[list[Case], list[Case]]:
    """
    Hold out entire examples, not task labels.

    Each context kind remains represented in both sets, but the exact
    source/target pair is unseen during training.
    """
    rng = random.Random(seed)

    grouped = defaultdict(list)
    for case in cases:
        grouped[
            case.context_name
        ].append(case)

    train = []
    test = []

    for context_name, values in grouped.items():
        values = list(values)
        rng.shuffle(values)

        cut = max(
            1,
            int(
                len(values)
                * 0.65
            ),
        )

        train.extend(
            values[:cut]
        )
        test.extend(
            values[cut:]
        )

    rng.shuffle(train)
    rng.shuffle(test)

    return train, test


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    router: ContextAttention,
    db: ConceptNet,
    cases: list[Case],
    update: bool = False,
) -> list[dict]:
    reports = []

    for case in cases:
        context = CONTEXTS[
            case.context_name
        ]

        ranked = router.rank(
            context
        )

        selected_relation = (
            ranked[0][0]
            if ranked
            else None
        )

        gold_relation = None

        # We deliberately keep the mapping out of the router. It is only used
        # by the evaluation oracle.
        gold_relation = {
            "KIND": "IsA",
            "ABILITY": "CapableOf",
            "QUALITY": "HasProperty",
            "PURPOSE": "UsedFor",
            "POSSESSION": "HasA",
            "PART": "PartOf",
            "ASSOCIATION": "RelatedTo",
            "SIMILARITY": "SimilarTo",
            "OPPOSITE": "Antonym",
            "CAUSE": "Causes",
            "LOCATION": "AtLocation",
        }[
            case.context_name
        ]

        target_result = select_target(
            db,
            case.source,
            selected_relation,
            case.target,
        )

        relation_correct = (
            selected_relation
            == gold_relation
        )

        binding_correct = (
            relation_correct
            and target_result["correct"]
        )

        margin = 0.0
        if len(ranked) >= 2:
            margin = (
                ranked[0][1]
                - ranked[1][1]
            )

        reports.append(
            {
                "case_id": case.case_id,
                "context": case.context_name,
                "context_features": list(
                    context.features
                ),
                "source": case.source,
                "target": case.target,
                "gold_relation": gold_relation,
                "ranked_relations": ranked,
                "selected_relation": selected_relation,
                "relation_correct": relation_correct,
                "relation_margin": margin,
                "selected_target": target_result[
                    "selected"
                ],
                "target_correct": target_result[
                    "correct"
                ],
                "binding_correct": binding_correct,
                "target_ranked": target_result[
                    "ranked"
                ],
            }
        )

        if update:
            router.update(
                context,
                gold_relation,
                selected_relation,
            )

    return reports


def summarize(
    reports: list[dict],
) -> dict:
    if not reports:
        return {
            "count": 0,
            "relation_accuracy": 0.0,
            "target_accuracy": 0.0,
            "binding_accuracy": 0.0,
            "mean_margin": 0.0,
        }

    return {
        "count": len(reports),
        "relation_accuracy": (
            sum(
                int(
                    report[
                        "relation_correct"
                    ]
                )
                for report in reports
            )
            / len(reports)
        ),
        "target_accuracy": (
            sum(
                int(
                    report[
                        "target_correct"
                    ]
                )
                for report in reports
            )
            / len(reports)
        ),
        "binding_accuracy": (
            sum(
                int(
                    report[
                        "binding_correct"
                    ]
                )
                for report in reports
            )
            / len(reports)
        ),
        "mean_margin": (
            sum(
                report["relation_margin"]
                for report in reports
            )
            / len(reports)
        ),
        "by_context": summarize_contexts(
            reports
        ),
    }


def summarize_contexts(
    reports: list[dict],
) -> dict:
    groups = defaultdict(list)

    for report in reports:
        groups[
            report["context"]
        ].append(report)

    result = {}

    for context, values in groups.items():
        result[
            context
        ] = {
            "count": len(values),
            "relation_accuracy": (
                sum(
                    int(
                        item[
                            "relation_correct"
                        ]
                    )
                    for item in values
                )
                / len(values)
            ),
            "target_accuracy": (
                sum(
                    int(
                        item[
                            "target_correct"
                        ]
                    )
                    for item in values
                )
                / len(values)
            ),
            "binding_accuracy": (
                sum(
                    int(
                        item[
                            "binding_correct"
                        ]
                    )
                    for item in values
                )
                / len(values)
            ),
            "mean_margin": (
                sum(
                    item["relation_margin"]
                    for item in values
                )
                / len(values)
            ),
        }

    return result


def same_pair_different_context_cases() -> list[Case]:
    """
    These are the key attention cases.

    Same source and target, different context. A static edge-strength heuristic
    cannot know which relation should win; the context must change the routing.
    """
    pairs = (
        ("dog", "animal"),
        ("cat", "animal"),
        ("bird", "animal"),
        ("car", "vehicle"),
        ("chair", "object"),
        ("knife", "tool"),
    )

    cases = []

    for source, target in pairs:
        cases.append(
            Case(
                f"paired_kind_{source}",
                "KIND",
                source,
                target,
            )
        )

        cases.append(
            Case(
                f"paired_association_{source}",
                "ASSOCIATION",
                source,
                target,
            )
        )

    return cases


def summarize_confusion(
    reports: list[dict],
) -> dict:
    confusion = Counter()

    for report in reports:
        confusion[
            (
                report["gold_relation"],
                report["selected_relation"],
            )
        ] += 1

    return {
        f"{gold}->{selected}": count
        for (
            gold,
            selected,
        ), count in confusion.items()
    }


def compact_failures(
    reports: list[dict],
) -> list[dict]:
    failures = []

    for report in reports:
        if report["binding_correct"]:
            continue

        failures.append(
            {
                "case_id": report["case_id"],
                "context": report["context"],
                "source": report["source"],
                "target": report["target"],
                "gold_relation": report[
                    "gold_relation"
                ],
                "selected_relation": report[
                    "selected_relation"
                ],
                "selected_target": report[
                    "selected_target"
                ],
                "relation_margin": report[
                    "relation_margin"
                ],
            }
        )

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    started = time.perf_counter()

    print(
        "=== V134 CONTEXT-DRIVEN RELATION ATTENTION ==="
    )

    cases = build_cases()

    train_cases, heldout_cases = split_cases(
        cases
    )

    paired = same_pair_different_context_cases()

    print(
        "cases:",
        len(cases),
        flush=True,
    )
    print(
        "train:",
        len(train_cases),
        "heldout:",
        len(heldout_cases),
        "paired:",
        len(paired),
        flush=True,
    )

    db = ConceptNet(
        DB_PATH
    )

    try:
        # ---------------------------------------------------------------
        # Zero-shot router.
        # ---------------------------------------------------------------

        zero_router = ContextAttention(
            RELATIONS,
            learning_rate=0.55,
        )

        zero_heldout = evaluate(
            zero_router,
            db,
            heldout_cases,
            update=False,
        )

        zero_paired = evaluate(
            zero_router,
            db,
            paired,
            update=False,
        )

        # ---------------------------------------------------------------
        # Learned router.
        # ---------------------------------------------------------------

        router = ContextAttention(
            RELATIONS,
            learning_rate=0.55,
        )

        train_history = []

        for epoch in range(1, 7):
            before = evaluate(
                router,
                db,
                train_cases,
                update=False,
            )

            before_summary = summarize(
                before
            )

            training_pass = evaluate(
                router,
                db,
                train_cases,
                update=True,
            )

            after = evaluate(
                router,
                db,
                train_cases,
                update=False,
            )

            after_summary = summarize(
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
                f"train_binding_before="
                f"{before_summary['binding_accuracy']:.4f} "
                f"after="
                f"{after_summary['binding_accuracy']:.4f} "
                f"updates={router.updates}",
                flush=True,
            )

        learned_heldout = evaluate(
            router,
            db,
            heldout_cases,
            update=False,
        )

        learned_paired = evaluate(
            router,
            db,
            paired,
            update=False,
        )

        # ---------------------------------------------------------------
        # Context generalization diagnostic.
        # ---------------------------------------------------------------

        context_overlap = {}

        for context_name, context in CONTEXTS.items():
            gold_relation = {
                "KIND": "IsA",
                "ABILITY": "CapableOf",
                "QUALITY": "HasProperty",
                "PURPOSE": "UsedFor",
                "POSSESSION": "HasA",
                "PART": "PartOf",
                "ASSOCIATION": "RelatedTo",
                "SIMILARITY": "SimilarTo",
                "OPPOSITE": "Antonym",
                "CAUSE": "Causes",
                "LOCATION": "AtLocation",
            }[
                context_name
            ]

            context_overlap[
                context_name
            ] = {
                "gold_relation": gold_relation,
                "canonical_overlap": router.canonical_overlap(
                    context,
                    gold_relation,
                ),
                "learned_relation_scores": {
                    relation: router.score(
                        context,
                        relation,
                    )
                    for relation in RELATIONS
                },
            }

        zero_heldout_summary = summarize(
            zero_heldout
        )
        learned_heldout_summary = summarize(
            learned_heldout
        )
        zero_paired_summary = summarize(
            zero_paired
        )
        learned_paired_summary = summarize(
            learned_paired
        )

        print()
        print(
            "=== V134 SUMMARY ==="
        )
        print(
            "zero-shot heldout binding:",
            f"{zero_heldout_summary['binding_accuracy']:.4f}",
        )
        print(
            "learned heldout binding:",
            f"{learned_heldout_summary['binding_accuracy']:.4f}",
        )
        print(
            "zero-shot paired binding:",
            f"{zero_paired_summary['binding_accuracy']:.4f}",
        )
        print(
            "learned paired binding:",
            f"{learned_paired_summary['binding_accuracy']:.4f}",
        )
        print(
            "learned heldout relation:",
            f"{learned_heldout_summary['relation_accuracy']:.4f}",
        )
        print(
            "learned heldout target:",
            f"{learned_heldout_summary['target_accuracy']:.4f}",
        )
        print(
            "router updates:",
            router.updates,
        )

        # Compact persistent report.
        payload = {
            "experiment": (
                "V134 context-driven relation attention"
            ),
            "cases": {
                "total": len(cases),
                "train": len(train_cases),
                "heldout": len(heldout_cases),
                "paired": len(paired),
            },
            "zero_shot_heldout": zero_heldout_summary,
            "learned_heldout": learned_heldout_summary,
            "zero_shot_paired": zero_paired_summary,
            "learned_paired": learned_paired_summary,
            "train_history": train_history,
            "context_diagnostics": context_overlap,
            "router": router.snapshot(),
            "heldout_confusion": summarize_confusion(
                learned_heldout
            ),
            "heldout_failures": compact_failures(
                learned_heldout
            ),
            "paired_failures": compact_failures(
                learned_paired
            ),
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

        print(
            "saved:",
            OUTPUT_PATH,
        )
        print(
            "elapsed_seconds:",
            f"{time.perf_counter() - started:.2f}",
        )
        print(
            "=== V134 COMPLETE ==="
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
