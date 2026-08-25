from __future__ import annotations

"""
V86 — SEMANTIC FEATURE GRAPH / TRANSFER TEST
MULTI-CATEGORY FIX

The semantic corpus permits a concept to occur with multiple superordinate
categories. Example:
    pin -> tool
    pin -> kitchen

Therefore:
    concept -> {category_1, category_2, ...}

is represented as a many-to-many graph relation.

Experiment:
    semantic concept
        ↓
    shared feature cells
        ↓
    concept-feature edges

    semantic concept
        ↓
    one or more superordinate category cells

Split:
    concept-disjoint 70 / 15 / 15

Measures:
    feature reuse
    category reuse
    new feature cells
    new category cells
    feature sharing/compression
    graph growth

No semantic category is treated as a unique ground-truth label.
"""

import csv
import hashlib
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from simulator import Config, Network
except ImportError:
    from .simulator import Config, Network


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "semantics.csv"

TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SemanticRow:
    item_picture: str
    concept: str
    category: str
    feature: str
    sensorimotor: str
    feature_prod_freq: int
    feature_occ: int
    feature_total_prod_freq: int
    feature_distinct: float
    feature_cv: float


@dataclass
class ConceptRecord:
    concept: str
    categories: set[str] = field(default_factory=set)
    features: set[str] = field(default_factory=set)


REQUIRED_COLUMNS = {
    "item_picture",
    "basic_level_concept",
    "superordinate_category",
    "feature_name",
    "feature_sensorimotor_dominant",
    "feature_prod_freq",
    "feature_occ",
    "feature_total_prod_freq",
    "feature_distinct",
    "feature_cv",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_semantics(path: Path) -> list[SemanticRow]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns

        if missing:
            raise RuntimeError(
                "semantics.csv missing columns: "
                + ", ".join(sorted(missing))
            )

        rows = []

        for raw in reader:
            concept = raw[
                "basic_level_concept"
            ].strip().lower()

            category = raw[
                "superordinate_category"
            ].strip().lower()

            feature = raw[
                "feature_name"
            ].strip().lower()

            if not concept or not feature:
                continue

            try:
                row = SemanticRow(
                    item_picture=raw[
                        "item_picture"
                    ].strip().lower(),
                    concept=concept,
                    category=category,
                    feature=feature,
                    sensorimotor=raw[
                        "feature_sensorimotor_dominant"
                    ].strip().lower(),
                    feature_prod_freq=int(
                        raw["feature_prod_freq"]
                    ),
                    feature_occ=int(
                        raw["feature_occ"]
                    ),
                    feature_total_prod_freq=int(
                        raw["feature_total_prod_freq"]
                    ),
                    feature_distinct=float(
                        raw["feature_distinct"]
                    ),
                    feature_cv=float(
                        raw["feature_cv"]
                    ),
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Bad numeric row: {raw}"
                ) from exc

            rows.append(row)

    if not rows:
        raise RuntimeError(
            "No semantic rows loaded."
        )

    return rows


def group_concepts(
    rows: list[SemanticRow],
) -> dict[str, ConceptRecord]:
    """
    Aggregate all categories/features observed for each concept.

    A concept may legitimately have multiple superordinate categories.
    """
    grouped: dict[str, ConceptRecord] = {}

    for row in rows:
        record = grouped.get(row.concept)

        if record is None:
            record = ConceptRecord(
                concept=row.concept
            )
            grouped[row.concept] = record

        if row.category:
            record.categories.add(
                row.category
            )

        record.features.add(
            row.feature
        )

    return grouped


def stable_rank(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def split_concepts(
    concepts: dict[str, ConceptRecord],
):
    ordered = sorted(
        concepts.values(),
        key=lambda record: (
            stable_rank(record.concept),
            record.concept,
        ),
    )

    n = len(ordered)
    train_end = int(n * TRAIN_FRACTION)
    validation_end = (
        train_end
        + int(n * VALID_FRACTION)
    )

    train = ordered[:train_end]
    validation = ordered[
        train_end:validation_end
    ]
    test = ordered[validation_end:]

    names = [
        {
            record.concept
            for record in subset
        }
        for subset in (
            train,
            validation,
            test,
        )
    ]

    assert not names[0] & names[1]
    assert not names[0] & names[2]
    assert not names[1] & names[2]

    return train, validation, test


# ---------------------------------------------------------------------------
# Real semantic graph
# ---------------------------------------------------------------------------

class SemanticGraphV86(Network):
    FEATURE = "semantic_feature"
    CONCEPT = "semantic_concept"
    CATEGORY = "semantic_category"

    CONCEPT_FEATURE = "SEMANTIC_CONCEPT_FEATURE"
    CONCEPT_CATEGORY = "SEMANTIC_CONCEPT_CATEGORY"

    def __init__(
        self,
        config: Optional[Config] = None,
    ) -> None:
        super().__init__(config)

        self.feature_cells: dict[str, int] = {}
        self.concept_cells: dict[str, int] = {}
        self.category_cells: dict[str, int] = {}

        self.concept_features: dict[
            str,
            set[str],
        ] = defaultdict(set)

        self.concept_categories: dict[
            str,
            set[str],
        ] = defaultdict(set)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def feature_cell(
        self,
        feature: str,
    ) -> tuple[int, bool]:
        existing = self.feature_cells.get(
            feature
        )

        if existing is not None:
            return existing, False

        cell_id = self.create_cell(
            self.FEATURE,
            symbol=feature,
        )

        self.feature_cells[feature] = cell_id

        return cell_id, True

    def category_cell(
        self,
        category: str,
    ) -> tuple[int, bool]:
        existing = self.category_cells.get(
            category
        )

        if existing is not None:
            return existing, False

        cell_id = self.create_cell(
            self.CATEGORY,
            symbol=category,
        )

        self.category_cells[category] = cell_id

        return cell_id, True

    def concept_cell(
        self,
        concept: str,
    ) -> tuple[int, bool]:
        existing = self.concept_cells.get(
            concept
        )

        if existing is not None:
            return existing, False

        cell_id = self.create_cell(
            self.CONCEPT,
            symbol=concept,
        )

        self.concept_cells[concept] = cell_id

        return cell_id, True

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn_concept(
        self,
        record: ConceptRecord,
    ) -> dict[str, int]:
        concept_id, concept_created = (
            self.concept_cell(
                record.concept
            )
        )

        created_features = 0
        reused_features = 0
        created_categories = 0
        reused_categories = 0

        for feature in sorted(
            record.features
        ):
            feature_id, created = (
                self.feature_cell(
                    feature
                )
            )

            if created:
                created_features += 1
            else:
                reused_features += 1

            edge_key = (
                concept_id,
                feature_id,
            )

            if edge_key not in self.synapses:
                self.connect(
                    concept_id,
                    feature_id,
                    self.CONCEPT_FEATURE,
                    1.0,
                )

            self.concept_features[
                record.concept
            ].add(feature)

        for category in sorted(
            record.categories
        ):
            category_id, created = (
                self.category_cell(
                    category
                )
            )

            if created:
                created_categories += 1
            else:
                reused_categories += 1

            edge_key = (
                concept_id,
                category_id,
            )

            if edge_key not in self.synapses:
                self.connect(
                    concept_id,
                    category_id,
                    self.CONCEPT_CATEGORY,
                    1.0,
                )

            self.concept_categories[
                record.concept
            ].add(category)

        return {
            "created_concept": int(
                concept_created
            ),
            "created_features": (
                created_features
            ),
            "reused_features": (
                reused_features
            ),
            "created_categories": (
                created_categories
            ),
            "reused_categories": (
                reused_categories
            ),
        }

    def learn(
        self,
        records: list[ConceptRecord],
    ) -> dict[str, int]:
        result = Counter()

        for record in records:
            result.update(
                self.learn_concept(
                    record
                )
            )

        return dict(result)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        return {
            "feature_cells": len(
                self.feature_cells
            ),
            "concept_cells": len(
                self.concept_cells
            ),
            "category_cells": len(
                self.category_cells
            ),
            "network_cells": len(
                self.cells
            ),
            "network_synapses": len(
                self.synapses
            ),
            "concept_feature_edges": sum(
                syn.kind
                == self.CONCEPT_FEATURE
                for syn in self.synapses.values()
            ),
            "concept_category_edges": sum(
                syn.kind
                == self.CONCEPT_CATEGORY
                for syn in self.synapses.values()
            ),
        }

    def feature_usage_counts(
        self,
    ) -> Counter[str]:
        counts = Counter()

        for features in (
            self.concept_features.values()
        ):
            counts.update(features)

        return counts

    def category_usage_counts(
        self,
    ) -> Counter[str]:
        counts = Counter()

        for categories in (
            self.concept_categories.values()
        ):
            counts.update(categories)

        return counts


# ---------------------------------------------------------------------------
# Transfer metrics
# ---------------------------------------------------------------------------

def incremental_metrics(
    graph: SemanticGraphV86,
    records: list[ConceptRecord],
    label: str,
) -> dict[str, float]:
    known_features = set(
        graph.feature_cells
    )

    known_categories = set(
        graph.category_cells
    )

    total_feature_observations = 0
    reused_features = 0
    new_features = 0

    total_category_links = 0
    reused_categories = 0
    new_categories = 0

    for record in records:
        for feature in record.features:
            total_feature_observations += 1

            if feature in known_features:
                reused_features += 1
            else:
                new_features += 1
                known_features.add(feature)

        for category in record.categories:
            total_category_links += 1

            if category in known_categories:
                reused_categories += 1
            else:
                new_categories += 1
                known_categories.add(category)

    result = {
        "concepts": float(len(records)),
        "feature_observations": float(
            total_feature_observations
        ),
        "reused_features": float(
            reused_features
        ),
        "new_features": float(
            new_features
        ),
        "feature_reuse_rate": (
            reused_features
            / max(1, total_feature_observations)
        ),
        "category_links": float(
            total_category_links
        ),
        "reused_categories": float(
            reused_categories
        ),
        "new_categories": float(
            new_categories
        ),
        "category_reuse_rate": (
            reused_categories
            / max(1, total_category_links)
        ),
    }

    print(
        f"=== V86 {label} INCREMENTAL ==="
    )

    for key, value in result.items():
        print(
            f"{key:24s}: {value}"
        )

    print()

    return result


def report_feature_sharing(
    graph: SemanticGraphV86,
) -> None:
    usage = graph.feature_usage_counts()

    repeated = [
        count
        for count in usage.values()
        if count > 1
    ]

    print(
        "=== V86 FEATURE SHARING ==="
    )
    print(
        "unique_features   :",
        len(usage),
    )
    print(
        "reused_features   :",
        len(repeated),
    )
    print(
        "singleton_features:",
        sum(
            count == 1
            for count in usage.values()
        ),
    )
    print(
        "mean_reuse_shared :",
        sum(repeated)
        / max(1, len(repeated)),
    )
    print(
        "max_feature_reuse :",
        max(
            usage.values(),
            default=0,
        ),
    )
    print()


def report_compression(
    graph: SemanticGraphV86,
    records: list[ConceptRecord],
    label: str,
) -> None:
    naive_feature_nodes = sum(
        len(record.features)
        for record in records
    )

    shared_feature_nodes = len(
        graph.feature_cells
    )

    print(
        f"=== V86 {label} FEATURE COMPRESSION ==="
    )
    print(
        "naive_feature_nodes   :",
        naive_feature_nodes,
    )
    print(
        "shared_feature_nodes  :",
        shared_feature_nodes,
    )
    print(
        "feature_node_reduction:",
        1.0
        - (
            shared_feature_nodes
            / max(
                1,
                naive_feature_nodes,
            )
        ),
    )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start = time.perf_counter()

    print(
        "=== V86 SEMANTIC FEATURE GRAPH / MULTI-CATEGORY ==="
    )
    print(
        "corpus:",
        DATA_PATH,
    )
    print()

    rows = load_semantics(
        DATA_PATH
    )

    concepts = group_concepts(
        rows
    )

    train, validation, test = (
        split_concepts(
            concepts
        )
    )

    all_records = (
        train + validation + test
    )

    all_categories = {
        category
        for record in all_records
        for category in record.categories
    }

    all_features = {
        feature
        for record in all_records
        for feature in record.features
    }

    multi_category_concepts = sum(
        len(record.categories) > 1
        for record in all_records
    )

    print(
        "semantic_rows            :",
        len(rows),
    )
    print(
        "unique_concepts          :",
        len(concepts),
    )
    print(
        "train_concepts           :",
        len(train),
    )
    print(
        "validation_concepts      :",
        len(validation),
    )
    print(
        "test_concepts            :",
        len(test),
    )
    print(
        "unique_features          :",
        len(all_features),
    )
    print(
        "unique_categories        :",
        len(all_categories),
    )
    print(
        "multi_category_concepts  :",
        multi_category_concepts,
    )
    print()

    graph = SemanticGraphV86()

    # ---------------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------------

    train_result = graph.learn(
        train
    )

    print(
        "=== V86 TRAIN ==="
    )
    print(train_result)
    print(graph.counts())
    print()

    report_feature_sharing(
        graph
    )

    report_compression(
        graph,
        train,
        "TRAIN",
    )

    # ---------------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------------

    validation_metrics = (
        incremental_metrics(
            graph,
            validation,
            "VALIDATION",
        )
    )

    validation_before = graph.counts()

    validation_result = graph.learn(
        validation
    )

    validation_after = graph.counts()

    print(
        "=== V86 VALIDATION GRAPH DELTA ==="
    )
    print(validation_result)
    print(
        "new_feature_cells  :",
        validation_after["feature_cells"]
        - validation_before["feature_cells"],
    )
    print(
        "new_category_cells :",
        validation_after["category_cells"]
        - validation_before["category_cells"],
    )
    print()

    report_compression(
        graph,
        train + validation,
        "TRAIN+VALIDATION",
    )

    # ---------------------------------------------------------------
    # TEST
    # ---------------------------------------------------------------

    test_metrics = (
        incremental_metrics(
            graph,
            test,
            "TEST",
        )
    )

    test_before = graph.counts()

    test_result = graph.learn(
        test
    )

    test_after = graph.counts()

    print(
        "=== V86 TEST GRAPH DELTA ==="
    )
    print(test_result)
    print(
        "new_feature_cells  :",
        test_after["feature_cells"]
        - test_before["feature_cells"],
    )
    print(
        "new_category_cells :",
        test_after["category_cells"]
        - test_before["category_cells"],
    )
    print()

    report_compression(
        graph,
        all_records,
        "FULL CORPUS",
    )

    report_feature_sharing(
        graph
    )

    final = graph.counts()

    print(
        "=== V86 SUMMARY ==="
    )
    print(
        "validation_feature_reuse :",
        validation_metrics[
            "feature_reuse_rate"
        ],
    )
    print(
        "test_feature_reuse       :",
        test_metrics[
            "feature_reuse_rate"
        ],
    )
    print(
        "validation_category_reuse:",
        validation_metrics[
            "category_reuse_rate"
        ],
    )
    print(
        "test_category_reuse      :",
        test_metrics[
            "category_reuse_rate"
        ],
    )
    print(
        "final_feature_cells      :",
        final["feature_cells"],
    )
    print(
        "final_concept_cells      :",
        final["concept_cells"],
    )
    print(
        "final_category_cells     :",
        final["category_cells"],
    )
    print(
        "final_concept_feature_edges:",
        final["concept_feature_edges"],
    )
    print(
        "final_concept_category_edges:",
        final["concept_category_edges"],
    )
    print(
        "elapsed_seconds          :",
        f"{time.perf_counter() - start:.2f}",
    )
    print(
        "=== V86 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
