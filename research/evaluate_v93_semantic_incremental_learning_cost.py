from __future__ import annotations

"""
V90 — FAST SEMANTIC INCREMENTAL TRANSFER

Purpose
-------
Measure the graph cost of introducing held-out concepts WITHOUT recursively
rebuilding combinations for every concept.

Training builds the reusable semantic substrate once:
    feature cells
    unordered feature-pair combination cells
    recursive higher-order combination cells

Validation/test are then evaluated NON-DESTRUCTIVELY by lookup only.

For each held-out concept:
    known_features
    new_features
    known_feature_pairs
    novel_feature_pairs
    estimated_new_pair_cells

No mutation during holdout.
No per-concept combinatorial recursive expansion.
"""

import csv
import hashlib
import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, Optional


try:
    from simulator import Config, Network
except ImportError:
    from .simulator import Config, Network


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "semantics.csv"

TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15

UnitSet = FrozenSet[int]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SemanticRow:
    concept: str
    category: str
    feature: str


@dataclass
class ConceptRecord:
    concept: str
    categories: set[str] = field(default_factory=set)
    features: set[str] = field(default_factory=set)


def load_rows(path: Path) -> list[SemanticRow]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows: list[SemanticRow] = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "basic_level_concept",
            "superordinate_category",
            "feature_name",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise RuntimeError(
                "Missing columns: "
                + ", ".join(sorted(missing))
            )

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

            if concept and feature:
                rows.append(
                    SemanticRow(
                        concept=concept,
                        category=category,
                        feature=feature,
                    )
                )

    if not rows:
        raise RuntimeError("No semantic rows loaded.")

    return rows


def group_concepts(
    rows: list[SemanticRow],
) -> dict[str, ConceptRecord]:
    grouped: dict[str, ConceptRecord] = {}

    for row in rows:
        record = grouped.get(row.concept)

        if record is None:
            record = ConceptRecord(
                concept=row.concept
            )
            grouped[row.concept] = record

        record.features.add(row.feature)

        if row.category:
            record.categories.add(row.category)

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

    train_end = int(
        n * TRAIN_FRACTION
    )

    validation_end = (
        train_end
        + int(n * VALID_FRACTION)
    )

    return (
        ordered[:train_end],
        ordered[train_end:validation_end],
        ordered[validation_end:],
    )


# ---------------------------------------------------------------------------
# Semantic graph
# ---------------------------------------------------------------------------

class SemanticTransferGraphV90(Network):
    FEATURE = "v90_feature"
    COMBINATION = "v90_combination"

    COMBINATION_MEMBER = "V90_COMBINATION_MEMBER"

    def __init__(
        self,
        config: Optional[Config] = None,
    ) -> None:
        super().__init__(config)

        self.feature_cells: dict[str, int] = {}

        self.combination_by_key: dict[
            UnitSet,
            int,
        ] = {}

        self.combination_level: dict[
            int,
            int,
        ] = {}

    # ------------------------------------------------------------------
    # Feature cells
    # ------------------------------------------------------------------

    def feature_cell(
        self,
        feature: str,
    ) -> int:
        existing = self.feature_cells.get(feature)

        if existing is not None:
            return existing

        cell_id = self.create_cell(
            self.FEATURE,
            symbol=feature,
        )

        self.feature_cells[feature] = cell_id

        return cell_id

    # ------------------------------------------------------------------
    # Combination cells
    # ------------------------------------------------------------------

    def combination_cell(
        self,
        key: UnitSet,
        level: int,
    ) -> tuple[int, bool]:
        existing = self.combination_by_key.get(key)

        if existing is not None:
            return existing, False

        cell_id = self.create_cell(
            self.COMBINATION
        )

        self.combination_by_key[key] = cell_id
        self.combination_level[cell_id] = level

        for member in sorted(key):
            self.connect(
                member,
                cell_id,
                self.COMBINATION_MEMBER,
                1.0,
            )

        return cell_id, True

    # ------------------------------------------------------------------
    # Training substrate
    # ------------------------------------------------------------------

    def concept_feature_set(
        self,
        record: ConceptRecord,
        learn: bool = True,
    ) -> UnitSet:
        ids = []

        for feature in record.features:
            if learn:
                feature_id = self.feature_cell(feature)
            else:
                feature_id = self.feature_cells.get(
                    feature,
                    -1,
                )

            if feature_id >= 0:
                ids.append(feature_id)

        return frozenset(ids)

    def train_feature_layer(
        self,
        records: list[ConceptRecord],
    ) -> list[UnitSet]:
        streams = []

        for record in records:
            streams.append(
                self.concept_feature_set(
                    record,
                    learn=True,
                )
            )

        return streams

    def discover_recursive_combinations(
        self,
        streams: list[UnitSet],
        max_levels: int = 8,
        min_occurrences: int = 2,
    ) -> list[dict[str, int]]:
        """
        Build the training combination substrate once.

        Unlike V89, this happens only during TRAINING.
        """
        current = streams
        results = []

        for level in range(
            1,
            max_levels + 1,
        ):
            occurrences: dict[UnitSet, int] = {}

            # Each concept contributes unordered pairs of its current units.
            for units in current:
                for pair in itertools.combinations(
                    sorted(units),
                    2,
                ):
                    key = frozenset(pair)
                    occurrences[key] = (
                        occurrences.get(key, 0)
                        + 1
                    )

            recurring = {
                key
                for key, count in occurrences.items()
                if count >= min_occurrences
            }

            before = len(
                self.combination_by_key
            )

            for key in recurring:
                self.combination_cell(
                    key,
                    level,
                )

            after = len(
                self.combination_by_key
            )

            new_cells = after - before

            # Compress each concept using already learned recurring sets.
            next_streams = []

            for units in current:
                remaining = set(units)
                output = set()

                candidates = sorted(
                    (
                        key
                        for key in recurring
                        if key.issubset(remaining)
                    ),
                    key=lambda key: (
                        -len(key),
                        tuple(sorted(key)),
                    ),
                )

                for key in candidates:
                    if not key.issubset(remaining):
                        continue

                    combination_id = (
                        self.combination_by_key[key]
                    )

                    remaining.difference_update(key)
                    output.add(combination_id)

                output.update(remaining)

                next_streams.append(
                    frozenset(output)
                )

            results.append(
                {
                    "level": level,
                    "candidate_pairs": len(
                        occurrences
                    ),
                    "recurring_pairs": len(
                        recurring
                    ),
                    "new_cells": new_cells,
                    "remaining_units": sum(
                        len(units)
                        for units in next_streams
                    ),
                }
            )

            current = next_streams

            if new_cells == 0:
                break

        return results

    # ------------------------------------------------------------------
    # Frozen transfer measurement
    # ------------------------------------------------------------------

    def measure_concept_transfer(
        self,
        record: ConceptRecord,
    ) -> dict[str, float]:
        """
        Fast, non-mutating transfer-cost measurement.

        Feature cost:
            features absent from the training feature dictionary.

        Pair cost:
            unordered feature pairs absent from the learned pair dictionary.

        Higher-order cost:
            observed only via existing learned combination membership.
            No new higher-order nodes are created during evaluation.
        """
        known_feature_ids = []
        unknown_features = 0

        for feature in record.features:
            feature_id = self.feature_cells.get(feature)

            if feature_id is None:
                unknown_features += 1
            else:
                known_feature_ids.append(feature_id)

        pairs = list(
            itertools.combinations(
                sorted(known_feature_ids),
                2,
            )
        )

        known_pairs = 0
        novel_pairs = 0

        for left, right in pairs:
            key = frozenset(
                (left, right)
            )

            if key in self.combination_by_key:
                known_pairs += 1
            else:
                novel_pairs += 1

        total_pairs = len(pairs)

        return {
            "feature_count": float(
                len(record.features)
            ),
            "known_features": float(
                len(known_feature_ids)
            ),
            "new_features": float(
                unknown_features
            ),
            "feature_reuse_rate": (
                len(known_feature_ids)
                / max(
                    1,
                    len(record.features),
                )
            ),
            "feature_pairs": float(
                total_pairs
            ),
            "known_pairs": float(
                known_pairs
            ),
            "novel_pairs": float(
                novel_pairs
            ),
            "pair_reuse_rate": (
                known_pairs
                / max(
                    1,
                    total_pairs,
                )
            ),
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_split(
    graph: SemanticTransferGraphV90,
    records: list[ConceptRecord],
    label: str,
) -> list[dict[str, object]]:
    print(
        f"=== V90 {label} INCREMENTAL TRANSFER ==="
    )

    rows = []

    for index, record in enumerate(
        records,
        start=1,
    ):
        metrics = graph.measure_concept_transfer(
            record
        )

        rows.append(
            {
                "concept": record.concept,
                **metrics,
            }
        )

        print(
            f"{index:3d}/{len(records):3d} "
            f"{record.concept:24s} "
            f"features={int(metrics['feature_count']):2d} "
            f"new_features={int(metrics['new_features']):2d} "
            f"pair_reuse={metrics['pair_reuse_rate']:.3f}"
        )

    print()

    return rows


def summarize(
    rows: list[dict[str, object]],
    label: str,
) -> None:
    if not rows:
        return

    def avg(key: str) -> float:
        return sum(
            float(row[key])
            for row in rows
        ) / len(rows)

    print(
        f"=== V90 {label} SUMMARY ==="
    )
    print(
        "concepts                 :",
        len(rows),
    )
    print(
        "mean_new_features        :",
        avg("new_features"),
    )
    print(
        "mean_feature_reuse_rate  :",
        avg("feature_reuse_rate"),
    )
    print(
        "mean_known_pairs         :",
        avg("known_pairs"),
    )
    print(
        "mean_novel_pairs         :",
        avg("novel_pairs"),
    )
    print(
        "mean_pair_reuse_rate     :",
        avg("pair_reuse_rate"),
    )

    lowest = sorted(
        rows,
        key=lambda row: (
            float(row["new_features"]),
            float(row["novel_pairs"]),
            str(row["concept"]),
        ),
    )

    highest = sorted(
        rows,
        key=lambda row: (
            -float(row["new_features"]),
            -float(row["novel_pairs"]),
            str(row["concept"]),
        ),
    )

    print()
    print(
        "--- LOWEST TRANSFER COST ---"
    )

    for row in lowest[:10]:
        print(
            f"{str(row['concept']):24s} "
            f"new_features={int(row['new_features']):2d} "
            f"novel_pairs={int(row['novel_pairs']):3d} "
            f"pair_reuse={float(row['pair_reuse_rate']):.3f}"
        )

    print()
    print(
        "--- HIGHEST TRANSFER COST ---"
    )

    for row in highest[:10]:
        print(
            f"{str(row['concept']):24s} "
            f"new_features={int(row['new_features']):2d} "
            f"novel_pairs={int(row['novel_pairs']):3d} "
            f"pair_reuse={float(row['pair_reuse_rate']):.3f}"
        )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start = time.perf_counter()

    print(
        "=== V90 FAST SEMANTIC INCREMENTAL TRANSFER ==="
    )
    print(
        "Training builds the substrate once; held-out concepts are lookup-only."
    )
    print(
        "corpus:",
        DATA_PATH,
    )
    print()

    rows = load_rows(
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

    print(
        "semantic_rows:",
        len(rows),
    )
    print(
        "concepts:",
        len(concepts),
    )
    print(
        "train:",
        len(train),
    )
    print(
        "validation:",
        len(validation),
    )
    print(
        "test:",
        len(test),
    )
    print()

    graph = SemanticTransferGraphV90()

    # ---------------------------------------------------------------
    # TRAIN — only expensive operation.
    # ---------------------------------------------------------------

    train_streams = graph.train_feature_layer(
        train
    )

    print(
        "train_feature_cells:",
        len(graph.feature_cells),
    )

    level_rows = (
        graph.discover_recursive_combinations(
            train_streams
        )
    )

    print(
        "=== V90 TRAIN COMBINATION LEVELS ==="
    )

    for row in level_rows:
        print(
            f"level={row['level']:2d} "
            f"candidate_pairs={row['candidate_pairs']:6d} "
            f"recurring={row['recurring_pairs']:6d} "
            f"new_cells={row['new_cells']:6d} "
            f"remaining_units={row['remaining_units']:6d}"
        )

    print()

    train_counts = {
        "feature_cells": len(
            graph.feature_cells
        ),
        "combination_cells": len(
            graph.combination_by_key
        ),
        "network_cells": len(
            graph.cells
        ),
        "network_synapses": len(
            graph.synapses
        ),
    }

    print(
        "=== V90 TRAIN SUBSTRATE ==="
    )

    for key, value in train_counts.items():
        print(
            f"{key:24s}: {value}"
        )

    print()

    # ---------------------------------------------------------------
    # VALIDATION / TEST — pure lookup.
    # ---------------------------------------------------------------

    validation_rows = evaluate_split(
        graph,
        validation,
        "VALIDATION",
    )

    summarize(
        validation_rows,
        "VALIDATION",
    )

    test_rows = evaluate_split(
        graph,
        test,
        "TEST",
    )

    summarize(
        test_rows,
        "TEST",
    )

    # ---------------------------------------------------------------
    # Overall summary.
    # ---------------------------------------------------------------

    def mean(
        rows: list[dict[str, object]],
        key: str,
    ) -> float:
        if not rows:
            return 0.0

        return sum(
            float(row[key])
            for row in rows
        ) / len(rows)

    print(
        "=== V90 SUMMARY ==="
    )
    print(
        "validation_feature_reuse :",
        mean(
            validation_rows,
            "feature_reuse_rate",
        ),
    )
    print(
        "validation_pair_reuse   :",
        mean(
            validation_rows,
            "pair_reuse_rate",
        ),
    )
    print(
        "test_feature_reuse      :",
        mean(
            test_rows,
            "feature_reuse_rate",
        ),
    )
    print(
        "test_pair_reuse         :",
        mean(
            test_rows,
            "pair_reuse_rate",
        ),
    )
    print(
        "training_feature_cells  :",
        train_counts["feature_cells"],
    )
    print(
        "training_combination_cells:",
        train_counts[
            "combination_cells"
        ],
    )
    print(
        "elapsed_seconds         :",
        f"{time.perf_counter() - start:.2f}",
    )

    print(
        "=== V90 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
