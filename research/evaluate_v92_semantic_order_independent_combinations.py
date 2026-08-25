from __future__ import annotations

"""
V88 — ORDER-INDEPENDENT SEMANTIC FEATURE COMBINATIONS

V87 exposed a representation error:
    sorted(feature_names) -> sequence

That ordering is arbitrary for semantic feature sets.

V88 uses SET semantics throughout.

Representation:
    concept = frozenset(feature IDs)

Combination discovery:
    repeated unordered feature pairs
        -> reusable combination unit

Recursive discovery:
    repeated unordered sets of lower-level units
        -> higher-order reusable combination unit

No feature ordering is assumed.

The experiment asks:

    Do recurring semantic feature combinations remain reusable when their
    representation is explicitly invariant to feature ordering?

Primary measurements:
    * unique feature cells
    * recurring unordered feature pairs
    * recurring higher-order combinations
    * combination reuse on concept-disjoint validation/test concepts
    * recursive compression of unordered concept feature sets
    * fixed-point convergence

No semantic label is used as a target.
"""

import csv
import hashlib
import itertools
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, Optional, Tuple

try:
    from simulator import Config, Network
except ImportError:
    from .simulator import Config, Network


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "semantics.csv"

TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15

MIN_OCCURRENCES = 2
MAX_LEVELS = 8


# ---------------------------------------------------------------------------
# Semantic corpus
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

    rows = []

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

        missing = required - set(
            reader.fieldnames or []
        )

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
        raise RuntimeError(
            "No semantic rows loaded."
        )

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
            record.categories.add(
                row.category
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
    validation = ordered[train_end:validation_end]
    test = ordered[validation_end:]

    return train, validation, test


# ---------------------------------------------------------------------------
# Order-independent graph
# ---------------------------------------------------------------------------

Unit = int
UnitSet = FrozenSet[Unit]


class SemanticOrderIndependentGraphV88(Network):
    FEATURE = "v88_feature"
    COMBINATION = "v88_combination"

    CONCEPT_FEATURE = "V88_CONCEPT_FEATURE"
    COMBINATION_MEMBER = "V88_COMBINATION_MEMBER"

    def __init__(
        self,
        config: Optional[Config] = None,
    ) -> None:
        super().__init__(config)

        self.feature_cells: dict[str, int] = {}

        # Canonical unordered combination key.
        # frozenset makes representation explicitly order-independent.
        self.combination_by_key: dict[
            UnitSet,
            int,
        ] = {}

        self.combination_level: dict[
            int,
            int,
        ] = {}

        self.combination_activations: Counter[
            int
        ] = Counter()

        self.concept_feature_ids: dict[
            str,
            UnitSet,
        ] = {}

    # ------------------------------------------------------------------
    # Feature nodes
    # ------------------------------------------------------------------

    def feature_cell(
        self,
        feature: str,
    ) -> int:
        existing = self.feature_cells.get(
            feature
        )

        if existing is not None:
            return existing

        cell_id = self.create_cell(
            self.FEATURE,
            symbol=feature,
        )

        self.feature_cells[feature] = cell_id

        return cell_id

    def concept_units(
        self,
        record: ConceptRecord,
    ) -> UnitSet:
        units = frozenset(
            self.feature_cell(
                feature
            )
            for feature in record.features
        )

        self.concept_feature_ids[
            record.concept
        ] = units

        return units

    # ------------------------------------------------------------------
    # Combination nodes
    # ------------------------------------------------------------------

    def combination_cell(
        self,
        key: UnitSet,
        level: int,
    ) -> tuple[int, bool]:
        existing = self.combination_by_key.get(
            key
        )

        if existing is not None:
            self.combination_activations[
                existing
            ] += 1

            return existing, False

        cell_id = self.create_cell(
            self.COMBINATION
        )

        self.combination_by_key[
            key
        ] = cell_id

        self.combination_level[
            cell_id
        ] = level

        self.combination_activations[
            cell_id
        ] = 1

        for member in sorted(key):
            self.connect(
                member,
                cell_id,
                self.COMBINATION_MEMBER,
                1.0,
            )

        return cell_id, True

    # ------------------------------------------------------------------
    # Order-independent discovery
    # ------------------------------------------------------------------

    def discover_pairs(
        self,
        concept_unit_sets: list[UnitSet],
    ) -> tuple[list[UnitSet], dict[str, int]]:
        """
        Count unordered feature pairs across concepts.

        A pair is represented canonically as:
            frozenset({A, B})

        Therefore:
            {A, B} == {B, A}
        """
        occurrences: Counter[UnitSet] = Counter()

        for units in concept_unit_sets:
            for pair in itertools.combinations(
                sorted(units),
                2,
            ):
                occurrences[
                    frozenset(pair)
                ] += 1

        recurring = {
            pair: count
            for pair, count in occurrences.items()
            if count >= MIN_OCCURRENCES
        }

        new_cells = 0
        reused_cells = 0

        for pair in sorted(
            recurring,
            key=lambda item: tuple(sorted(item)),
        ):
            _cell_id, created = (
                self.combination_cell(
                    pair,
                    level=1,
                )
            )

            if created:
                new_cells += 1
            else:
                reused_cells += 1

        return list(recurring.keys()), {
            "candidate_pairs": len(occurrences),
            "recurring_pairs": len(recurring),
            "new_cells": new_cells,
            "reused_cells": reused_cells,
        }

    def compress_one(
        self,
        units: UnitSet,
        available_combinations: set[UnitSet],
    ) -> tuple[UnitSet, int]:
        """
        Greedily replace recurring unordered subsets with a combination unit.

        The underlying semantic content remains a set; replacement is only a
        graph encoding operation.
        """
        remaining = set(units)
        output: set[int] = set()
        replacements = 0

        # Prefer larger combinations when several overlap.
        candidates = sorted(
            (
                combo
                for combo in available_combinations
                if combo.issubset(remaining)
            ),
            key=lambda combo: (
                -len(combo),
                tuple(sorted(combo)),
            ),
        )

        for combo in candidates:
            if not combo.issubset(remaining):
                continue

            assembly_id = (
                self.combination_by_key[combo]
            )

            remaining.difference_update(
                combo
            )
            output.add(
                assembly_id
            )
            replacements += 1

        output.update(remaining)

        return frozenset(output), replacements

    def discover_next_level(
        self,
        concept_unit_sets: list[UnitSet],
        level: int,
    ) -> tuple[list[UnitSet], dict[str, int]]:
        """
        Same operator at every level.

        Each concept is an unordered set of units.
        Recurring unordered pairs among those units become the next-level
        combination candidates.
        """
        occurrences: Counter[UnitSet] = Counter()

        for units in concept_unit_sets:
            for pair in itertools.combinations(
                sorted(units),
                2,
            ):
                occurrences[
                    frozenset(pair)
                ] += 1

        recurring = {
            pair: count
            for pair, count in occurrences.items()
            if count >= MIN_OCCURRENCES
        }

        new_cells = 0
        reused_cells = 0

        for pair in sorted(
            recurring,
            key=lambda item: tuple(
                sorted(item)
            ),
        ):
            _cell_id, created = (
                self.combination_cell(
                    pair,
                    level=level,
                )
            )

            if created:
                new_cells += 1
            else:
                reused_cells += 1

        compressed = []
        replacements = 0

        available = set(recurring)

        for units in concept_unit_sets:
            result, count = self.compress_one(
                units,
                available,
            )

            compressed.append(result)
            replacements += count

        return compressed, {
            "candidate_pairs": len(occurrences),
            "recurring_pairs": len(recurring),
            "new_cells": new_cells,
            "reused_cells": reused_cells,
            "replacements": replacements,
        }

    # ------------------------------------------------------------------
    # Graph metrics
    # ------------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        return {
            "feature_cells": sum(
                cell.kind == self.FEATURE
                for cell in self.cells.values()
            ),
            "combination_cells": sum(
                cell.kind == self.COMBINATION
                for cell in self.cells.values()
            ),
            "network_cells": len(
                self.cells
            ),
            "network_synapses": len(
                self.synapses
            ),
            "combination_member_synapses": sum(
                syn.kind
                == self.COMBINATION_MEMBER
                for syn in self.synapses.values()
            ),
        }


# ---------------------------------------------------------------------------
# Transfer metrics
# ---------------------------------------------------------------------------

def feature_transfer(
    graph: SemanticOrderIndependentGraphV88,
    records: list[ConceptRecord],
    label: str,
) -> dict[str, float]:
    known = set(
        graph.feature_cells
    )

    total = 0
    reused = 0
    new = 0

    for record in records:
        for feature in record.features:
            total += 1

            if feature in known:
                reused += 1
            else:
                new += 1
                known.add(feature)

    result = {
        "concepts": float(len(records)),
        "feature_observations": float(total),
        "reused_features": float(reused),
        "new_features": float(new),
        "feature_reuse_rate": (
            reused / max(1, total)
        ),
    }

    print(
        f"=== V88 {label} FEATURE TRANSFER ==="
    )

    for key, value in result.items():
        print(
            f"{key:26s}: {value}"
        )

    print()

    return result


def heldout_combination_reuse(
    graph: SemanticOrderIndependentGraphV88,
    records: list[ConceptRecord],
    label: str,
) -> dict[str, float]:
    """
    For every held-out concept, enumerate unordered feature pairs and ask
    whether the corresponding pair combination was already learned.

    Because pairs are frozensets, ordering is irrelevant.
    """
    total_pairs = 0
    known_pairs = 0

    for record in records:
        feature_ids = []

        for feature in sorted(
            record.features
        ):
            feature_id = graph.feature_cells.get(
                feature
            )

            if feature_id is not None:
                feature_ids.append(
                    feature_id
                )

        for left, right in itertools.combinations(
            sorted(feature_ids),
            2,
        ):
            total_pairs += 1

            key = frozenset(
                (left, right)
            )

            if key in graph.combination_by_key:
                known_pairs += 1

    result = {
        "pairs": float(total_pairs),
        "known_pairs": float(known_pairs),
        "reuse_rate": (
            known_pairs
            / max(1, total_pairs)
        ),
    }

    print(
        f"=== V88 {label} COMBINATION TRANSFER ==="
    )

    for key, value in result.items():
        print(
            f"{key:20s}: {value}"
        )

    print()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start = time.perf_counter()

    print(
        "=== V88 ORDER-INDEPENDENT SEMANTIC COMBINATIONS ==="
    )
    print(
        "Feature combinations are frozensets; no arbitrary feature ordering."
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

    train, validation, test = split_concepts(
        concepts
    )

    print(
        "semantic_rows :",
        len(rows),
    )
    print(
        "concepts      :",
        len(concepts),
    )
    print(
        "train         :",
        len(train),
    )
    print(
        "validation    :",
        len(validation),
    )
    print(
        "test          :",
        len(test),
    )
    print()

    graph = SemanticOrderIndependentGraphV88()

    # ---------------------------------------------------------------
    # TRAIN FEATURES
    # ---------------------------------------------------------------

    train_units = [
        graph.concept_units(
            record
        )
        for record in train
    ]

    print(
        "train_feature_cells:",
        len(graph.feature_cells),
    )

    train_transfer = feature_transfer(
        graph,
        train,
        "TRAIN",
    )

    # ---------------------------------------------------------------
    # TRAIN RECURSIVE COMBINATIONS
    # ---------------------------------------------------------------

    current = train_units
    level_rows = []

    for level in range(
        1,
        MAX_LEVELS + 1,
    ):
        before = graph.counts()

        current, stats = (
            graph.discover_next_level(
                current,
                level,
            )
        )

        after = graph.counts()

        new_cells = (
            after["combination_cells"]
            - before["combination_cells"]
        )

        level_rows.append(
            {
                "level": level,
                "candidate_pairs": stats[
                    "candidate_pairs"
                ],
                "recurring_pairs": stats[
                    "recurring_pairs"
                ],
                "new_cells": new_cells,
                "replacements": stats[
                    "replacements"
                ],
                "combination_cells_total": after[
                    "combination_cells"
                ],
                "remaining_units": sum(
                    len(units)
                    for units in current
                ),
            }
        )

        print(
            f"TRAIN level={level:2d} "
            f"candidate_pairs={stats['candidate_pairs']:6d} "
            f"recurring={stats['recurring_pairs']:6d} "
            f"new_cells={new_cells:6d} "
            f"replacements={stats['replacements']:6d} "
            f"remaining_units={level_rows[-1]['remaining_units']:6d}"
        )

        if new_cells == 0:
            break

    # ---------------------------------------------------------------
    # HELD-OUT TRANSFER
    # ---------------------------------------------------------------

    validation_transfer = feature_transfer(
        graph,
        validation,
        "VALIDATION",
    )

    test_transfer = feature_transfer(
        graph,
        test,
        "TEST",
    )

    validation_combo = heldout_combination_reuse(
        graph,
        validation,
        "VALIDATION",
    )

    test_combo = heldout_combination_reuse(
        graph,
        test,
        "TEST",
    )

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------

    final = graph.counts()

    print(
        "=== V88 SUMMARY ==="
    )
    print(
        "feature_cells          :",
        final["feature_cells"],
    )
    print(
        "combination_cells     :",
        final["combination_cells"],
    )
    print(
        "validation_feature_reuse:",
        validation_transfer[
            "feature_reuse_rate"
        ],
    )
    print(
        "test_feature_reuse      :",
        test_transfer[
            "feature_reuse_rate"
        ],
    )
    print(
        "validation_pair_reuse   :",
        validation_combo[
            "reuse_rate"
        ],
    )
    print(
        "test_pair_reuse         :",
        test_combo[
            "reuse_rate"
        ],
    )

    print()
    print(
        "=== V88 LEVEL SUMMARY ==="
    )

    for row in level_rows:
        print(
            f"level={row['level']:2d} "
            f"candidate_pairs={row['candidate_pairs']:6d} "
            f"recurring={row['recurring_pairs']:6d} "
            f"new_cells={row['new_cells']:6d} "
            f"replacements={row['replacements']:6d} "
            f"remaining_units={row['remaining_units']:6d}"
        )

    print()
    print(
        "elapsed_seconds :",
        f"{time.perf_counter() - start:.2f}",
    )

    print(
        "=== V88 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
