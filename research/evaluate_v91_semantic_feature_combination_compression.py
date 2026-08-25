from __future__ import annotations

"""
V87 — SHARED SEMANTIC FEATURE COMBINATION COMPRESSION

Builds on the successful V86 semantic feature graph.

V86 established:
    unseen concepts reuse a large fraction of existing feature nodes.

V87 asks the next structural question:

    Do REPEATED COMBINATIONS of semantic features also become reusable
    higher-order graph units?

Representation
--------------
    concept
       ↓
    feature cells

Then recursively:
    repeated adjacent feature pairs
        ↓
    feature-combination assembly

and:
    repeated adjacent assembly pairs
        ↓
    higher-order semantic assembly

No semantic labels are inferred.
No category is used as a combination target.
No hand-authored feature groups are supplied.

The only source of combination discovery is repeated feature structure in the
human-elicited semantic corpus.

Primary metrics
---------------
    unique feature cells
    unique feature pairs
    recurring feature pairs
    new combination cells
    combination reuse
    concept-disjoint validation/test transfer
    recursive compression of concept feature streams
"""

import csv
import hashlib
import time
from collections import Counter
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

MIN_OCCURRENCES = 2
MAX_LEVELS = 8


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

    train_end = int(
        n * TRAIN_FRACTION
    )

    validation_end = (
        train_end
        + int(n * VALID_FRACTION)
    )

    return (
        ordered[:train_end],
        ordered[
            train_end:validation_end
        ],
        ordered[validation_end:],
    )


# ---------------------------------------------------------------------------
# Semantic graph
# ---------------------------------------------------------------------------

class SemanticCombinationGraphV87(Network):
    FEATURE = "v87_feature"
    COMBINATION = "v87_combination"

    CONCEPT_FEATURE = "V87_CONCEPT_FEATURE"
    COMBINATION_MEMBER = "V87_COMBINATION_MEMBER"

    def __init__(
        self,
        config: Optional[Config] = None,
    ) -> None:
        super().__init__(config)

        self.feature_cells: dict[str, int] = {}

        # key:
        #     ordered tuple of lower-level unit IDs
        # value:
        #     discovered combination cell
        self.combination_by_key: dict[
            tuple[int, ...],
            int,
        ] = {}

        self.combination_level: dict[
            int,
            int,
        ] = {}

        self.combination_activations: Counter[
            int
        ] = Counter()

    # ------------------------------------------------------------------
    # Feature cells
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

    def learn_concept(
        self,
        record: ConceptRecord,
    ) -> list[int]:
        sequence = []

        for feature in sorted(
            record.features
        ):
            feature_id = self.feature_cell(
                feature
            )

            sequence.append(
                feature_id
            )

        return sequence

    # ------------------------------------------------------------------
    # Combination discovery
    # ------------------------------------------------------------------

    def combination_cell(
        self,
        key: tuple[int, ...],
        level: int,
    ) -> int:
        existing = self.combination_by_key.get(
            key
        )

        if existing is not None:
            self.combination_activations[
                existing
            ] += 1
            return existing

        cell_id = self.create_cell(
            self.COMBINATION,
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

        for member in key:
            self.connect(
                member,
                cell_id,
                self.COMBINATION_MEMBER,
                1.0,
            )

        return cell_id

    def discover_level(
        self,
        streams: list[list[int]],
        level: int,
        min_occurrences: int = MIN_OCCURRENCES,
    ) -> tuple[list[list[int]], dict[str, int]]:
        """
        Generic semantic combination operator.

        It does not care whether the input units are:
            level-0 features
            level-1 combinations
            level-2 combinations
            ...

        Same operator at every level.
        """
        occurrences = Counter()

        for stream in streams:
            for left, right in zip(
                stream,
                stream[1:],
            ):
                occurrences[
                    (left, right)
                ] += 1

        recurring = {
            key: count
            for key, count in occurrences.items()
            if count >= min_occurrences
        }

        next_streams = []
        reused = 0
        discovered = 0

        for stream in streams:
            output = []
            i = 0

            while i < len(stream):
                if i + 1 < len(stream):
                    key = (
                        stream[i],
                        stream[i + 1],
                    )

                    if key in recurring:
                        combination = (
                            self.combination_cell(
                                key,
                                level,
                            )
                        )

                        if self.combination_activations[
                            combination
                        ] > 1:
                            reused += 1

                        output.append(
                            combination
                        )

                        # Track whether it was just created by checking
                        # membership after creation is insufficient, so use
                        # the global creation delta below.
                        i += 2
                        continue

                output.append(
                    stream[i]
                )
                i += 1

            next_streams.append(
                output
            )

        # Count unique level-specific cells.
        discovered = sum(
            level_value == level
            for level_value
            in self.combination_level.values()
        )

        return next_streams, {
            "candidate_pairs": len(
                occurrences
            ),
            "recurring_pairs": len(
                recurring
            ),
            "level_combination_cells": (
                discovered
            ),
            "reuse_events": reused,
        }

    # ------------------------------------------------------------------
    # Stream helpers
    # ------------------------------------------------------------------

    def streams_for(
        self,
        records: list[ConceptRecord],
    ) -> list[list[int]]:
        streams = []

        for record in records:
            sequence = (
                self.learn_concept(
                    record
                )
            )

            if sequence:
                streams.append(
                    sequence
                )

        return streams

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
# Transfer measurement
# ---------------------------------------------------------------------------

def feature_transfer(
    graph: SemanticCombinationGraphV87,
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
        "concepts": float(
            len(records)
        ),
        "feature_observations": float(total),
        "reused_features": float(reused),
        "new_features": float(new),
        "feature_reuse_rate": (
            reused / max(1, total)
        ),
    }

    print(
        f"=== V87 {label} FEATURE TRANSFER ==="
    )

    for key, value in result.items():
        print(
            f"{key:24s}: {value}"
        )

    print()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start = time.perf_counter()

    print(
        "=== V87 SHARED SEMANTIC FEATURE COMBINATIONS ==="
    )
    print(
        "Objective: discover repeated semantic feature combinations "
        "and recursively compress them."
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
        "semantic_rows     :",
        len(rows),
    )
    print(
        "concepts          :",
        len(concepts),
    )
    print(
        "train             :",
        len(train),
    )
    print(
        "validation        :",
        len(validation),
    )
    print(
        "test              :",
        len(test),
    )
    print()

    graph = SemanticCombinationGraphV87()

    # ---------------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------------

    train_streams = graph.streams_for(
        train
    )

    print(
        "train_feature_cells:",
        len(graph.feature_cells),
    )

    train_transfer = feature_transfer(
        graph,
        train,
        "TRAIN",
    )

    # Only training feature streams are used to seed the combination graph.
    current_streams = train_streams

    rows_out = []

    for level in range(
        1,
        MAX_LEVELS + 1,
    ):
        before = graph.counts()

        current_streams, stats = (
            graph.discover_level(
                current_streams,
                level,
            )
        )

        after = graph.counts()

        new_cells = (
            after["combination_cells"]
            - before["combination_cells"]
        )

        total_input = sum(
            len(stream)
            for stream
            in current_streams
        )

        rows_out.append(
            {
                "level": level,
                "input_streams": len(
                    current_streams
                ),
                "output_units": total_input,
                "candidate_pairs": stats[
                    "candidate_pairs"
                ],
                "recurring_pairs": stats[
                    "recurring_pairs"
                ],
                "new_cells": new_cells,
                "combination_cells_total": (
                    after[
                        "combination_cells"
                    ]
                ),
            }
        )

        print(
            f"TRAIN level={level} "
            f"candidate_pairs={stats['candidate_pairs']:5d} "
            f"recurring={stats['recurring_pairs']:5d} "
            f"new_cells={new_cells:5d} "
            f"total_combinations={after['combination_cells']:5d}"
        )

        if new_cells == 0:
            break

    # ---------------------------------------------------------------
    # HELD-OUT CONCEPTS
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

    # Important:
    # use ONLY the already learned feature cells when constructing the
    # held-out streams. Unknown features are represented by -1 and excluded.
    def heldout_streams(
        records: list[ConceptRecord],
    ) -> list[list[int]]:
        result = []

        for record in records:
            stream = []

            for feature in sorted(
                record.features
            ):
                feature_id = graph.feature_cells.get(
                    feature,
                    -1,
                )

                if feature_id >= 0:
                    stream.append(
                        feature_id
                    )

            if stream:
                result.append(
                    stream
                )

        return result

    validation_streams = (
        heldout_streams(validation)
    )

    test_streams = (
        heldout_streams(test)
    )

    # Measure how many HELD-OUT feature combinations already exist as
    # learned combination cells.
    def count_known_combinations(
        streams: list[list[int]],
        label: str,
    ) -> dict[str, int]:
        total_pairs = 0
        known_pairs = 0

        for stream in streams:
            for pair in zip(
                stream,
                stream[1:],
            ):
                total_pairs += 1

                if (
                    pair
                    in graph.combination_by_key
                ):
                    known_pairs += 1

        print(
            f"=== V87 {label} COMBINATION TRANSFER ==="
        )
        print(
            "feature_pairs          :",
            total_pairs,
        )
        print(
            "known_combinations     :",
            known_pairs,
        )
        print(
            "combination_reuse_rate:",
            known_pairs
            / max(1, total_pairs),
        )
        print()

        return {
            "pairs": total_pairs,
            "known": known_pairs,
        }

    validation_combo = (
        count_known_combinations(
            validation_streams,
            "VALIDATION",
        )
    )

    test_combo = (
        count_known_combinations(
            test_streams,
            "TEST",
        )
    )

    # ---------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------

    final = graph.counts()

    print(
        "=== V87 SUMMARY ==="
    )
    print(
        "train_feature_cells      :",
        len(graph.feature_cells),
    )
    print(
        "train_combination_cells :",
        final["combination_cells"],
    )
    print(
        "validation_feature_reuse :",
        validation_transfer[
            "feature_reuse_rate"
        ],
    )
    print(
        "test_feature_reuse       :",
        test_transfer[
            "feature_reuse_rate"
        ],
    )
    print(
        "validation_combo_reuse   :",
        validation_combo["known"]
        / max(
            1,
            validation_combo["pairs"],
        ),
    )
    print(
        "test_combo_reuse         :",
        test_combo["known"]
        / max(
            1,
            test_combo["pairs"],
        ),
    )

    print()
    print(
        "=== V87 LEVEL SUMMARY ==="
    )

    for row in rows_out:
        print(
            f"level={row['level']:2d} "
            f"candidate_pairs={row['candidate_pairs']:6d} "
            f"recurring={row['recurring_pairs']:6d} "
            f"new_cells={row['new_cells']:6d} "
            f"total={row['combination_cells_total']:6d}"
        )

    print()
    print(
        "elapsed_seconds :",
        f"{time.perf_counter() - start:.2f}",
    )
    print(
        "=== V87 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
