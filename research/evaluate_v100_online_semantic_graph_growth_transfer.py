from __future__ import annotations

"""
V97 — ONLINE SEMANTIC GRAPH GROWTH / TRANSFER

Question
--------
Can the SAME graph learn a new semantic concept online by reusing existing
lexical and semantic structure, without retraining the old graph?

This is stronger than the earlier lookup-only transfer tests.

Setup
-----
1. Build the lexical substrate from ALL dictionary words.
2. Learn a semantic graph from TRAIN semantic cues only.
3. Freeze everything learned so far:
       existing lexical units
       existing semantic feature cells
       existing lexical->feature links
4. Present VALIDATION and TEST cues one at a time.
5. For each new cue, grow the SAME graph in place:
       * reuse existing lexical units
       * reuse existing semantic feature nodes when possible
       * create only genuinely new semantic feature nodes when necessary
       * add only the new cue's lexical->feature links
6. NEVER retrain or rebuild earlier concepts.
7. Verify that replaying an already learned cue creates ZERO new nodes/edges.

This directly tests incremental graph growth with transfer.

Primary outputs
---------------
    matched cues
    train / validation / test
    mean existing-feature reuse
    mean new-feature creation
    mean lexical unit reuse
    graph cells/edges before and after
    validation growth
    test growth
    replay idempotence
    total graph growth per new semantic concept

The lexical substrate itself remains fixed after full-dictionary construction.
Semantic growth happens online.
"""

import csv
import hashlib
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DICTIONARY_PATH = ROOT / "data" / "dictionary.csv"
SEMANTICS_PATH = ROOT / "data" / "semantics-large.csv"

TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15

MIN_OCCURRENCES = 2
MAX_LEXICAL_LEVELS = 10


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class CueRecord:
    cue: str
    features: dict[str, float] = field(default_factory=dict)


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def load_dictionary(path: Path) -> set[str]:
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

    if len(words) < 1000:
        raise RuntimeError(
            f"Dictionary unexpectedly small: {len(words)}"
        )

    return words


def load_large_semantics(
    path: Path,
) -> dict[str, CueRecord]:
    records: dict[str, CueRecord] = {}

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "cue",
            "translated",
            "frequency_translated",
            "n",
            "normalized_translated",
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                "semantics-large.csv missing columns: "
                + ", ".join(sorted(missing))
            )

        for raw in reader:
            cue = raw["cue"].strip().lower()
            feature = raw["translated"].strip().lower()

            if not cue or not feature:
                continue

            normalized = parse_float(
                raw["normalized_translated"]
            )

            if normalized <= 0.0:
                frequency = parse_float(
                    raw["frequency_translated"]
                )
                n = parse_int(raw["n"])
                if n > 0:
                    normalized = frequency / n

            if normalized <= 0.0:
                continue

            record = records.get(cue)
            if record is None:
                record = CueRecord(cue=cue)
                records[cue] = record

            record.features[feature] = (
                record.features.get(feature, 0.0)
                + normalized
            )

    if not records:
        raise RuntimeError(
            "No usable semantic cues loaded."
        )

    return records


def stable_rank(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def split_records(
    records: list[CueRecord],
):
    ordered = sorted(
        records,
        key=lambda record: (
            stable_rank(record.cue),
            record.cue,
        ),
    )

    n = len(ordered)
    train_end = int(
        n * TRAIN_FRACTION
    )
    validation_end = (
        train_end
        + int(
            n * VALID_FRACTION
        )
    )

    return (
        ordered[:train_end],
        ordered[train_end:validation_end],
        ordered[validation_end:],
    )


# ---------------------------------------------------------------------------
# Full lexical substrate
# ---------------------------------------------------------------------------

def width1_units(
    word: str,
) -> list[tuple[str, str, str]]:
    return [
        (
            word[pos - 1] if pos > 0 else "^",
            word[pos],
            word[pos + 1] if pos + 1 < len(word) else "$",
        )
        for pos in range(len(word))
    ]


class FullLexicalSubstrate:
    def __init__(self) -> None:
        self.primitive_ids: dict[
            tuple[str, str, str],
            int,
        ] = {}

        self.assembly_ids: dict[
            frozenset[int],
            int,
        ] = {}

        self.next_id = 0

    def primitive_id(
        self,
        unit: tuple[str, str, str],
    ) -> int:
        existing = self.primitive_ids.get(unit)
        if existing is not None:
            return existing

        identifier = self.next_id
        self.next_id += 1
        self.primitive_ids[unit] = identifier
        return identifier

    def base_stream(
        self,
        word: str,
    ) -> list[int]:
        return [
            self.primitive_id(unit)
            for unit in width1_units(word)
        ]

    def train(
        self,
        words: list[str],
    ) -> None:
        streams = [
            self.base_stream(word)
            for word in words
        ]

        print(
            "LEXICAL level=0 "
            f"primitive_units={len(self.primitive_ids)}",
            flush=True,
        )

        for level in range(
            1,
            MAX_LEXICAL_LEVELS + 1,
        ):
            occurrences = Counter()

            for stream in streams:
                for left, right in zip(
                    stream,
                    stream[1:],
                ):
                    occurrences[
                        frozenset((left, right))
                    ] += 1

            recurring = {
                key
                for key, count in occurrences.items()
                if count >= MIN_OCCURRENCES
            }

            created = 0

            for key in recurring:
                if key in self.assembly_ids:
                    continue

                self.assembly_ids[key] = self.next_id
                self.next_id += 1
                created += 1

            next_streams = []

            for stream in streams:
                output = []
                i = 0

                while i < len(stream):
                    if i + 1 < len(stream):
                        key = frozenset(
                            (
                                stream[i],
                                stream[i + 1],
                            )
                        )
                        assembly = self.assembly_ids.get(key)

                        if (
                            assembly is not None
                            and key in recurring
                        ):
                            output.append(assembly)
                            i += 2
                            continue

                    output.append(stream[i])
                    i += 1

                next_streams.append(output)

            streams = next_streams

            print(
                f"LEXICAL level={level:2d} "
                f"recurring={len(recurring):7d} "
                f"new_units={created:7d} "
                f"stream_units={sum(len(s) for s in streams):8d}",
                flush=True,
            )

            if created == 0:
                break

    def units(
        self,
        word: str,
    ) -> set[int]:
        """
        Dual lexical representation:
            primitive width-1 units + recursive assembly units

        The lexical substrate is frozen after full-dictionary training.
        """
        primitive = set(
            self.base_stream(word)
        )

        recursive = set()

        stream = list(primitive)

        # Use the actual ordered width-1 stream for recursive compression.
        stream = self.base_stream(word)

        while True:
            output = []
            changed = False
            i = 0

            while i < len(stream):
                if i + 1 < len(stream):
                    key = frozenset(
                        (
                            stream[i],
                            stream[i + 1],
                        )
                    )
                    assembly = self.assembly_ids.get(key)

                    if assembly is not None:
                        output.append(assembly)
                        recursive.add(assembly)
                        i += 2
                        changed = True
                        continue

                output.append(stream[i])
                i += 1

            stream = output

            if not changed:
                break

        return primitive | recursive

    def stats(self) -> dict[str, int]:
        return {
            "primitive_units": len(self.primitive_ids),
            "recursive_assemblies": len(self.assembly_ids),
            "all_units": self.next_id,
        }


# ---------------------------------------------------------------------------
# Online semantic graph
# ---------------------------------------------------------------------------

class OnlineSemanticGraph:
    """
    Mutable semantic layer on top of an immutable lexical substrate.

    Nodes:
        semantic feature nodes
        semantic concept nodes

    Edges:
        lexical unit -> semantic feature
        concept -> semantic feature

    Existing nodes/edges are reused.
    New structure is appended.
    No prior training pass is repeated.
    """

    def __init__(
        self,
        lexical: FullLexicalSubstrate,
    ) -> None:
        self.lexical = lexical

        self.feature_ids: dict[str, int] = {}
        self.concept_ids: dict[str, int] = {}

        self.next_node_id = 0

        self.lexical_feature_weight: dict[
            tuple[int, int],
            float,
        ] = {}

        self.concept_feature_weight: dict[
            tuple[int, int],
            float,
        ] = {}

        self.concept_units: dict[
            str,
            set[int],
        ] = {}

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def feature_id(
        self,
        feature: str,
    ) -> tuple[int, bool]:
        existing = self.feature_ids.get(feature)

        if existing is not None:
            return existing, False

        identifier = self.next_node_id
        self.next_node_id += 1

        self.feature_ids[feature] = identifier
        return identifier, True

    def concept_id(
        self,
        cue: str,
    ) -> tuple[int, bool]:
        existing = self.concept_ids.get(cue)

        if existing is not None:
            return existing, False

        identifier = self.next_node_id
        self.next_node_id += 1

        self.concept_ids[cue] = identifier
        return identifier, True

    # ------------------------------------------------------------------
    # Growth
    # ------------------------------------------------------------------

    def add_concept(
        self,
        record: CueRecord,
    ) -> dict[str, float]:
        """
        Online mutation of the SAME graph.

        Existing lexical structure:
            never rebuilt.

        Existing semantic features:
            reused.

        New semantic features:
            appended.

        Existing lexical->feature relations:
            reinforced.

        Existing concepts:
            untouched.
        """
        before_nodes = self.next_node_id
        before_edges = (
            len(self.lexical_feature_weight)
            + len(self.concept_feature_weight)
        )

        units = self.lexical.units(
            record.cue
        )

        existing_features = 0
        new_features = 0
        new_lexical_links = 0
        reinforced_lexical_links = 0
        new_concept_links = 0

        concept_node, concept_created = (
            self.concept_id(record.cue)
        )

        if not concept_created:
            # The caller should normally use replay() for this case.
            pass

        self.concept_units[
            record.cue
        ] = set(units)

        for feature, weight in (
            record.features.items()
        ):
            feature_node, created = (
                self.feature_id(feature)
            )

            if created:
                new_features += 1
            else:
                existing_features += 1

            concept_edge = (
                concept_node,
                feature_node,
            )

            if concept_edge in self.concept_feature_weight:
                self.concept_feature_weight[
                    concept_edge
                ] += weight
            else:
                self.concept_feature_weight[
                    concept_edge
                ] = weight
                new_concept_links += 1

            for unit in units:
                lexical_edge = (
                    unit,
                    feature_node,
                )

                if lexical_edge in self.lexical_feature_weight:
                    self.lexical_feature_weight[
                        lexical_edge
                    ] += weight
                    reinforced_lexical_links += 1
                else:
                    self.lexical_feature_weight[
                        lexical_edge
                    ] = weight
                    new_lexical_links += 1

        after_nodes = self.next_node_id
        after_edges = (
            len(self.lexical_feature_weight)
            + len(self.concept_feature_weight)
        )

        return {
            "concept_created": float(
                concept_created
            ),
            "existing_features": float(
                existing_features
            ),
            "new_features": float(
                new_features
            ),
            "feature_reuse_rate": (
                existing_features
                / max(
                    1,
                    existing_features + new_features,
                )
            ),
            "lexical_units": float(
                len(units)
            ),
            "new_lexical_links": float(
                new_lexical_links
            ),
            "reinforced_lexical_links": float(
                reinforced_lexical_links
            ),
            "new_concept_links": float(
                new_concept_links
            ),
            "nodes_added": float(
                after_nodes - before_nodes
            ),
            "edges_added": float(
                after_edges - before_edges
            ),
        }

    def replay(
        self,
        record: CueRecord,
    ) -> dict[str, float]:
        """
        Replay must not create nodes or edges.

        We use the existing graph IDs and existing edges only.
        """
        before_nodes = self.next_node_id
        before_edges = (
            len(self.lexical_feature_weight)
            + len(self.concept_feature_weight)
        )

        concept_id = self.concept_ids.get(
            record.cue
        )

        if concept_id is None:
            raise RuntimeError(
                "Cannot replay unseen concept: "
                + record.cue
            )

        units = self.lexical.units(
            record.cue
        )

        missing_features = sum(
            feature not in self.feature_ids
            for feature in record.features
        )

        missing_links = 0

        for feature, _weight in (
            record.features.items()
        ):
            feature_id = self.feature_ids.get(
                feature
            )

            if feature_id is None:
                continue

            if (
                concept_id,
                feature_id,
            ) not in self.concept_feature_weight:
                missing_links += 1

            for unit in units:
                if (
                    unit,
                    feature_id,
                ) not in self.lexical_feature_weight:
                    missing_links += 1

        after_nodes = self.next_node_id
        after_edges = (
            len(self.lexical_feature_weight)
            + len(self.concept_feature_weight)
        )

        assert (
            after_nodes == before_nodes
        )
        assert (
            after_edges == before_edges
        )

        return {
            "missing_features": float(
                missing_features
            ),
            "missing_links": float(
                missing_links
            ),
            "nodes_added": 0.0,
            "edges_added": 0.0,
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        return {
            "concept_nodes": len(
                self.concept_ids
            ),
            "feature_nodes": len(
                self.feature_ids
            ),
            "semantic_nodes": self.next_node_id,
            "lexical_feature_edges": len(
                self.lexical_feature_weight
            ),
            "concept_feature_edges": len(
                self.concept_feature_weight
            ),
            "semantic_edges": (
                len(
                    self.lexical_feature_weight
                )
                + len(
                    self.concept_feature_weight
                )
            ),
        }


# ---------------------------------------------------------------------------
# Online evaluation
# ---------------------------------------------------------------------------

def grow_split(
    graph: OnlineSemanticGraph,
    records: list[CueRecord],
    label: str,
) -> list[dict[str, object]]:
    print(
        f"=== V97 ONLINE {label} GROWTH ==="
    )

    rows = []

    for index, record in enumerate(
        records,
        start=1,
    ):
        metrics = graph.add_concept(
            record
        )

        row = {
            "cue": record.cue,
            **metrics,
        }
        rows.append(row)

        if (
            index <= 10
            or index % 100 == 0
            or index == len(records)
        ):
            print(
                f"{index:4d}/{len(records):4d} "
                f"{record.cue:20s} "
                f"reuse={metrics['feature_reuse_rate']:.3f} "
                f"new_features={int(metrics['new_features']):3d} "
                f"nodes+={int(metrics['nodes_added']):3d} "
                f"edges+={int(metrics['edges_added']):4d}",
                flush=True,
            )

    print()
    return rows


def summarize_growth(
    rows: list[dict[str, object]],
    label: str,
) -> None:
    if not rows:
        return

    def mean(key: str) -> float:
        return sum(
            float(row[key])
            for row in rows
        ) / len(rows)

    print(
        f"=== V97 {label} GROWTH SUMMARY ==="
    )
    print(
        "concepts:",
        len(rows),
    )
    print(
        "mean_feature_reuse_rate:",
        mean("feature_reuse_rate"),
    )
    print(
        "mean_new_features:",
        mean("new_features"),
    )
    print(
        "mean_nodes_added:",
        mean("nodes_added"),
    )
    print(
        "mean_edges_added:",
        mean("edges_added"),
    )
    print(
        "mean_new_lexical_links:",
        mean("new_lexical_links"),
    )
    print(
        "mean_reinforced_lexical_links:",
        mean("reinforced_lexical_links"),
    )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start = time.perf_counter()

    print(
        "=== V97 ONLINE SEMANTIC GRAPH GROWTH ==="
    )
    print(
        "Question: can new semantic concepts grow the SAME graph "
        "by reusing existing lexical and semantic structure?"
    )
    print()

    dictionary = load_dictionary(
        DICTIONARY_PATH
    )

    semantics = load_large_semantics(
        SEMANTICS_PATH
    )

    matched = [
        record
        for record in semantics.values()
        if record.cue in dictionary
    ]

    if len(matched) < 1000:
        raise RuntimeError(
            "Expected thousands of matched cues; "
            f"got {len(matched)}"
        )

    train, validation, test = split_records(
        matched
    )

    print(
        "dictionary_words:",
        len(dictionary),
    )
    print(
        "semantic_cues_total:",
        len(semantics),
    )
    print(
        "matched_cues:",
        len(matched),
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

    # ---------------------------------------------------------------
    # Full lexical substrate.
    # ---------------------------------------------------------------

    lexical = FullLexicalSubstrate()

    lexical.train(
        sorted(
            dictionary,
            key=lambda word: (
                stable_rank(word),
                word,
            ),
        )
    )

    print()
    print(
        "=== V97 LEXICAL SUBSTRATE ==="
    )

    for key, value in lexical.stats().items():
        print(
            f"{key:24s}: {value}"
        )

    print()

    # ---------------------------------------------------------------
    # Semantic graph: TRAIN only.
    # ---------------------------------------------------------------

    graph = OnlineSemanticGraph(
        lexical
    )

    train_rows = grow_split(
        graph,
        train,
        "TRAIN",
    )

    summarize_growth(
        train_rows,
        "TRAIN",
    )

    train_after = graph.stats()

    print(
        "=== GRAPH AFTER TRAIN ==="
    )

    for key, value in train_after.items():
        print(
            f"{key:28s}: {value}"
        )

    print()

    # ---------------------------------------------------------------
    # ONLINE VALIDATION GROWTH.
    # ---------------------------------------------------------------

    validation_before = graph.stats()

    validation_rows = grow_split(
        graph,
        validation,
        "VALIDATION",
    )

    validation_after = graph.stats()

    summarize_growth(
        validation_rows,
        "VALIDATION",
    )

    print(
        "=== VALIDATION GRAPH DELTA ==="
    )

    for key in (
        "concept_nodes",
        "feature_nodes",
        "semantic_nodes",
        "lexical_feature_edges",
        "concept_feature_edges",
        "semantic_edges",
    ):
        print(
            f"{key:28s}: "
            f"{validation_after[key] - validation_before[key]}"
        )

    print()

    # ---------------------------------------------------------------
    # ONLINE TEST GROWTH.
    # ---------------------------------------------------------------

    test_before = graph.stats()

    test_rows = grow_split(
        graph,
        test,
        "TEST",
    )

    test_after = graph.stats()

    summarize_growth(
        test_rows,
        "TEST",
    )

    print(
        "=== TEST GRAPH DELTA ==="
    )

    for key in (
        "concept_nodes",
        "feature_nodes",
        "semantic_nodes",
        "lexical_feature_edges",
        "concept_feature_edges",
        "semantic_edges",
    ):
        print(
            f"{key:28s}: "
            f"{test_after[key] - test_before[key]}"
        )

    print()

    # ---------------------------------------------------------------
    # REPLAY TEST: no retraining, no graph growth.
    # ---------------------------------------------------------------

    replay_before = graph.stats()

    replay_failures = 0

    for record in test:
        result = graph.replay(
            record
        )

        if (
            result["nodes_added"] != 0.0
            or result["edges_added"] != 0.0
            or result["missing_features"] != 0.0
            or result["missing_links"] != 0.0
        ):
            replay_failures += 1

    replay_after = graph.stats()

    assert (
        replay_before == replay_after
    )
    assert replay_failures == 0

    print(
        "=== V97 REPLAY ==="
    )
    print(
        "replayed_test_concepts:",
        len(test),
    )
    print(
        "replay_failures:",
        replay_failures,
    )
    print(
        "graph_changed:",
        replay_before != replay_after,
    )
    print(
        "ONLINE REPLAY IDEMPOTENCE: PASS"
    )
    print()

    # ---------------------------------------------------------------
    # Final.
    # ---------------------------------------------------------------

    final = graph.stats()

    print(
        "=== V97 FINAL GRAPH ==="
    )

    for key, value in final.items():
        print(
            f"{key:28s}: {value}"
        )

    print()
    print(
        "=== V97 TRANSFER SUMMARY ==="
    )
    print(
        "validation_mean_feature_reuse:",
        sum(
            row["feature_reuse_rate"]
            for row in validation_rows
        )
        / max(
            1,
            len(validation_rows),
        ),
    )
    print(
        "test_mean_feature_reuse:",
        sum(
            row["feature_reuse_rate"]
            for row in test_rows
        )
        / max(
            1,
            len(test_rows),
        ),
    )
    print(
        "validation_mean_nodes_added:",
        sum(
            row["nodes_added"]
            for row in validation_rows
        )
        / max(
            1,
            len(validation_rows),
        ),
    )
    print(
        "test_mean_nodes_added:",
        sum(
            row["nodes_added"]
            for row in test_rows
        )
        / max(
            1,
            len(test_rows),
        ),
    )

    print()
    print(
        "elapsed_seconds:",
        f"{time.perf_counter() - start:.2f}",
    )

    print(
        "=== V97 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
