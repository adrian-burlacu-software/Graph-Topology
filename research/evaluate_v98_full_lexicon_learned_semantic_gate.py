from __future__ import annotations

"""
V95 — FULL LEXICON + LEARNED SEMANTIC GATE

One serious attempt at the missing architectural layer.

Problem exposed by V93/V94:
    * full recursive lexical graph is structurally useful
    * recursive-only semantic transfer loses coverage
    * dual-level exposure restores coverage but does not beat width-1

Hypothesis:
    the semantic interface should NOT consume every lexical unit equally.

V95 adds one learned selector / gate:

    lexical substrate
       |
       +-- primitive width-1 units
       |
       +-- recursive assemblies
                |
                v
        semantic feature gate
                |
                v
        selected form->feature links

The gate is learned ONLY from TRAIN semantic anchors.

For each lexical unit u and semantic feature f, estimate whether f is more
strongly associated with u than its corpus-wide feature prior would predict.

Score:
    association(u,f)
        = P(f | u) / P(f)

A unit is allowed to contribute to a feature when its association clears a
training-derived gate.

This is NOT a hand-coded semantic rule:
    * the lexical graph still comes from ALL dictionary words
    * semantic features are human-elicited
    * the selector is learned from train anchors
    * test is concept-disjoint
    * one shuffled-control model uses exactly the same machinery

Representations compared:
    1. WIDTH1
    2. RECURSIVE
    3. GATED-DUAL
    4. SHUFFLED GATED-DUAL

The gated dual model exposes both primitive and recursive lexical units, but
lets learned association strength decide which units actually participate in
predicting a feature.

Primary outcome:
    Does GATED-DUAL beat both WIDTH1 and the shuffled control on held-out
    semantic feature prediction?

This is intended as a "one serious shot" at the semantic interface, not another
ladder of tiny representation tweaks.
"""

import csv
import hashlib
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DICTIONARY_PATH = ROOT / "data" / "dictionary.csv"
SEMANTICS_PATH = ROOT / "data" / "semantics.csv"

TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15

TOP_K = 10
SEED = 9173

# Learned gate controls.
MIN_UNIT_ANCHORS = 2
ASSOCIATION_FLOOR = 1.0
MAX_SELECTED_UNITS = 24


# ---------------------------------------------------------------------------
# Semantic data
# ---------------------------------------------------------------------------

@dataclass
class ConceptRecord:
    concept: str
    features: set[str] = field(default_factory=set)
    categories: set[str] = field(default_factory=set)


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


def load_semantics(
    path: Path,
) -> dict[str, ConceptRecord]:
    grouped: dict[str, ConceptRecord] = {}

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
                "semantics.csv missing columns: "
                + ", ".join(sorted(missing))
            )

        for raw in reader:
            concept = raw[
                "basic_level_concept"
            ].strip().lower()

            feature = raw[
                "feature_name"
            ].strip().lower()

            category = raw[
                "superordinate_category"
            ].strip().lower()

            if not concept or not feature:
                continue

            record = grouped.get(concept)

            if record is None:
                record = ConceptRecord(
                    concept=concept
                )
                grouped[concept] = record

            record.features.add(feature)

            if category:
                record.categories.add(category)

    return grouped


def stable_rank(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def split_records(
    records: list[ConceptRecord],
):
    ordered = sorted(
        records,
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
        ordered[
            validation_end:
        ],
    )


# ---------------------------------------------------------------------------
# Lexical substrate
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
    """
    Entire dictionary builds lexical structure.

    Level 0:
        width-1 local units

    Higher levels:
        recursively discovered reusable assemblies
    """

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

        for level in range(1, 10 + 1):
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
                for key, count
                in occurrences.items()
                if count >= 2
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

                        assembly = self.assembly_ids.get(
                            key
                        )

                        if (
                            assembly is not None
                            and key in recurring
                        ):
                            output.append(
                                assembly
                            )
                            i += 2
                            continue

                    output.append(stream[i])
                    i += 1

                next_streams.append(output)

            streams = next_streams

            print(
                f"LEXICAL level={level:2d} "
                f"recurring={len(recurring):6d} "
                f"new_units={created:6d} "
                f"stream_units={sum(len(s) for s in streams):7d}",
                flush=True,
            )

            if created == 0:
                break

    def recursive_units(
        self,
        word: str,
    ) -> list[int]:
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

                    assembly = self.assembly_ids.get(
                        key
                    )

                    if assembly is not None:
                        output.append(assembly)
                        i += 2
                        changed = True
                        continue

                output.append(stream[i])
                i += 1

            stream = output

            if not changed:
                break

        return stream

    def dual_units(
        self,
        word: str,
    ) -> list[int]:
        return list(
            set(
                self.base_stream(word)
            )
            | set(
                self.recursive_units(word)
            )
        )

    def stats(self) -> dict[str, int]:
        return {
            "primitive_units": len(
                self.primitive_ids
            ),
            "recursive_assemblies": len(
                self.assembly_ids
            ),
            "all_units": self.next_id,
        }


# ---------------------------------------------------------------------------
# Learned semantic gate
# ---------------------------------------------------------------------------

class LearnedSemanticGate:
    """
    Learns lexical-unit -> feature association using train anchors.

    The gate is learned from the SAME evidence used by the predictor.

    For each unit:
        unit_count = number of training anchors containing the unit

    For each unit/feature pair:
        co_count = number of training anchors containing both

    Global feature prior:
        P(feature)

    Association:
        P(feature | unit) / P(feature)

    Only units with enough anchor coverage are permitted to vote.
    """

    def __init__(
        self,
        substrate: FullLexicalSubstrate,
        mode: str,
        use_gate: bool,
    ) -> None:
        if mode not in {
            "width1",
            "recursive",
            "dual",
        }:
            raise ValueError(mode)

        self.substrate = substrate
        self.mode = mode
        self.use_gate = use_gate

        self.feature_ids: dict[str, int] = {}
        self.feature_names: dict[int, str] = {}

        self.feature_anchor_count: Counter[int] = Counter()
        self.unit_anchor_count: Counter[int] = Counter()

        self.unit_feature_count: dict[
            int,
            Counter[int],
        ] = defaultdict(Counter)

        self.selected_units: dict[
            int,
            set[int],
        ] = defaultdict(set)

        self.total_anchors = 0

        self.selection_count = 0

    def representation(
        self,
        word: str,
    ) -> list[int]:
        if self.mode == "width1":
            return self.substrate.base_stream(word)

        if self.mode == "recursive":
            return self.substrate.recursive_units(word)

        return self.substrate.dual_units(word)

    def feature_id(
        self,
        feature: str,
    ) -> int:
        existing = self.feature_ids.get(feature)

        if existing is not None:
            return existing

        identifier = len(
            self.feature_ids
        )

        self.feature_ids[feature] = identifier
        self.feature_names[identifier] = feature

        return identifier

    def learn(
        self,
        record: ConceptRecord,
    ) -> None:
        self.total_anchors += 1

        feature_ids = {
            self.feature_id(feature)
            for feature in record.features
        }

        for feature_id in feature_ids:
            self.feature_anchor_count[
                feature_id
            ] += 1

        units = set(
            self.representation(
                record.concept
            )
        )

        for unit_id in units:
            self.unit_anchor_count[
                unit_id
            ] += 1

            for feature_id in feature_ids:
                self.unit_feature_count[
                    unit_id
                ][feature_id] += 1

    def finalize_gate(self) -> None:
        """
        Learn which unit-feature links deserve semantic participation.

        A positive association above 1.0 means:
            feature is more prevalent in this unit's anchors than globally.

        The strongest links are retained per unit, bounded by
        MAX_SELECTED_UNITS only on the unit side at prediction time.
        """
        if not self.use_gate:
            return

        for unit_id, feature_counts in (
            self.unit_feature_count.items()
        ):
            if (
                self.unit_anchor_count[unit_id]
                < MIN_UNIT_ANCHORS
            ):
                continue

            scored = []

            unit_total = self.unit_anchor_count[
                unit_id
            ]

            for feature_id, co_count in feature_counts.items():
                feature_prior = (
                    self.feature_anchor_count[
                        feature_id
                    ]
                    / max(
                        1,
                        self.total_anchors,
                    )
                )

                conditional = (
                    co_count
                    / unit_total
                )

                association = (
                    conditional
                    / max(
                        1e-9,
                        feature_prior,
                    )
                )

                if association >= ASSOCIATION_FLOOR:
                    scored.append(
                        (
                            association,
                            feature_id,
                        )
                    )

            scored.sort(
                reverse=True
            )

            self.selected_units[
                unit_id
            ] = {
                feature_id
                for _score, feature_id
                in scored
            }

            self.selection_count += len(
                self.selected_units[
                    unit_id
                ]
            )

    def score(
        self,
        word: str,
    ) -> Counter[int]:
        units = list(
            set(
                self.representation(word)
            )
        )

        # Prefer units with learned semantic coverage.
        units.sort(
            key=lambda unit_id: (
                -self.unit_anchor_count.get(
                    unit_id,
                    0,
                ),
                unit_id,
            )
        )

        units = units[:MAX_SELECTED_UNITS]

        scores = Counter()

        for unit_id in units:
            unit_total = self.unit_anchor_count.get(
                unit_id,
                0,
            )

            if unit_total == 0:
                continue

            allowed = None

            if self.use_gate:
                allowed = self.selected_units.get(
                    unit_id,
                    set(),
                )

                if not allowed:
                    continue

            for feature_id, co_count in (
                self.unit_feature_count.get(
                    unit_id,
                    Counter(),
                ).items()
            ):
                if (
                    allowed is not None
                    and feature_id not in allowed
                ):
                    continue

                feature_prior = (
                    self.feature_anchor_count[
                        feature_id
                    ]
                    / max(
                        1,
                        self.total_anchors,
                    )
                )

                conditional = (
                    co_count
                    / unit_total
                )

                association = (
                    conditional
                    / max(
                        1e-9,
                        feature_prior,
                    )
                )

                # Log association behaves like a compact evidence score.
                scores[feature_id] += math.log(
                    max(
                        1.0,
                        association,
                    )
                )

        return scores

    def predict(
        self,
        word: str,
        k: int,
    ) -> list[str]:
        ranked = sorted(
            self.score(word).items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        return [
            self.feature_names[feature_id]
            for feature_id, _score
            in ranked[:k]
        ]

    def coverage(
        self,
        word: str,
    ) -> float:
        units = set(
            self.representation(word)
        )

        covered = sum(
            self.unit_anchor_count.get(
                unit_id,
                0,
            )
            > 0
            for unit_id in units
        )

        return covered / max(
            1,
            len(units),
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def prf(
    predicted: list[str],
    actual: set[str],
):
    predicted_set = set(predicted)

    if not predicted_set:
        return 0.0, 0.0, 0.0

    tp = len(
        predicted_set & actual
    )

    precision = (
        tp / len(predicted_set)
    )

    recall = (
        tp / max(1, len(actual))
    )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2.0
            * precision
            * recall
            / (precision + recall)
        )

    return precision, recall, f1


def evaluate(
    model: LearnedSemanticGate,
    records: list[ConceptRecord],
    label: str,
) -> dict[str, float]:
    precisions = []
    recalls = []
    f1s = []
    coverages = []

    for record in records:
        predicted = model.predict(
            record.concept,
            TOP_K,
        )

        precision, recall, f1 = prf(
            predicted,
            record.features,
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        coverages.append(
            model.coverage(
                record.concept
            )
        )

    result = {
        "precision_at_k": sum(
            precisions
        )
        / max(
            1,
            len(precisions),
        ),
        "recall_at_k": sum(
            recalls
        )
        / max(
            1,
            len(recalls),
        ),
        "f1_at_k": sum(
            f1s
        )
        / max(
            1,
            len(f1s),
        ),
        "coverage": sum(
            coverages
        )
        / max(
            1,
            len(coverages),
        ),
    }

    print(
        f"=== V95 {label} ==="
    )

    for key, value in result.items():
        print(
            f"{key:24s}: {value}"
        )

    print()

    return result


def shuffled_records(
    records: list[ConceptRecord],
    seed: int,
) -> list[ConceptRecord]:
    rng = random.Random(seed)

    feature_sets = [
        frozenset(record.features)
        for record in records
    ]

    rng.shuffle(
        feature_sets
    )

    return [
        ConceptRecord(
            concept=record.concept,
            features=set(feature_set),
            categories=set(record.categories),
        )
        for record, feature_set in zip(
            records,
            feature_sets,
        )
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start = time.perf_counter()

    print(
        "=== V95 FULL LEXICON LEARNED SEMANTIC GATE ==="
    )
    print(
        "One-shot attempt at the semantic interface."
    )
    print(
        "Full dictionary -> recursive lexical graph -> learned semantic gate."
    )
    print()

    dictionary = load_dictionary(
        DICTIONARY_PATH
    )

    semantic_map = load_semantics(
        SEMANTICS_PATH
    )

    anchors = [
        record
        for record in semantic_map.values()
        if record.concept in dictionary
    ]

    train, validation, test = split_records(
        anchors
    )

    print(
        "dictionary_words:",
        len(dictionary),
    )
    print(
        "semantic_concepts:",
        len(semantic_map),
    )
    print(
        "matched_anchors:",
        len(anchors),
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

    substrate = FullLexicalSubstrate()

    all_words = sorted(
        dictionary,
        key=lambda word: (
            stable_rank(word),
            word,
        ),
    )

    substrate.train(
        all_words
    )

    print()
    print(
        "=== FULL LEXICAL SUBSTRATE ==="
    )

    for key, value in substrate.stats().items():
        print(
            f"{key:24s}: {value}"
        )

    print()

    # ---------------------------------------------------------------
    # Four semantic interfaces.
    # ---------------------------------------------------------------

    width1 = LearnedSemanticGate(
        substrate,
        mode="width1",
        use_gate=False,
    )

    recursive = LearnedSemanticGate(
        substrate,
        mode="recursive",
        use_gate=False,
    )

    gated_dual = LearnedSemanticGate(
        substrate,
        mode="dual",
        use_gate=True,
    )

    shuffled_dual = LearnedSemanticGate(
        substrate,
        mode="dual",
        use_gate=True,
    )

    for record in train:
        width1.learn(record)
        recursive.learn(record)
        gated_dual.learn(record)
        shuffled_dual.learn(record)

    gated_dual.finalize_gate()

    for record in shuffled_records(
        train,
        SEED,
    ):
        # Replace the shuffled model's learned counts by clearing and
        # retraining it from the permuted semantic anchors.
        pass

    # Build the shuffled model independently so the real model is untouched.
    shuffled_dual = LearnedSemanticGate(
        substrate,
        mode="dual",
        use_gate=True,
    )

    shuffled_train = shuffled_records(
        train,
        SEED,
    )

    for record in shuffled_train:
        shuffled_dual.learn(record)

    shuffled_dual.finalize_gate()

    # ---------------------------------------------------------------
    # Evaluate.
    # ---------------------------------------------------------------

    width1_test = evaluate(
        width1,
        test,
        "WIDTH1 TEST",
    )

    recursive_test = evaluate(
        recursive,
        test,
        "RECURSIVE TEST",
    )

    dual_test = evaluate(
        gated_dual,
        test,
        "GATED-DUAL TEST",
    )

    shuffled_test = evaluate(
        shuffled_dual,
        test,
        "SHUFFLED GATED-DUAL TEST",
    )

    # Validation is included as a sanity check but test is the headline.
    evaluate(
        gated_dual,
        validation,
        "GATED-DUAL VALIDATION",
    )

    # ---------------------------------------------------------------
    # Final comparison.
    # ---------------------------------------------------------------

    print(
        "=== V95 COMPARISON ==="
    )

    for metric in (
        "precision_at_k",
        "recall_at_k",
        "f1_at_k",
    ):
        w = width1_test[metric]
        r = recursive_test[metric]
        d = dual_test[metric]
        s = shuffled_test[metric]

        print(
            f"{metric:18s} "
            f"width1={w:.6f} "
            f"recursive={r:.6f} "
            f"gated_dual={d:.6f} "
            f"shuffled={s:.6f}"
        )

        print(
            f"{'':18s}"
            f" gated-width1={d-w:+.6f} "
            f" gated-shuffle={d-s:+.6f} "
            f"lift={d/max(1e-9,s):.3f}x"
        )

    print()
    print(
        "gate_selected_links:",
        gated_dual.selection_count,
    )

    print(
        "width1_coverage:",
        width1_test["coverage"],
    )
    print(
        "recursive_coverage:",
        recursive_test["coverage"],
    )
    print(
        "gated_dual_coverage:",
        dual_test["coverage"],
    )

    print()
    print(
        "=== V95 INTERPRETATION ==="
    )
    print(
        "The lexical substrate was learned from the entire dictionary."
    )
    print(
        "The semantic gate learned which lexical units are useful from TRAIN anchors."
    )
    print(
        "GATED-DUAL > WIDTH1 means the selector successfully extracts "
        "useful higher-order lexical structure without sacrificing all local coverage."
    )
    print(
        "GATED-DUAL > SHUFFLED means the selected form->feature associations "
        "generalize beyond random word/feature alignment."
    )

    print()
    print(
        "elapsed_seconds:",
        f"{time.perf_counter() - start:.2f}",
    )

    print(
        "=== V95 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
