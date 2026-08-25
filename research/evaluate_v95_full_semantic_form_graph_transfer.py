from __future__ import annotations

"""
V92 — FULL MATCHED-CONCEPT FORM -> SEMANTICS GRAPH TRANSFER

Why all matched concepts?
-------------------------
The previous experiment had only ~153 exact word/concept matches because it
required a dictionary word to exactly equal the semantic basic-level concept.

That is unnecessarily small.

V92 uses ALL exact matches available across the uploaded dictionary + semantic
corpus, rather than holding back a tiny 24-concept test merely because the
intersection is small.

For evaluation, we still split the matched concepts into:
    70% train
    15% validation
    15% test

The key comparison is:

    MODEL A:
        isolated width-1 lexical units
        -> semantic feature predictions

    MODEL B:
        recursively discovered lexical assemblies
        -> semantic feature predictions

Both are compared against the SAME shuffled alignment control and the SAME
concept-disjoint test set.

The lexical representation is the graph representation already developed:
    characters
       ↓
    width-1 local units
       ↓
    recursive reusable lexical assemblies

The semantic target is:
    human-elicited feature set

This tests whether recursive lexical structure carries more reusable
form->feature information than isolated local units.
"""

import csv
import hashlib
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]

DICTIONARY_PATH = ROOT / "data" / "dictionary.csv"
SEMANTICS_PATH = ROOT / "data" / "semantics.csv"

TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15

TOP_K = 10
SEED = 9173

MIN_OCCURRENCES = 2
MAX_LEXICAL_LEVELS = 8


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
                "Missing columns: "
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
        ordered[train_end:validation_end],
        ordered[validation_end:],
    )


# ---------------------------------------------------------------------------
# Lexical width-1 units
# ---------------------------------------------------------------------------

def width1_units(
    word: str,
) -> list[tuple[str, str, str]]:
    units = []

    for pos, symbol in enumerate(word):
        left = (
            word[pos - 1]
            if pos > 0
            else "^"
        )

        right = (
            word[pos + 1]
            if pos + 1 < len(word)
            else "$"
        )

        units.append(
            (
                left,
                symbol,
                right,
            )
        )

    return units


# ---------------------------------------------------------------------------
# Recursive lexical assembly substrate
# ---------------------------------------------------------------------------

class LexicalAssemblySubstrate:
    """
    Builds the same width-1 -> recursive compression hierarchy used in the
    lexical experiments.

    It is independent of the semantic predictor.
    """

    def __init__(self) -> None:
        self.unit_ids: dict[
            tuple[str, ...],
            int,
        ] = {}

        self.unit_level: dict[int, int] = {}

        self.key_to_id: dict[
            frozenset[int],
            int,
        ] = {}

        self.next_id = 0

    def primitive_id(
        self,
        unit: tuple[str, str, str],
    ) -> int:
        key = tuple(unit)

        existing = self.unit_ids.get(key)

        if existing is not None:
            return existing

        identifier = self.next_id
        self.next_id += 1

        self.unit_ids[key] = identifier
        self.unit_level[identifier] = 0

        return identifier

    def base_stream(
        self,
        word: str,
    ) -> list[int]:
        return [
            self.primitive_id(unit)
            for unit in width1_units(word)
        ]

    def discover_level(
        self,
        streams: list[list[int]],
        level: int,
    ) -> list[list[int]]:
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

        for key in recurring:
            if key in self.key_to_id:
                continue

            identifier = self.next_id
            self.next_id += 1

            self.key_to_id[key] = identifier
            self.unit_level[identifier] = level

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

                    assembly = self.key_to_id.get(key)

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

        return next_streams

    def train(
        self,
        words: list[str],
    ) -> None:
        streams = [
            self.base_stream(word)
            for word in words
        ]

        for level in range(
            1,
            MAX_LEXICAL_LEVELS + 1,
        ):
            before = self.next_id

            streams = self.discover_level(
                streams,
                level,
            )

            created = self.next_id - before

            print(
                f"LEXICAL level={level} "
                f"new_units={created} "
                f"stream_units={sum(len(s) for s in streams)}",
                flush=True,
            )

            if created == 0:
                break

    def representation(
        self,
        word: str,
        recursive: bool,
    ) -> list[int]:
        if not recursive:
            return self.base_stream(word)

        streams = [
            self.base_stream(word)
        ]

        for level in range(
            1,
            MAX_LEXICAL_LEVELS + 1,
        ):
            before = self.next_id
            streams = self.discover_level(
                streams,
                level,
            )

            # IMPORTANT:
            # At inference time, no new assemblies should be learned.
            # discover_level above could create them, so the recursive test
            # below uses a separate frozen compression path instead.
            del before

        return streams[0]

    def frozen_recursive_representation(
        self,
        word: str,
    ) -> list[int]:
        stream = self.base_stream(word)

        for level in range(
            1,
            MAX_LEXICAL_LEVELS + 1,
        ):
            output = []
            i = 0
            changed = False

            while i < len(stream):
                if i + 1 < len(stream):
                    key = frozenset(
                        (
                            stream[i],
                            stream[i + 1],
                        )
                    )

                    assembly = self.key_to_id.get(key)

                    if assembly is not None:
                        output.append(
                            assembly
                        )
                        i += 2
                        changed = True
                        continue

                output.append(stream[i])
                i += 1

            stream = output

            if not changed:
                break

        return stream

    def known_primitive_units(
        self,
        word: str,
    ) -> tuple[int, int]:
        units = set(
            width1_units(word)
        )

        known = sum(
            unit in self.unit_ids
            for unit in units
        )

        return (
            known,
            len(units),
        )


# ---------------------------------------------------------------------------
# Cross-modal predictors
# ---------------------------------------------------------------------------

class FormFeatureModel:
    """
    Maps lexical units -> semantic features.

    Two instances are used:
        recursive=False : width-1 units only
        recursive=True  : recursive lexical units
    """

    def __init__(
        self,
        substrate: LexicalAssemblySubstrate,
        recursive: bool,
    ) -> None:
        self.substrate = substrate
        self.recursive = recursive

        self.feature_ids: dict[
            str,
            int,
        ] = {}

        self.unit_feature_counts: dict[
            int,
            Counter[int],
        ] = defaultdict(Counter)

        self.unit_document_counts: Counter[int] = Counter()

    def feature_id(
        self,
        feature: str,
    ) -> int:
        existing = self.feature_ids.get(feature)

        if existing is not None:
            return existing

        identifier = len(self.feature_ids)
        self.feature_ids[feature] = identifier

        return identifier

    def learn(
        self,
        record: ConceptRecord,
    ) -> None:
        features = [
            self.feature_id(feature)
            for feature in record.features
        ]

        units = (
            self.substrate.frozen_recursive_representation(
                record.concept
            )
            if self.recursive
            else self.substrate.base_stream(
                record.concept
            )
        )

        for unit_id in set(units):
            self.unit_document_counts[
                unit_id
            ] += 1

            for feature_id in features:
                self.unit_feature_counts[
                    unit_id
                ][feature_id] += 1

    def predict(
        self,
        word: str,
        k: int,
    ) -> list[str]:
        units = (
            self.substrate.frozen_recursive_representation(
                word
            )
            if self.recursive
            else self.substrate.base_stream(
                word
            )
        )

        scores = Counter()

        for unit_id in set(units):
            denominator = max(
                1,
                self.unit_document_counts.get(
                    unit_id,
                    0,
                ),
            )

            for feature_id, count in (
                self.unit_feature_counts.get(
                    unit_id,
                    Counter(),
                ).items()
            ):
                scores[feature_id] += (
                    count / denominator
                )

        reverse = {
            identifier: feature
            for feature, identifier
            in self.feature_ids.items()
        }

        ranked = sorted(
            scores.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        return [
            reverse[identifier]
            for identifier, _score
            in ranked[:k]
        ]

    def lexical_reuse_rate(
        self,
        word: str,
    ) -> float:
        if self.recursive:
            units = set(
                self.substrate.frozen_recursive_representation(
                    word
                )
            )

            known = sum(
                unit_id
                in self.unit_document_counts
                for unit_id in units
            )
        else:
            primitive_units = set(
                width1_units(word)
            )

            known = sum(
                unit in self.substrate.unit_ids
                for unit in primitive_units
            )

            units = primitive_units

        return (
            known
            / max(
                1,
                len(units),
            )
        )


# ---------------------------------------------------------------------------
# Metrics
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
        tp
        / len(predicted_set)
    )

    recall = (
        tp
        / max(
            1,
            len(actual),
        )
    )

    f1 = (
        0.0
        if precision + recall == 0.0
        else (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
        )
    )

    return precision, recall, f1


def evaluate(
    model: FormFeatureModel,
    records: list[ConceptRecord],
    label: str,
) -> dict[str, float]:
    precision = []
    recall = []
    f1 = []
    reuse = []

    for record in records:
        predicted = model.predict(
            record.concept,
            TOP_K,
        )

        p, r, score = prf(
            predicted,
            record.features,
        )

        precision.append(p)
        recall.append(r)
        f1.append(score)
        reuse.append(
            model.lexical_reuse_rate(
                record.concept
            )
        )

    result = {
        "precision_at_k": sum(precision) / max(1, len(precision)),
        "recall_at_k": sum(recall) / max(1, len(recall)),
        "f1_at_k": sum(f1) / max(1, len(f1)),
        "lexical_reuse": sum(reuse) / max(1, len(reuse)),
    }

    print(
        f"=== V92 {label} ==="
    )

    for key, value in result.items():
        print(
            f"{key:20s}: {value}"
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

    rng.shuffle(feature_sets)

    return [
        ConceptRecord(
            concept=record.concept,
            features=set(features),
            categories=set(record.categories),
        )
        for record, features in zip(
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
        "=== V92 FULL MATCHED-CONCEPT FORM -> SEMANTICS ==="
    )
    print(
        "Compare width-1 lexical units vs recursively discovered lexical units."
    )
    print(
        "ASCII-safe output for Windows consoles."
    )
    print()

    dictionary = load_dictionary(
        DICTIONARY_PATH
    )

    semantics = load_semantics(
        SEMANTICS_PATH
    )

    matched = [
        record
        for record in semantics.values()
        if record.concept in dictionary
    ]

    if len(matched) < 30:
        raise RuntimeError(
            f"Too few exact matches: {len(matched)}"
        )

    train, validation, test = split_records(
        matched
    )

    print(
        "dictionary_words :",
        len(dictionary),
    )
    print(
        "semantic_concepts:",
        len(semantics),
    )
    print(
        "matched_concepts :",
        len(matched),
    )
    print(
        "train            :",
        len(train),
    )
    print(
        "validation       :",
        len(validation),
    )
    print(
        "test             :",
        len(test),
    )
    print()

    # ---------------------------------------------------------------
    # Build lexical substrate on TRAIN only.
    # ---------------------------------------------------------------

    substrate = LexicalAssemblySubstrate()

    train_words = [
        record.concept
        for record in train
    ]

    substrate.train(
        train_words
    )

    print()
    print(
        "=== LEXICAL SUBSTRATE ==="
    )
    print(
        "primitive_units:",
        sum(
            level == 0
            for level
            in substrate.unit_level.values()
        ),
    )
    print(
        "all_units:",
        substrate.next_id,
    )
    print()

    # ---------------------------------------------------------------
    # Baseline: width-1 units.
    # ---------------------------------------------------------------

    baseline = FormFeatureModel(
        substrate,
        recursive=False,
    )

    for record in train:
        baseline.learn(record)

    # ---------------------------------------------------------------
    # Recursive lexical model.
    # ---------------------------------------------------------------

    recursive = FormFeatureModel(
        substrate,
        recursive=True,
    )

    for record in train:
        recursive.learn(record)

    # ---------------------------------------------------------------
    # REAL alignment, held-out.
    # ---------------------------------------------------------------

    baseline_validation = evaluate(
        baseline,
        validation,
        "WIDTH-1 VALIDATION",
    )

    baseline_test = evaluate(
        baseline,
        test,
        "WIDTH-1 TEST",
    )

    recursive_validation = evaluate(
        recursive,
        validation,
        "RECURSIVE VALIDATION",
    )

    recursive_test = evaluate(
        recursive,
        test,
        "RECURSIVE TEST",
    )

    # ---------------------------------------------------------------
    # ONE shuffled control for recursive model.
    # ---------------------------------------------------------------

    shuffled = shuffled_records(
        train,
        SEED,
    )

    shuffled_model = FormFeatureModel(
        substrate,
        recursive=True,
    )

    for record in shuffled:
        shuffled_model.learn(record)

    shuffled_test = evaluate(
        shuffled_model,
        test,
        "RECURSIVE SHUFFLED TEST",
    )

    # ---------------------------------------------------------------
    # Comparison.
    # ---------------------------------------------------------------

    print(
        "=== V92 CROSS-MODAL COMPARISON ==="
    )

    for metric in (
        "precision_at_k",
        "recall_at_k",
        "f1_at_k",
    ):
        b = baseline_test[metric]
        r = recursive_test[metric]
        s = shuffled_test[metric]

        print(
            f"{metric:18s} "
            f"width1={b:.6f} "
            f"recursive={r:.6f} "
            f"shuffled={s:.6f} "
            f"recursive_delta={r - b:+.6f} "
            f"recursive_lift={r / max(1e-9, s):.3f}x"
        )

    print(
        "width1_lexical_reuse:",
        baseline_test[
            "lexical_reuse"
        ],
    )

    print(
        "recursive_lexical_reuse:",
        recursive_test[
            "lexical_reuse"
        ],
    )

    print()
    print(
        "=== V92 INTERPRETATION ==="
    )
    print(
        "recursive > width1 means higher-order lexical structure "
        "adds form->feature information beyond isolated local units."
    )
    print(
        "recursive > shuffled means the effect is above a "
        "word/feature marginal control."
    )

    print()
    print(
        "elapsed_seconds:",
        f"{time.perf_counter() - start:.2f}",
    )

    print(
        "=== V92 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
