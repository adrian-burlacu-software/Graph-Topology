from __future__ import annotations

"""
V93 — FULL LEXICON + SEMANTIC ANCHOR TRANSFER

This is the big-jump version.

The COMPLETE dictionary is used to build the lexical substrate:
    data/dictionary.csv  (~4925 words)

The COMPLETE semantic corpus is used as an independently observed semantic
anchor set:
    data/semantics.csv

We do NOT restrict lexical learning to the exact semantic intersection.

Architecture
------------
ALL DICTIONARY WORDS
    ↓
width-1 lexical units
    ↓
recursive lexical assemblies
    ↓
full lexical graph

SEMANTIC ANCHORS
    ↓
human-elicited feature sets
    ↓
cross-modal links from lexical units / assemblies
       to semantic features

Evaluation
----------
Only semantic concepts that have an exact spelling match in the dictionary
can be direct anchors for form -> feature supervision.

That is a limitation of the available paired data, but the lexical substrate
is still learned from the ENTIRE dictionary, so semantic prediction is being
performed against a much richer lexical graph than V92.

We compare:

    A) width-1 lexical representation
    B) recursively discovered lexical representation
    C) shuffled word <-> semantic-anchor alignment

We also evaluate:

    * feature precision / recall / F1
    * semantic-neighborhood coherence
    * lexical reuse coverage
    * recursive vs width-1 delta
    * real vs shuffled lift

The main question is NOT:
    "Does spelling inherently determine meaning?"

It is:
    "Does a recursively compressed lexical topology contain reusable
     structure that transfers statistically to independently elicited
     semantic features better than local lexical fragments or chance
     alignment?"

No semantic labels are synthesized.
"""

import csv
import hashlib
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

MIN_OCCURRENCES = 2
MAX_LEXICAL_LEVELS = 10


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


def split_anchor_records(
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
# Lexical representation
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


class FullLexiconSubstrate:
    """
    Learns lexical structure from ALL dictionary words.

    Level 0:
        width-1 local units.

    Higher levels:
        recursively discovered unordered pair assemblies.

    The semantic corpus is completely absent from this stage.
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

        self.unit_level: dict[int, int] = {}

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
    ) -> tuple[list[list[int]], int, int]:
        """
        Discover recurring unordered adjacent transitions.

        The entire dictionary contributes evidence.
        """
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

            identifier = self.next_id
            self.next_id += 1

            self.assembly_ids[key] = identifier
            self.unit_level[identifier] = level

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

                    if assembly is not None and key in recurring:
                        output.append(assembly)
                        i += 2
                        continue

                output.append(stream[i])
                i += 1

            next_streams.append(output)

        return (
            next_streams,
            created,
            len(recurring),
        )

    def train(
        self,
        words: list[str],
    ) -> None:
        streams = [
            self.base_stream(word)
            for word in words
        ]

        print(
            "FULL LEXICON LEVEL 0 "
            f"primitive_units={len(self.primitive_ids)}",
            flush=True,
        )

        for level in range(
            1,
            MAX_LEXICAL_LEVELS + 1,
        ):
            streams, created, recurring = (
                self.discover_level(
                    streams,
                    level,
                )
            )

            print(
                f"FULL LEXICON level={level:2d} "
                f"recurring={recurring:6d} "
                f"new_units={created:6d} "
                f"stream_units={sum(len(s) for s in streams):7d}",
                flush=True,
            )

            if created == 0:
                break

    def frozen_recursive_units(
        self,
        word: str,
    ) -> list[int]:
        """
        Apply the trained lexical hierarchy without creating anything new.
        """
        stream = self.base_stream(word)

        while True:
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

                    assembly = self.assembly_ids.get(key)

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


# ---------------------------------------------------------------------------
# Cross-modal mapping
# ---------------------------------------------------------------------------

class FormFeatureAssociator:
    """
    Learn lexical-unit -> semantic-feature evidence from semantic anchors.

    The lexical substrate itself was trained on ALL dictionary words.
    """

    def __init__(
        self,
        substrate: FullLexiconSubstrate,
        recursive: bool,
    ) -> None:
        self.substrate = substrate
        self.recursive = recursive

        self.feature_ids: dict[str, int] = {}
        self.feature_by_id: dict[int, str] = {}

        self.unit_feature_counts: dict[
            int,
            Counter[int],
        ] = defaultdict(Counter)

        self.unit_anchor_counts: Counter[int] = Counter()

    def feature_id(
        self,
        feature: str,
    ) -> int:
        existing = self.feature_ids.get(feature)

        if existing is not None:
            return existing

        identifier = len(self.feature_ids)

        self.feature_ids[feature] = identifier
        self.feature_by_id[identifier] = feature

        return identifier

    def representation(
        self,
        word: str,
    ) -> list[int]:
        if self.recursive:
            return self.substrate.frozen_recursive_units(word)

        return self.substrate.base_stream(word)

    def learn(
        self,
        record: ConceptRecord,
    ) -> None:
        features = [
            self.feature_id(feature)
            for feature in record.features
        ]

        units = set(
            self.representation(
                record.concept
            )
        )

        for unit_id in units:
            self.unit_anchor_counts[
                unit_id
            ] += 1

            for feature_id in features:
                self.unit_feature_counts[
                    unit_id
                ][feature_id] += 1

    def score(
        self,
        word: str,
    ) -> Counter[int]:
        scores = Counter()

        for unit_id in set(
            self.representation(word)
        ):
            total_anchors = max(
                1,
                self.unit_anchor_counts.get(
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
                    count / total_anchors
                )

        return scores

    def predict(
        self,
        word: str,
        k: int,
    ) -> list[str]:
        scores = self.score(word)

        ranked = sorted(
            scores.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        return [
            self.feature_by_id[feature_id]
            for feature_id, _score in ranked[:k]
        ]

    def representation_coverage(
        self,
        word: str,
    ) -> float:
        units = set(
            self.representation(word)
        )

        known = sum(
            unit_id
            in self.unit_anchor_counts
            for unit_id in units
        )

        return (
            known
            / max(1, len(units))
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
        if precision + recall == 0
        else (
            2
            * precision
            * recall
            / (precision + recall)
        )
    )

    return precision, recall, f1


def evaluate(
    model: FormFeatureAssociator,
    records: list[ConceptRecord],
    label: str,
) -> dict[str, float]:
    p_values = []
    r_values = []
    f1_values = []
    coverage = []

    for record in records:
        predicted = model.predict(
            record.concept,
            TOP_K,
        )

        p, r, f1 = prf(
            predicted,
            record.features,
        )

        p_values.append(p)
        r_values.append(r)
        f1_values.append(f1)
        coverage.append(
            model.representation_coverage(
                record.concept
            )
        )

    result = {
        "concepts": float(len(records)),
        "precision_at_k": sum(p_values)
        / max(1, len(p_values)),
        "recall_at_k": sum(r_values)
        / max(1, len(r_values)),
        "f1_at_k": sum(f1_values)
        / max(1, len(f1_values)),
        "representation_coverage": sum(coverage)
        / max(1, len(coverage)),
    }

    print(
        f"=== V93 {label} ==="
    )

    for key, value in result.items():
        print(
            f"{key:28s}: {value}"
        )

    print()

    return result


def shuffled_training(
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
        "=== V93 FULL LEXICON SEMANTIC ANCHOR TRANSFER ==="
    )
    print(
        "Full dictionary builds lexical structure."
    )
    print(
        "Semantic concepts supply independent feature anchors."
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

    train, validation, test = (
        split_anchor_records(
            matched
        )
    )

    print(
        "dictionary_words      :",
        len(dictionary),
    )
    print(
        "semantic_concepts     :",
        len(semantics),
    )
    print(
        "exact_anchor_matches  :",
        len(matched),
    )
    print(
        "unmatched_semantics   :",
        len(semantics) - len(matched),
    )
    print(
        "train_anchors         :",
        len(train),
    )
    print(
        "validation_anchors    :",
        len(validation),
    )
    print(
        "test_anchors          :",
        len(test),
    )
    print()

    # ---------------------------------------------------------------
    # BIG LEAP:
    # Build lexical substrate from ALL dictionary words.
    # ---------------------------------------------------------------

    substrate = FullLexiconSubstrate()

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
    print(
        "dictionary_words :",
        len(all_words),
    )
    print(
        "primitive_units  :",
        len(substrate.primitive_ids),
    )
    print(
        "all_units        :",
        substrate.next_id,
    )
    print()

    # ---------------------------------------------------------------
    # Train width-1 and recursive cross-modal associators on TRAIN anchors.
    # ---------------------------------------------------------------

    width1 = FormFeatureAssociator(
        substrate,
        recursive=False,
    )

    recursive = FormFeatureAssociator(
        substrate,
        recursive=True,
    )

    for record in train:
        width1.learn(record)
        recursive.learn(record)

    print(
        "semantic_train_anchors:",
        len(train),
    )
    print(
        "width1_feature_vocab  :",
        len(width1.feature_ids),
    )
    print(
        "recursive_feature_vocab:",
        len(recursive.feature_ids),
    )
    print()

    # ---------------------------------------------------------------
    # REAL TEST
    # ---------------------------------------------------------------

    width1_validation = evaluate(
        width1,
        validation,
        "WIDTH1 VALIDATION",
    )

    width1_test = evaluate(
        width1,
        test,
        "WIDTH1 TEST",
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
    # SHUFFLED CONTROL
    # ---------------------------------------------------------------

    shuffled_records = shuffled_training(
        train,
        SEED,
    )

    shuffled = FormFeatureAssociator(
        substrate,
        recursive=True,
    )

    for record in shuffled_records:
        shuffled.learn(record)

    shuffled_test = evaluate(
        shuffled,
        test,
        "RECURSIVE SHUFFLED TEST",
    )

    # ---------------------------------------------------------------
    # BIG COMPARISON
    # ---------------------------------------------------------------

    print(
        "=== V93 CROSS-MODAL COMPARISON ==="
    )

    for metric in (
        "precision_at_k",
        "recall_at_k",
        "f1_at_k",
    ):
        w = width1_test[metric]
        r = recursive_test[metric]
        s = shuffled_test[metric]

        print(
            f"{metric:20s} "
            f"width1={w:.6f} "
            f"recursive={r:.6f} "
            f"shuffled={s:.6f} "
            f"rec-vs-width={r - w:+.6f} "
            f"rec-vs-shuffle={r - s:+.6f} "
            f"lift={r / max(1e-9, s):.3f}x"
        )

    print()
    print(
        "width1_representation_coverage:",
        width1_test[
            "representation_coverage"
        ],
    )
    print(
        "recursive_representation_coverage:",
        recursive_test[
            "representation_coverage"
        ],
    )

    print()
    print(
        "=== V93 INTERPRETATION ==="
    )
    print(
        "The lexical graph was trained on ALL dictionary words, "
        "not merely semantic matches."
    )
    print(
        "Real recursive > shuffled supports reusable form->feature "
        "information beyond random word/feature alignment."
    )
    print(
        "Recursive > width1 indicates higher-order lexical structure "
        "adds information beyond isolated local units."
    )

    print()
    print(
        "elapsed_seconds:",
        f"{time.perf_counter() - start:.2f}",
    )
    print(
        "=== V93 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
