from __future__ import annotations

"""
V91 - LEXICON <-> SEMANTICS CROSS-MODAL TRANSFER

Question
--------
Can the graph discover reusable structure linking WORD FORM to independently
elicited SEMANTIC FEATURES?

We already have:
    lexical width-1 local binding reuse
    semantic feature / combination reuse

V91 connects the two spaces and tests whether form-to-meaning associations
generalize to concept-disjoint words.

Pipeline
--------
1. Load dictionary.csv and semantics.csv.
2. Match semantic basic-level concepts to exact dictionary words.
3. Split matched concepts deterministically into train / validation / test.
4. Build lexical width-1 binding units from training words.
5. Build semantic feature nodes from training concepts.
6. Learn cross-modal links:

       lexical binding unit -> semantic feature

   from TRAIN concepts only.
7. For each TEST concept:
       use the lexical units of its word
       score semantic features through learned cross-modal links
       compare predictions with the human-elicited features for that concept.
8. Run one permutation control:
       shuffle which semantic concept is attached to each TRAIN word,
       repeat the same predictor,
       compare real vs shuffled performance.

Important
---------
This is NOT a claim that spelling "contains" meaning.
The experiment asks whether independently observed word-form fragments have
statistically reusable associations with independently elicited semantic
features.

No semantic labels are invented.
No test concepts are used during training.
No threshold is tuned on TEST.

Primary metrics
---------------
    lexical unit reuse on TEST
    feature prediction precision@k
    feature prediction recall@k
    feature F1@k
    real-vs-shuffled lift

The lexical representation is the already validated width-1 form:
    previous character | current character | next character
"""

import csv
import hashlib
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Paths / split
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DICTIONARY_PATH = ROOT / "data" / "dictionary.csv"
SEMANTICS_PATH = ROOT / "data" / "semantics.csv"

TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15

TOP_K = 10
SEED = 9173


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

    if not words:
        raise RuntimeError(
            "dictionary.csv produced zero words"
        )

    return words


def load_semantics(path: Path) -> list[SemanticRow]:
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

            if concept and feature:
                rows.append(
                    SemanticRow(
                        concept=concept,
                        category=category,
                        feature=feature,
                    )
                )

    return rows


def group_semantics(
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

        record.features.add(
            row.feature
        )

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
    matched_words: set[str],
):
    matched = [
        record
        for record in concepts.values()
        if record.concept in matched_words
    ]

    matched.sort(
        key=lambda record: (
            stable_rank(record.concept),
            record.concept,
        )
    )

    n = len(matched)

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
        matched[:train_end],
        matched[train_end:validation_end],
        matched[validation_end:],
    )


# ---------------------------------------------------------------------------
# Lexical width-1 representation
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
# Cross-modal learner
# ---------------------------------------------------------------------------

class CrossModalModel:
    """
    Learns:
        lexical local unit -> semantic feature

    The mapping is just a frequency graph.

    A lexical unit receives votes for the semantic features that occur on the
    same concept during training.
    """

    def __init__(self) -> None:
        self.unit_ids: dict[
            tuple[str, str, str],
            int,
        ] = {}

        self.feature_ids: dict[
            str,
            int,
        ] = {}

        self.unit_feature_counts: dict[
            int,
            Counter[int],
        ] = defaultdict(Counter)

        self.unit_document_counts: Counter[int] = Counter()

    def unit_id(
        self,
        unit: tuple[str, str, str],
        learn: bool,
    ) -> int:
        existing = self.unit_ids.get(unit)

        if existing is not None:
            return existing

        if not learn:
            return -1

        identifier = len(self.unit_ids)
        self.unit_ids[unit] = identifier

        return identifier

    def feature_id(
        self,
        feature: str,
        learn: bool,
    ) -> int:
        existing = self.feature_ids.get(feature)

        if existing is not None:
            return existing

        if not learn:
            return -1

        identifier = len(self.feature_ids)
        self.feature_ids[feature] = identifier

        return identifier

    def learn(
        self,
        record: ConceptRecord,
    ) -> None:
        features = [
            self.feature_id(
                feature,
                learn=True,
            )
            for feature in record.features
        ]

        units = [
            self.unit_id(
                unit,
                learn=True,
            )
            for unit in width1_units(
                record.concept
            )
        ]

        # A feature is counted once per lexical unit per concept.
        for unit_id in set(units):
            self.unit_document_counts[
                unit_id
            ] += 1

            for feature_id in features:
                self.unit_feature_counts[
                    unit_id
                ][feature_id] += 1

    def score_features(
        self,
        word: str,
    ) -> Counter[int]:
        scores = Counter()

        for unit in set(
            width1_units(word)
        ):
            unit_id = self.unit_ids.get(unit)

            if unit_id is None:
                continue

            for feature_id, count in (
                self.unit_feature_counts[
                    unit_id
                ].items()
            ):
                # Normalize by how broadly the lexical unit appears.
                # This keeps common generic local units from dominating solely
                # because they occur in many training concepts.
                denominator = max(
                    1,
                    self.unit_document_counts[
                        unit_id
                    ],
                )

                scores[feature_id] += (
                    count / denominator
                )

        return scores

    def predict(
        self,
        word: str,
        k: int,
    ) -> list[str]:
        scores = self.score_features(
            word
        )

        ranked = sorted(
            scores.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        reverse = {
            identifier: feature
            for feature, identifier
            in self.feature_ids.items()
        }

        return [
            reverse[identifier]
            for identifier, _score in ranked[:k]
        ]

    def lexical_reuse_rate(
        self,
        word: str,
    ) -> float:
        units = set(
            width1_units(word)
        )

        if not units:
            return 0.0

        known = sum(
            unit in self.unit_ids
            for unit in units
        )

        return (
            known
            / len(units)
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_recall_f1(
    predicted: list[str],
    actual: set[str],
) -> tuple[float, float, float]:
    predicted_set = set(predicted)

    if not predicted_set:
        return 0.0, 0.0, 0.0

    true_positive = len(
        predicted_set & actual
    )

    precision = (
        true_positive
        / len(predicted_set)
    )

    recall = (
        true_positive
        / max(
            1,
            len(actual),
        )
    )

    if (
        precision
        + recall
    ) == 0.0:
        f1 = 0.0
    else:
        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    return precision, recall, f1


def evaluate_model(
    model: CrossModalModel,
    records: list[ConceptRecord],
    label: str,
) -> dict[str, float]:
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_reuse = 0.0

    rows = []

    for record in records:
        predicted = model.predict(
            record.concept,
            TOP_K,
        )

        precision, recall, f1 = (
            precision_recall_f1(
                predicted,
                record.features,
            )
        )

        reuse = model.lexical_reuse_rate(
            record.concept
        )

        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_reuse += reuse

        rows.append(
            (
                record.concept,
                precision,
                recall,
                f1,
                reuse,
                predicted,
                record.features,
            )
        )

    count = max(
        1,
        len(records),
    )

    result = {
        "concepts": float(len(records)),
        "precision_at_k": (
            total_precision / count
        ),
        "recall_at_k": (
            total_recall / count
        ),
        "f1_at_k": (
            total_f1 / count
        ),
        "lexical_unit_reuse": (
            total_reuse / count
        ),
    }

    print(
        f"=== V91 {label} ==="
    )

    for key, value in result.items():
        print(
            f"{key:24s}: {value}"
        )

    print()

    # A few examples make the cross-modal graph inspectable.
    print(
        "--- examples ---"
    )

    for (
        concept,
        precision,
        recall,
        f1,
        reuse,
        predicted,
        actual,
    ) in rows[:10]:
        print(
            f"{concept:18s} "
            f"reuse={reuse:.3f} "
            f"p={precision:.3f} "
            f"r={recall:.3f} "
            f"f1={f1:.3f} "
            f"pred={predicted[:5]}"
        )

    print()

    return result


# ---------------------------------------------------------------------------
# One permutation control
# ---------------------------------------------------------------------------

def shuffled_training_records(
    records: list[ConceptRecord],
    seed: int,
) -> list[ConceptRecord]:
    """
    Keep each word and each feature vector intact, but randomly permute the
    feature vectors across TRAIN words.

    This destroys genuine word-form -> semantic-feature alignment while
    preserving:
        lexical distribution
        feature distribution
        number of features/concept
    """
    rng = random.Random(seed)

    feature_sets = [
        frozenset(record.features)
        for record in records
    ]

    rng.shuffle(
        feature_sets
    )

    shuffled = []

    for record, feature_set in zip(
        records,
        feature_sets,
    ):
        shuffled.append(
            ConceptRecord(
                concept=record.concept,
                categories=set(
                    record.categories
                ),
                features=set(
                    feature_set
                ),
            )
        )

    return shuffled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start = time.perf_counter()

    print(
        "=== V91 LEXICON <-> SEMANTICS CROSS-MODAL TRANSFER ==="
    )
    print(
        "Question: do reusable lexical structures carry reusable "
        "form->semantic-feature information?"
    )
    print()

    dictionary = load_dictionary(
        DICTIONARY_PATH
    )

    semantic_rows = load_semantics(
        SEMANTICS_PATH
    )

    concepts = group_semantics(
        semantic_rows
    )

    matched_words = {
        concept
        for concept in concepts
        if concept in dictionary
    }

    train, validation, test = (
        split_concepts(
            concepts,
            matched_words,
        )
    )

    print(
        "dictionary_words :",
        len(dictionary),
    )
    print(
        "semantic_rows    :",
        len(semantic_rows),
    )
    print(
        "semantic_concepts:",
        len(concepts),
    )
    print(
        "matched_concepts :",
        len(matched_words),
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

    if len(matched_words) < 30:
        raise RuntimeError(
            "Too few exact word/concept matches for cross-modal test."
        )

    # ---------------------------------------------------------------
    # REAL ALIGNMENT
    # ---------------------------------------------------------------

    real = CrossModalModel()

    for record in train:
        real.learn(
            record
        )

    print(
        "=== REAL CROSS-MODAL MODEL ==="
    )
    print(
        "lexical_units :",
        len(real.unit_ids),
    )
    print(
        "semantic_features:",
        len(real.feature_ids),
    )
    print()

    validation_result = evaluate_model(
        real,
        validation,
        "VALIDATION REAL",
    )

    test_result = evaluate_model(
        real,
        test,
        "TEST REAL",
    )

    # ---------------------------------------------------------------
    # PERMUTATION CONTROL
    # ---------------------------------------------------------------

    shuffled_records = (
        shuffled_training_records(
            train,
            SEED,
        )
    )

    shuffled = CrossModalModel()

    for record in shuffled_records:
        shuffled.learn(
            record
        )

    print(
        "=== SHUFFLED CONTROL ==="
    )
    print(
        "lexical_units :",
        len(shuffled.unit_ids),
    )
    print(
        "semantic_features:",
        len(shuffled.feature_ids),
    )
    print()

    shuffled_result = evaluate_model(
        shuffled,
        test,
        "TEST SHUFFLED",
    )

    # ---------------------------------------------------------------
    # FORM -> MEANING LIFT
    # ---------------------------------------------------------------

    print(
        "=== V91 FORM->MEANING LIFT ==="
    )

    for metric in (
        "precision_at_k",
        "recall_at_k",
        "f1_at_k",
    ):
        real_value = test_result[metric]
        shuffled_value = shuffled_result[metric]

        lift = (
            real_value
            / max(
                1e-9,
                shuffled_value,
            )
        )

        delta = (
            real_value
            - shuffled_value
        )

        print(
            f"{metric:18s} "
            f"real={real_value:.6f} "
            f"shuffle={shuffled_value:.6f} "
            f"delta={delta:+.6f} "
            f"lift={lift:.3f}x"
        )

    print(
        "test_lexical_unit_reuse:",
        test_result[
            "lexical_unit_reuse"
        ],
    )

    print()

    print(
        "=== V91 INTERPRETATION ==="
    )
    print(
        "Real > shuffled means lexical local structure carries "
        "information about independently elicited semantic features "
        "beyond the corpus marginal distributions."
    )
    print(
        "This is a statistical form->meaning result, not a claim that "
        "word spelling intrinsically determines semantics."
    )

    print()
    print(
        "elapsed_seconds:",
        f"{time.perf_counter() - start:.2f}",
    )

    print(
        "=== V91 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
