from __future__ import annotations

"""
V96 — FULL LEXICON + LARGE HUMAN SEMANTIC TRANSFER

Semantic corpus:
    data/semantics-large.csv

This replaces the tiny 153-anchor bottleneck.

The file is the large human feature-production table with columns including:
    cue
    feature
    translated
    frequency_feature
    frequency_translated
    n
    normalized_feature
    normalized_translated
    pos_cue
    pos_feature
    pos_translated
    FSG
    BSG
    word_list
    school_code

Canonical semantic feature:
    translated

Canonical feature weight:
    normalized_translated

Lexical substrate:
    ALL words in data/dictionary.csv

Lexical representations:
    1. WIDTH1
    2. RECURSIVE
    3. DUAL (width1 + recursive)

Cross-modal learner:
    lexical unit -> canonical semantic feature
    weighted by human production frequency

Evaluation:
    concept/cue-disjoint train / validation / test
    real alignment
    shuffled control
    top-k precision/recall/F1
    weighted semantic recall@k
    lexical representation coverage

This is intentionally one large experiment rather than another chain of
micro-tests.

No semantic labels are invented.
No test examples are used to train the graph.
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
SEMANTICS_PATH = ROOT / "data" / "semantics-large.csv"

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
class CueRecord:
    cue: str
    features: dict[str, float] = field(default_factory=dict)
    raw_feature_forms: Counter[str] = field(default_factory=Counter)
    pos_cue: set[str] = field(default_factory=set)

    def add(
        self,
        translated: str,
        weight: float,
        raw_feature: str,
        pos_cue: str,
    ) -> None:
        self.features[translated] = (
            self.features.get(translated, 0.0)
            + weight
        )
        self.raw_feature_forms[raw_feature] += 1

        if pos_cue:
            self.pos_cue.add(pos_cue)


def parse_float(
    value: str,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_int(
    value: str,
) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def load_dictionary(
    path: Path,
) -> set[str]:
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
    """
    Uses the large table's `translated` form as the canonical feature.

    Rows with empty translated/cue are discarded.
    """
    records: dict[str, CueRecord] = {}

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "cue",
            "feature",
            "translated",
            "frequency_translated",
            "n",
            "normalized_translated",
            "pos_cue",
        }

        missing = required - set(
            reader.fieldnames or []
        )

        if missing:
            raise RuntimeError(
                "semantics-large.csv missing columns: "
                + ", ".join(sorted(missing))
            )

        row_count = 0

        for raw in reader:
            cue = raw["cue"].strip().lower()
            translated = raw["translated"].strip().lower()
            raw_feature = raw["feature"].strip().lower()
            pos_cue = raw["pos_cue"].strip().lower()

            if not cue or not translated:
                continue

            normalized = parse_float(
                raw["normalized_translated"]
            )

            frequency = parse_float(
                raw["frequency_translated"]
            )

            n = parse_int(
                raw["n"]
            )

            # Prefer the provided normalized human production frequency.
            # Fall back to frequency / n if the normalized field is absent or
            # malformed.
            weight = normalized

            if weight <= 0.0 and n > 0:
                weight = frequency / n

            if weight <= 0.0:
                continue

            record = records.get(cue)

            if record is None:
                record = CueRecord(
                    cue=cue
                )
                records[cue] = record

            record.add(
                translated=translated,
                weight=weight,
                raw_feature=raw_feature,
                pos_cue=pos_cue,
            )

            row_count += 1

    if not records:
        raise RuntimeError(
            "No usable semantic cues loaded."
        )

    print(
        "semantic_rows_used:",
        row_count,
    )

    return records


def stable_rank(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def split_cues(
    cues: list[CueRecord],
):
    ordered = sorted(
        cues,
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
    Builds the lexical graph from ALL dictionary words before semantic
    supervision is introduced.
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
                for key, count
                in occurrences.items()
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

                    output.append(
                        stream[i]
                    )
                    i += 1

                next_streams.append(
                    output
                )

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
                        output.append(
                            assembly
                        )
                        i += 2
                        changed = True
                        continue

                output.append(
                    stream[i]
                )
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
# Cross-modal weighted learner
# ---------------------------------------------------------------------------

class WeightedFormFeatureModel:
    """
    Maps lexical units -> weighted semantic features.

    Training weights come directly from human semantic production frequency.

    For a lexical unit u and feature f:

        accumulated_weight[u, f]

    The score is normalized by the total semantic mass attached to the unit,
    preventing globally common features from dominating every unit equally.
    """

    def __init__(
        self,
        substrate: FullLexicalSubstrate,
        mode: str,
    ) -> None:
        if mode not in {
            "width1",
            "recursive",
            "dual",
        }:
            raise ValueError(mode)

        self.substrate = substrate
        self.mode = mode

        self.unit_feature_mass: dict[
            int,
            Counter[str],
        ] = defaultdict(Counter)

        self.unit_total_mass: Counter[int] = Counter()

        self.feature_global_mass: Counter[str] = Counter()

        self.total_mass = 0.0

        self.anchor_count = 0

    def representation(
        self,
        word: str,
    ) -> list[int]:
        if self.mode == "width1":
            return self.substrate.base_stream(word)

        if self.mode == "recursive":
            return self.substrate.recursive_units(word)

        return self.substrate.dual_units(word)

    def learn(
        self,
        record: CueRecord,
    ) -> None:
        self.anchor_count += 1

        units = set(
            self.representation(
                record.cue
            )
        )

        for feature, weight in (
            record.features.items()
        ):
            self.feature_global_mass[
                feature
            ] += weight

            self.total_mass += weight

            for unit_id in units:
                self.unit_feature_mass[
                    unit_id
                ][feature] += weight

                self.unit_total_mass[
                    unit_id
                ] += weight

    def score(
        self,
        word: str,
    ) -> Counter[str]:
        scores = Counter()

        for unit_id in set(
            self.representation(word)
        ):
            unit_mass = self.unit_total_mass.get(
                unit_id,
                0.0,
            )

            if unit_mass <= 0.0:
                continue

            for feature, mass in (
                self.unit_feature_mass.get(
                    unit_id,
                    Counter(),
                ).items()
            ):
                # Conditional semantic weight for the unit.
                conditional = (
                    mass
                    / unit_mass
                )

                global_prior = (
                    self.feature_global_mass[
                        feature
                    ]
                    / max(
                        1e-12,
                        self.total_mass,
                    )
                )

                # Lift above the global feature prior.
                lift = (
                    conditional
                    / max(
                        1e-12,
                        global_prior,
                    )
                )

                scores[feature] += (
                    math.log(
                        max(
                            1.0,
                            lift,
                        )
                    )
                )

        return scores

    def predict(
        self,
        word: str,
        k: int = TOP_K,
    ) -> list[str]:
        ranked = sorted(
            self.score(word).items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        return [
            feature
            for feature, _score
            in ranked[:k]
        ]

    def coverage(
        self,
        word: str,
    ) -> float:
        units = set(
            self.representation(word)
        )

        if not units:
            return 0.0

        known = sum(
            self.unit_total_mass.get(
                unit_id,
                0.0,
            )
            > 0.0
            for unit_id in units
        )

        return (
            known
            / len(units)
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def weighted_actual_features(
    record: CueRecord,
) -> set[str]:
    return set(
        record.features
    )


def precision_recall_f1(
    predicted: list[str],
    actual: set[str],
):
    predicted_set = set(predicted)

    if not predicted_set:
        return 0.0, 0.0, 0.0

    tp = len(
        predicted_set
        & actual
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

    if precision + recall == 0.0:
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

    return (
        precision,
        recall,
        f1,
    )


def weighted_recall_at_k(
    predicted: list[str],
    record: CueRecord,
) -> float:
    total = sum(
        record.features.values()
    )

    if total <= 0.0:
        return 0.0

    recovered = sum(
        record.features.get(
            feature,
            0.0,
        )
        for feature in predicted
    )

    return (
        recovered
        / total
    )


def evaluate(
    model: WeightedFormFeatureModel,
    records: list[CueRecord],
    label: str,
) -> dict[str, float]:
    p_values = []
    r_values = []
    f1_values = []
    weighted_recall = []
    coverage = []

    for record in records:
        predicted = model.predict(
            record.cue
        )

        actual = weighted_actual_features(
            record
        )

        precision, recall, f1 = (
            precision_recall_f1(
                predicted,
                actual,
            )
        )

        p_values.append(
            precision
        )
        r_values.append(
            recall
        )
        f1_values.append(
            f1
        )
        weighted_recall.append(
            weighted_recall_at_k(
                predicted,
                record,
            )
        )
        coverage.append(
            model.coverage(
                record.cue
            )
        )

    count = max(
        1,
        len(records),
    )

    result = {
        "cues": float(len(records)),
        "precision_at_k": (
            sum(p_values)
            / count
        ),
        "recall_at_k": (
            sum(r_values)
            / count
        ),
        "f1_at_k": (
            sum(f1_values)
            / count
        ),
        "weighted_recall_at_k": (
            sum(weighted_recall)
            / count
        ),
        "lexical_coverage": (
            sum(coverage)
            / count
        ),
    }

    print(
        f"=== V96 {label} ==="
    )

    for key, value in result.items():
        print(
            f"{key:26s}: {value}"
        )

    print()

    return result


def shuffle_semantic_assignments(
    records: list[CueRecord],
    seed: int,
) -> list[CueRecord]:
    """
    Preserve the entire feature distribution and feature weights, but destroy
    cue -> feature alignment.
    """
    rng = random.Random(seed)

    payloads = [
        (
            dict(record.features),
            set(),
            set(record.pos_cue),
        )
        for record in records
    ]

    rng.shuffle(
        payloads
    )

    shuffled = []

    for record, (
        features,
        categories,
        pos_cue,
    ) in zip(
        records,
        payloads,
    ):
        shuffled.append(
            CueRecord(
                cue=record.cue,
                features=features,
                                pos_cue=pos_cue,
            )
        )

    return shuffled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start = time.perf_counter()

    print(
        "=== V96 FULL LEXICON + LARGE SEMANTIC CORPUS ==="
    )
    print(
        "Human feature weights are retained."
    )
    print(
        "Full dictionary builds lexical substrate."
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

    unmatched = [
        record
        for record in semantics.values()
        if record.cue not in dictionary
    ]

    train, validation, test = split_cues(
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
        "matched_dictionary_cues:",
        len(matched),
    )
    print(
        "unmatched_semantic_cues:",
        len(unmatched),
    )
    print(
        "train_cues:",
        len(train),
    )
    print(
        "validation_cues:",
        len(validation),
    )
    print(
        "test_cues:",
        len(test),
    )
    print()

    if len(matched) < 1000:
        raise RuntimeError(
            "Expected thousands of matched semantic cues; "
            f"only {len(matched)} matched."
        )

    # ---------------------------------------------------------------
    # FULL LEXICAL GRAPH
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
        "=== LEXICAL SUBSTRATE ==="
    )

    for key, value in substrate.stats().items():
        print(
            f"{key:24s}: {value}"
        )

    print()

    # ---------------------------------------------------------------
    # REAL MODELS
    # ---------------------------------------------------------------

    width1 = WeightedFormFeatureModel(
        substrate,
        "width1",
    )

    recursive = WeightedFormFeatureModel(
        substrate,
        "recursive",
    )

    dual = WeightedFormFeatureModel(
        substrate,
        "dual",
    )

    for record in train:
        width1.learn(record)
        recursive.learn(record)
        dual.learn(record)

    # ---------------------------------------------------------------
    # TEST REAL ALIGNMENT
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

    dual_validation = evaluate(
        dual,
        validation,
        "DUAL VALIDATION",
    )

    dual_test = evaluate(
        dual,
        test,
        "DUAL TEST",
    )

    # ---------------------------------------------------------------
    # SHUFFLED CONTROL
    # ---------------------------------------------------------------

    shuffled_dual = WeightedFormFeatureModel(
        substrate,
        "dual",
    )

    shuffled_train = shuffle_semantic_assignments(
        train,
        SEED,
    )

    for record in shuffled_train:
        shuffled_dual.learn(record)

    shuffled_test = evaluate(
        shuffled_dual,
        test,
        "SHUFFLED DUAL TEST",
    )

    # ---------------------------------------------------------------
    # Big comparison
    # ---------------------------------------------------------------

    print(
        "=== V96 CROSS-MODAL COMPARISON ==="
    )

    for metric in (
        "precision_at_k",
        "recall_at_k",
        "f1_at_k",
        "weighted_recall_at_k",
    ):
        w = width1_test[metric]
        r = recursive_test[metric]
        d = dual_test[metric]
        s = shuffled_test[metric]

        print(
            f"{metric:24s} "
            f"width1={w:.6f} "
            f"recursive={r:.6f} "
            f"dual={d:.6f} "
            f"shuffle={s:.6f}"
        )

        print(
            f"{'':24s}"
            f"dual-width1={d-w:+.6f} "
            f"dual-shuffle={d-s:+.6f} "
            f"lift={d/max(1e-9,s):.3f}x"
        )

    print()
    print(
        "=== V96 COVERAGE ==="
    )

    print(
        "width1:",
        width1_test[
            "lexical_coverage"
        ],
    )

    print(
        "recursive:",
        recursive_test[
            "lexical_coverage"
        ],
    )

    print(
        "dual:",
        dual_test[
            "lexical_coverage"
        ],
    )

    print()
    print(
        "=== V96 CORPUS SCALE ==="
    )

    print(
        "dictionary_words:",
        len(dictionary),
    )

    print(
        "semantic_cues:",
        len(semantics),
    )

    print(
        "matched_cues:",
        len(matched),
    )

    print(
        "train_cues:",
        len(train),
    )

    print(
        "test_cues:",
        len(test),
    )

    print()
    print(
        "=== V96 INTERPRETATION ==="
    )

    print(
        "The lexical graph uses the entire dictionary."
    )

    print(
        "The semantic graph uses the large human feature-production corpus."
    )

    print(
        "translated is the canonical semantic feature; "
        "human production frequency is retained as edge weight."
    )

    print(
        "Real > shuffled is the cross-modal signal."
    )

    print(
        "Dual > width1 is evidence that reusable higher-order lexical "
        "structure adds semantic information beyond local fragments."
    )

    print()
    print(
        "elapsed_seconds:",
        f"{time.perf_counter() - start:.2f}",
    )

    print(
        "=== V96 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
