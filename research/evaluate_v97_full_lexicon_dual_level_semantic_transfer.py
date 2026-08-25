from __future__ import annotations

"""
V94 — FULL LEXICON + DUAL-LEVEL FORM/SEMANTICS TRANSFER

Why this version
----------------
V93 showed:

    recursive lexical structure > shuffled control

but:

    recursive lexical structure < width-1

The main reason was coverage:
    width-1 units covered ~53.6% of held-out lexical representations
    recursive units covered ~19.4%

So V94 does NOT throw away the compressed hierarchy.

It exposes BOTH levels to the semantic learner:

    primitive width-1 units
            +
    recursively discovered lexical assemblies
            ↓
    semantic feature predictor

The lexical substrate is still learned from ALL 4,925 dictionary words.

Semantic supervision is still limited to the independently observed concepts
present in semantics.csv. We never invent labels for the remaining dictionary.

This tests the more natural architecture:

    raw local structure
           \
            +--> semantic interface
           /
    compressed reusable structure

instead of forcing semantics to consume only the maximally compressed form.

Controls
--------
    1. width-1 only
    2. recursive only
    3. dual-level (width-1 + recursive)
    4. shuffled dual-level

The single primary comparison:
    dual-level vs width-1
    dual-level vs shuffled

All semantic evaluation remains concept-disjoint.
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
# Lexical recursive substrate
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
    Entire dictionary is learned here.

    No semantic information enters this stage.
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

    def dual_units(
        self,
        word: str,
    ) -> list[int]:
        # Keep BOTH representations. This restores semantic coverage while
        # preserving the information in reusable higher-order assemblies.
        primitive = self.base_stream(word)
        recursive = self.recursive_units(word)
        return list(set(primitive) | set(recursive))

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
# Cross-modal learner
# ---------------------------------------------------------------------------

class SemanticAssociator:
    """
    One learner over one chosen lexical interface.

    mode:
        width1
        recursive
        dual
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

        self.feature_ids: dict[str, int] = {}
        self.feature_names: dict[int, str] = {}

        self.unit_feature_counts: dict[
            int,
            Counter[int],
        ] = defaultdict(Counter)

        self.unit_anchor_counts: Counter[int] = Counter()

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

        identifier = len(self.feature_ids)

        self.feature_ids[feature] = identifier
        self.feature_names[identifier] = feature

        return identifier

    def learn(
        self,
        record: ConceptRecord,
    ) -> None:
        feature_ids = [
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

            for feature_id in feature_ids:
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
            denominator = max(
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
                    count / denominator
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
            for feature_id, _score in ranked[:k]
        ]

    def coverage(
        self,
        word: str,
    ) -> float:
        units = set(
            self.representation(word)
        )

        known = sum(
            unit_id in self.unit_anchor_counts
            for unit_id in units
        )

        return known / max(
            1,
            len(units),
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
        tp / len(predicted_set)
    )

    recall = (
        tp / max(1, len(actual))
    )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    return precision, recall, f1


def evaluate(
    model: SemanticAssociator,
    records: list[ConceptRecord],
    label: str,
) -> dict[str, float]:
    p = []
    r = []
    f = []
    c = []

    for record in records:
        predicted = model.predict(
            record.concept,
            TOP_K,
        )

        precision, recall, f1 = prf(
            predicted,
            record.features,
        )

        p.append(precision)
        r.append(recall)
        f.append(f1)
        c.append(
            model.coverage(
                record.concept
            )
        )

    result = {
        "precision_at_k": sum(p)
        / max(1, len(p)),
        "recall_at_k": sum(r)
        / max(1, len(r)),
        "f1_at_k": sum(f)
        / max(1, len(f)),
        "coverage": sum(c)
        / max(1, len(c)),
    }

    print(
        f"=== V94 {label} ==="
    )

    for key, value in result.items():
        print(
            f"{key:26s}: {value}"
        )

    print()

    return result


def shuffle_training(
    records: list[ConceptRecord],
    seed: int,
) -> list[ConceptRecord]:
    rng = random.Random(seed)

    feature_sets = [
        frozenset(
            record.features
        )
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
        "=== V94 FULL LEXICON DUAL-LEVEL SEMANTIC TRANSFER ==="
    )
    print(
        "ASCII-safe Windows output."
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
        "unmatched_semantics:",
        len(semantic_map) - len(anchors),
    )
    print(
        "train_anchors:",
        len(train),
    )
    print(
        "validation_anchors:",
        len(validation),
    )
    print(
        "test_anchors:",
        len(test),
    )
    print()

    # ---------------------------------------------------------------
    # BIG LEXICON
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
        "=== FULL LEXICAL GRAPH ==="
    )

    for key, value in substrate.stats().items():
        print(
            f"{key:24s}: {value}"
        )

    print()

    # ---------------------------------------------------------------
    # THREE INTERFACES
    # ---------------------------------------------------------------

    models = {
        "width1": SemanticAssociator(
            substrate,
            "width1",
        ),
        "recursive": SemanticAssociator(
            substrate,
            "recursive",
        ),
        "dual": SemanticAssociator(
            substrate,
            "dual",
        ),
    }

    for record in train:
        for model in models.values():
            model.learn(record)

    # ---------------------------------------------------------------
    # TEST
    # ---------------------------------------------------------------

    results = {}

    for name, model in models.items():
        results[
            f"{name}_validation"
        ] = evaluate(
            model,
            validation,
            f"{name.upper()} VALIDATION",
        )

        results[
            f"{name}_test"
        ] = evaluate(
            model,
            test,
            f"{name.upper()} TEST",
        )

    # ---------------------------------------------------------------
    # SHUFFLED DUAL CONTROL
    # ---------------------------------------------------------------

    shuffled_model = SemanticAssociator(
        substrate,
        "dual",
    )

    for record in shuffle_training(
        train,
        SEED,
    ):
        shuffled_model.learn(record)

    results[
        "shuffled_test"
    ] = evaluate(
        shuffled_model,
        test,
        "DUAL SHUFFLED TEST",
    )

    # ---------------------------------------------------------------
    # COMPARISON
    # ---------------------------------------------------------------

    print(
        "=== V94 CROSS-MODAL COMPARISON ==="
    )

    for metric in (
        "precision_at_k",
        "recall_at_k",
        "f1_at_k",
    ):
        w = results[
            "width1_test"
        ][metric]

        r = results[
            "recursive_test"
        ][metric]

        d = results[
            "dual_test"
        ][metric]

        s = results[
            "shuffled_test"
        ][metric]

        print(
            f"{metric:18s} "
            f"width1={w:.6f} "
            f"recursive={r:.6f} "
            f"dual={d:.6f} "
            f"shuffle={s:.6f}"
        )

        print(
            f"{'':18s}"
            f" dual-width1={d-w:+.6f} "
            f" dual-shuffle={d-s:+.6f} "
            f"lift={d/max(1e-9,s):.3f}x"
        )

    print()
    print(
        "=== V94 COVERAGE ==="
    )

    for name in (
        "width1_test",
        "recursive_test",
        "dual_test",
        "shuffled_test",
    ):
        print(
            f"{name:20s}:",
            results[name]["coverage"],
        )

    print()
    print(
        "=== V94 INTERPRETATION ==="
    )
    print(
        "All dictionary words participate in the lexical graph."
    )
    print(
        "Semantic training is limited to independently observed matched anchors."
    )
    print(
        "Dual-level asks whether semantics benefits from BOTH raw local "
        "coverage and recursive reusable structure."
    )
    print(
        "Dual > shuffled is the form->feature signal."
    )
    print(
        "Dual > width1 is the higher-order structural gain."
    )

    print()
    print(
        "elapsed_seconds:",
        f"{time.perf_counter() - start:.2f}",
    )
    print(
        "=== V94 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
