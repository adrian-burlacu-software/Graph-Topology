from __future__ import annotations

"""
V72 DICTIONARY CORPUS — FAST / PROFILED HARNESS

Key changes from the stalled corpus run:
  * NO deepcopy of the trained Network.
  * NO Cartesian-product explosion across all prefix/symbol/suffix factors.
  * Novel-combination candidates come from real validation/test positions.
  * Progress + elapsed time is printed for every expensive stage.
  * Only a bounded number of autonomous novel probes are committed.
  * Frozen evaluation never calls Network.process_word(); it reads the
    factorized binding graph directly.

Corpus:
    data/dictionary.csv

The current repository corpus is treated as one word per non-empty line.
"""

import time
import hashlib
from pathlib import Path
from collections import Counter

from evaluate_factorized_composition_v74_real_graph_layer import (
    V72RealGraphComposition,
    REUSE,
    BRANCH,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "dictionary.csv"

TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15

MAX_NOVEL_PROBES = 100
PROGRESS_WORDS = 250


# ---------------------------------------------------------------------------
# Timing / logging
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self) -> None:
        self.started = time.perf_counter()

    def mark(self, label: str) -> None:
        elapsed = time.perf_counter() - self.started
        print(
            f"[{elapsed:8.2f}s] {label}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def load_dictionary(path: Path) -> list[str]:
    words = []

    for raw in path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        word = raw.strip().lower()

        if word and word.isalpha():
            words.append(word)

    return sorted(set(words))


def stable_rank(word: str) -> str:
    return hashlib.sha256(
        word.encode("utf-8")
    ).hexdigest()


def split_words(
    words: list[str],
) -> tuple[list[str], list[str], list[str]]:
    ordered = sorted(
        words,
        key=lambda word: (stable_rank(word), word),
    )

    n = len(ordered)

    train_end = int(n * TRAIN_FRACTION)
    valid_end = train_end + int(
        n * VALID_FRACTION
    )

    train = ordered[:train_end]
    validation = ordered[train_end:valid_end]
    test = ordered[valid_end:]

    assert not set(train) & set(validation)
    assert not set(train) & set(test)
    assert not set(validation) & set(test)

    return train, validation, test


# ---------------------------------------------------------------------------
# Structural sets
# ---------------------------------------------------------------------------

def composition_of(
    word: str,
    pos: int,
) -> tuple[str, str, str]:
    return (
        word[:pos],
        word[pos],
        word[pos + 1:],
    )


def training_compositions(
    words: list[str],
) -> set[tuple[str, str, str]]:
    return {
        composition_of(word, pos)
        for word in words
        for pos in range(len(word))
    }


def training_factors(
    words: list[str],
) -> dict[str, set[str]]:
    return {
        "prefix": {
            word[:pos]
            for word in words
            for pos in range(len(word))
        },
        "symbol": {
            word[pos]
            for word in words
            for pos in range(len(word))
        },
        "suffix": {
            word[pos + 1:]
            for word in words
            for pos in range(len(word))
        },
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_network(
    train: list[str],
    timer: Timer,
) -> V72RealGraphComposition:
    network = V72RealGraphComposition()

    network.train(
        train,
        epochs=1,
    )
    timer.mark(
        f"real Network training complete ({len(train)} words)"
    )

    # V72 composition layer: explicit loop so progress is visible and
    # exceptions are localized.
    for index, word in enumerate(train, start=1):
        for pos in range(len(word)):
            network.learn_composition_position(
                word,
                pos,
            )

        if (
            index % PROGRESS_WORDS == 0
            or index == len(train)
        ):
            timer.mark(
                f"V72 factor/binding training "
                f"{index}/{len(train)} words"
            )

    network.calibrate_v72_threshold()

    timer.mark("V72 training + calibration complete")
    return network


# ---------------------------------------------------------------------------
# Frozen evaluation WITHOUT Network.process_word/deepcopy
# ---------------------------------------------------------------------------

def evaluate_holdout(
    network: V72RealGraphComposition,
    words: list[str],
    train_bindings: set[tuple[str, str, str]],
    train_factors: dict[str, set[str]],
    label: str,
) -> dict[str, int]:
    counts = Counter()

    for word in words:
        for pos in range(len(word)):
            comp = composition_of(word, pos)

            if (
                comp[0] in train_factors["prefix"]
                and comp[1] in train_factors["symbol"]
                and comp[2] in train_factors["suffix"]
            ):
                if comp in train_bindings:
                    counts["EXACT_REUSE"] += 1
                else:
                    counts["KNOWN_FACTORS_NOVEL_BINDING"] += 1
            else:
                counts["UNKNOWN_FACTOR"] += 1

    total = sum(counts.values())

    print(
        f"=== V72 {label} HOLDOUT ==="
    )
    print("positions              :", total)
    print("exact_reuse            :", counts["EXACT_REUSE"])
    print(
        "known_factor_novel     :",
        counts["KNOWN_FACTORS_NOVEL_BINDING"],
    )
    print("unknown_factor         :", counts["UNKNOWN_FACTOR"])

    print(
        "exact_reuse_rate       :",
        counts["EXACT_REUSE"] / max(1, total),
    )
    print(
        "known_factor_novel_rate:",
        counts["KNOWN_FACTORS_NOVEL_BINDING"] / max(1, total),
    )
    print(
        "unknown_factor_rate    :",
        counts["UNKNOWN_FACTOR"] / max(1, total),
    )
    print(
        f"=== END V72 {label} HOLDOUT ==="
    )
    print()

    return dict(counts)


# ---------------------------------------------------------------------------
# Novel candidates FROM REAL HOLDOUT POSITIONS
# ---------------------------------------------------------------------------

def build_real_novel_candidates(
    words: list[str],
    network: V72RealGraphComposition,
    train_factors: dict[str, set[str]],
    limit: int,
) -> list[tuple[str, int, object, dict[str, float]]]:
    candidates = []
    seen = set()

    for word in words:
        for pos in range(len(word)):
            comp = composition_of(word, pos)

            if comp in seen:
                continue

            if not (
                comp[0] in train_factors["prefix"]
                and comp[1] in train_factors["symbol"]
                and comp[2] in train_factors["suffix"]
            ):
                continue

            factors = network.factorize_position(
                word,
                pos,
                learn=False,
            )

            if network.exact_binding(factors) is not None:
                continue

            evidence = network.pair_evidence(
                factors
            )

            seen.add(comp)

            candidates.append(
                (
                    word,
                    pos,
                    factors,
                    evidence,
                )
            )

    # Strongest evidence first so the bounded probe is informative.
    candidates.sort(
        key=lambda row: (
            -row[3]["minimum"],
            -row[3]["sum"],
            row[0],
            row[1],
        )
    )

    return candidates[:limit]


# ---------------------------------------------------------------------------
# Bounded autonomous probe
# ---------------------------------------------------------------------------

def autonomous_probe(
    network: V72RealGraphComposition,
    candidates,
    timer: Timer,
) -> dict[str, int]:
    counts = Counter()

    before_bindings = len(
        network.v72_binding_by_key
    )
    before_factors = len(
        network.v72_factor_by_value
    )

    print("=== V72 AUTONOMOUS NOVEL PROBE ===")

    for index, (
        word,
        pos,
        factors,
        evidence,
    ) in enumerate(
        candidates,
        start=1,
    ):
        baseline = network.v72_baseline(
            factors
        )

        assert baseline == BRANCH

        action = network.v72_autonomous(
            factors
        )

        counts[action] += 1

        print(
            f"{index:3d}/{len(candidates):3d} "
            f"{word:12s} "
            f"pos={pos:2d} "
            f"baseline={baseline:6s} "
            f"action={action:7s} "
            f"min={evidence['minimum']:.1f} "
            f"sum={evidence['sum']:.1f}",
            flush=True,
        )

    after_bindings = len(
        network.v72_binding_by_key
    )
    after_factors = len(
        network.v72_factor_by_value
    )

    print()
    print("candidate_count    :", len(candidates))
    print("composed           :", counts["COMPOSE"])
    print("branched           :", counts[BRANCH])
    print(
        "new_bindings       :",
        after_bindings - before_bindings,
    )
    print(
        "new_factors        :",
        after_factors - before_factors,
    )

    assert after_factors == before_factors

    timer.mark("bounded autonomous novel probe complete")

    print("=== END V72 AUTONOMOUS NOVEL PROBE ===")
    print()

    return dict(counts)


# ---------------------------------------------------------------------------
# Growth
# ---------------------------------------------------------------------------

def print_graph_counts(
    network: V72RealGraphComposition,
    label: str,
) -> dict[str, int]:
    result = {
        "network_cells": len(network.cells),
        "network_synapses": len(network.synapses),
        "v72_factor_cells": len(network.v72_factor_cells()),
        "v72_binding_cells": len(network.v72_binding_cells()),
        "v72_pair_synapses": len(network.v72_pair_synapses()),
    }

    print(f"=== {label} GRAPH COUNTS ===")
    for key, value in result.items():
        print(f"{key:24s}: {value}")
    print()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    timer = Timer()

    print("=== V72 DICTIONARY CORPUS FAST PROFILED RUN ===")
    print("corpus_path:", CORPUS_PATH)
    print()

    words = load_dictionary(
        CORPUS_PATH
    )
    timer.mark(
        f"corpus loaded: {len(words)} unique words"
    )

    train, validation, test = split_words(
        words
    )
    timer.mark(
        f"split complete: "
        f"train={len(train)} "
        f"validation={len(validation)} "
        f"test={len(test)}"
    )

    train_bindings = training_compositions(
        train
    )
    train_factor_sets = training_factors(
        train
    )

    print()
    print("=== TRAINING STRUCTURE ===")
    print("train_words       :", len(train))
    print("train_compositions:", len(train_bindings))
    print(
        "train_prefixes    :",
        len(train_factor_sets["prefix"]),
    )
    print(
        "train_symbols     :",
        len(train_factor_sets["symbol"]),
    )
    print(
        "train_suffixes    :",
        len(train_factor_sets["suffix"]),
    )
    print()

    network = build_network(
        train,
        timer,
    )

    before_probe = print_graph_counts(
        network,
        "TRAINED",
    )

    evaluate_holdout(
        network,
        validation,
        train_bindings,
        train_factor_sets,
        "VALIDATION",
    )
    timer.mark("validation classification complete")

    evaluate_holdout(
        network,
        test,
        train_bindings,
        train_factor_sets,
        "TEST",
    )
    timer.mark("test classification complete")

    candidates = build_real_novel_candidates(
        test,
        network,
        train_factor_sets,
        MAX_NOVEL_PROBES,
    )
    timer.mark(
        f"novel candidate selection complete: "
        f"{len(candidates)} candidates"
    )

    if candidates:
        autonomous_probe(
            network,
            candidates,
            timer,
        )
    else:
        print(
            "No known-factor novel bindings found in test; "
            "skipping autonomous probe."
        )

    after_probe = print_graph_counts(
        network,
        "AFTER AUTONOMOUS PROBE",
    )

    print("=== V72 GRAPH DELTAS ===")
    for key in before_probe:
        print(
            f"{key:24s}: "
            f"{after_probe[key] - before_probe[key]:+d}"
        )
    print()

    timer.mark("RUN COMPLETE")
    print("=== V72 FAST PROFILED CORPUS COMPLETE ===")


if __name__ == "__main__":
    main()
