from __future__ import annotations

"""
V74 — LOCAL CONTEXT WIDTH SWEEP

Compare bounded local factor representations on the exact same dictionary
corpus split:

    width = 1
    width = 2
    width = 3
    width = 4

For each width measure:
    * validation/test exact-binding reuse rate
    * known-factor novel rate
    * unknown-factor rate
    * factor-cell count
    * binding-cell count
    * pair-support synapse count
    * novel known-factor COMPOSE / BRANCH rate

No architecture is selected beforehand. The goal is to determine whether
"local context" is robust and whether there is a useful representation scale.
"""

import hashlib
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from simulator import (
        BRANCH,
        REUSE,
        Config,
        Network,
    )
except ImportError:
    from .simulator import (
        BRANCH,
        REUSE,
        Config,
        Network,
    )


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "dictionary.csv"

TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15

WIDTHS = (1, 2, 3, 4)
MAX_NOVEL_PROBES = 100


# ---------------------------------------------------------------------------
# Local factors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LocalFactors:
    left: str
    symbol: str
    right: str


def local_factors(
    word: str,
    pos: int,
    width: int,
) -> LocalFactors:
    if not 0 <= pos < len(word):
        raise IndexError(
            f"pos={pos} outside word length={len(word)}"
        )

    left = word[
        max(0, pos - width):pos
    ]
    right = word[
        pos + 1:pos + 1 + width
    ]

    left = (
        "^" * (width - len(left))
        + left
    )

    right = (
        right
        + "$" * (width - len(right))
    )

    return LocalFactors(
        left=left,
        symbol=word[pos],
        right=right,
    )


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def load_dictionary(
    path: Path,
) -> list[str]:
    words = []

    for raw in path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        word = raw.strip().lower()

        if word and word.isalpha():
            words.append(word)

    words = sorted(set(words))

    if not words:
        raise RuntimeError(
            f"No words loaded from {path}"
        )

    return words


def stable_rank(word: str) -> str:
    return hashlib.sha256(
        word.encode("utf-8")
    ).hexdigest()


def split_words(
    words: list[str],
) -> tuple[list[str], list[str], list[str]]:
    ordered = sorted(
        words,
        key=lambda word: (
            stable_rank(word),
            word,
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

    train = ordered[:train_end]
    validation = ordered[
        train_end:validation_end
    ]
    test = ordered[validation_end:]

    assert not set(train) & set(validation)
    assert not set(train) & set(test)
    assert not set(validation) & set(test)

    return train, validation, test


# ---------------------------------------------------------------------------
# Width-specific real graph
# ---------------------------------------------------------------------------

class WidthNetwork(Network):
    V_FACTOR = "v74_factor"
    V_BINDING = "v74_binding"

    V_PAIR = "V74_PAIR"
    V_BINDING_EDGE = "V74_BINDING"

    def __init__(
        self,
        width: int,
        config: Optional[Config] = None,
    ) -> None:
        super().__init__(config)

        self.width = width

        self.factor_by_value: dict[
            tuple[str, str],
            int,
        ] = {}

        self.binding_by_key: dict[
            tuple[int, int, int],
            int,
        ] = {}

        self.pair_support: dict[
            tuple[int, int],
            float,
        ] = {}

        self.compose_threshold = 1.0

    # ------------------------------------------------------------------
    # Factors
    # ------------------------------------------------------------------

    def factorize(
        self,
        word: str,
        pos: int,
        learn: bool,
    ) -> tuple[int, int, int]:
        factors = local_factors(
            word,
            pos,
            self.width,
        )

        ids = []

        for kind, value in (
            ("left", factors.left),
            ("symbol", factors.symbol),
            ("right", factors.right),
        ):
            key = (kind, value)

            if key in self.factor_by_value:
                ids.append(
                    self.factor_by_value[key]
                )
                continue

            if not learn:
                ids.append(-1)
                continue

            cell_id = self.create_cell(
                self.V_FACTOR,
                symbol=value,
            )

            self.factor_by_value[key] = cell_id
            ids.append(cell_id)

        return tuple(ids)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Pair support
    # ------------------------------------------------------------------

    def _reinforce_pair(
        self,
        left: int,
        right: int,
    ) -> None:
        if left < 0 or right < 0:
            return

        key = (left, right)

        synapse = self.synapses.get(key)

        if synapse is None:
            synapse = self.connect(
                left,
                right,
                self.V_PAIR,
                1.0,
            )

        synapse.weight += 1.0
        synapse.learning += 1.0

        self.pair_support[key] = synapse.weight

    def observe_factors(
        self,
        factors: tuple[int, int, int],
    ) -> None:
        left, symbol, right = factors

        self._reinforce_pair(
            left,
            symbol,
        )
        self._reinforce_pair(
            symbol,
            right,
        )
        self._reinforce_pair(
            left,
            right,
        )

    def pair_evidence(
        self,
        factors: tuple[int, int, int],
    ) -> dict[str, float]:
        if min(factors) < 0:
            return {
                "left_symbol": 0.0,
                "symbol_right": 0.0,
                "left_right": 0.0,
                "minimum": 0.0,
                "sum": 0.0,
            }

        left, symbol, right = factors

        ls = self.pair_support.get(
            (left, symbol),
            0.0,
        )
        sr = self.pair_support.get(
            (symbol, right),
            0.0,
        )
        lr = self.pair_support.get(
            (left, right),
            0.0,
        )

        return {
            "left_symbol": ls,
            "symbol_right": sr,
            "left_right": lr,
            "minimum": min(ls, sr, lr),
            "sum": ls + sr + lr,
        }

    # ------------------------------------------------------------------
    # Exact binding
    # ------------------------------------------------------------------

    def exact_binding(
        self,
        factors: tuple[int, int, int],
    ) -> Optional[int]:
        if min(factors) < 0:
            return None

        return self.binding_by_key.get(
            factors
        )

    def bind(
        self,
        factors: tuple[int, int, int],
    ) -> int:
        existing = self.binding_by_key.get(
            factors
        )

        if existing is not None:
            return existing

        binding_id = self.create_cell(
            self.V_BINDING
        )

        self.binding_by_key[
            factors
        ] = binding_id

        for factor_id in factors:
            self.connect(
                factor_id,
                binding_id,
                self.V_BINDING_EDGE,
                1.0,
            )

        return binding_id

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_local(
        self,
        words: list[str],
    ) -> None:
        for word_index, word in enumerate(
            words,
            start=1,
        ):
            for pos in range(len(word)):
                factors = self.factorize(
                    word,
                    pos,
                    learn=True,
                )
                self.observe_factors(
                    factors
                )
                self.bind(
                    factors
                )

            if (
                word_index % 1000 == 0
                or word_index == len(words)
            ):
                print(
                    f"    width={self.width} "
                    f"binding_train={word_index}/{len(words)}",
                    flush=True,
                )

    def calibrate(
        self,
    ) -> float:
        minimums = []

        for factors in self.binding_by_key:
            minimums.append(
                self.pair_evidence(
                    factors
                )["minimum"]
            )

        self.compose_threshold = (
            min(minimums)
            if minimums
            else 1.0
        )

        return self.compose_threshold

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    def baseline(
        self,
        factors: tuple[int, int, int],
    ) -> str:
        if min(factors) < 0:
            return BRANCH

        return (
            REUSE
            if self.exact_binding(factors)
            is not None
            else BRANCH
        )

    def autonomous(
        self,
        factors: tuple[int, int, int],
    ) -> str:
        if min(factors) < 0:
            return BRANCH

        if self.exact_binding(factors) is not None:
            return REUSE

        if (
            self.pair_evidence(
                factors
            )["minimum"]
            >= self.compose_threshold
        ):
            self.bind(
                factors
            )
            return "COMPOSE"

        return BRANCH

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        return {
            "factor_cells": sum(
                1
                for cell in self.cells.values()
                if cell.kind == self.V_FACTOR
            ),
            "binding_cells": sum(
                1
                for cell in self.cells.values()
                if cell.kind == self.V_BINDING
            ),
            "pair_synapses": sum(
                1
                for synapse in self.synapses.values()
                if synapse.kind == self.V_PAIR
            ),
            "network_cells": len(self.cells),
            "network_synapses": len(self.synapses),
        }


# ---------------------------------------------------------------------------
# Width run
# ---------------------------------------------------------------------------

def evaluate_width(
    width: int,
    train: list[str],
    validation: list[str],
    test: list[str],
) -> dict[str, object]:
    start = time.perf_counter()

    print()
    print(
        "============================================================"
    )
    print(
        f"V74 WIDTH = {width}"
    )
    print(
        "============================================================"
    )

    network = WidthNetwork(
        width
    )

    network.train(
        train,
        epochs=1,
    )

    print(
        f"  [{time.perf_counter() - start:7.2f}s] "
        "real Network training complete",
        flush=True,
    )

    network.train_local(
        train
    )

    threshold = network.calibrate()

    print(
        f"  [{time.perf_counter() - start:7.2f}s] "
        f"V74 local training complete; "
        f"threshold={threshold}",
        flush=True,
    )

    counts = network.counts()

    def classify(
        words: list[str],
    ) -> Counter:
        result = Counter()

        for word in words:
            for pos in range(len(word)):
                factors = network.factorize(
                    word,
                    pos,
                    learn=False,
                )

                if min(factors) < 0:
                    result["UNKNOWN_FACTOR"] += 1
                elif network.exact_binding(
                    factors
                ) is not None:
                    result["EXACT_REUSE"] += 1
                else:
                    result["KNOWN_FACTORS_NOVEL"] += 1

        return result

    validation_counts = classify(
        validation
    )
    test_counts = classify(
        test
    )

    def rates(
        counter: Counter,
    ) -> dict[str, float]:
        total = sum(counter.values())

        return {
            key: counter[key] / max(1, total)
            for key in (
                "EXACT_REUSE",
                "KNOWN_FACTORS_NOVEL",
                "UNKNOWN_FACTOR",
            )
        }

    # Novel candidates are selected from actual test positions, preventing
    # Cartesian-product blowups.
    candidates = []
    seen = set()

    for word in test:
        for pos in range(len(word)):
            factors = network.factorize(
                word,
                pos,
                learn=False,
            )

            if min(factors) < 0:
                continue

            if network.exact_binding(
                factors
            ) is not None:
                continue

            if factors in seen:
                continue

            evidence = network.pair_evidence(
                factors
            )

            candidates.append(
                (
                    word,
                    pos,
                    factors,
                    evidence,
                )
            )
            seen.add(factors)

    candidates.sort(
        key=lambda row: (
            -row[3]["minimum"],
            -row[3]["sum"],
            row[0],
            row[1],
        )
    )

    candidates = candidates[:MAX_NOVEL_PROBES]

    before_bindings = len(
        network.binding_by_key
    )
    composed = 0
    branched = 0

    for _word, _pos, factors, _evidence in candidates:
        assert network.baseline(
            factors
        ) == BRANCH

        action = network.autonomous(
            factors
        )

        if action == "COMPOSE":
            composed += 1
        else:
            branched += 1

    after_bindings = len(
        network.binding_by_key
    )

    assert (
        after_bindings - before_bindings
        == composed
    )

    final_counts = network.counts()

    elapsed = time.perf_counter() - start

    result = {
        "width": width,
        "threshold": threshold,
        "validation_counts": validation_counts,
        "validation_rates": rates(
            validation_counts
        ),
        "test_counts": test_counts,
        "test_rates": rates(
            test_counts
        ),
        "novel_candidates": len(candidates),
        "novel_composed": composed,
        "novel_branched": branched,
        "counts": final_counts,
        "elapsed": elapsed,
    }

    print(
        "  validation:",
        dict(validation_counts),
    )
    print(
        "  validation_rates:",
        result["validation_rates"],
    )
    print(
        "  test:",
        dict(test_counts),
    )
    print(
        "  test_rates:",
        result["test_rates"],
    )
    print(
        "  novel_candidates:",
        len(candidates),
    )
    print(
        "  novel_composed:",
        composed,
    )
    print(
        "  novel_branched:",
        branched,
    )
    print(
        "  graph_counts:",
        final_counts,
    )
    print(
        f"  elapsed={elapsed:.2f}s"
    )

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    total_start = time.perf_counter()

    print("=== V74 LOCAL CONTEXT WIDTH SWEEP ===")
    print(
        "widths :",
        WIDTHS,
    )
    print(
        "corpus :",
        CORPUS_PATH,
    )
    print()

    words = load_dictionary(
        CORPUS_PATH
    )

    train, validation, test = split_words(
        words
    )

    print("corpus_words :", len(words))
    print("train_words  :", len(train))
    print("validation   :", len(validation))
    print("test         :", len(test))
    print()

    results = []

    for width in WIDTHS:
        results.append(
            evaluate_width(
                width,
                train,
                validation,
                test,
            )
        )

    print()
    print(
        "================================================================"
    )
    print(
        "V74 WIDTH SWEEP SUMMARY"
    )
    print(
        "================================================================"
    )

    print(
        "width | "
        "val_reuse | "
        "test_reuse | "
        "test_novel | "
        "test_unknown | "
        "composed | "
        "branched | "
        "factors | "
        "bindings | "
        "pair_synapses | "
        "seconds"
    )

    for row in results:
        validation_rates = row[
            "validation_rates"
        ]
        test_rates = row[
            "test_rates"
        ]
        counts = row["counts"]

        print(
            f"{row['width']:5d} | "
            f"{validation_rates['EXACT_REUSE']:.4f} | "
            f"{test_rates['EXACT_REUSE']:.4f} | "
            f"{test_rates['KNOWN_FACTORS_NOVEL']:.4f} | "
            f"{test_rates['UNKNOWN_FACTOR']:.4f} | "
            f"{row['novel_composed']:8d} | "
            f"{row['novel_branched']:8d} | "
            f"{counts['factor_cells']:7d} | "
            f"{counts['binding_cells']:8d} | "
            f"{counts['pair_synapses']:12d} | "
            f"{row['elapsed']:.2f}"
        )

    print()
    print(
        "total_seconds :",
        f"{time.perf_counter() - total_start:.2f}",
    )
    print(
        "=== V74 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
