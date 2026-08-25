from __future__ import annotations

"""
V76 — WIDTH-1 BLIND COMPOSITION POLICY

Threshold selection uses VALIDATION only.
TEST is never used for calibration.

Width-1 local factor:
    previous character | current symbol | next character

Corpus:
    data/dictionary.csv
"""

import hashlib
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from simulator import BRANCH, REUSE, Config, Network
except ImportError:
    from .simulator import BRANCH, REUSE, Config, Network


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "dictionary.csv"

TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15
MAX_NOVEL_PROBES = 100


# ---------------------------------------------------------------------------
# Width-1 factors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LocalFactors:
    left: str
    symbol: str
    right: str


def local_factors(word: str, pos: int) -> LocalFactors:
    if not 0 <= pos < len(word):
        raise IndexError(pos)

    left = word[pos - 1] if pos > 0 else "^"
    right = word[pos + 1] if pos + 1 < len(word) else "$"

    return LocalFactors(left, word[pos], right)


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

    words = sorted(set(words))
    if not words:
        raise RuntimeError(f"No words loaded from {path}")
    return words


def stable_rank(word: str) -> str:
    return hashlib.sha256(word.encode("utf-8")).hexdigest()


def split_words(words: list[str]):
    ordered = sorted(words, key=lambda w: (stable_rank(w), w))

    n = len(ordered)
    train_end = int(n * TRAIN_FRACTION)
    validation_end = train_end + int(n * VALID_FRACTION)

    train = ordered[:train_end]
    validation = ordered[train_end:validation_end]
    test = ordered[validation_end:]

    assert not set(train) & set(validation)
    assert not set(train) & set(test)
    assert not set(validation) & set(test)

    return train, validation, test


# ---------------------------------------------------------------------------
# Width-1 graph layer
# ---------------------------------------------------------------------------

class Width1CompositionNetwork(Network):
    V_FACTOR = "v76_factor"
    V_BINDING = "v76_binding"
    V_PAIR = "V76_PAIR"
    V_BINDING_EDGE = "V76_BINDING"

    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__(config)

        self.factor_by_value: dict[tuple[str, str], int] = {}
        self.binding_by_key: dict[tuple[int, int, int], int] = {}

        self.pair_support: dict[tuple[str, int, int], float] = {}
        self.triple_support: dict[tuple[int, int, int], float] = {}

        self.compose_threshold = 1.0

    def _factor(self, kind: str, value: str, learn: bool) -> int:
        key = (kind, value)
        existing = self.factor_by_value.get(key)

        if existing is not None:
            return existing

        if not learn:
            return -1

        cell_id = self.create_cell(
            self.V_FACTOR,
            symbol=value,
        )
        self.factor_by_value[key] = cell_id
        return cell_id

    def factorize(
        self,
        word: str,
        pos: int,
        learn: bool,
    ) -> tuple[int, int, int]:
        factors = local_factors(word, pos)

        return (
            self._factor("left", factors.left, learn),
            self._factor("symbol", factors.symbol, learn),
            self._factor("right", factors.right, learn),
        )

    def _reinforce_pair(
        self,
        kind: str,
        left: int,
        right: int,
    ) -> None:
        if left < 0 or right < 0:
            return

        key = (kind, left, right)
        self.pair_support[key] = self.pair_support.get(key, 0.0) + 1.0

    def observe(self, factors: tuple[int, int, int]) -> None:
        left, symbol, right = factors

        self._reinforce_pair("ls", left, symbol)
        self._reinforce_pair("sr", symbol, right)
        self._reinforce_pair("lr", left, right)

        self.triple_support[factors] = (
            self.triple_support.get(factors, 0.0) + 1.0
        )

    def bind(self, factors: tuple[int, int, int]) -> int:
        existing = self.binding_by_key.get(factors)
        if existing is not None:
            return existing

        binding_id = self.create_cell(self.V_BINDING)
        self.binding_by_key[factors] = binding_id

        for factor_id in factors:
            self.connect(
                factor_id,
                binding_id,
                self.V_BINDING_EDGE,
                1.0,
            )

        return binding_id

    def train_local(self, words: list[str]) -> None:
        for index, word in enumerate(words, start=1):
            for pos in range(len(word)):
                factors = self.factorize(word, pos, learn=True)
                self.observe(factors)
                self.bind(factors)

            if index % 1000 == 0 or index == len(words):
                print(
                    f"width1 training {index}/{len(words)}",
                    flush=True,
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

        ls = self.pair_support.get(("ls", left, symbol), 0.0)
        sr = self.pair_support.get(("sr", symbol, right), 0.0)
        lr = self.pair_support.get(("lr", left, right), 0.0)

        return {
            "left_symbol": ls,
            "symbol_right": sr,
            "left_right": lr,
            "minimum": min(ls, sr, lr),
            "sum": ls + sr + lr,
        }

    def exact_binding(
        self,
        factors: tuple[int, int, int],
    ) -> Optional[int]:
        if min(factors) < 0:
            return None
        return self.binding_by_key.get(factors)

    def baseline(self, factors: tuple[int, int, int]) -> str:
        if min(factors) < 0:
            return BRANCH
        return REUSE if self.exact_binding(factors) is not None else BRANCH

    def calibrate_from_validation(
        self,
        validation_rows,
    ) -> tuple[float, dict[str, float]]:
        """
        Calibration is performed exclusively on novel validation cases.

        We do not use TEST to set the threshold.
        """
        zero_scores = []
        positive_scores = []

        for _word, _pos, _factors, evidence in validation_rows:
            score = evidence["minimum"]
            if score <= 0.0:
                zero_scores.append(score)
            else:
                positive_scores.append(score)

        if zero_scores and positive_scores:
            threshold = (
                max(zero_scores) + min(positive_scores)
            ) / 2.0
            threshold = max(threshold, 1e-9)
        elif positive_scores:
            threshold = min(positive_scores)
        else:
            threshold = 1e-9

        self.compose_threshold = threshold

        stats = {
            "validation_candidates": float(len(validation_rows)),
            "validation_zero_support": float(len(zero_scores)),
            "validation_positive_support": float(len(positive_scores)),
            "threshold": float(threshold),
        }
        return threshold, stats

    def autonomous(
        self,
        factors: tuple[int, int, int],
    ) -> str:
        if min(factors) < 0:
            return BRANCH

        if self.exact_binding(factors) is not None:
            return REUSE

        evidence = self.pair_evidence(factors)

        if evidence["minimum"] >= self.compose_threshold:
            self.bind(factors)
            return "COMPOSE"

        return BRANCH

    def graph_counts(self) -> dict[str, int]:
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
            "network_cells": len(self.cells),
            "network_synapses": len(self.synapses),
            "pair_support_edges": len(self.pair_support),
            "triple_support_keys": len(self.triple_support),
        }


# ---------------------------------------------------------------------------
# Candidate collection
# ---------------------------------------------------------------------------

def build_novel_rows(
    network: Width1CompositionNetwork,
    words: list[str],
):
    rows = []
    seen = set()

    for word in words:
        for pos in range(len(word)):
            factors = network.factorize(word, pos, learn=False)

            if min(factors) < 0:
                continue

            if network.exact_binding(factors) is not None:
                continue

            if factors in seen:
                continue

            rows.append(
                (
                    word,
                    pos,
                    factors,
                    network.pair_evidence(factors),
                )
            )
            seen.add(factors)

    rows.sort(
        key=lambda row: (
            row[3]["minimum"],
            row[3]["sum"],
            row[0],
            row[1],
        )
    )
    return rows


def print_score_distribution(
    rows,
    threshold: float,
    label: str,
) -> None:
    scores = sorted(
        evidence["minimum"]
        for _word, _pos, _factors, evidence in rows
    )

    print(f"=== V76 {label} SCORE DISTRIBUTION ===")

    if not scores:
        print("count :", 0)
        print()
        return

    print("count :", len(scores))
    print("min   :", scores[0])
    print("max   :", scores[-1])

    for percentile in (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0):
        index = min(
            len(scores) - 1,
            int(percentile * (len(scores) - 1)),
        )
        print(
            f"p{int(percentile * 100):02d}  :",
            scores[index],
        )

    print("threshold :", threshold)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    total_start = time.perf_counter()

    print("=== V76 WIDTH-1 BLIND COMPOSITION POLICY ===")
    print("threshold source: VALIDATION ONLY")
    print("test set: untouched until policy application")
    print("corpus:", CORPUS_PATH)
    print()

    words = load_dictionary(CORPUS_PATH)
    train, validation, test = split_words(words)

    print("corpus_words :", len(words))
    print("train_words  :", len(train))
    print("validation   :", len(validation))
    print("test         :", len(test))
    print()

    network = Width1CompositionNetwork()

    t = time.perf_counter()

    network.train(train, epochs=1)
    print(
        f"[{time.perf_counter() - t:.2f}s] "
        "real Network training complete"
    )

    network.train_local(train)
    print(
        f"[{time.perf_counter() - t:.2f}s] "
        "width-1 factor/binding training complete"
    )

    validation_rows = build_novel_rows(
        network,
        validation,
    )

    test_rows = build_novel_rows(
        network,
        test,
    )

    threshold, calibration = network.calibrate_from_validation(
        validation_rows
    )

    print("=== V76 VALIDATION CALIBRATION ===")
    for key, value in calibration.items():
        print(f"{key:30s}: {value}")
    print()

    print_score_distribution(
        validation_rows,
        threshold,
        "VALIDATION",
    )

    print_score_distribution(
        test_rows,
        threshold,
        "TEST",
    )

    # Count the untouched test distribution before committing anything.
    test_compose = sum(
        evidence["minimum"] >= threshold
        for _word, _pos, _factors, evidence in test_rows
    )
    test_branch = len(test_rows) - test_compose

    print("=== V76 BLIND TEST POLICY ===")
    print("test_novel_cases :", len(test_rows))
    print("test_compose     :", test_compose)
    print("test_branch      :", test_branch)
    print(
        "test_compose_rate:",
        test_compose / max(1, len(test_rows)),
    )
    print(
        "test_branch_rate :",
        test_branch / max(1, len(test_rows)),
    )
    print()

    # Apply the frozen validation-derived threshold to TEST exactly once.
    before_bindings = len(network.binding_by_key)

    committed_compose = 0
    committed_branch = 0

    for _word, _pos, factors, evidence in test_rows[:MAX_NOVEL_PROBES]:
        assert network.baseline(factors) == BRANCH

        if evidence["minimum"] >= threshold:
            network.bind(factors)
            committed_compose += 1
        else:
            committed_branch += 1

    after_bindings = len(network.binding_by_key)

    assert after_bindings - before_bindings == committed_compose

    print("=== V76 COMMIT SAMPLE ===")
    print("probe_limit       :", min(MAX_NOVEL_PROBES, len(test_rows)))
    print("committed_compose  :", committed_compose)
    print("committed_branch   :", committed_branch)
    print("new_bindings       :", after_bindings - before_bindings)
    print()

    print("=== V76 FINAL GRAPH ===")
    for key, value in network.graph_counts().items():
        print(f"{key:24s}: {value}")
    print("compose_threshold      :", network.compose_threshold)
    print()

    print(
        "total_seconds :",
        f"{time.perf_counter() - total_start:.2f}",
    )
    print("=== V76 COMPLETE ===")


if __name__ == "__main__":
    main()
