from __future__ import annotations

"""
V75 — WIDTH-1 COMPOSITION SELECTIVITY

Width 1 is now frozen from V74:
    left-neighbor | current-symbol | right-neighbor

This experiment does NOT revisit factor width.

It focuses on the remaining question:

    When should a novel local factor combination COMPOSE?

The experiment builds two classes of novel local triples from the same real
dictionary corpus:

    POSITIVE structural support:
        the exact local factor components occur in compatible training
        neighborhoods often enough to justify a composition.

    NEGATIVE coincidence:
        individual factors are known, but the exact pair structure is absent
        or insufficient.

The goal is to measure COMPOSE / BRANCH selectivity rather than merely proving
that composition can happen.

No arbitrary Cartesian product is used. Candidate triples are drawn from
actual held-out word positions and from controlled negatives constructed from
the corpus factor vocabulary.
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

WIDTH = 1
MAX_POSITIVE = 100
MAX_NEGATIVE = 100


# ---------------------------------------------------------------------------
# Local factor representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LocalFactors:
    left: str
    symbol: str
    right: str


def local_factors(
    word: str,
    pos: int,
) -> LocalFactors:
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

    return LocalFactors(
        left=left,
        symbol=word[pos],
        right=right,
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
# Width-1 graph layer
# ---------------------------------------------------------------------------

class Width1CompositionNetwork(Network):
    V_FACTOR = "v75_factor"
    V_BINDING = "v75_binding"

    V_PAIR = "V75_PAIR"
    V_BINDING_EDGE = "V75_BINDING"

    def __init__(
        self,
        config: Optional[Config] = None,
    ) -> None:
        super().__init__(config)

        self.factor_by_value: dict[
            tuple[str, str],
            int,
        ] = {}

        self.binding_by_key: dict[
            tuple[int, int, int],
            int,
        ] = {}

        # Specific pair support:
        #   left-symbol
        #   symbol-right
        #   left-right
        self.pair_support: dict[
            tuple[str, int, int],
            float,
        ] = {}

        # Higher-order context support:
        # exact ordered neighborhood occurrence:
        #   (left_factor, symbol_factor, right_factor)
        self.triple_support: dict[
            tuple[int, int, int],
            float,
        ] = {}

        self.compose_threshold = 1.0

    # ------------------------------------------------------------------
    # Factors
    # ------------------------------------------------------------------

    def _factor(
        self,
        kind: str,
        value: str,
        learn: bool,
    ) -> int:
        key = (kind, value)

        existing = self.factor_by_value.get(
            key
        )

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
        f = local_factors(
            word,
            pos,
        )

        return (
            self._factor(
                "left",
                f.left,
                learn,
            ),
            self._factor(
                "symbol",
                f.symbol,
                learn,
            ),
            self._factor(
                "right",
                f.right,
                learn,
            ),
        )

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _reinforce_pair(
        self,
        kind: str,
        left: int,
        right: int,
    ) -> None:
        if left < 0 or right < 0:
            return

        pair_key = (
            kind,
            left,
            right,
        )

        self.pair_support[pair_key] = (
            self.pair_support.get(
                pair_key,
                0.0,
            )
            + 1.0
        )

    def observe(
        self,
        factors: tuple[int, int, int],
    ) -> None:
        left, symbol, right = factors

        self._reinforce_pair(
            "ls",
            left,
            symbol,
        )
        self._reinforce_pair(
            "sr",
            symbol,
            right,
        )
        self._reinforce_pair(
            "lr",
            left,
            right,
        )

        self.triple_support[factors] = (
            self.triple_support.get(
                factors,
                0.0,
            )
            + 1.0
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

        self.binding_by_key[factors] = binding_id

        for factor_id in factors:
            self.connect(
                factor_id,
                binding_id,
                self.V_BINDING_EDGE,
                1.0,
            )

        return binding_id

    def train_local(
        self,
        words: list[str],
    ) -> None:
        for index, word in enumerate(
            words,
            start=1,
        ):
            for pos in range(len(word)):
                factors = self.factorize(
                    word,
                    pos,
                    learn=True,
                )

                self.observe(factors)
                self.bind(factors)

            if (
                index % 1000 == 0
                or index == len(words)
            ):
                print(
                    f"width1 training "
                    f"{index}/{len(words)}",
                    flush=True,
                )

    # ------------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------------

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
            ("ls", left, symbol),
            0.0,
        )

        sr = self.pair_support.get(
            ("sr", symbol, right),
            0.0,
        )

        lr = self.pair_support.get(
            ("lr", left, right),
            0.0,
        )

        return {
            "left_symbol": ls,
            "symbol_right": sr,
            "left_right": lr,
            "minimum": min(ls, sr, lr),
            "sum": ls + sr + lr,
        }

    def triple_evidence(
        self,
        factors: tuple[int, int, int],
    ) -> float:
        return self.triple_support.get(
            factors,
            0.0,
        )

    # ------------------------------------------------------------------
    # Decision
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

    def calibrate(
        self,
    ) -> float:
        # Training triples are exact learned compositions. We deliberately
        # calibrate a minimum triple support > 0 for the compose gate.
        observed = [
            support
            for support in self.triple_support.values()
            if support > 0
        ]

        self.compose_threshold = (
            min(observed)
            if observed
            else 1.0
        )

        return self.compose_threshold

    def autonomous(
        self,
        factors: tuple[int, int, int],
    ) -> str:
        if min(factors) < 0:
            return BRANCH

        if self.exact_binding(factors) is not None:
            return REUSE

        pair = self.pair_evidence(
            factors
        )

        # New V75 rule:
        # generic pair support alone does NOT authorize composition.
        # We require at least one strong specific pair AND all primitive
        # factors to be known. The exact triple remains novel.
        #
        # This gate is deliberately exposed as a metric rather than hidden in
        # the baseline REUSE semantics.
        max_pair = max(
            pair["left_symbol"],
            pair["symbol_right"],
            pair["left_right"],
        )

        if (
            max_pair >= self.compose_threshold
            and pair["minimum"] > 0.0
        ):
            self.bind(factors)
            return "COMPOSE"

        return BRANCH

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

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
            "pair_support_edges": len(
                self.pair_support
            ),
            "triple_support_keys": len(
                self.triple_support
            ),
        }


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------

def build_positive_candidates(
    network: Width1CompositionNetwork,
    validation: list[str],
    test: list[str],
    limit: int,
):
    """
    Positive structural candidates:
      * exact local triple absent from training;
      * all factors known;
      * all three pair relationships have non-zero support.

    These are the best cases for genuine compositional binding.
    """
    selected = []
    seen = set()

    for word in validation + test:
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

            pair = network.pair_evidence(
                factors
            )

            if pair["minimum"] <= 0:
                continue

            if factors in seen:
                continue

            selected.append(
                (
                    word,
                    pos,
                    factors,
                    pair,
                )
            )
            seen.add(factors)

    selected.sort(
        key=lambda row: (
            -row[3]["minimum"],
            -row[3]["sum"],
            row[0],
            row[1],
        )
    )

    return selected[:limit]


def build_negative_candidates(
    network: Width1CompositionNetwork,
    validation: list[str],
    test: list[str],
    limit: int,
):
    """
    Negative candidates:
      * exact local triple novel;
      * all factors individually known;
      * at least one required pair is absent.

    These are the critical false-composition controls.
    """
    selected = []
    seen = set()

    for word in validation + test:
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

            pair = network.pair_evidence(
                factors
            )

            if pair["minimum"] > 0:
                continue

            if factors in seen:
                continue

            selected.append(
                (
                    word,
                    pos,
                    factors,
                    pair,
                )
            )
            seen.add(factors)

    selected.sort(
        key=lambda row: (
            row[3]["sum"],
            row[0],
            row[1],
        )
    )

    return selected[:limit]


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------

def evaluate_candidates(
    network: Width1CompositionNetwork,
    positive,
    negative,
) -> None:
    print("=== V75 COMPOSITION SELECTIVITY ===")

    positive_compose = 0
    positive_branch = 0

    print(
        "--- POSITIVE SUPPORTED NOVEL CASES ---"
    )

    for word, pos, factors, evidence in positive:
        baseline = network.baseline(
            factors
        )
        assert baseline == BRANCH

        action = network.autonomous(
            factors
        )

        print(
            f"{word:12s} "
            f"pos={pos:2d} "
            f"baseline={baseline:6s} "
            f"action={action:7s} "
            f"min={evidence['minimum']:.1f} "
            f"sum={evidence['sum']:.1f}"
        )

        if action == "COMPOSE":
            positive_compose += 1
        else:
            positive_branch += 1

    negative_compose = 0
    negative_branch = 0

    print()
    print(
        "--- NEGATIVE UNSUPPORTED NOVEL CASES ---"
    )

    for word, pos, factors, evidence in negative:
        # Evaluate on the current network, but these exact negative triples
        # remain novel unless an earlier positive case accidentally created
        # the same factor triple.
        if network.exact_binding(
            factors
        ) is not None:
            # This can only happen if a candidate duplicated a positive case.
            # Such a collision is excluded from the final negative set.
            continue

        baseline = network.baseline(
            factors
        )
        assert baseline == BRANCH

        action = network.autonomous(
            factors
        )

        print(
            f"{word:12s} "
            f"pos={pos:2d} "
            f"baseline={baseline:6s} "
            f"action={action:7s} "
            f"min={evidence['minimum']:.1f} "
            f"sum={evidence['sum']:.1f}"
        )

        if action == "COMPOSE":
            negative_compose += 1
        else:
            negative_branch += 1

    print()
    print(
        "positive_cases       :",
        len(positive),
    )
    print(
        "positive_composed    :",
        positive_compose,
    )
    print(
        "positive_branched    :",
        positive_branch,
    )
    print(
        "negative_cases       :",
        len(negative),
    )
    print(
        "negative_composed    :",
        negative_compose,
    )
    print(
        "negative_branched    :",
        negative_branch,
    )

    positive_precision = (
        positive_compose / max(1, len(positive))
    )

    negative_branch_rate = (
        negative_branch / max(1, len(negative))
    )

    print(
        "positive_compose_rate:",
        positive_precision,
    )
    print(
        "negative_branch_rate :",
        negative_branch_rate,
    )

    print("=== END V75 COMPOSITION SELECTIVITY ===")
    print()


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

    return sorted(set(words))


def stable_rank(word: str) -> str:
    return hashlib.sha256(
        word.encode("utf-8")
    ).hexdigest()


def split_words(
    words: list[str],
):
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

    return (
        ordered[:train_end],
        ordered[
            train_end:validation_end
        ],
        ordered[validation_end:],
    )


# ---------------------------------------------------------------------------
# Corpus holdout representation stats
# ---------------------------------------------------------------------------

def holdout_stats(
    network: Width1CompositionNetwork,
    words: list[str],
    label: str,
) -> Counter:
    counts = Counter()

    for word in words:
        for pos in range(len(word)):
            factors = network.factorize(
                word,
                pos,
                learn=False,
            )

            if min(factors) < 0:
                counts["UNKNOWN_FACTOR"] += 1

            elif network.exact_binding(
                factors
            ) is not None:
                counts["EXACT_REUSE"] += 1

            else:
                counts["KNOWN_FACTORS_NOVEL"] += 1

    total = sum(counts.values())

    print(f"=== V75 {label} HOLDOUT ===")
    print("positions             :", total)
    print(
        "exact_reuse           :",
        counts["EXACT_REUSE"],
    )
    print(
        "known_factors_novel   :",
        counts["KNOWN_FACTORS_NOVEL"],
    )
    print(
        "unknown_factor        :",
        counts["UNKNOWN_FACTOR"],
    )
    print(
        "exact_reuse_rate      :",
        counts["EXACT_REUSE"]
        / max(1, total),
    )
    print(
        "known_factor_novel_rate:",
        counts["KNOWN_FACTORS_NOVEL"]
        / max(1, total),
    )
    print()

    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    total_start = time.perf_counter()

    print(
        "=== V75 WIDTH-1 COMPOSITION SELECTIVITY ==="
    )
    print(
        "corpus:",
        CORPUS_PATH,
    )
    print()

    words = load_dictionary(
        CORPUS_PATH
    )

    train, validation, test = split_words(
        words
    )

    print(
        "corpus_words :",
        len(words),
    )
    print(
        "train_words  :",
        len(train),
    )
    print(
        "validation   :",
        len(validation),
    )
    print(
        "test         :",
        len(test),
    )
    print()

    network = Width1CompositionNetwork()

    t = time.perf_counter()

    print(
        "=== V75 REAL NETWORK TRAINING ==="
    )

    network.train(
        train,
        epochs=1,
    )

    print(
        f"[{time.perf_counter() - t:.2f}s] "
        "real Network training complete"
    )

    network.train_local(
        train
    )

    threshold = network.calibrate()

    print(
        f"[{time.perf_counter() - t:.2f}s] "
        f"width-1 factor training complete "
        f"threshold={threshold}"
    )

    print(
        "graph_counts :",
        network.graph_counts(),
    )
    print()

    holdout_stats(
        network,
        validation,
        "VALIDATION",
    )

    holdout_stats(
        network,
        test,
        "TEST",
    )

    positive = build_positive_candidates(
        network,
        validation,
        test,
        MAX_POSITIVE,
    )

    # Build negative candidates BEFORE any autonomous composition mutates the
    # binding graph. This preserves their intended meaning.
    negative = build_negative_candidates(
        network,
        validation,
        test,
        MAX_NEGATIVE,
    )

    print(
        "positive_candidates :",
        len(positive),
    )
    print(
        "negative_candidates :",
        len(negative),
    )
    print()

    assert positive, (
        "V75 requires supported novel local combinations"
    )
    assert negative, (
        "V75 requires unsupported novel local combinations"
    )

    before_bindings = (
        len(network.binding_by_key)
    )

    evaluate_candidates(
        network,
        positive,
        negative,
    )

    after_bindings = (
        len(network.binding_by_key)
    )

    print()
    print(
        "=== V75 GRAPH GROWTH ==="
    )
    print(
        "bindings_before :",
        before_bindings,
    )
    print(
        "bindings_after  :",
        after_bindings,
    )
    print(
        "new_bindings    :",
        after_bindings - before_bindings,
    )
    print(
        "factor_cells    :",
        network.graph_counts()["factor_cells"],
    )

    assert (
        after_bindings
        >= before_bindings
    )

    print()
    print(
        "total_seconds :",
        f"{time.perf_counter() - total_start:.2f}",
    )
    print(
        "=== V75 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
