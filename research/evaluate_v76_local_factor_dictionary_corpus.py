from __future__ import annotations

"""
V73 — LOCAL FACTOR REPRESENTATION ON THE REAL GRAPH

Motivation
----------
The full-prefix/full-suffix representation made an exact binding almost
equivalent to memorizing a whole word position. With a word-disjoint corpus,
that produced ~0 exact REUSE positions.

V73 changes ONLY the factorization granularity.

Instead of:
    full prefix + symbol + full suffix

we use bounded local context:
    left window + symbol + right window

with boundary markers.

Default:
    left_width  = 2
    right_width = 2

The rest of the V72 composition architecture remains conceptually identical:
    factor cells
    specific pair support
    exact binding cells
    stable REUSE / BRANCH
    autonomous COMPOSE / BRANCH

This script is a corpus-scale comparison harness.

It intentionally avoids:
    * deepcopy of the trained network
    * Cartesian factor explosions
    * mutation during holdout evaluation
"""

import copy
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


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "dictionary.csv"

TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15

LEFT_WIDTH = 2
RIGHT_WIDTH = 2

MAX_NOVEL_PROBES = 100


# ---------------------------------------------------------------------------
# Factor representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LocalFactors:
    left: str
    symbol: str
    right: str


BOUNDARY = "^"
END = "$"


def local_factors(
    word: str,
    pos: int,
    left_width: int = LEFT_WIDTH,
    right_width: int = RIGHT_WIDTH,
) -> LocalFactors:
    if not 0 <= pos < len(word):
        raise IndexError(
            f"pos={pos} outside word length={len(word)}"
        )

    left = word[max(0, pos - left_width):pos]
    right = word[pos + 1:pos + 1 + right_width]

    left = (
        BOUNDARY * (left_width - len(left))
        + left
    )

    right = (
        right
        + END * (right_width - len(right))
    )

    return LocalFactors(
        left=left,
        symbol=word[pos],
        right=right,
    )


# ---------------------------------------------------------------------------
# Corpus split
# ---------------------------------------------------------------------------

def load_dictionary(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)

    words = []

    for raw in path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        word = raw.strip().lower()

        if word and word.isalpha():
            words.append(word)

    words = sorted(set(words))

    if len(words) < 100:
        raise RuntimeError(
            f"Corpus too small: {len(words)}"
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
# Local-factor graph
# ---------------------------------------------------------------------------

class V73LocalFactorNetwork(Network):
    """
    Existing Network + independent local-factor composition layer.

    The existing path graph remains untouched.

    V73 graph state:
        local factor cell:
            (kind, local-value)

        binding cell:
            (left_factor, symbol_factor, right_factor)

        pair support:
            left-symbol
            symbol-right
            left-right
    """

    V73_FACTOR = "v73_factor"
    V73_BINDING = "v73_binding"

    V73_PAIR = "V73_PAIR"
    V73_BINDING_EDGE = "V73_BINDING"

    def __init__(
        self,
        config: Optional[Config] = None,
    ) -> None:
        super().__init__(config)

        self.v73_factor_by_value: dict[
            tuple[str, str],
            int,
        ] = {}

        self.v73_binding_by_key: dict[
            tuple[int, int, int],
            int,
        ] = {}

        self.v73_pair_support: dict[
            tuple[int, int],
            float,
        ] = {}

        self.v73_threshold = 1.0

    # ------------------------------------------------------------------
    # Factor cells
    # ------------------------------------------------------------------

    def _factor_cell(
        self,
        kind: str,
        value: str,
        learn: bool,
    ) -> int:
        key = (kind, value)

        existing = self.v73_factor_by_value.get(key)
        if existing is not None:
            return existing

        if not learn:
            return -1

        cell_id = self.create_cell(
            self.V73_FACTOR,
            symbol=value,
        )

        self.v73_factor_by_value[key] = cell_id

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
            self._factor_cell(
                "left",
                f.left,
                learn,
            ),
            self._factor_cell(
                "symbol",
                f.symbol,
                learn,
            ),
            self._factor_cell(
                "right",
                f.right,
                learn,
            ),
        )

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
                self.V73_PAIR,
                1.0,
            )

        synapse.weight += 1.0
        synapse.learning += 1.0

        self.v73_pair_support[key] = synapse.weight

    def _observe_factors(
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

        ls = self.v73_pair_support.get(
            (left, symbol),
            0.0,
        )
        sr = self.v73_pair_support.get(
            (symbol, right),
            0.0,
        )
        lr = self.v73_pair_support.get(
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
    # Exact bindings
    # ------------------------------------------------------------------

    def exact_binding(
        self,
        factors: tuple[int, int, int],
    ) -> Optional[int]:
        if min(factors) < 0:
            return None

        return self.v73_binding_by_key.get(
            factors
        )

    def _bind(
        self,
        factors: tuple[int, int, int],
    ) -> int:
        existing = self.v73_binding_by_key.get(
            factors
        )

        if existing is not None:
            return existing

        binding_id = self.create_cell(
            self.V73_BINDING
        )

        self.v73_binding_by_key[
            factors
        ] = binding_id

        for factor_id in factors:
            self.connect(
                factor_id,
                binding_id,
                self.V73_BINDING_EDGE,
                1.0,
            )

        return binding_id

    def train_local_bindings(
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

                self._observe_factors(
                    factors
                )
                self._bind(
                    factors
                )

            if (
                word_index % 500 == 0
                or word_index == len(words)
            ):
                print(
                    f"V73 binding training "
                    f"{word_index}/{len(words)}",
                    flush=True,
                )

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def calibrate_threshold(self) -> float:
        minima = []

        for factors in self.v73_binding_by_key:
            minima.append(
                self.pair_evidence(
                    factors
                )["minimum"]
            )

        self.v73_threshold = (
            min(minima)
            if minima
            else 1.0
        )

        return self.v73_threshold

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
            >= self.v73_threshold
        ):
            self._bind(factors)
            return "COMPOSE"

        return BRANCH

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    def factor_cell_count(self) -> int:
        return sum(
            1
            for cell in self.cells.values()
            if cell.kind == self.V73_FACTOR
        )

    def binding_cell_count(self) -> int:
        return sum(
            1
            for cell in self.cells.values()
            if cell.kind == self.V73_BINDING
        )

    def pair_synapse_count(self) -> int:
        return sum(
            1
            for synapse in self.synapses.values()
            if synapse.kind == self.V73_PAIR
        )


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def elapsed(start: float) -> float:
    return time.perf_counter() - start


# ---------------------------------------------------------------------------
# Structure statistics
# ---------------------------------------------------------------------------

def composition_label(
    factors: LocalFactors,
) -> str:
    return (
        f"{factors.left}|"
        f"{factors.symbol}|"
        f"{factors.right}"
    )


def analyze_local_factor_space(
    train: list[str],
    validation: list[str],
    test: list[str],
) -> None:
    train_factor_set = {
        local_factors(word, pos)
        for word in train
        for pos in range(len(word))
    }

    print("=== V73 LOCAL FACTOR STRUCTURE ===")
    print("train_words :", len(train))
    print(
        "train_unique_local_factors :",
        len(train_factor_set),
    )

    for label, words in (
        ("VALIDATION", validation),
        ("TEST", test),
    ):
        exact = 0
        novel = 0

        for word in words:
            for pos in range(len(word)):
                factors = local_factors(
                    word,
                    pos,
                )

                if factors in train_factor_set:
                    exact += 1
                else:
                    novel += 1

        total = exact + novel

        print(
            f"{label:12s} positions={total} "
            f"known_local={exact} "
            f"novel_local={novel}"
        )

    print("=== END V73 LOCAL FACTOR STRUCTURE ===")
    print()


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def build_network(
    train: list[str],
) -> V73LocalFactorNetwork:
    start = time.perf_counter()

    network = V73LocalFactorNetwork()

    print(
        f"[{elapsed(start):7.2f}s] "
        f"starting real Network training"
    )

    network.train(
        train,
        epochs=1,
    )

    print(
        f"[{elapsed(start):7.2f}s] "
        f"real Network training complete"
    )

    network.train_local_bindings(
        train
    )

    print(
        f"[{elapsed(start):7.2f}s] "
        f"V73 local binding training complete"
    )

    network.calibrate_threshold()

    print(
        f"[{elapsed(start):7.2f}s] "
        f"threshold calibrated = "
        f"{network.v73_threshold}"
    )

    return network


# ---------------------------------------------------------------------------
# Holdout evaluation
# ---------------------------------------------------------------------------

def evaluate_holdout(
    network: V73LocalFactorNetwork,
    words: list[str],
    train_binding_keys: set[
        tuple[int, int, int]
    ],
    label: str,
) -> None:
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
            elif factors in train_binding_keys:
                counts["EXACT_REUSE"] += 1
            else:
                counts["KNOWN_FACTORS_NOVEL"] += 1

    total = sum(counts.values())

    print(f"=== V73 {label} ===")
    print("positions           :", total)
    print("exact_reuse         :", counts["EXACT_REUSE"])
    print(
        "known_factor_novel  :",
        counts["KNOWN_FACTORS_NOVEL"],
    )
    print(
        "unknown_factor     :",
        counts["UNKNOWN_FACTOR"],
    )
    print(
        "exact_reuse_rate    :",
        counts["EXACT_REUSE"] / max(1, total),
    )
    print(
        "known_factor_rate   :",
        counts["KNOWN_FACTORS_NOVEL"]
        / max(1, total),
    )
    print(
        "unknown_factor_rate :",
        counts["UNKNOWN_FACTOR"]
        / max(1, total),
    )
    print()
    print(f"=== END V73 {label} ===")
    print()


# ---------------------------------------------------------------------------
# Real novel local cases
# ---------------------------------------------------------------------------

def build_novel_local_cases(
    network: V73LocalFactorNetwork,
    words: list[str],
    limit: int = 100,
):
    selected = []
    seen = set()

    for word in words:
        for pos in range(len(word)):
            factors = network.factorize(
                word,
                pos,
                learn=False,
            )

            if min(factors) < 0:
                continue

            if (
                factors
                in network.v73_binding_by_key
            ):
                continue

            if factors in seen:
                continue

            evidence = network.pair_evidence(
                factors
            )

            selected.append(
                (
                    word,
                    pos,
                    factors,
                    evidence,
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


def evaluate_novel_local_cases(
    network: V73LocalFactorNetwork,
    candidates,
) -> None:
    print(
        "=== V73 NOVEL LOCAL COMPOSITIONS ==="
    )

    composed = 0
    branched = 0

    before_bindings = (
        network.binding_cell_count()
    )
    before_factors = (
        network.factor_cell_count()
    )

    for index, (
        word,
        pos,
        factors,
        evidence,
    ) in enumerate(
        candidates,
        start=1,
    ):
        baseline = network.baseline(
            factors
        )

        assert baseline == BRANCH

        action = network.autonomous(
            factors
        )

        if action == "COMPOSE":
            composed += 1
        elif action == BRANCH:
            branched += 1

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

    after_bindings = (
        network.binding_cell_count()
    )
    after_factors = (
        network.factor_cell_count()
    )

    print()
    print("candidates          :", len(candidates))
    print("composed            :", composed)
    print("branched            :", branched)
    print(
        "new_binding_cells   :",
        after_bindings - before_bindings,
    )
    print(
        "new_factor_cells    :",
        after_factors - before_factors,
    )

    assert (
        after_factors
        == before_factors
    )

    print(
        "=== END V73 NOVEL LOCAL COMPOSITIONS ==="
    )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    total_start = time.perf_counter()

    print(
        "=== V73 LOCAL-FACTOR DICTIONARY CORPUS ==="
    )
    print(
        "left_width =",
        LEFT_WIDTH,
        "right_width =",
        RIGHT_WIDTH,
    )
    print(
        "corpus =",
        CORPUS_PATH,
    )
    print()

    t = time.perf_counter()
    words = load_dictionary(
        CORPUS_PATH
    )
    print(
        f"[{elapsed(total_start):7.2f}s] "
        f"loaded {len(words)} unique words"
    )

    train, validation, test = split_words(
        words
    )
    print(
        f"[{elapsed(total_start):7.2f}s] "
        f"split train={len(train)} "
        f"validation={len(validation)} "
        f"test={len(test)}"
    )

    analyze_local_factor_space(
        train,
        validation,
        test,
    )

    network = build_network(
        train
    )

    train_binding_keys = set(
        network.v73_binding_by_key.keys()
    )

    print("=== V73 TRAINED GRAPH ===")
    print(
        "network_cells     :",
        len(network.cells),
    )
    print(
        "network_synapses  :",
        len(network.synapses),
    )
    print(
        "factor_cells      :",
        network.factor_cell_count(),
    )
    print(
        "binding_cells     :",
        network.binding_cell_count(),
    )
    print(
        "pair_synapses     :",
        network.pair_synapse_count(),
    )
    print(
        "compose_threshold :",
        network.v73_threshold,
    )
    print()

    evaluate_holdout(
        network,
        validation,
        train_binding_keys,
        "VALIDATION",
    )

    evaluate_holdout(
        network,
        test,
        train_binding_keys,
        "TEST",
    )

    candidates = build_novel_local_cases(
        network,
        test,
        limit=MAX_NOVEL_PROBES,
    )

    print(
        f"[{elapsed(total_start):7.2f}s] "
        f"selected {len(candidates)} novel local cases"
    )

    evaluate_novel_local_cases(
        network,
        candidates,
    )

    print(
        "=== V73 COMPLETE ==="
    )
    print(
        "total_elapsed :",
        f"{elapsed(total_start):.2f}s",
    )


if __name__ == "__main__":
    main()
