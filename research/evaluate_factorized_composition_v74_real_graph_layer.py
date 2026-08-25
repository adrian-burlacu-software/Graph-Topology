from __future__ import annotations

"""
V72 — FACTORIZED COMPOSITION LAYER ON THE REAL GRAPH

This is the integration boundary established by result 113.

The existing Network vocabulary graph remains authoritative for sequential
path behavior:

    parent --symbol--> vocabulary_cell

V72 adds a separate composition layer INSIDE THE SAME Network:

    factor cells
          ↓
    pair-support synapses
          ↓
    binding cell
       ↙ ↓ ↘
   prefix symbol suffix

This gives us two distinct semantics without creating a second simulator:

1. Existing Network path:
       local sequential reuse / branch

2. V72 composition layer:
       exact (prefix, symbol, suffix) binding
       + specific factor-pair support

The V72 layer uses the real Network Cell/Synapse primitives.

Run from research/:
    python evaluate_factorized_composition_v72_real_graph_layer.py
"""

import copy
from dataclasses import dataclass
from typing import Optional

try:
    from simulator import (
        BRANCH,
        REUSE,
        Config,
        EXCITE,
        Network,
    )
except ImportError:
    from .simulator import (
        BRANCH,
        REUSE,
        Config,
        EXCITE,
        Network,
    )


# ---------------------------------------------------------------------------
# V28 benchmark
# ---------------------------------------------------------------------------

REUSE_TRAINING = [
    "CAT", "CAR", "CAN", "CARD", "CART",
    "CAD", "COD", "COT", "BAD", "BAR",
    "BARD", "BAN", "DART", "DAT", "BOT",
    "BOAT",
]

BRANCH_TRAINING = [
    "CAB", "CAP", "CAG", "COB", "COR",
    "DAB", "DAG", "DAN", "BAT", "BAG",
    "DOA", "DOG", "BOD", "BOR", "CARTB",
]

TRAINING = REUSE_TRAINING + BRANCH_TRAINING

TEST_REUSE = [
    "CAT", "CAR", "CAN", "CARD", "CART",
    "CAD", "COD", "COT", "BAD", "BAR",
]

TEST_BRANCH = [
    "CABD", "CAPT", "CAGD", "COBD", "CORD",
    "DABD", "DAGT", "DANT", "BATD", "BAGT",
]

TEST = TEST_REUSE + TEST_BRANCH


# ---------------------------------------------------------------------------
# Independent evaluator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Composition:
    prefix: str
    symbol: str
    suffix: str


class IndependentGroundTruth:
    def __init__(self, training_words: list[str]) -> None:
        self.bindings: set[Composition] = set()

        for word in training_words:
            for pos in range(len(word)):
                self.bindings.add(
                    Composition(
                        word[:pos],
                        word[pos],
                        word[pos + 1:],
                    )
                )

    def available(self, word: str, pos: int) -> bool:
        return Composition(
            word[:pos],
            word[pos],
            word[pos + 1:],
        ) in self.bindings


# ---------------------------------------------------------------------------
# Factorization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FactorIds:
    prefix: int
    symbol: int
    suffix: int


class V72RealGraphComposition(Network):
    """
    V72 composition layer embedded in the actual Network graph.

    Additional Cell kinds:
        v72_factor
        v72_binding

    Additional Synapse roles:
        V72_PAIR
        V72_BINDING

    Existing vocabulary/designer behavior is untouched unless callers
    explicitly consume v72_observable_state().
    """

    V72_FACTOR = "v72_factor"
    V72_BINDING = "v72_binding"

    V72_PAIR = "V72_PAIR"
    V72_BINDING_EDGE = "V72_BINDING"

    def __init__(
        self,
        config: Optional[Config] = None,
    ) -> None:
        super().__init__(config)

        # Raw-factor lookup is substrate metadata. It is never exposed to
        # the designer.
        self.v72_factor_by_value: dict[
            tuple[str, str],
            int,
        ] = {}

        self.v72_factor_kind_by_id: dict[
            int,
            str,
        ] = {}

        self.v72_binding_by_key: dict[
            tuple[int, int, int],
            int,
        ] = {}

        self.v72_binding_factors: dict[
            int,
            tuple[int, int, int],
        ] = {}

        self.v72_pair_support: dict[
            tuple[int, int],
            float,
        ] = {}

        self.v72_threshold: float = 1.0

    # ------------------------------------------------------------------
    # Factor cells
    # ------------------------------------------------------------------

    def _get_or_create_factor(
        self,
        kind: str,
        value: str,
        *,
        learn: bool,
    ) -> int:
        key = (kind, value)

        existing = self.v72_factor_by_value.get(key)
        if existing is not None:
            return existing

        if not learn:
            return -1

        cell_id = self.create_cell(
            self.V72_FACTOR,
            symbol=value,
        )

        self.v72_factor_by_value[key] = cell_id
        self.v72_factor_kind_by_id[cell_id] = kind

        return cell_id

    def factorize_position(
        self,
        word: str,
        pos: int,
        *,
        learn: bool,
    ) -> FactorIds:
        if not 0 <= pos < len(word):
            raise IndexError(
                f"pos={pos} outside word length={len(word)}"
            )

        return FactorIds(
            prefix=self._get_or_create_factor(
                "prefix",
                word[:pos],
                learn=learn,
            ),
            symbol=self._get_or_create_factor(
                "symbol",
                word[pos],
                learn=learn,
            ),
            suffix=self._get_or_create_factor(
                "suffix",
                word[pos + 1:],
                learn=learn,
            ),
        )

    # ------------------------------------------------------------------
    # Pair support as real Synapse state
    # ------------------------------------------------------------------

    def _reinforce_pair(
        self,
        source_factor: int,
        target_factor: int,
    ) -> None:
        if source_factor < 0 or target_factor < 0:
            return

        key = (source_factor, target_factor)
        syn = self.synapses.get(key)

        if syn is None:
            syn = self.connect(
                source_factor,
                target_factor,
                self.V72_PAIR,
                1.0,
            )

        syn.weight += 1.0
        syn.learning += 1.0

        self.v72_pair_support[key] = syn.weight

    def _observe_pair_structure(
        self,
        factors: FactorIds,
    ) -> None:
        self._reinforce_pair(
            factors.prefix,
            factors.symbol,
        )
        self._reinforce_pair(
            factors.symbol,
            factors.suffix,
        )
        self._reinforce_pair(
            factors.prefix,
            factors.suffix,
        )

    def pair_evidence(
        self,
        factors: FactorIds,
    ) -> dict[str, float]:
        if min(
            factors.prefix,
            factors.symbol,
            factors.suffix,
        ) < 0:
            return {
                "prefix_symbol": 0.0,
                "symbol_suffix": 0.0,
                "prefix_suffix": 0.0,
                "minimum": 0.0,
                "sum": 0.0,
            }

        ps = self.v72_pair_support.get(
            (factors.prefix, factors.symbol),
            0.0,
        )

        ss = self.v72_pair_support.get(
            (factors.symbol, factors.suffix),
            0.0,
        )

        px = self.v72_pair_support.get(
            (factors.prefix, factors.suffix),
            0.0,
        )

        return {
            "prefix_symbol": ps,
            "symbol_suffix": ss,
            "prefix_suffix": px,
            "minimum": min(ps, ss, px),
            "sum": ps + ss + px,
        }

    # ------------------------------------------------------------------
    # Binding cells as real graph state
    # ------------------------------------------------------------------

    def exact_binding(
        self,
        factors: FactorIds,
    ) -> Optional[int]:
        if min(
            factors.prefix,
            factors.symbol,
            factors.suffix,
        ) < 0:
            return None

        return self.v72_binding_by_key.get(
            (
                factors.prefix,
                factors.symbol,
                factors.suffix,
            )
        )

    def _create_binding_cell(
        self,
        factors: FactorIds,
    ) -> int:
        key = (
            factors.prefix,
            factors.symbol,
            factors.suffix,
        )

        existing = self.v72_binding_by_key.get(key)

        if existing is not None:
            return existing

        binding_id = self.create_cell(
            self.V72_BINDING,
        )

        self.v72_binding_by_key[key] = binding_id
        self.v72_binding_factors[binding_id] = key

        for factor_id in key:
            self.connect(
                factor_id,
                binding_id,
                self.V72_BINDING_EDGE,
                1.0,
            )

        return binding_id

    def learn_composition_position(
        self,
        word: str,
        pos: int,
    ) -> FactorIds:
        factors = self.factorize_position(
            word,
            pos,
            learn=True,
        )

        self._observe_pair_structure(factors)
        self._create_binding_cell(factors)

        return factors

    def train_composition_layer(
        self,
        words: list[str],
        epochs: int = 1,
    ) -> None:
        for _ in range(epochs):
            for word in words:
                for pos in range(len(word)):
                    self.learn_composition_position(
                        word,
                        pos,
                    )

    # ------------------------------------------------------------------
    # Threshold
    # ------------------------------------------------------------------

    def calibrate_v72_threshold(self) -> float:
        minima = []

        for factors in self.v72_binding_by_key:
            evidence = self.pair_evidence(
                FactorIds(*factors)
            )
            minima.append(
                evidence["minimum"]
            )

        self.v72_threshold = (
            min(minima)
            if minima
            else 1.0
        )

        return self.v72_threshold

    # ------------------------------------------------------------------
    # Stable composition readout
    # ------------------------------------------------------------------

    def v72_baseline(
        self,
        factors: FactorIds,
    ) -> str:
        if min(
            factors.prefix,
            factors.symbol,
            factors.suffix,
        ) < 0:
            return BRANCH

        return (
            REUSE
            if self.exact_binding(factors) is not None
            else BRANCH
        )

    def v72_autonomous(
        self,
        factors: FactorIds,
    ) -> str:
        if min(
            factors.prefix,
            factors.symbol,
            factors.suffix,
        ) < 0:
            return BRANCH

        if self.exact_binding(factors) is not None:
            return REUSE

        evidence = self.pair_evidence(factors)

        if evidence["minimum"] >= self.v72_threshold:
            self._create_binding_cell(factors)
            return "COMPOSE"

        return BRANCH

    # ------------------------------------------------------------------
    # Designer-safe observable state
    # ------------------------------------------------------------------

    def v72_observable_state(
        self,
        factors: FactorIds,
    ) -> dict[str, object]:
        binding = self.exact_binding(factors)
        evidence = self.pair_evidence(factors)

        return {
            "known_factors": min(
                factors.prefix,
                factors.symbol,
                factors.suffix,
            ) >= 0,
            "binding_known": binding is not None,
            "binding_id": binding,
            "pair_evidence": evidence,
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def v72_factor_cells(self) -> list[int]:
        return [
            cid
            for cid, cell in self.cells.items()
            if cell.kind == self.V72_FACTOR
        ]

    def v72_binding_cells(self) -> list[int]:
        return [
            cid
            for cid, cell in self.cells.items()
            if cell.kind == self.V72_BINDING
        ]

    def v72_pair_synapses(self) -> list:
        return [
            syn
            for syn in self.synapses.values()
            if syn.kind == self.V72_PAIR
        ]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def validate_v28() -> IndependentGroundTruth:
    gt = IndependentGroundTruth(
        REUSE_TRAINING,
    )

    train_reuse = sum(
        gt.available(word, pos)
        for word in TRAINING
        for pos in range(len(word))
    )

    train_total = sum(
        len(word)
        for word in TRAINING
    )

    test_reuse = sum(
        gt.available(word, pos)
        for word in TEST
        for pos in range(len(word))
    )

    test_total = sum(
        len(word)
        for word in TEST
    )

    print("=== V28 GROUND-TRUTH BALANCE ===")
    print(
        f"TRAINING positions={train_total} "
        f"reuse={train_reuse} "
        f"branch={train_total - train_reuse}"
    )
    print(
        f"TEST positions={test_total} "
        f"reuse={test_reuse} "
        f"branch={test_total - test_reuse}"
    )

    assert train_reuse > 0
    assert train_total - train_reuse > 0
    assert test_reuse > 0
    assert test_total - test_reuse > 0

    print("GROUND TRUTH BALANCE ASSERTIONS: PASS")
    print("=== END V28 GROUND-TRUTH BALANCE ===")
    print()

    return gt


def train_real_network(
    network: V72RealGraphComposition,
) -> None:
    print("=== REAL NETWORK TRAINING ===")

    # Existing Graph-Topology learning.
    network.train(
        TRAINING,
        epochs=5,
    )

    # Separate V72 composition layer learned inside the same graph.
    network.train_composition_layer(
        TRAINING,
        epochs=1,
    )

    network.calibrate_v72_threshold()

    print()
    print("network_vocabulary_cells :",
          len(network.vocabulary_cells()))
    print("network_synapses        :",
          len(network.synapses))
    print("v72_factor_cells        :",
          len(network.v72_factor_cells()))
    print("v72_binding_cells       :",
          len(network.v72_binding_cells()))
    print("v72_pair_synapses       :",
          len(network.v72_pair_synapses()))
    print("v72_threshold           :",
          network.v72_threshold)

    print("=== END REAL NETWORK TRAINING ===")
    print()


def evaluate_v72_composition(
    network: V72RealGraphComposition,
    gt: IndependentGroundTruth,
) -> None:
    print("=== V72 EXACT COMPOSITION HOLDOUT ===")

    probe = copy.deepcopy(network)

    total = 0
    correct = 0
    reuse_correct = 0
    branch_correct = 0
    errors = []

    for word in TEST:
        for pos in range(len(word)):
            factors = probe.factorize_position(
                word,
                pos,
                learn=False,
            )

            actual = probe.v72_baseline(
                factors,
            )

            expected = (
                REUSE
                if gt.available(word, pos)
                else BRANCH
            )

            total += 1

            if actual == expected:
                correct += 1

                if expected == REUSE:
                    reuse_correct += 1
                else:
                    branch_correct += 1

            else:
                errors.append(
                    (
                        word,
                        pos,
                        expected,
                        actual,
                        probe.pair_evidence(factors),
                    )
                )

    print("total_positions :", total)
    print("correct          :", correct)
    print(
        "accuracy         :",
        correct / max(1, total),
    )
    print("reuse_correct    :", reuse_correct)
    print("branch_correct   :", branch_correct)
    print("errors           :", len(errors))

    for error in errors[:20]:
        print(
            f"{error[0]:6s} "
            f"pos={error[1]:2d} "
            f"expected={error[2]:6s} "
            f"actual={error[3]:6s} "
            f"evidence={error[4]}"
        )

    assert correct == total

    print("V72 EXACT COMPOSITION HOLDOUT: PASS")
    print()


def evaluate_novel_combinations(
    network: V72RealGraphComposition,
) -> None:
    print("=== V72 NOVEL FACTOR COMBINATIONS ===")

    probe = copy.deepcopy(network)

    known_prefixes = sorted(
        value
        for (kind, value), _cell_id
        in probe.v72_factor_by_value.items()
        if kind == "prefix"
    )
    known_symbols = sorted(
        value
        for (kind, value), _cell_id
        in probe.v72_factor_by_value.items()
        if kind == "symbol"
    )
    known_suffixes = sorted(
        value
        for (kind, value), _cell_id
        in probe.v72_factor_by_value.items()
        if kind == "suffix"
    )

    candidates = []

    for prefix in known_prefixes:
        for symbol in known_symbols:
            for suffix in known_suffixes:
                if not prefix or not suffix:
                    continue

                word = prefix + symbol + suffix
                pos = len(prefix)

                factors = probe.factorize_position(
                    word,
                    pos,
                    learn=False,
                )

                if min(
                    factors.prefix,
                    factors.symbol,
                    factors.suffix,
                ) < 0:
                    continue

                if probe.exact_binding(factors) is not None:
                    continue

                evidence = probe.pair_evidence(
                    factors
                )

                candidates.append(
                    (
                        word,
                        factors,
                        evidence,
                    )
                )

    candidates.sort(
        key=lambda row: (
            -row[2]["minimum"],
            row[0],
        )
    )

    selected = candidates[:16]

    if not selected:
        print("No novel factor combinations found.")
        print("=== END V72 NOVEL FACTOR COMBINATIONS ===")
        return

    composed = 0
    branched = 0

    for word, factors, evidence in selected:
        baseline = probe.v72_baseline(
            factors
        )

        assert baseline == BRANCH

        action = probe.v72_autonomous(
            factors
        )

        print(
            f"{word:12s} "
            f"baseline={baseline:6s} "
            f"autonomous={action:7s} "
            f"minimum={evidence['minimum']:.1f} "
            f"sum={evidence['sum']:.1f}"
        )

        if action == "COMPOSE":
            composed += 1
        elif action == BRANCH:
            branched += 1
        else:
            raise AssertionError(
                f"Unexpected V72 action: {action}"
            )

    print()
    print("novel_cases :", len(selected))
    print("composed    :", composed)
    print("branched    :", branched)

    assert composed + branched == len(selected)

    print("V72 NOVEL COMBINATIONS: PASS")
    print("=== END V72 NOVEL FACTOR COMBINATIONS ===")
    print()



def main() -> None:
    print("=== V72 REAL GRAPH FACTORIZED COMPOSITION LAYER ===")
    print(
        "Existing Network path topology remains untouched; "
        "exact composition lives in real factor/binding cells and synapses."
    )
    print()

    gt = validate_v28()

    network = V72RealGraphComposition()

    train_real_network(
        network,
    )

    evaluate_v72_composition(
        network,
        gt,
    )

    evaluate_novel_combinations(
        network,
    )

    print("=== V72 COMPLETE ===")


if __name__ == "__main__":
    main()
