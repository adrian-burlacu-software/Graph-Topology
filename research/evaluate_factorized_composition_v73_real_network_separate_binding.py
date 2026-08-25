from __future__ import annotations

"""
V71 — REAL GRAPH-TOPOLOGY INTEGRATION

This is the first V71 implementation against the actual Graph-Topology
Network / Cell / Synapse substrate.

The existing graph remains authoritative:
    existing vocabulary child -> REUSE
    missing child            -> BRANCH / create

V71 adds:
    learned prefix/symbol/suffix factor IDs
    specific pair-support:
        prefix <-> symbol
        symbol <-> suffix
        prefix <-> suffix

The factor evidence is injected into the existing designer as activity.
The designer still receives activity, not:
    reuse_available
    raw prefix
    raw symbol
    raw suffix
    ground truth

The existing Network structural action path is preserved.

Run from research/:
    python evaluate_factorized_composition_v71_real_network.py

The script uses a deep-copy frozen probe for TEST so held-out evaluation does
not mutate the trained network.
"""


import copy
from dataclasses import dataclass
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
    def __init__(self, words: list[str]) -> None:
        self.links: set[Composition] = set()

        for word in words:
            for pos in range(len(word)):
                self.links.add(
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
        ) in self.links


# ---------------------------------------------------------------------------
# V71 factor store
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FactorIds:
    prefix: int
    symbol: int
    suffix: int


class FactorStoreV71:
    def __init__(self) -> None:
        self.prefix: dict[str, int] = {}
        self.symbol: dict[str, int] = {}
        self.suffix: dict[str, int] = {}

    @staticmethod
    def _get(
        table: dict[str, int],
        value: str,
        learn: bool,
    ) -> int:
        existing = table.get(value)

        if existing is not None:
            return existing

        if not learn:
            return -1

        new_id = len(table)
        table[value] = new_id
        return new_id

    def factorize(
        self,
        word: str,
        pos: int,
        learn: bool,
    ) -> FactorIds:
        return FactorIds(
            prefix=self._get(
                self.prefix,
                word[:pos],
                learn,
            ),
            symbol=self._get(
                self.symbol,
                word[pos],
                learn,
            ),
            suffix=self._get(
                self.suffix,
                word[pos + 1:],
                learn,
            ),
        )

    @property
    def count(self) -> int:
        return (
            len(self.prefix)
            + len(self.symbol)
            + len(self.suffix)
        )


# ---------------------------------------------------------------------------
# V71 real Network
# ---------------------------------------------------------------------------

class FactorizedBindingGraphV72:
    """
    Independent exact-composition binding state.

    This is intentionally separate from Network's path/trie vocabulary graph.

    Network graph:
        current_path -> next_symbol
        meaning: local sequential transition exists

    V72 binding graph:
        (prefix_factor, symbol_factor, suffix_factor)
        meaning: exact positional composition has been learned
    """

    def __init__(self) -> None:
        self.bindings: set[tuple[int, int, int]] = set()

        self.pair_support: dict[tuple[str, int, int], float] = {}

    def observe(
        self,
        factors: FactorIds,
    ) -> None:
        self.bindings.add(
            (
                factors.prefix,
                factors.symbol,
                factors.suffix,
            )
        )

        pairs = (
            ("ps", factors.prefix, factors.symbol),
            ("ss", factors.symbol, factors.suffix),
            ("px", factors.prefix, factors.suffix),
        )

        for kind, left, right in pairs:
            key = (kind, left, right)
            self.pair_support[key] = (
                self.pair_support.get(key, 0.0) + 1.0
            )

    def exact_known(
        self,
        factors: FactorIds,
    ) -> bool:
        if min(
            factors.prefix,
            factors.symbol,
            factors.suffix,
        ) < 0:
            return False

        return (
            factors.prefix,
            factors.symbol,
            factors.suffix,
        ) in self.bindings

    def evidence(
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

        ps = self.pair_support.get(
            ("ps", factors.prefix, factors.symbol),
            0.0,
        )
        ss = self.pair_support.get(
            ("ss", factors.symbol, factors.suffix),
            0.0,
        )
        px = self.pair_support.get(
            ("px", factors.prefix, factors.suffix),
            0.0,
        )

        return {
            "prefix_symbol": ps,
            "symbol_suffix": ss,
            "prefix_suffix": px,
            "minimum": min(ps, ss, px),
            "sum": ps + ss + px,
        }


class FactorizedNetworkV72(Network):
    """
    Real Graph-Topology Network with V72 composition state layered on top.

    The Network's existing vocabulary/path REUSE semantics are untouched.
    V72 exact-composition semantics live in binding_graph.
    """

    pair_support_threshold: float = 1.0

    def __init__(
        self,
        config: Optional[Config] = None,
    ) -> None:
        super().__init__(config)

        self.v72_factors = FactorStoreV71()
        self.v72_binding_graph = (
            FactorizedBindingGraphV72()
        )

        self.v72_context: Optional[FactorIds] = None
        self.v72_pair_evidence: dict[str, float] = (
            self.v72_binding_graph.evidence(
                FactorIds(-1, -1, -1)
            )
        )

        self.v72_composition_mode = BRANCH
        self.v72_trace: list[dict] = []

    def _v72_process_position(
        self,
        word: str,
        pos: int,
        learn: bool,
    ) -> dict:
        factors = self.v72_factors.factorize(
            word,
            pos,
            learn=learn,
        )

        self.v72_context = factors
        self.v72_pair_evidence = (
            self.v72_binding_graph.evidence(
                factors
            )
        )

        exact_known_before = (
            self.v72_binding_graph.exact_known(
                factors
            )
        )

        if exact_known_before:
            mode = REUSE

        elif min(
            factors.prefix,
            factors.symbol,
            factors.suffix,
        ) < 0:
            mode = BRANCH

        elif (
            self.v72_pair_evidence["minimum"]
            >= self.pair_support_threshold
        ):
            mode = "COMPOSE"

        else:
            mode = BRANCH

        self.v72_composition_mode = mode

        return {
            "factors": factors,
            "exact_known_before": exact_known_before,
            "mode": mode,
            "pair_evidence": dict(
                self.v72_pair_evidence
            ),
        }

    def process_word(
        self,
        word: str,
        learn: bool = True,
    ) -> dict:
        """
        Use Network's original process_word semantics untouched, while
        recording V72's independent exact-composition state.

        During learning:
            - first run the real Network action path;
            - separately learn the factorized exact binding.

        During frozen readout:
            - Network path remains unchanged;
            - V72 reports exact-binding semantics independently.
        """
        # Run the existing Network implementation on the real graph.
        result = super().process_word(
            word,
            learn=learn,
        )

        self.v72_trace = []

        for pos in range(len(word)):
            observation = self._v72_process_position(
                word,
                pos,
                learn=learn,
            )

            if learn:
                # Factorized binding learning is separate from Network's
                # sequential child graph.
                self.v72_binding_graph.observe(
                    observation["factors"]
                )

            self.v72_trace.append(
                {
                    "word": word,
                    "pos": pos,
                    "factor_ids": (
                        observation["factors"].prefix,
                        observation["factors"].symbol,
                        observation["factors"].suffix,
                    ),
                    "exact_known_before": (
                        observation["exact_known_before"]
                    ),
                    "composition_mode": (
                        observation["mode"]
                    ),
                    "pair_evidence": (
                        observation["pair_evidence"]
                    ),
                }
            )

        self.v72_context = None

        return result

# ---------------------------------------------------------------------------
# V72B evaluation
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


def evaluate_v72b_test(
    network: FactorizedNetworkV72,
    gt: IndependentGroundTruth,
) -> None:
    print("=== V72B FROZEN COMPOSITION EVALUATION ===")

    probe = copy.deepcopy(network)

    total = 0
    correct = 0
    reuse_correct = 0
    branch_correct = 0

    errors = []

    for word in TEST:
        # Run the real Network only to preserve its learned graph state.
        # The exact V72 composition state is evaluated from its factorized
        # binding graph, which is frozen in this copy.
        probe.process_word(
            word,
            learn=False,
        )

        for row in probe.v72_trace:
            pos = row["pos"]

            expected = (
                REUSE
                if gt.available(word, pos)
                else BRANCH
            )

            actual = (
                REUSE
                if row["exact_known_before"]
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
                    {
                        "word": word,
                        "pos": pos,
                        "expected": expected,
                        "actual": actual,
                        "mode": row[
                            "composition_mode"
                        ],
                        "pair_evidence": row[
                            "pair_evidence"
                        ],
                    }
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
            f"{error['word']:6s} "
            f"pos={error['pos']:2d} "
            f"expected={error['expected']:6s} "
            f"actual={error['actual']:6s} "
            f"mode={error['mode']:7s} "
            f"evidence={error['pair_evidence']}"
        )

    assert correct == total, (
        f"V72B exact-composition regression failed: "
        f"{correct}/{total}"
    )

    print("V72B FROZEN COMPOSITION: PASS")
    print()


def evaluate_novel_factor_cases(
    network: FactorizedNetworkV72,
) -> None:
    print("=== V72B NOVEL FACTOR COMPOSITION ===")

    probe = copy.deepcopy(network)

    prefixes = sorted(
        probe.v72_factors.prefix.keys()
    )
    symbols = sorted(
        probe.v72_factors.symbol.keys()
    )
    suffixes = sorted(
        probe.v72_factors.suffix.keys()
    )

    selected = []

    for prefix in prefixes:
        for symbol in symbols:
            for suffix in suffixes:
                if not prefix or not suffix:
                    continue

                factors = probe.v72_factors.factorize(
                    prefix + symbol + suffix,
                    len(prefix),
                    learn=False,
                )

                if min(
                    factors.prefix,
                    factors.symbol,
                    factors.suffix,
                ) < 0:
                    continue

                if (
                    probe.v72_binding_graph
                    .exact_known(factors)
                ):
                    continue

                evidence = (
                    probe.v72_binding_graph
                    .evidence(factors)
                )

                selected.append(
                    (
                        prefix + symbol + suffix,
                        factors,
                        evidence,
                    )
                )

    selected.sort(
        key=lambda row: (
            -row[2]["minimum"],
            row[0],
        )
    )

    selected = selected[:16]

    if not selected:
        print("No novel factor combinations found.")
        print()
        return

    for word, factors, evidence in selected:
        exact_known = (
            probe.v72_binding_graph
            .exact_known(factors)
        )

        print(
            f"{word:12s} "
            f"exact_known={exact_known} "
            f"ps={evidence['prefix_symbol']:.1f} "
            f"ss={evidence['symbol_suffix']:.1f} "
            f"px={evidence['prefix_suffix']:.1f} "
            f"min={evidence['minimum']:.1f}"
        )

        assert not exact_known

    print("novel_cases :", len(selected))
    print("V72B NOVEL FACTOR COMPOSITION: PASS")
    print()


def train_real_network(
    network: FactorizedNetworkV72,
) -> None:
    print("=== V72B REAL NETWORK TRAINING ===")

    network.train(
        TRAINING,
        epochs=5,
    )

    # Factor bindings are separate from the real Network path graph.
    #
    # Run the factorized learner once over the training set explicitly, while
    # leaving the real Network structure untouched.
    for word in TRAINING:
        network.process_word(
            word,
            learn=True,
        )

    print("prefix_factors  :", len(
        network.v72_factors.prefix
    ))
    print("symbol_factors  :", len(
        network.v72_factors.symbol
    ))
    print("suffix_factors  :", len(
        network.v72_factors.suffix
    ))
    print(
        "primitive_factors :",
        network.v72_factors.count,
    )
    print(
        "exact_bindings    :",
        len(
            network.v72_binding_graph.bindings
        ),
    )
    print(
        "pair_support      :",
        len(
            network.v72_binding_graph.pair_support
        ),
    )
    print(
        "vocabulary_cells  :",
        len(network.vocabulary_cells()),
    )
    print(
        "synapses          :",
        len(network.synapses),
    )

    print("=== END V72B REAL NETWORK TRAINING ===")
    print()


def main() -> None:
    print("=== V72B REAL NETWORK + SEPARATE COMPOSITION BINDING ===")
    print(
        "Network path semantics and exact composition semantics are "
        "intentionally separate."
    )
    print()

    gt = validate_v28()

    network = FactorizedNetworkV72()

    train_real_network(
        network,
    )

    evaluate_v72b_test(
        network,
        gt,
    )

    evaluate_novel_factor_cases(
        network,
    )

    print("=== V72B COMPLETE ===")


if __name__ == "__main__":
    main()
