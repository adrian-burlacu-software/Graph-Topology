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

class FactorizedNetworkV71(Network):
    """
    Actual Graph-Topology Network with V71 factorized evidence.

    No separate binding graph is introduced: the existing vocabulary graph is
    the binding graph. A composition binding exists when the actual graph
    contains the corresponding child edge.

    Pair support is auxiliary learned state used to produce designer activity.
    """

    pair_support_threshold: float = 1.0
    pair_support_gain: float = 0.35

    def __init__(
        self,
        config: Optional[Config] = None,
    ) -> None:
        super().__init__(config)

        self.v71_factors = FactorStoreV71()
        self.v71_pair_support: dict[
            tuple[str, int, int],
            float,
        ] = {}

        self.v71_context: Optional[FactorIds] = None
        self.v71_pair_evidence: dict[str, float] = {
            "prefix_symbol": 0.0,
            "symbol_suffix": 0.0,
            "prefix_suffix": 0.0,
            "minimum": 0.0,
            "sum": 0.0,
        }

        self.v71_composition_mode = BRANCH
        self.v71_trace: list[dict] = []

    # ------------------------------------------------------------------
    # Factor evidence
    # ------------------------------------------------------------------

    def _v71_pair_key(
        self,
        kind: str,
        left: int,
        right: int,
    ) -> tuple[str, int, int]:
        return kind, left, right

    def _v71_observe_factors(
        self,
        factors: FactorIds,
    ) -> None:
        self.v71_pair_support[
            self._v71_pair_key(
                "ps",
                factors.prefix,
                factors.symbol,
            )
        ] = self.v71_pair_support.get(
            self._v71_pair_key(
                "ps",
                factors.prefix,
                factors.symbol,
            ),
            0.0,
        ) + 1.0

        self.v71_pair_support[
            self._v71_pair_key(
                "ss",
                factors.symbol,
                factors.suffix,
            )
        ] = self.v71_pair_support.get(
            self._v71_pair_key(
                "ss",
                factors.symbol,
                factors.suffix,
            ),
            0.0,
        ) + 1.0

        self.v71_pair_support[
            self._v71_pair_key(
                "px",
                factors.prefix,
                factors.suffix,
            )
        ] = self.v71_pair_support.get(
            self._v71_pair_key(
                "px",
                factors.prefix,
                factors.suffix,
            ),
            0.0,
        ) + 1.0

    def _v71_evidence(
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

        ps = self.v71_pair_support.get(
            self._v71_pair_key(
                "ps",
                factors.prefix,
                factors.symbol,
            ),
            0.0,
        )

        ss = self.v71_pair_support.get(
            self._v71_pair_key(
                "ss",
                factors.symbol,
                factors.suffix,
            ),
            0.0,
        )

        px = self.v71_pair_support.get(
            self._v71_pair_key(
                "px",
                factors.prefix,
                factors.suffix,
            ),
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
    # Designer integration
    # ------------------------------------------------------------------

    def _stimulate_local_context(
        self,
        current_id: Optional[int],
        symbol: str,
    ) -> tuple[float, float]:
        """
        Preserve the real simulator's sensory signal and add V71-specific
        pair-support activity to the existing context channel.

        Exact vocabulary matching remains represented by the graph's normal
        matching activity. V71 does not hand the designer a boolean lookup.
        """
        match_activity, context_activity = (
            super()._stimulate_local_context(
                current_id,
                symbol,
            )
        )

        factors = self.v71_context

        if factors is None:
            return match_activity, context_activity

        evidence = self.v71_pair_evidence

        # Only add V71 support when there is no exact existing child.
        # Existing matches already have their own graph activity path.
        exact_exists = bool(
            current_id is not None
            and self.find_child(current_id, symbol) is not None
        )

        if not exact_exists:
            normalized = min(
                1.0,
                evidence["minimum"]
                / self.pair_support_threshold,
            )

            context_activity += (
                normalized
                * self.pair_support_gain
            )

        return match_activity, context_activity

    # ------------------------------------------------------------------
    # Process path — mirrors current Network.process_word with V71 hooks
    # ------------------------------------------------------------------

    def process_word(
        self,
        word: str,
        learn: bool = True,
    ) -> dict:
        current_id: Optional[int] = None

        created = 0
        reused = 0
        branched = 0

        self.v71_trace = []

        for order, symbol in enumerate(word):
            factors = self.v71_factors.factorize(
                word,
                order,
                learn=learn,
            )

            self.v71_context = factors
            self.v71_pair_evidence = (
                self._v71_evidence(factors)
            )

            existing = self.find_child(
                current_id,
                symbol,
            )

            correct = (
                REUSE
                if existing is not None
                else BRANCH
            )

            # V71 semantic mode is separate from the existing graph action.
            if existing is not None:
                self.v71_composition_mode = REUSE

            elif (
                min(
                    factors.prefix,
                    factors.symbol,
                    factors.suffix,
                ) < 0
            ):
                self.v71_composition_mode = BRANCH

            elif (
                self.v71_pair_evidence["minimum"]
                >= self.pair_support_threshold
            ):
                # In the real simulator, BRANCH is the structural create path.
                # COMPOSE therefore remains an internal semantic label.
                self.v71_composition_mode = "COMPOSE"

            else:
                self.v71_composition_mode = BRANCH

            self._reset_designer_input()

            self.spike_designer(
                current_id,
                symbol,
            )

            action = self.designer_signal(
                current_id,
                symbol,
            )

            new_id, made, reused_now, reward = (
                self._apply_decision(
                    current_id,
                    symbol,
                    order,
                    action,
                )
            )

            current_id = new_id

            created += made
            reused += reused_now

            if action == BRANCH:
                branched += 1

            if made:
                self.total_create += made

            if reused_now:
                self.total_reuse += reused_now

            if learn:
                self.learn_designer(
                    action,
                    correct,
                    reward,
                )

                # Learn the factor relationships AFTER the structural action,
                # so first exposure cannot fabricate prior evidence.
                self._v71_observe_factors(factors)

            self.v71_trace.append(
                {
                    "word": word,
                    "pos": order,
                    "symbol": symbol,
                    "factor_ids": (
                        factors.prefix,
                        factors.symbol,
                        factors.suffix,
                    ),
                    "pair_evidence": dict(
                        self.v71_pair_evidence
                    ),
                    "composition_mode": (
                        self.v71_composition_mode
                    ),
                    "existing_before": existing is not None,
                    "action": action,
                    "created": made,
                    "reused": reused_now,
                    "reward": reward,
                }
            )

        self.v71_context = None

        return {
            "word": word,
            "created": created,
            "reused": reused,
            "branched": branched,
        }


# ---------------------------------------------------------------------------
# Evaluation helpers
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
    network: FactorizedNetworkV71,
) -> None:
    print("=== V71 REAL NETWORK TRAINING ===")
    network.train(
        TRAINING,
        epochs=5,
    )

    print()
    print("factor_prefixes :", len(network.v71_factors.prefix))
    print("factor_symbols  :", len(network.v71_factors.symbol))
    print("factor_suffixes :", len(network.v71_factors.suffix))
    print("primitive_factors :", network.v71_factors.count)
    print("pair_support_edges :", len(network.v71_pair_support))
    print("vocabulary_cells :", len(network.vocabulary_cells()))
    print("synapses :", len(network.synapses))
    print("=== END V71 REAL NETWORK TRAINING ===")
    print()


def frozen_probe_word(
    network: FactorizedNetworkV71,
    word: str,
) -> list[dict]:
    """
    Probe a deep copy so TEST does not mutate the trained graph.
    """
    probe = copy.deepcopy(network)

    result = probe.process_word(
        word,
        learn=False,
    )

    del result

    return probe.v71_trace


def evaluate_test(
    network: FactorizedNetworkV71,
    gt: IndependentGroundTruth,
) -> None:
    print("=== V71 FROZEN HELD-OUT EVALUATION ===")

    total = 0
    correct = 0

    reuse_correct = 0
    branch_correct = 0

    errors = []

    composition_modes = {
        REUSE: 0,
        "COMPOSE": 0,
        BRANCH: 0,
    }

    for word in TEST:
        trace = frozen_probe_word(
            network,
            word,
        )

        assert len(trace) == len(word)

        for row in trace:
            pos = row["pos"]
            expected = (
                REUSE
                if gt.available(word, pos)
                else BRANCH
            )

            actual = row["action"]

            total += 1

            if row["composition_mode"] in composition_modes:
                composition_modes[
                    row["composition_mode"]
                ] += 1

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
                        "composition_mode": row[
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
    print(
        "composition_modes:",
        composition_modes,
    )
    print("errors           :", len(errors))

    for error in errors[:20]:
        print(
            f"{error['word']:6s} "
            f"pos={error['pos']:2d} "
            f"expected={error['expected']:6s} "
            f"actual={error['actual']:6s} "
            f"mode={error['composition_mode']:7s} "
            f"evidence={error['pair_evidence']}"
        )

    print("=== END V71 FROZEN HELD-OUT EVALUATION ===")
    print()


def factorized_novel_probe(
    network: FactorizedNetworkV71,
) -> None:
    print("=== V71 NOVEL FACTOR COMBINATION PROBE ===")

    known_prefixes = sorted(
        network.v71_factors.prefix.keys()
    )
    known_symbols = sorted(
        network.v71_factors.symbol.keys()
    )
    known_suffixes = sorted(
        network.v71_factors.suffix.keys()
    )

    seen = set()

    for word in TRAINING:
        for pos in range(len(word)):
            seen.add(
                (
                    word[:pos],
                    word[pos],
                    word[pos + 1:],
                )
            )

    selected = []

    for prefix in known_prefixes:
        for symbol in known_symbols:
            for suffix in known_suffixes:
                if not prefix or not suffix:
                    continue

                triple = (
                    prefix,
                    symbol,
                    suffix,
                )

                if triple in seen:
                    continue

                selected.append(triple)

    selected = selected[:12]

    if not selected:
        print("No novel combinations available.")
        print("=== END V71 NOVEL FACTOR COMBINATION PROBE ===")
        return

    probe = copy.deepcopy(network)

    # We only inspect. We do not mutate the trained network.
    for prefix, symbol, suffix in selected:
        synthetic_word = prefix + symbol + suffix
        pos = len(prefix)

        factors = probe.v71_factors.factorize(
            synthetic_word,
            pos,
            learn=False,
        )

        evidence = probe._v71_evidence(
            factors
        )

        exact = (
            probe.find_child(
                None,
                symbol,
            )
            if pos == 0
            else None
        )

        del exact

        print(
            f"{synthetic_word:12s} "
            f"factor_ids="
            f"{factors.prefix}/"
            f"{factors.symbol}/"
            f"{factors.suffix} "
            f"pair_min={evidence['minimum']:.1f} "
            f"pair_sum={evidence['sum']:.1f}"
        )

    print("novel_cases :", len(selected))
    print("=== END V71 NOVEL FACTOR COMBINATION PROBE ===")
    print()


def main() -> None:
    print("=== V71 REAL GRAPH-TOPOLOGY INTEGRATION ===")
    print(
        "Existing Network vocabulary topology + "
        "factorized pair-support designer evidence."
    )
    print()

    gt = validate_v28()

    network = FactorizedNetworkV71()

    train_real_network(
        network,
    )

    evaluate_test(
        network,
        gt,
    )

    factorized_novel_probe(
        network,
    )

    print("=== V71 INTEGRATION COMPLETE ===")


if __name__ == "__main__":
    main()
