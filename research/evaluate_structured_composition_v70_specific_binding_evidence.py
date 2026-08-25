from __future__ import annotations

"""
V68 — FACTORIZED COMPOSITION ENGINE

Integration baseline derived from the validated V67 architecture.

Core contract
-------------
Primitive factors:
    prefix / symbol / suffix

Binding graph:
    exact learned combinations of primitive factor IDs

Baseline readout:
    exact binding exists -> REUSE
    otherwise            -> BRANCH

Autonomous learning:
    known primitive factors + learned structural evidence -> COMPOSE

The designer interface receives only observable substrate state. It never
receives raw prefix/symbol/suffix strings, ground truth, or an external
assembly list.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# V70 BENCHMARK CORPUS
# ---------------------------------------------------------------------------

V70_REUSE_TRAINING = [
    "CAT", "CAR", "CAN", "CARD", "CART",
    "CAD", "COD", "COT", "BAD", "BAR",
    "BARD", "BAN", "DART", "DAT", "BOT",
    "BOAT",
]

V70_BRANCH_TRAINING = [
    "CAB", "CAP", "CAG", "COB", "COR",
    "DAB", "DAG", "DAN", "BAT", "BAG",
    "DOA", "DOG", "BOD", "BOR", "CARTB",
]

V70_TRAINING = (
    V70_REUSE_TRAINING
    + V70_BRANCH_TRAINING
)

V70_TEST_REUSE = [
    "CAT", "CAR", "CAN", "CARD", "CART",
    "CAD", "COD", "COT", "BAD", "BAR",
]

V70_TEST_BRANCH = [
    "CABD", "CAPT", "CAGD", "COBD", "CORD",
    "DABD", "DAGT", "DANT", "BATD", "BAGT",
]

V70_TEST = V70_TEST_REUSE + V70_TEST_BRANCH


def v70_independent_ground_truth() -> dict[tuple[str, int], bool]:
    known = set()

    for word in V70_REUSE_TRAINING:
        for pos in range(len(word)):
            known.add(
                (
                    word[:pos],
                    word[pos],
                    word[pos + 1:],
                )
            )

    result = {}

    for word in V70_TEST:
        for pos in range(len(word)):
            result[(word, pos)] = (
                (
                    word[:pos],
                    word[pos],
                    word[pos + 1:],
                )
                in known
            )

    return result


from typing import Iterable


@dataclass(frozen=True)
class BindingRef:
    prefix: int
    symbol: int
    suffix: int


@dataclass
class BindingNode:
    binding_id: int
    factors: BindingRef
    activations: int = 0


class PrimitiveFactorStore:
    """Reusable primitive factor vocabulary."""

    def __init__(self) -> None:
        self._tables: dict[str, dict[str, int]] = {
            "prefix": {},
            "symbol": {},
            "suffix": {},
        }

    def learn(self, kind: str, value: str) -> int:
        table = self._tables[kind]
        existing = table.get(value)
        if existing is not None:
            return existing

        value_id = len(table)
        table[value] = value_id
        return value_id

    def lookup(self, kind: str, value: str) -> int:
        return self._tables[kind].get(value, -1)

    @property
    def count(self) -> int:
        return sum(len(table) for table in self._tables.values())

    @property
    def counts(self) -> dict[str, int]:
        return {
            kind: len(table)
            for kind, table in self._tables.items()
        }


class BindingGraph:
    """Exact bindings plus learned factor transition support."""

    def __init__(self) -> None:
        self.nodes: dict[BindingRef, BindingNode] = {}
        self.by_id: dict[int, BindingNode] = {}
        self.next_id = 0
        self.transition_weights: dict[tuple[int, int], float] = {}

    def lookup(self, factors: BindingRef) -> BindingNode | None:
        return self.nodes.get(factors)

    def bind(self, factors: BindingRef) -> BindingNode:
        existing = self.nodes.get(factors)
        if existing is not None:
            existing.activations += 1
            return existing

        node = BindingNode(
            binding_id=self.next_id,
            factors=factors,
            activations=1,
        )
        self.next_id += 1
        self.nodes[factors] = node
        self.by_id[node.binding_id] = node
        return node

    def reinforce_transition(
        self,
        previous: BindingNode,
        current: BindingNode,
    ) -> None:
        previous_ids = (
            previous.factors.prefix,
            previous.factors.symbol,
            previous.factors.suffix,
        )
        current_ids = (
            current.factors.prefix,
            current.factors.symbol,
            current.factors.suffix,
        )

        for source in previous_ids:
            for target in current_ids:
                key = (source, target)
                self.transition_weights[key] = (
                    self.transition_weights.get(key, 0.0) + 1.0
                )

    def transition_evidence(
        self,
        factors: BindingRef,
    ) -> float:
        ids = (
            factors.prefix,
            factors.symbol,
            factors.suffix,
        )
        masses = [
            self.transition_weights.get((source, target), 0.0)
            for source in ids
            for target in ids
        ]
        return sum(masses) / len(masses) if masses else 0.0

    @property
    def count(self) -> int:
        return len(self.nodes)


class FactorizedCompositionEngine:
    """
    Production-shaped implementation of the validated V67 architecture.
    """

    def __init__(self) -> None:
        self.factors = PrimitiveFactorStore()
        self.bindings = BindingGraph()
        self._calibration_threshold = 1.0

    # ------------------------------------------------------------------
    # Structural factorization
    # ------------------------------------------------------------------

    def factorize(
        self,
        prefix: str,
        symbol: str,
        suffix: str,
        *,
        learn: bool = False,
    ) -> BindingRef:
        def resolve(kind: str, value: str) -> int:
            if learn:
                return self.factors.learn(kind, value)
            return self.factors.lookup(kind, value)

        return BindingRef(
            prefix=resolve("prefix", prefix),
            symbol=resolve("symbol", symbol),
            suffix=resolve("suffix", suffix),
        )

    def factorize_position(
        self,
        word: str,
        pos: int,
        *,
        learn: bool = False,
    ) -> BindingRef:
        if not (0 <= pos < len(word)):
            raise IndexError(
                f"position {pos} outside word of length {len(word)}"
            )

        return self.factorize(
            word[:pos],
            word[pos],
            word[pos + 1:],
            learn=learn,
        )

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def observe(self, factors: BindingRef) -> BindingNode:
        if min(
            factors.prefix,
            factors.symbol,
            factors.suffix,
        ) < 0:
            raise ValueError(
                "Cannot observe a binding with unknown primitive factors."
            )
        return self.bindings.bind(factors)

    def train_word(self, word: str) -> None:
        previous: BindingNode | None = None

        for pos in range(len(word)):
            factors = self.factorize_position(
                word,
                pos,
                learn=True,
            )
            current = self.observe(factors)

            if previous is not None:
                self.bindings.reinforce_transition(
                    previous,
                    current,
                )

            previous = current

    def train(self, words: Iterable[str]) -> None:
        for word in words:
            self.train_word(word)

    # ------------------------------------------------------------------
    # Decision policies
    # ------------------------------------------------------------------

    def baseline_readout(
        self,
        factors: BindingRef,
    ) -> tuple[str, BindingNode | None]:
        """
        Stable benchmark/readout contract.

        REUSE means exact binding already exists.
        """
        if min(
            factors.prefix,
            factors.symbol,
            factors.suffix,
        ) < 0:
            return "BRANCH", None

        binding = self.bindings.lookup(factors)
        if binding is None:
            return "BRANCH", None

        binding.activations += 1
        return "REUSE", binding

    def calibrate_autonomous_threshold(self) -> float:
        scores: list[float] = []

        max_transition = max(
            self.bindings.transition_weights.values(),
            default=1.0,
        )

        for factors in self.bindings.nodes:
            transition = self.bindings.transition_evidence(factors)
            normalized = (
                transition / max_transition
                if max_transition > 0.0
                else 0.0
            )
            scores.append(0.5 + 0.5 * normalized)

        self._calibration_threshold = (
            min(scores) if scores else 1.0
        )
        return self._calibration_threshold

    def autonomous_readout(
        self,
        factors: BindingRef,
    ) -> tuple[str, BindingNode | None, dict[str, float]]:
        """
        Autonomous learning policy.

        Existing binding:
            REUSE

        Novel known-factor combination with sufficient evidence:
            COMPOSE

        Otherwise:
            BRANCH
        """
        if min(
            factors.prefix,
            factors.symbol,
            factors.suffix,
        ) < 0:
            return "BRANCH", None, {
                "familiarity": 0.0,
                "transition": 0.0,
                "score": 0.0,
            }

        existing = self.bindings.lookup(factors)

        if existing is not None:
            existing.activations += 1
            return "REUSE", existing, {
                "familiarity": 1.0,
                "transition": self.bindings.transition_evidence(factors),
                "score": 1.0,
            }

        transition = self.bindings.transition_evidence(factors)
        max_transition = max(
            self.bindings.transition_weights.values(),
            default=1.0,
        )
        normalized = (
            transition / max_transition
            if max_transition > 0.0
            else 0.0
        )

        score = 0.5 + 0.5 * normalized

        evidence = {
            "familiarity": 1.0,
            "transition": transition,
            "normalized_transition": normalized,
            "score": score,
        }

        if score >= self._calibration_threshold:
            node = self.bindings.bind(factors)
            return "COMPOSE", node, evidence

        return "BRANCH", None, evidence

    # ------------------------------------------------------------------
    # Designer-safe observable state
    # ------------------------------------------------------------------

    def observable_state(
        self,
        factors: BindingRef,
    ) -> dict[str, object]:
        binding = self.bindings.lookup(factors)

        return {
            "factor_ids": (
                factors.prefix,
                factors.symbol,
                factors.suffix,
            ),
            "known_factors": min(
                factors.prefix,
                factors.symbol,
                factors.suffix,
            ) >= 0,
            "binding_known": binding is not None,
            "binding_id": (
                binding.binding_id
                if binding is not None
                else None
            ),
            "transition_evidence": (
                self.bindings.transition_evidence(factors)
                if min(
                    factors.prefix,
                    factors.symbol,
                    factors.suffix,
                ) >= 0
                else 0.0
            ),
        }


class DecoupledDesigner:
    """
    Designer-facing interface.

    This object deliberately has no access to raw structural strings.
    """

    @staticmethod
    def decide(
        observable: dict[str, object],
    ) -> str:
        if not observable["known_factors"]:
            return "BRANCH"

        if observable["binding_known"]:
            return "REUSE"

        return "BRANCH"


# ---------------------------------------------------------------------------
# Small regression contract
# ---------------------------------------------------------------------------

def regression_smoke() -> None:
    """
    Lightweight architectural smoke test.

    The full V28/V66 benchmark remains the research test. This smoke test
    verifies the core lifecycle without importing the benchmark corpus.
    """
    engine = FactorizedCompositionEngine()
    designer = DecoupledDesigner()

    engine.train([
        "CAT",
        "CAR",
        "CAN",
        "BOAT",
    ])

    engine.calibrate_autonomous_threshold()

    known = engine.factorize(
        "CA",
        "T",
        "",
        learn=False,
    )

    action, binding = engine.baseline_readout(known)
    assert action == "REUSE"
    assert binding is not None

    novel = engine.factorize(
        "CA",
        "D",
        "X",
        learn=False,
    )

    observable = engine.observable_state(novel)

    # Novel exact bindings remain BRANCH under the stable baseline.
    assert designer.decide(observable) == "BRANCH"

    # Autonomous policy may compose, but the final observable state must
    # become REUSE after composition.
    decision, created, _ = engine.autonomous_readout(novel)

    if decision == "COMPOSE":
        assert created is not None
        final_state = engine.observable_state(novel)
        assert designer.decide(final_state) == "REUSE"

    print("V68 SMOKE TEST: PASS")




# ---------------------------------------------------------------------------
# V70 SPECIFIC COMPOSITION EVIDENCE
# ---------------------------------------------------------------------------

class SpecificCompositionEngine(FactorizedCompositionEngine):
    """
    V70 refinement.

    In addition to the primitive-factor transition graph inherited from V68,
    learn a pairwise support table for the exact role combinations:

        (prefix_id, symbol_id)
        (symbol_id, suffix_id)
        (prefix_id, suffix_id)

    A novel triple is eligible for COMPOSE only when the specific triple has
    sufficient pairwise support.

    This deliberately avoids treating generic factor popularity as evidence
    that an arbitrary combination is meaningful.
    """

    def __init__(self) -> None:
        super().__init__()

        self.pair_support: dict[
            tuple[str, int, int],
            float,
        ] = {}

    def _reinforce_pair(
        self,
        kind: str,
        left: int,
        right: int,
    ) -> None:
        key = (kind, left, right)
        self.pair_support[key] = (
            self.pair_support.get(key, 0.0) + 1.0
        )

    def observe(self, factors: BindingRef) -> BindingNode:
        node = super().observe(factors)

        self._reinforce_pair(
            "ps",
            factors.prefix,
            factors.symbol,
        )
        self._reinforce_pair(
            "ss",
            factors.symbol,
            factors.suffix,
        )
        self._reinforce_pair(
            "px",
            factors.prefix,
            factors.suffix,
        )

        return node

    def pair_evidence(
        self,
        factors: BindingRef,
    ) -> dict[str, float]:
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

    def calibrate_specific_threshold(self) -> float:
        """
        Calibrate from learned exact bindings.

        Use the minimum pairwise support among observed training bindings.
        A novel triple must meet at least this level across all three pairings.
        """
        observed = []

        for factors in self.bindings.nodes:
            evidence = self.pair_evidence(factors)
            observed.append(evidence["minimum"])

        self._calibration_threshold = (
            min(observed)
            if observed
            else 1.0
        )

        return self._calibration_threshold

    def specific_autonomous_readout(
        self,
        factors: BindingRef,
    ) -> tuple[str, BindingNode | None, dict[str, float]]:
        if min(
            factors.prefix,
            factors.symbol,
            factors.suffix,
        ) < 0:
            return "BRANCH", None, {
                "minimum_pair_support": 0.0,
                "pair_sum": 0.0,
            }

        existing = self.bindings.lookup(factors)

        if existing is not None:
            existing.activations += 1

            evidence = self.pair_evidence(factors)
            return "REUSE", existing, evidence

        evidence = self.pair_evidence(factors)

        if (
            evidence["minimum"]
            >= self._calibration_threshold
        ):
            node = self.bindings.bind(factors)
            return "COMPOSE", node, evidence

        return "BRANCH", None, evidence


# ---------------------------------------------------------------------------
# V70 focused case selection
# ---------------------------------------------------------------------------

def v70_case_pool(
    engine: SpecificCompositionEngine,
):
    prefixes = sorted(
        engine.factors._tables["prefix"].keys()
    )
    symbols = sorted(
        engine.factors._tables["symbol"].keys()
    )
    suffixes = sorted(
        engine.factors._tables["suffix"].keys()
    )

    known = set(engine.bindings.nodes.keys())

    cases = []

    for prefix in prefixes:
        for symbol in symbols:
            for suffix in suffixes:
                if not prefix or not suffix:
                    continue

                factors = engine.factorize(
                    prefix,
                    symbol,
                    suffix,
                    learn=False,
                )

                if min(
                    factors.prefix,
                    factors.symbol,
                    factors.suffix,
                ) < 0:
                    continue

                if factors in known:
                    continue

                evidence = engine.pair_evidence(factors)

                cases.append(
                    (
                        prefix + symbol + suffix,
                        factors,
                        evidence,
                    )
                )

    return cases


def v70_run_specificity_test(
    engine: SpecificCompositionEngine,
) -> None:
    print("=== V70 SPECIFIC COMPOSITION SELECTIVITY ===")

    threshold = engine.calibrate_specific_threshold()

    cases = v70_case_pool(engine)

    # Split by minimum pairwise support. We want both strongly supported and
    # unsupported novel combinations where the corpus allows them.
    supported = [
        row for row in cases
        if row[2]["minimum"] >= threshold
    ]

    unsupported = [
        row for row in cases
        if row[2]["minimum"] < threshold
    ]

    supported.sort(
        key=lambda row: (
            -row[2]["minimum"],
            row[0],
        )
    )
    unsupported.sort(
        key=lambda row: (
            row[2]["minimum"],
            row[0],
        )
    )

    supported = supported[:12]
    unsupported = unsupported[:12]

    print("calibrated_pair_threshold :", threshold)
    print("supported_novel_cases     :", len(supported))
    print("unsupported_novel_cases   :", len(unsupported))

    assert supported, (
        "V70 needs at least one structurally supported novel combination"
    )
    assert unsupported, (
        "V70 needs at least one structurally unsupported novel combination"
    )

    print()
    print("--- SUPPORTED NOVEL COMBINATIONS ---")

    supported_composed = 0

    for word, factors, evidence in supported:
        decision, binding, _ = (
            engine.specific_autonomous_readout(factors)
        )

        print(
            f"{word:12s} "
            f"decision={decision:7s} "
            f"ps={evidence['prefix_symbol']:.1f} "
            f"ss={evidence['symbol_suffix']:.1f} "
            f"px={evidence['prefix_suffix']:.1f} "
            f"min={evidence['minimum']:.1f}"
        )

        assert decision == "COMPOSE"
        assert binding is not None
        supported_composed += 1

    print()
    print("--- UNSUPPORTED NOVEL COMBINATIONS ---")

    unsupported_branched = 0

    for word, factors, evidence in unsupported:
        decision, binding, _ = (
            engine.specific_autonomous_readout(factors)
        )

        print(
            f"{word:12s} "
            f"decision={decision:7s} "
            f"ps={evidence['prefix_symbol']:.1f} "
            f"ss={evidence['symbol_suffix']:.1f} "
            f"px={evidence['prefix_suffix']:.1f} "
            f"min={evidence['minimum']:.1f}"
        )

        assert decision == "BRANCH"
        assert binding is None
        unsupported_branched += 1

    print()
    print("supported_composed   :", supported_composed)
    print("unsupported_branched  :", unsupported_branched)

    assert supported_composed == len(supported)
    assert unsupported_branched == len(unsupported)

    print("V70 SPECIFIC SELECTIVITY: PASS")
    print()


def v70_v28_regression(
    engine: SpecificCompositionEngine,
) -> None:
    """
    Stable baseline: exact known bindings are REUSE; novel test positions
    remain BRANCH. This does not use autonomous composition during the test.
    """
    gt = v70_independent_ground_truth()

    correct = 0
    total = 0

    for word in V70_TEST:
        for pos in range(len(word)):
            factors = engine.factorize_position(
                word,
                pos,
                learn=False,
            )

            action = (
                "REUSE"
                if engine.bindings.lookup(factors)
                is not None
                else "BRANCH"
            )

            expected = (
                "REUSE"
                if gt[(word, pos)]
                else "BRANCH"
            )

            total += 1
            correct += int(action == expected)

    print("V70 baseline accuracy :", f"{correct}/{total}")
    assert correct == total

    print("V70 BASELINE REGRESSION: PASS")
    print()


def main() -> None:
    print("=== V70 SPECIFIC COMPOSITION EVIDENCE ===")
    print(
        "Novel bindings must be supported by the specific factor pairings, "
        "not generic factor transition mass."
    )
    print()

    engine = SpecificCompositionEngine()
    engine.train(V70_TRAINING)

    print("=== V70 TRAINED STATE ===")
    print("primitive_factors :", engine.factors.count)
    print("bindings          :", engine.bindings.count)
    print("transitions       :", len(engine.bindings.transition_weights))
    print("pair_support      :", len(engine.pair_support))
    print()

    v70_v28_regression(engine)
    v70_run_specificity_test(engine)

    print("=== V70 COMPLETE ===")


if __name__ == "__main__":
    main()
