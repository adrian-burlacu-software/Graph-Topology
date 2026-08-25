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


if __name__ == "__main__":
    regression_smoke()
