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


# ---------------------------------------------------------------------------
# V71 INTEGRATION ENGINE
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FactorBinding:
    prefix_id: int
    symbol_id: int
    suffix_id: int


@dataclass
class BindingNode:
    binding_id: int
    factors: FactorBinding
    activations: int = 0


class PrimitiveFactorStore:
    def __init__(self) -> None:
        self._tables: dict[str, dict[str, int]] = {
            "prefix": {},
            "symbol": {},
            "suffix": {},
        }

    def learn(self, kind: str, value: str) -> int:
        table = self._tables[kind]
        if value in table:
            return table[value]

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
    def __init__(self) -> None:
        self.nodes: dict[FactorBinding, BindingNode] = {}
        self.by_id: dict[int, BindingNode] = {}
        self.next_id = 0

        # Exact role-pair support learned from observed bindings.
        self.pair_support: dict[
            tuple[str, int, int],
            float,
        ] = {}

        # Binding-to-binding / factor transition support.
        self.transitions: dict[tuple[int, int], float] = {}

    def lookup(
        self,
        factors: FactorBinding,
    ) -> BindingNode | None:
        return self.nodes.get(factors)

    def bind(
        self,
        factors: FactorBinding,
    ) -> BindingNode:
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

        self._reinforce_pair(
            "ps",
            factors.prefix_id,
            factors.symbol_id,
        )
        self._reinforce_pair(
            "ss",
            factors.symbol_id,
            factors.suffix_id,
        )
        self._reinforce_pair(
            "px",
            factors.prefix_id,
            factors.suffix_id,
        )

        return node

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

    def pair_evidence(
        self,
        factors: FactorBinding,
    ) -> dict[str, float]:
        ps = self.pair_support.get(
            ("ps", factors.prefix_id, factors.symbol_id),
            0.0,
        )
        ss = self.pair_support.get(
            ("ss", factors.symbol_id, factors.suffix_id),
            0.0,
        )
        px = self.pair_support.get(
            ("px", factors.prefix_id, factors.suffix_id),
            0.0,
        )

        return {
            "prefix_symbol": ps,
            "symbol_suffix": ss,
            "prefix_suffix": px,
            "minimum": min(ps, ss, px),
            "sum": ps + ss + px,
        }

    def reinforce_transition(
        self,
        previous: BindingNode,
        current: BindingNode,
    ) -> None:
        previous_ids = (
            previous.factors.prefix_id,
            previous.factors.symbol_id,
            previous.factors.suffix_id,
        )
        current_ids = (
            current.factors.prefix_id,
            current.factors.symbol_id,
            current.factors.suffix_id,
        )

        for source in previous_ids:
            for target in current_ids:
                key = (source, target)
                self.transitions[key] = (
                    self.transitions.get(key, 0.0) + 1.0
                )

    @property
    def count(self) -> int:
        return len(self.nodes)


class FactorizedCompositionEngineV71:
    """
    Runtime integration surface.

    The engine learns reusable primitive factors and exact bindings.
    Specific pair-support is the evidence used for autonomous composition.

    Baseline:
        known exact binding -> REUSE
        otherwise           -> BRANCH

    Autonomous:
        known factors + sufficient specific pair evidence -> COMPOSE
        otherwise                                      -> BRANCH
    """

    def __init__(self) -> None:
        self.factors = PrimitiveFactorStore()
        self.graph = BindingGraph()

        self._compose_threshold = 1.0

    # ------------------------------------------------------------------
    # Factorization
    # ------------------------------------------------------------------

    def factorize(
        self,
        prefix: str,
        symbol: str,
        suffix: str,
        *,
        learn: bool = False,
    ) -> FactorBinding:
        def resolve(kind: str, value: str) -> int:
            if learn:
                return self.factors.learn(kind, value)
            return self.factors.lookup(kind, value)

        return FactorBinding(
            prefix_id=resolve("prefix", prefix),
            symbol_id=resolve("symbol", symbol),
            suffix_id=resolve("suffix", suffix),
        )

    def factorize_position(
        self,
        word: str,
        pos: int,
        *,
        learn: bool = False,
    ) -> FactorBinding:
        if not 0 <= pos < len(word):
            raise IndexError(
                f"position={pos} out of range for length={len(word)}"
            )

        return self.factorize(
            word[:pos],
            word[pos],
            word[pos + 1:],
            learn=learn,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def observe(
        self,
        factors: FactorBinding,
    ) -> BindingNode:
        if min(
            factors.prefix_id,
            factors.symbol_id,
            factors.suffix_id,
        ) < 0:
            raise ValueError(
                "Cannot observe unknown primitive factors."
            )

        return self.graph.bind(factors)

    def train_word(self, word: str) -> None:
        previous: BindingNode | None = None

        for pos in range(len(word)):
            current_factors = self.factorize_position(
                word,
                pos,
                learn=True,
            )
            current = self.observe(current_factors)

            if previous is not None:
                self.graph.reinforce_transition(
                    previous,
                    current,
                )

            previous = current

    def train(
        self,
        words: Iterable[str],
    ) -> None:
        for word in words:
            self.train_word(word)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate_compose_threshold(self) -> float:
        observed = []

        for factors in self.graph.nodes:
            evidence = self.graph.pair_evidence(factors)
            observed.append(evidence["minimum"])

        self._compose_threshold = (
            min(observed)
            if observed
            else 1.0
        )

        return self._compose_threshold

    @property
    def compose_threshold(self) -> float:
        return self._compose_threshold

    # ------------------------------------------------------------------
    # Stable readout
    # ------------------------------------------------------------------

    def baseline_readout(
        self,
        factors: FactorBinding,
    ) -> tuple[str, BindingNode | None]:
        """
        Stable external readout.

        This path never composes. A composition is REUSE only after its exact
        binding exists.
        """
        if min(
            factors.prefix_id,
            factors.symbol_id,
            factors.suffix_id,
        ) < 0:
            return "BRANCH", None

        binding = self.graph.lookup(factors)

        if binding is None:
            return "BRANCH", None

        binding.activations += 1
        return "REUSE", binding

    # ------------------------------------------------------------------
    # Autonomous composition
    # ------------------------------------------------------------------

    def autonomous_readout(
        self,
        factors: FactorBinding,
    ) -> tuple[
        str,
        BindingNode | None,
        dict[str, float],
    ]:
        """
        Decide whether a novel combination has enough SPECIFIC evidence to
        compose.

        Generic factor popularity is not enough. All three pair roles are
        considered and the minimum support is the gate.
        """
        if min(
            factors.prefix_id,
            factors.symbol_id,
            factors.suffix_id,
        ) < 0:
            return "BRANCH", None, {
                "prefix_symbol": 0.0,
                "symbol_suffix": 0.0,
                "prefix_suffix": 0.0,
                "minimum": 0.0,
                "sum": 0.0,
            }

        existing = self.graph.lookup(factors)

        if existing is not None:
            existing.activations += 1
            return "REUSE", existing, (
                self.graph.pair_evidence(factors)
            )

        evidence = self.graph.pair_evidence(factors)

        if evidence["minimum"] >= self._compose_threshold:
            node = self.graph.bind(factors)
            return "COMPOSE", node, evidence

        return "BRANCH", None, evidence

    # ------------------------------------------------------------------
    # Designer-safe observable state
    # ------------------------------------------------------------------

    def observable_state(
        self,
        factors: FactorBinding,
    ) -> dict[str, object]:
        binding = self.graph.lookup(factors)

        return {
            "factor_ids": (
                factors.prefix_id,
                factors.symbol_id,
                factors.suffix_id,
            ),
            "known_factors": min(
                factors.prefix_id,
                factors.symbol_id,
                factors.suffix_id,
            ) >= 0,
            "binding_known": binding is not None,
            "binding_id": (
                binding.binding_id
                if binding is not None
                else None
            ),
            "pair_evidence": self.graph.pair_evidence(
                factors
            ),
        }


class DecoupledDesignerV71:
    """
    Designer-facing interface.

    It receives IDs and learned evidence only.
    It never receives raw structural strings.
    """

    @staticmethod
    def baseline(
        observable: dict[str, object],
    ) -> str:
        if not observable["known_factors"]:
            return "BRANCH"

        return (
            "REUSE"
            if observable["binding_known"]
            else "BRANCH"
        )

    @staticmethod
    def autonomous(
        observable: dict[str, object],
        compose_threshold: float,
    ) -> str:
        if not observable["known_factors"]:
            return "BRANCH"

        if observable["binding_known"]:
            return "REUSE"

        evidence = observable["pair_evidence"]

        if (
            isinstance(evidence, dict)
            and float(evidence["minimum"]) >= compose_threshold
        ):
            return "COMPOSE"

        return "BRANCH"


# ---------------------------------------------------------------------------
# Example integration contract
# ---------------------------------------------------------------------------

def smoke_test_v71() -> None:
    engine = FactorizedCompositionEngineV71()
    designer = DecoupledDesignerV71()

    training = [
        "CAT",
        "CAR",
        "CAN",
        "BOAT",
    ]

    engine.train(training)
    engine.calibrate_compose_threshold()

    # ------------------------------------------------------------------
    # 1. Known exact composition -> REUSE
    # ------------------------------------------------------------------

    known = engine.factorize(
        "CA",
        "T",
        "",
        learn=False,
    )

    observable = engine.observable_state(known)

    assert observable["known_factors"]
    assert designer.baseline(observable) == "REUSE"

    # ------------------------------------------------------------------
    # 2. Build a genuinely novel triple from factors that are already
    #    known individually.
    #
    #    We construct the candidate from the actual factor tables rather
    #    than assuming a literal suffix such as "AT" was learned.
    # ------------------------------------------------------------------

    known_prefixes = sorted(
        engine.factors._tables["prefix"].keys()
    )
    known_symbols = sorted(
        engine.factors._tables["symbol"].keys()
    )
    known_suffixes = sorted(
        engine.factors._tables["suffix"].keys()
    )

    known_bindings = set(
        engine.graph.nodes.keys()
    )

    novel = None

    for prefix in known_prefixes:
        for symbol in known_symbols:
            for suffix in known_suffixes:
                if not prefix or not suffix:
                    continue

                factors = engine.factorize(
                    prefix,
                    symbol,
                    suffix,
                    learn=False,
                )

                if min(
                    factors.prefix_id,
                    factors.symbol_id,
                    factors.suffix_id,
                ) < 0:
                    continue

                if factors in known_bindings:
                    continue

                novel = (
                    prefix + symbol + suffix,
                    factors,
                )
                break

            if novel is not None:
                break

        if novel is not None:
            break

    assert novel is not None, (
        "V71 smoke test could not construct a novel combination "
        "from already-learned factors"
    )

    novel_word, novel_factors = novel

    observable = engine.observable_state(
        novel_factors
    )

    assert observable["known_factors"]
    assert not observable["binding_known"]

    # Stable baseline: novel exact binding is BRANCH.
    assert designer.baseline(observable) == "BRANCH"

    # ------------------------------------------------------------------
    # 3. Autonomous policy may COMPOSE or BRANCH depending on learned
    #    specific evidence.
    # ------------------------------------------------------------------

    decision, node, evidence = (
        engine.autonomous_readout(
            novel_factors
        )
    )

    assert decision in {
        "COMPOSE",
        "BRANCH",
    }

    if decision == "COMPOSE":
        assert node is not None

        after = engine.observable_state(
            novel_factors
        )

        # Once composed, the stable baseline sees the exact binding as
        # REUSE.
        assert designer.baseline(after) == "REUSE"

    else:
        assert node is None

    # ------------------------------------------------------------------
    # 4. Unknown primitive factor -> BRANCH.
    # ------------------------------------------------------------------

    unknown = engine.factorize(
        "__UNKNOWN_PREFIX__",
        "__UNKNOWN_SYMBOL__",
        "__UNKNOWN_SUFFIX__",
        learn=False,
    )

    observable = engine.observable_state(
        unknown
    )

    assert not observable["known_factors"]
    assert designer.baseline(observable) == "BRANCH"
    assert (
        designer.autonomous(
            observable,
            engine.compose_threshold,
        )
        == "BRANCH"
    )

    print(
        "V71 novel smoke case :",
        novel_word,
    )
    print(
        "V71 novel decision   :",
        decision,
    )
    print(
        "V71 pair evidence    :",
        evidence,
    )
    print("V71 SMOKE TEST: PASS")


if __name__ == "__main__":
    smoke_test_v71()
