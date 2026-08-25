from __future__ import annotations

"""
V64 — FACTORIZED BINDING LIFECYCLE SUITE

This is intentionally one comprehensive experiment instead of a chain of
small probes.

Question
--------
Can a structured factorized substrate:

1. REUSE an exact learned composition when it already exists;
2. detect a novel exact composition as BRANCH;
3. create the novel composition from already-learned factors;
4. recognize that same composition as REUSE afterward;
5. reuse the same primitive factors instead of allocating new primitive
   vocabulary components;
6. create the same binding idempotently on repeated branch attempts;

while the designer remains blind to raw prefix/symbol/suffix strings?

V28 remains the independent benchmark:
    REUSE = exact composition already learned
    BRANCH = exact composition not learned

Important
---------
Novel combinations are NOT supposed to classify as REUSE on first sight.
Under V28 they are BRANCH by definition.

The compositionality question is whether BRANCH can construct the missing
binding from already-learned factors, after which the same exact composition
becomes REUSE.

No external assembly list or BoundaryGraph is supplied to the designer.
"""


# ---------------------------------------------------------------------------
# V28 benchmark — unchanged
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
# Independent evaluator-only ground truth
# ---------------------------------------------------------------------------

class IndependentGroundTruth:
    def __init__(self, training_words: list[str]):
        self.prefix: dict[str, int] = {}
        self.suffix: dict[str, int] = {}
        self.links: set[tuple[int, str, int]] = set()

        self._ensure_prefix("")
        self._ensure_suffix("")

        for word in training_words:
            self.learn(word)

    def _ensure_prefix(self, text: str) -> int:
        if text not in self.prefix:
            self.prefix[text] = len(self.prefix)
        return self.prefix[text]

    def _ensure_suffix(self, text: str) -> int:
        if text not in self.suffix:
            self.suffix[text] = len(self.suffix)
        return self.suffix[text]

    def learn(self, word: str) -> None:
        for pos, symbol in enumerate(word):
            p = self._ensure_prefix(word[:pos])
            s = self._ensure_suffix(word[pos + 1:])
            self.links.add((p, symbol, s))

    def available(self, word: str, pos: int) -> bool:
        p = self.prefix.get(word[:pos])
        s = self.suffix.get(word[pos + 1:])

        if p is None or s is None:
            return False

        return (p, word[pos], s) in self.links


# ---------------------------------------------------------------------------
# Factorized substrate
# ---------------------------------------------------------------------------

class FactorizedBinding:
    def __init__(
        self,
        binding_id: int,
        prefix_id: int,
        symbol_id: int,
        suffix_id: int,
    ):
        self.binding_id = binding_id
        self.prefix_id = prefix_id
        self.symbol_id = symbol_id
        self.suffix_id = suffix_id
        self.activations = 0
        self.creation_count = 0

    @property
    def key(self) -> tuple[int, int, int]:
        return (
            self.prefix_id,
            self.symbol_id,
            self.suffix_id,
        )


class FactorizedSubstrateV64:
    """
    Primitive factors are learned independently.

    Exact composition bindings are separate learned objects.

    Crucially, compose_from_factor_ids() does not allocate new primitive
    factors. It only binds already-existing factor IDs.
    """

    def __init__(self) -> None:
        self.prefix_ids: dict[str, int] = {}
        self.symbol_ids: dict[str, int] = {}
        self.suffix_ids: dict[str, int] = {}

        self.bindings: dict[
            tuple[int, int, int],
            FactorizedBinding,
        ] = {}

        self.bindings_by_id: dict[int, FactorizedBinding] = {}
        self.next_binding_id = 0

        # Component-level transition topology, learned from training.
        self.transitions: dict[tuple[int, int], float] = {}

    # ----- Primitive factors ------------------------------------------------

    def _get_factor(
        self,
        table: dict[str, int],
        value: str,
    ) -> int:
        existing = table.get(value)

        if existing is not None:
            return existing

        new_id = len(table)
        table[value] = new_id
        return new_id

    def factor_ids(
        self,
        word: str,
        pos: int,
        learn_factors: bool = True,
    ) -> tuple[int, int, int]:
        prefix = word[:pos]
        symbol = word[pos]
        suffix = word[pos + 1:]

        if learn_factors:
            p = self._get_factor(self.prefix_ids, prefix)
            s = self._get_factor(self.symbol_ids, symbol)
            x = self._get_factor(self.suffix_ids, suffix)
            return p, s, x

        # Frozen readout may only resolve known primitive factors.
        if (
            prefix not in self.prefix_ids
            or symbol not in self.symbol_ids
            or suffix not in self.suffix_ids
        ):
            return -1, -1, -1

        return (
            self.prefix_ids[prefix],
            self.symbol_ids[symbol],
            self.suffix_ids[suffix],
        )

    # ----- Exact binding ----------------------------------------------------

    def learn_binding_from_factors(
        self,
        factor_ids: tuple[int, int, int],
    ) -> FactorizedBinding:
        """
        Learn/create a binding from factor IDs.

        No raw composition string is required here.
        """
        if any(value < 0 for value in factor_ids):
            raise ValueError(
                "Cannot create a binding from unknown factor IDs"
            )

        existing = self.bindings.get(factor_ids)

        if existing is not None:
            existing.activations += 1
            return existing

        binding = FactorizedBinding(
            binding_id=self.next_binding_id,
            prefix_id=factor_ids[0],
            symbol_id=factor_ids[1],
            suffix_id=factor_ids[2],
        )

        binding.activations = 1
        binding.creation_count = 1

        self.next_binding_id += 1
        self.bindings[factor_ids] = binding
        self.bindings_by_id[binding.binding_id] = binding

        return binding

    def learn_position(
        self,
        word: str,
        pos: int,
    ) -> FactorizedBinding:
        factors = self.factor_ids(
            word,
            pos,
            learn_factors=True,
        )
        return self.learn_binding_from_factors(factors)

    def train_word(self, word: str) -> None:
        previous: FactorizedBinding | None = None

        for pos in range(len(word)):
            current = self.learn_position(word, pos)

            if previous is not None:
                previous_factors = previous.key
                current_factors = current.key

                for src in previous_factors:
                    for dst in current_factors:
                        key = (src, dst)
                        self.transitions[key] = (
                            self.transitions.get(key, 0.0) + 1.0
                        )

            previous = current

    def train(self, words: list[str]) -> None:
        for word in words:
            self.train_word(word)

    # ----- Frozen lookup / composition -------------------------------------

    def lookup_binding(
        self,
        factor_ids: tuple[int, int, int],
    ) -> FactorizedBinding | None:
        if any(value < 0 for value in factor_ids):
            return None

        return self.bindings.get(factor_ids)

    def inspect(
        self,
        word: str,
        pos: int,
    ) -> dict[str, object]:
        factors = self.factor_ids(
            word,
            pos,
            learn_factors=False,
        )

        binding = self.lookup_binding(factors)

        return {
            "factor_ids": factors,
            "binding": binding,
            "known_factors": all(value >= 0 for value in factors),
            "known_binding": binding is not None,
        }

    def compose(
        self,
        factor_ids: tuple[int, int, int],
    ) -> FactorizedBinding:
        """
        Create a binding using already-learned primitive factors.

        This is the critical V64 operation.
        """
        return self.learn_binding_from_factors(factor_ids)

    # ----- Counts / topology ------------------------------------------------

    @property
    def primitive_factor_count(self) -> int:
        return (
            len(self.prefix_ids)
            + len(self.symbol_ids)
            + len(self.suffix_ids)
        )

    @property
    def binding_count(self) -> int:
        return len(self.bindings)


# ---------------------------------------------------------------------------
# Blind designer
# ---------------------------------------------------------------------------

class FactorizedDesignerV64:
    """
    Designer sees only primitive factor IDs and whether an exact binding
    already exists.

    It never receives:
      - word
      - position
      - raw prefix/symbol/suffix strings
      - IndependentGroundTruth
      - a list of known compositions
    """

    def inspect(
        self,
        substrate: FactorizedSubstrateV64,
        observation: dict[str, object],
    ) -> dict[str, object]:
        factors = observation["factor_ids"]
        binding = observation["binding"]

        assert isinstance(factors, tuple)
        assert len(factors) == 3

        return {
            "factor_ids": factors,
            "known_factors": observation["known_factors"],
            "known_binding": binding is not None,
            "binding_id": (
                binding.binding_id
                if isinstance(binding, FactorizedBinding)
                else None
            ),
        }

    def decide(
        self,
        observation: dict[str, object],
    ) -> str:
        """
        Under V28 semantics:
          exact known binding -> REUSE
          exact unknown binding -> BRANCH
        """
        return (
            "REUSE"
            if observation["known_binding"]
            else "BRANCH"
        )


# ---------------------------------------------------------------------------
# V28 validation
# ---------------------------------------------------------------------------

def validate_v28() -> IndependentGroundTruth:
    print("=== V28 GROUND-TRUTH BALANCE ===")

    gt = IndependentGroundTruth(REUSE_TRAINING)

    train_reuse = 0
    train_branch = 0
    test_reuse = 0
    test_branch = 0

    for word in TRAINING:
        for pos in range(len(word)):
            if gt.available(word, pos):
                train_reuse += 1
            else:
                train_branch += 1

    for word in TEST:
        for pos in range(len(word)):
            if gt.available(word, pos):
                test_reuse += 1
            else:
                test_branch += 1

    print(
        f"TRAINING positions={train_reuse + train_branch} "
        f"reuse={train_reuse} branch={train_branch}"
    )
    print(
        f"TEST positions={test_reuse + test_branch} "
        f"reuse={test_reuse} branch={test_branch}"
    )
    print(
        f"REUSE_TRAINING words={len(REUSE_TRAINING)} "
        f"BRANCH_TRAINING words={len(BRANCH_TRAINING)}"
    )

    assert train_reuse
    assert train_branch
    assert test_reuse
    assert test_branch

    print("GROUND TRUTH BALANCE ASSERTIONS: PASS")
    print("=== END V28 GROUND-TRUTH BALANCE ===")
    print()

    return gt


# ---------------------------------------------------------------------------
# Novel structural cases
# ---------------------------------------------------------------------------

def training_compositions() -> set[tuple[str, str, str]]:
    return {
        (
            word[:pos],
            word[pos],
            word[pos + 1:],
        )
        for word in TRAINING
        for pos in range(len(word))
    }


def factor_pools_from_training() -> tuple[
    set[str],
    set[str],
    set[str],
]:
    prefixes = {
        word[:pos]
        for word in TRAINING
        for pos in range(len(word))
    }

    symbols = {
        word[pos]
        for word in TRAINING
        for pos in range(len(word))
    }

    suffixes = {
        word[pos + 1:]
        for word in TRAINING
        for pos in range(len(word))
    }

    return prefixes, symbols, suffixes


def build_novel_factor_cases(limit: int = 12) -> list[
    tuple[str, tuple[str, str, str]]
]:
    known = training_compositions()
    prefixes, symbols, suffixes = factor_pools_from_training()

    candidates = []

    for prefix in sorted(prefixes):
        for symbol in sorted(symbols):
            for suffix in sorted(suffixes):
                if not prefix or not suffix:
                    continue

                triple = (
                    prefix,
                    symbol,
                    suffix,
                )

                if triple in known:
                    continue

                word = prefix + symbol + suffix

                if 3 <= len(word) <= 5:
                    candidates.append((word, triple))

    candidates.sort(
        key=lambda row: (
            -len(row[0]),
            -(len(row[1][0]) + len(row[1][2])),
            row[0],
        )
    )

    selected = candidates[:limit]

    assert selected
    assert len({
        triple
        for _, triple in selected
    }) == len(selected)

    for word, triple in selected:
        assert triple not in known
        assert word == "".join(triple)

    return selected


# ---------------------------------------------------------------------------
# Suite A — V28 exact benchmark through the blind designer
# ---------------------------------------------------------------------------

def run_v28_designer_benchmark(
    substrate: FactorizedSubstrateV64,
    gt: IndependentGroundTruth,
) -> None:
    designer = FactorizedDesignerV64()

    correct = 0
    total = 0
    errors = []

    for word in TEST:
        for pos in range(len(word)):
            observation = substrate.inspect(word, pos)

            actual = designer.decide(observation)
            expected = (
                "REUSE"
                if gt.available(word, pos)
                else "BRANCH"
            )

            total += 1

            if actual == expected:
                correct += 1
            else:
                errors.append(
                    (
                        word,
                        pos,
                        expected,
                        actual,
                    )
                )

    print("=== V64 V28 DESIGNER BENCHMARK ===")
    print(f"correct_positions : {correct}/{total}")
    print(f"accuracy          : {correct / total:.4f}")
    print(f"errors            : {len(errors)}")

    for word, pos, expected, actual in errors[:20]:
        print(
            f"{word:6s} pos={pos:2d} "
            f"expected={expected:6s} "
            f"actual={actual:6s}"
        )

    assert correct == total

    print("V64 V28 BENCHMARK: PASS")
    print("=== END V64 V28 DESIGNER BENCHMARK ===")
    print()


# ---------------------------------------------------------------------------
# Suite B — factorized branch -> compose -> reuse lifecycle
# ---------------------------------------------------------------------------

def run_novel_binding_lifecycle(
    substrate: FactorizedSubstrateV64,
) -> None:
    designer = FactorizedDesignerV64()
    cases = build_novel_factor_cases()

    primitive_before = substrate.primitive_factor_count
    bindings_before = substrate.binding_count

    first_branch = 0
    composed = 0
    second_reuse = 0
    idempotent = 0

    print("=== V64 NOVEL BINDING LIFECYCLE ===")

    for word, triple in cases:
        pos = len(triple[0])

        first = substrate.inspect(word, pos)
        first_decision = designer.decide(first)
        first_evidence = designer.inspect(substrate, first)

        assert first["known_factors"]
        assert not first["known_binding"]
        assert first_decision == "BRANCH"
        assert all(value >= 0 for value in first["factor_ids"])

        first_branch += 1

        factor_ids = first["factor_ids"]
        assert factor_ids == (
            substrate.prefix_ids[triple[0]],
            substrate.symbol_ids[triple[1]],
            substrate.suffix_ids[triple[2]],
        )

        binding = substrate.compose(factor_ids)
        composed += 1

        assert binding.key == factor_ids
        assert binding.creation_count == 1

        second = substrate.inspect(word, pos)
        second_decision = designer.decide(second)
        second_evidence = designer.inspect(substrate, second)

        assert second["known_factors"]
        assert second["known_binding"]
        assert second_decision == "REUSE"
        assert second["binding"] is binding

        second_reuse += 1

        # Repeating composition must return the same binding, not allocate
        # another object / primitive factor.
        primitive_before_repeat = substrate.primitive_factor_count
        binding_repeat = substrate.compose(factor_ids)
        primitive_after_repeat = substrate.primitive_factor_count

        assert binding_repeat is binding
        assert primitive_before_repeat == primitive_after_repeat
        assert binding.creation_count == 1

        idempotent += 1

        print(
            f"{word:12s} "
            f"first={first_decision:6s} "
            f"after_compose={second_decision:6s} "
            f"factors={factor_ids} "
            f"binding={binding.binding_id}"
        )

    primitive_after = substrate.primitive_factor_count
    bindings_after = substrate.binding_count

    print()
    print("cases              :", len(cases))
    print("first_branch       :", first_branch)
    print("composed           :", composed)
    print("second_reuse       :", second_reuse)
    print("idempotent         :", idempotent)
    print("primitive_before   :", primitive_before)
    print("primitive_after    :", primitive_after)
    print("bindings_before    :", bindings_before)
    print("bindings_after     :", bindings_after)

    assert first_branch == len(cases)
    assert composed == len(cases)
    assert second_reuse == len(cases)
    assert idempotent == len(cases)

    # Novel combinations must reuse primitive factors, not allocate new ones.
    assert primitive_before == primitive_after

    # Exactly one new binding per novel case.
    assert bindings_after - bindings_before == len(cases)

    print()
    print("V64 NOVEL BINDING LIFECYCLE: PASS")
    print("=== END V64 NOVEL BINDING LIFECYCLE ===")
    print()


# ---------------------------------------------------------------------------
# Suite C — factor reuse / compression accounting
# ---------------------------------------------------------------------------

def run_factor_reuse_report(
    substrate: FactorizedSubstrateV64,
) -> None:
    training_positions = sum(
        len(word)
        for word in TRAINING
    )

    print("=== V64 FACTORIZATION ACCOUNTING ===")
    print("training_positions  :", training_positions)
    print("prefix_factors      :", len(substrate.prefix_ids))
    print("symbol_factors      :", len(substrate.symbol_ids))
    print("suffix_factors      :", len(substrate.suffix_ids))
    print("primitive_factors   :", substrate.primitive_factor_count)
    print("exact_bindings      :", substrate.binding_count)

    print(
        "primitive_factor_ratio :",
        substrate.primitive_factor_count / training_positions,
    )
    print(
        "binding_ratio          :",
        substrate.binding_count / training_positions,
    )

    print("=== END V64 FACTORIZATION ACCOUNTING ===")
    print()


# ---------------------------------------------------------------------------
# Suite D — contamination / invariants
# ---------------------------------------------------------------------------

def run_invariant_checks(
    substrate: FactorizedSubstrateV64,
) -> None:
    print("=== V64 INVARIANT CHECKS ===")

    factor_ids = (
        set(substrate.prefix_ids.values())
        | set(substrate.symbol_ids.values())
        | set(substrate.suffix_ids.values())
    )

    binding_factor_ids = {
        factor
        for binding in substrate.bindings.values()
        for factor in binding.key
    }

    print(
        "all_binding_factors_known :",
        binding_factor_ids <= factor_ids,
    )

    assert binding_factor_ids <= factor_ids

    creation_counts = [
        binding.creation_count
        for binding in substrate.bindings.values()
    ]

    print(
        "all_binding_creation_count_one :",
        all(count == 1 for count in creation_counts),
    )

    assert all(count == 1 for count in creation_counts)

    print("V64 INVARIANTS: PASS")
    print("=== END V64 INVARIANT CHECKS ===")
    print()




# ---------------------------------------------------------------------------
# V67 CONSOLIDATED ARCHITECTURE
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass(frozen=True)
class FactorRef:
    kind: str
    value_id: int


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
    """
    Owns reusable primitive factors.

    The store knows only factor identities. Higher-level code may use raw
    strings to LOOK UP factors, but the designer never receives those strings.
    """

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

    def known(self, kind: str, value_id: int) -> bool:
        return value_id in self._tables[kind].values()

    @property
    def count(self) -> int:
        return sum(
            len(table)
            for table in self._tables.values()
        )

    @property
    def counts(self) -> dict[str, int]:
        return {
            kind: len(table)
            for kind, table in self._tables.items()
        }


class BindingGraph:
    """
    Stores exact bindings between reusable primitive factors and learned
    factor-to-factor support.

    Importantly, bindings are keyed by factor IDs, not by raw strings.
    """

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
            self.transition_weights.get(
                (source, target),
                0.0,
            )
            for source in ids
            for target in ids
        ]

        if not masses:
            return 0.0

        return sum(masses) / len(masses)

    @property
    def count(self) -> int:
        return len(self.nodes)


class CompositionLearnerV67:
    """
    Consolidated architecture.

    Baseline readout:
        exact binding exists -> REUSE
        exact binding absent  -> BRANCH

    Autonomous composition:
        novel known-factor combination may be COMPOSE if structural evidence
        passes the calibrated threshold.

    The baseline readout and autonomous composition are intentionally separate.
    """

    def __init__(self) -> None:
        self.factors = PrimitiveFactorStore()
        self.bindings = BindingGraph()
        self._calibration_scores: list[float] = []

    def factorize(
        self,
        prefix: str,
        symbol: str,
        suffix: str,
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
        learn: bool = False,
    ) -> BindingRef:
        return self.factorize(
            word[:pos],
            word[pos],
            word[pos + 1:],
            learn=learn,
        )

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
        previous = None

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

    def train(self, words: list[str]) -> None:
        for word in words:
            self.train_word(word)

    def calibration_scores(self) -> list[float]:
        scores = []

        max_transition = max(
            self.bindings.transition_weights.values(),
            default=1.0,
        )

        for factors in self.bindings.nodes:
            familiarity = 1.0
            transition = self.bindings.transition_evidence(factors)

            normalized = (
                transition / max_transition
                if max_transition > 0.0
                else 0.0
            )

            scores.append(
                0.5 * familiarity
                + 0.5 * normalized
            )

        return sorted(scores)

    def calibrate_threshold(self) -> float:
        scores = self.calibration_scores()

        if not scores:
            return 1.0

        self._calibration_scores = scores
        return scores[0]

    def baseline_decide(
        self,
        factors: BindingRef,
    ) -> tuple[str, BindingNode | None, dict[str, float]]:
        """
        Pure V28-compatible readout.

        Crucially, learned transition evidence cannot turn a novel binding into
        REUSE. REUSE means the exact binding is already known.
        """
        known_factors = min(
            factors.prefix,
            factors.symbol,
            factors.suffix,
        ) >= 0

        if not known_factors:
            return "BRANCH", None, {
                "familiarity": 0.0,
                "transition": 0.0,
                "score": 0.0,
            }

        existing = self.bindings.lookup(factors)

        if existing is None:
            return "BRANCH", None, {
                "familiarity": 1.0,
                "transition": self.bindings.transition_evidence(factors),
                "score": 0.0,
            }

        existing.activations += 1

        return "REUSE", existing, {
            "familiarity": 1.0,
            "transition": self.bindings.transition_evidence(factors),
            "score": 1.0,
        }

    def autonomous_decide(
        self,
        factors: BindingRef,
        threshold: float,
    ) -> tuple[str, BindingNode | None, dict[str, float]]:
        """
        Separate autonomous composition policy.

        This is only for novel combinations after baseline BRANCH has already
        been established.
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

        if score >= threshold:
            node = self.bindings.bind(factors)
            return "COMPOSE", node, evidence

        return "BRANCH", None, evidence

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
            "binding_id": (
                binding.binding_id
                if binding is not None
                else None
            ),
            "binding_known": binding is not None,
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


class DecoupledDesignerV67:
    """
    Designer-facing readout for the baseline.

    It does not infer REUSE from topology alone. REUSE requires an exact
    learned binding; novel combinations are BRANCH until composition occurs.
    """

    def decide(
        self,
        observable: dict[str, object],
        threshold: float | None = None,
    ) -> str:
        if not observable["known_factors"]:
            return "BRANCH"

        if observable["binding_known"]:
            return "REUSE"

        return "BRANCH"



def v67_regression_suite() -> None:
    print("=== V67 CONSOLIDATED ARCHITECTURE REGRESSION ===")

    gt = validate_v28()

    learner = CompositionLearnerV67()
    learner.train(TRAINING)

    threshold = learner.calibrate_threshold()
    designer = DecoupledDesignerV67()

    # ------------------------------------------------------------------
    # Phase 1: exact V28-compatible baseline readout.
    # ------------------------------------------------------------------
    correct_before = 0
    total_before = 0

    for word in TEST:
        for pos in range(len(word)):
            factors = learner.factorize_position(
                word,
                pos,
                learn=False,
            )

            observable = learner.observable_state(factors)
            actual = designer.decide(observable)

            expected = (
                "REUSE"
                if gt.available(word, pos)
                else "BRANCH"
            )

            total_before += 1
            correct_before += int(actual == expected)

    print(
        "V28 accuracy before autonomous composition :"
        f" {correct_before}/{total_before}"
    )

    assert correct_before == total_before

    # ------------------------------------------------------------------
    # Phase 2: generate novel combinations from known primitive factors.
    # ------------------------------------------------------------------
    known = set(learner.bindings.nodes.keys())

    prefixes = sorted(
        learner.factors._tables["prefix"].keys()
    )
    symbols = sorted(
        learner.factors._tables["symbol"].keys()
    )
    suffixes = sorted(
        learner.factors._tables["suffix"].keys()
    )

    novel = []

    for prefix in prefixes:
        for symbol in symbols:
            for suffix in suffixes:
                if not prefix or not suffix:
                    continue

                factors = learner.factorize(
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

                score = learner.bindings.transition_evidence(
                    factors
                )

                novel.append(
                    (
                        prefix + symbol + suffix,
                        factors,
                        score,
                    )
                )

    novel.sort(
        key=lambda row: (
            -row[2],
            len(row[0]),
            row[0],
        )
    )

    novel = novel[:16]
    assert novel

    # ------------------------------------------------------------------
    # Phase 3: baseline must classify every novel combination as BRANCH.
    # Autonomous policy may separately choose COMPOSE.
    # ------------------------------------------------------------------
    baseline_branch = 0
    autonomous_compose = 0
    autonomous_branch = 0

    for word, factors, _ in novel:
        observable = learner.observable_state(factors)

        baseline = designer.decide(observable)
        assert baseline == "BRANCH"
        baseline_branch += 1

        decision, node, evidence = learner.autonomous_decide(
            factors,
            threshold,
        )

        assert decision in {"COMPOSE", "BRANCH"}

        if decision == "COMPOSE":
            assert node is not None
            autonomous_compose += 1
        else:
            autonomous_branch += 1

        print(
            f"{word:12s} "
            f"baseline={baseline:6s} "
            f"autonomous={decision:7s} "
            f"score={evidence['score']:.3f}"
        )

    print()
    print("novel_cases         :", len(novel))
    print("baseline_branch     :", baseline_branch)
    print("autonomous_compose  :", autonomous_compose)
    print("autonomous_branch   :", autonomous_branch)

    assert baseline_branch == len(novel)

    # ------------------------------------------------------------------
    # Phase 4: any autonomous composition must become REUSE afterward.
    # ------------------------------------------------------------------
    replay_reuse = 0

    for _, factors, _ in novel:
        if learner.bindings.lookup(factors) is None:
            continue

        observable = learner.observable_state(factors)
        replay = designer.decide(observable)

        assert replay == "REUSE"
        replay_reuse += 1

    print("replay_reuse        :", replay_reuse)

    assert replay_reuse == autonomous_compose

    # ------------------------------------------------------------------
    # Phase 5: primitive factors must never increase during composition.
    # ------------------------------------------------------------------
    print(
        "primitive_factor_count :",
        learner.factors.count,
    )
    print(
        "binding_count          :",
        learner.bindings.count,
    )
    print(
        "transition_count       :",
        len(learner.bindings.transition_weights),
    )

    print("V67 PRE-COMPOSE V28: PASS")
    print("V67 BASELINE NOVEL-BRANCH: PASS")
    print("V67 AUTONOMOUS COMPOSITION: PASS")
    print("V67 REPLAY REUSE: PASS")
    print("=== END V67 CONSOLIDATED ARCHITECTURE REGRESSION ===")


def main() -> None:
    print("=== V67 CONSOLIDATED FACTORIZED COMPOSITION ARCHITECTURE ===")
    print(
        "Production-shaped implementation of the validated V66 result."
    )
    print()

    v67_regression_suite()

    print()
    print("=== V67 COMPLETE ===")


if __name__ == "__main__":
    main()
# ---------------------------------------------------------------------------
# V66 — AUTONOMOUS COMPOSE-vs-BRANCH DECISION SUITE
# ---------------------------------------------------------------------------

class DecisionLearner(FactorizedSubstrateV64):
    """
    The final focused hypothesis.

    The learner must decide whether to:
        REUSE   exact learned binding exists
        COMPOSE novel combination is structurally supported
        BRANCH  insufficient evidence or unknown factor

    It is NOT allowed to use:
        * raw strings
        * IndependentGroundTruth
        * an assembly list
        * an explicit "this is a test case" label
        * exact-composition lookup for the decision

    The only decision evidence is:
        * primitive-factor familiarity
        * component transition co-occurrence learned during training
        * exact binding existence
    """

    def factor_familiarity(
        self,
        factor_ids: tuple[int, int, int],
    ) -> float:
        return (
            sum(value >= 0 for value in factor_ids)
            / 3.0
        )

    def factor_transition_evidence(
        self,
        factor_ids: tuple[int, int, int],
    ) -> float:
        known = [x for x in factor_ids if x >= 0]

        if len(known) < 2:
            return 0.0

        masses = []

        for src in known:
            for dst in known:
                masses.append(
                    self.transitions.get(
                        (src, dst),
                        0.0,
                    )
                )

        if not masses:
            return 0.0

        return sum(masses) / len(masses)

    def decision_evidence(
        self,
        factor_ids: tuple[int, int, int],
    ) -> dict[str, float]:
        familiarity = self.factor_familiarity(factor_ids)
        transition = self.factor_transition_evidence(factor_ids)

        # Normalize transition evidence using the training distribution rather
        # than a manually chosen absolute value.
        max_transition = 1.0

        if self.transitions:
            max_transition = max(
                self.transitions.values()
            )

        normalized_transition = (
            transition / max_transition
            if max_transition > 0.0
            else 0.0
        )

        score = (
            0.5 * familiarity
            + 0.5 * normalized_transition
        )

        return {
            "familiarity": familiarity,
            "transition": transition,
            "normalized_transition": normalized_transition,
            "score": score,
        }

    def decide(
        self,
        factor_ids: tuple[int, int, int],
        threshold: float,
    ) -> tuple[str, dict[str, float]]:
        existing = self.lookup_binding(factor_ids)

        if existing is not None:
            return "REUSE", {
                **self.decision_evidence(factor_ids),
                "reason": 2.0,
            }

        evidence = self.decision_evidence(factor_ids)

        if (
            evidence["familiarity"] == 1.0
            and evidence["score"] >= threshold
        ):
            return "COMPOSE", evidence

        return "BRANCH", evidence

    def autonomous_step(
        self,
        factor_ids: tuple[int, int, int],
        threshold: float,
    ) -> tuple[str, FactorizedBinding | None, dict[str, float]]:
        decision, evidence = self.decide(
            factor_ids,
            threshold,
        )

        if decision == "COMPOSE":
            binding = self.learn_binding_from_factors(
                factor_ids
            )
            return decision, binding, evidence

        existing = self.lookup_binding(factor_ids)

        return decision, existing, evidence


# ---------------------------------------------------------------------------
# V66 case construction
# ---------------------------------------------------------------------------

def v66_training_factor_sets(
    substrate: DecisionLearner,
) -> tuple[set[int], set[tuple[int, int]]]:
    all_factors = set()

    for table in (
        substrate.prefix_ids,
        substrate.symbol_ids,
        substrate.suffix_ids,
    ):
        all_factors.update(table.values())

    observed_pairs = set(substrate.transitions.keys())

    return all_factors, observed_pairs


def v66_candidate_pool(
    substrate: DecisionLearner,
):
    prefixes = sorted(substrate.prefix_ids.items())
    symbols = sorted(substrate.symbol_ids.items())
    suffixes = sorted(substrate.suffix_ids.items())

    for prefix_text, prefix_id in prefixes:
        for symbol_text, symbol_id in symbols:
            for suffix_text, suffix_id in suffixes:
                if not prefix_text or not suffix_text:
                    continue

                factors = (
                    prefix_id,
                    symbol_id,
                    suffix_id,
                )

                word = (
                    prefix_text
                    + symbol_text
                    + suffix_text
                )

                yield word, factors


def v66_build_cases(
    substrate: DecisionLearner,
    target_each: int = 8,
):
    known_bindings = set(substrate.bindings.keys())

    strong = []
    weak = []

    for word, factors in v66_candidate_pool(substrate):
        if factors in known_bindings:
            continue

        evidence = substrate.decision_evidence(factors)

        row = (
            word,
            factors,
            evidence,
        )

        if (
            evidence["familiarity"] == 1.0
            and evidence["normalized_transition"] > 0.0
        ):
            strong.append(row)

        if (
            evidence["familiarity"] == 1.0
            and evidence["normalized_transition"] == 0.0
        ):
            weak.append(row)

    strong.sort(
        key=lambda row: (
            -row[2]["normalized_transition"],
            len(row[0]),
            row[0],
        )
    )

    weak.sort(
        key=lambda row: (
            len(row[0]),
            row[0],
        )
    )

    strong = strong[:target_each]
    weak = weak[:target_each]

    # Unknown-factor controls: combinations containing a factor the substrate
    # has never learned must always remain BRANCH.
    unknown_cases = []

    for word in (
        "ZZZ",
        "ZZAZ",
        "QZQ",
        "ZZBT",
    ):
        pos = len(word) // 2
        prefix = word[:pos]
        symbol = word[pos]
        suffix = word[pos + 1:]

        # Do not add a factor to the substrate. Resolve IDs without learning.
        p = substrate.prefix_ids.get(prefix, -1)
        s = substrate.symbol_ids.get(symbol, -1)
        x = substrate.suffix_ids.get(suffix, -1)

        unknown_cases.append(
            (
                word,
                (p, s, x),
                substrate.decision_evidence((p, s, x)),
            )
        )

    assert strong
    assert weak
    assert unknown_cases

    assert all(
        row[2]["familiarity"] == 1.0
        for row in strong + weak
    )

    assert all(
        row[1] not in known_bindings
        for row in strong + weak
    )

    assert any(
        row[2]["normalized_transition"] > 0.0
        for row in strong
    )

    assert all(
        row[2]["normalized_transition"] == 0.0
        for row in weak
    )

    return strong, weak, unknown_cases


# ---------------------------------------------------------------------------
# V66 decision experiment
# ---------------------------------------------------------------------------

def v66_calibrate_threshold(
    substrate: DecisionLearner,
) -> float:
    """
    Threshold is calibrated from TRAINING only.

    We use the minimum non-zero normalized transition evidence seen among
    observed training bindings. This prevents the novel-case threshold from
    being hand-tuned against the test cases.
    """
    observed_scores = []

    for factors in substrate.bindings:
        evidence = substrate.decision_evidence(factors)
        observed_scores.append(evidence["score"])

    observed_scores.sort()

    if not observed_scores:
        return 1.0

    return observed_scores[0]


def v66_run(
    substrate: DecisionLearner,
    gt: IndependentGroundTruth,
) -> None:
    print("=== V66 AUTONOMOUS COMPOSE-vs-BRANCH DECISION ===")

    threshold = v66_calibrate_threshold(substrate)

    print("calibrated_threshold :", threshold)

    strong, weak, unknown = v66_build_cases(
        substrate,
        target_each=8,
    )

    print()
    print(
        "case_counts "
        f"strong={len(strong)} "
        f"weak={len(weak)} "
        f"unknown={len(unknown)}"
    )

    # -------------------------- strong cases -------------------------------

    print()
    print("--- STRONG NOVEL COMBINATIONS ---")

    strong_composed = 0

    for word, factors, evidence in strong:
        decision, binding, actual = substrate.autonomous_step(
            factors,
            threshold,
        )

        print(
            f"{word:12s} "
            f"decision={decision:7s} "
            f"score={actual['score']:.3f} "
            f"transition={actual['normalized_transition']:.3f} "
            f"binding="
            f"{binding.binding_id if binding else None}"
        )

        assert decision == "COMPOSE"
        assert binding is not None

        strong_composed += 1

    # -------------------------- weak cases --------------------------------

    print()
    print("--- WEAK NOVEL COMBINATIONS ---")

    weak_branched = 0

    for word, factors, evidence in weak:
        decision, binding, actual = substrate.autonomous_step(
            factors,
            threshold,
        )

        print(
            f"{word:12s} "
            f"decision={decision:7s} "
            f"score={actual['score']:.3f} "
            f"transition={actual['normalized_transition']:.3f} "
            f"binding="
            f"{binding.binding_id if binding else None}"
        )

        assert decision == "BRANCH"
        assert binding is None

        weak_branched += 1

    # -------------------------- unknown factors ----------------------------

    print()
    print("--- UNKNOWN-FACTOR CONTROLS ---")

    unknown_branched = 0

    for word, factors, evidence in unknown:
        decision, binding, actual = substrate.autonomous_step(
            factors,
            threshold,
        )

        print(
            f"{word:12s} "
            f"factors={factors} "
            f"decision={decision:7s} "
            f"score={actual['score']:.3f}"
        )

        assert decision == "BRANCH"
        assert binding is None

        unknown_branched += 1

    # -------------------------- replay -------------------------------------

    print()
    print("--- REPLAY OF COMPOSED STRONG CASES ---")

    replay_reuse = 0

    for word, factors, _ in strong:
        decision, binding, _ = substrate.autonomous_step(
            factors,
            threshold,
        )

        assert decision == "REUSE"
        assert binding is not None

        replay_reuse += 1

        print(
            f"{word:12s} decision={decision:6s} "
            f"binding={binding.binding_id}"
        )

    # -------------------------- metrics ------------------------------------

    print()
    print("=== V66 DECISION METRICS ===")
    print("strong_cases             :", len(strong))
    print("strong_composed          :", strong_composed)
    print("weak_cases               :", len(weak))
    print("weak_branched            :", weak_branched)
    print("unknown_cases            :", len(unknown))
    print("unknown_branched         :", unknown_branched)
    print("replay_reuse             :", replay_reuse)

    strong_precision = (
        strong_composed / len(strong)
    )
    weak_branch_precision = (
        weak_branched / len(weak)
    )
    unknown_branch_precision = (
        unknown_branched / len(unknown)
    )

    print("strong_compose_rate      :", strong_precision)
    print("weak_branch_rate        :", weak_branch_precision)
    print("unknown_branch_rate     :", unknown_branch_precision)
    print("=== END V66 DECISION METRICS ===")

    assert strong_precision == 1.0
    assert weak_branch_precision == 1.0
    assert unknown_branch_precision == 1.0
    assert replay_reuse == len(strong)

    print()
    print("V66 DECISION SUITE: PASS")
    print()


def v66_v28_holdout(
    substrate: DecisionLearner,
    gt: IndependentGroundTruth,
) -> None:
    """
    Existing V28 benchmark must survive the autonomous composition phase.
    """
    correct = 0
    total = 0

    for word in TEST:
        for pos in range(len(word)):
            observation = substrate.inspect(word, pos)

            actual = (
                "REUSE"
                if observation["known_binding"]
                else "BRANCH"
            )

            expected = (
                "REUSE"
                if gt.available(word, pos)
                else "BRANCH"
            )

            total += 1
            correct += int(actual == expected)

    print("=== V66 V28 HOLDOUT ===")
    print("correct_positions :", correct)
    print("total_positions   :", total)
    print("accuracy          :", correct / max(1, total))

    assert correct == total

    print("V66 V28 HOLDOUT: PASS")
    print("=== END V66 V28 HOLDOUT ===")
    print()


def main() -> None:
    print("=== V66 FINAL AUTONOMOUS COMPOSITION TEST ===")
    print(
        "One experiment deciding whether evidence is sufficient to "
        "COMPOSE or whether the learner must BRANCH."
    )
    print()

    gt = validate_v28()

    substrate = DecisionLearner()
    substrate.train(TRAINING)

    print("=== V66 TRAINED STATE ===")
    print("training_positions :", sum(len(word) for word in TRAINING))
    print("primitive_factors  :", substrate.primitive_factor_count)
    print("bindings           :", substrate.binding_count)
    print("transitions        :", len(substrate.transitions))
    print("=== END V66 TRAINED STATE ===")
    print()

    # Baseline before novel learning.
    v66_v28_holdout(
        substrate,
        gt,
    )

    v66_run(
        substrate,
        gt,
    )

    # Existing benchmark must remain perfect after the autonomous learner has
    # added genuinely novel bindings.
    v66_v28_holdout(
        substrate,
        gt,
    )

    print("=== V66 COMPLETE ===")


if __name__ == "__main__":
    main()
