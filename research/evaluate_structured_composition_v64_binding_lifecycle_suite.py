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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== V64 FACTORIZED BINDING LIFECYCLE SUITE ===")
    print(
        "One comprehensive run: exact reuse, novel branch, "
        "factor-based composition, post-composition reuse, "
        "idempotence, compression, and invariants."
    )
    print()

    gt = validate_v28()

    substrate = FactorizedSubstrateV64()
    substrate.train(TRAINING)

    print("=== V64 TRAINED SUBSTRATE ===")
    print("training_positions :", sum(len(word) for word in TRAINING))
    print("primitive_factors  :", substrate.primitive_factor_count)
    print("exact_bindings     :", substrate.binding_count)
    print("transitions        :", len(substrate.transitions))
    print("=== END V64 TRAINED SUBSTRATE ===")
    print()

    run_v28_designer_benchmark(
        substrate,
        gt,
    )

    run_novel_binding_lifecycle(
        substrate,
    )

    run_factor_reuse_report(
        substrate,
    )

    run_invariant_checks(
        substrate,
    )

    print("=== V64 COMPLETE ===")


if __name__ == "__main__":
    main()
