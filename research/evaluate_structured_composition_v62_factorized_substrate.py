from __future__ import annotations

"""
V62 — FACTORIZED STRUCTURED SUBSTRATE

Important correction from V61:
    CAT / CAR / CAN
and
    CAB / CAP / CAG

have the SAME structural shape: (prefix="CA", symbol, suffix="").
They differ only in the symbol value.

V62 therefore does not invent a semantic distinction between those groups.

Instead, it asks a cleaner question:

    Can the substrate factor a composition into reusable prefix/symbol/suffix
    components, while learning a separate binding for the exact combination?

Architecture
------------
prefix component       ─┐
symbol component        ├── learned binding ──> composition representation
suffix component       ─┘

The designer receives only:
    component IDs
    learned binding strength
    learned transition topology

The designer never receives:
    word
    position
    raw prefix/symbol/suffix strings
    independent ground truth
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


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
    def __init__(self, training_words: Iterable[str]):
        self.prefix: Dict[str, int] = {}
        self.suffix: Dict[str, int] = {}
        self.links = set()

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

@dataclass(frozen=True)
class ComponentKey:
    kind: str
    value: str


@dataclass
class FactorizedBinding:
    binding_id: int
    prefix_id: int
    symbol_id: int
    suffix_id: int
    strength: float = 0.0
    activations: int = 0


class FactorizedSubstrateV62:
    """
    Factorized representation.

    Prefix, symbol, and suffix components are shared independently.

    Exact compositions are represented by learned bindings between those
    components.

    Repeated exposure strengthens an existing binding instead of allocating
    another independent cell.
    """

    def __init__(self) -> None:
        self.components: Dict[ComponentKey, int] = {}
        self.component_by_id: Dict[int, ComponentKey] = {}
        self.next_component_id = 0

        self.bindings: Dict[Tuple[int, int, int], FactorizedBinding] = {}
        self.next_binding_id = 0

        # Learned component-to-component transition topology.
        self.transitions: Dict[Tuple[int, int], float] = defaultdict(float)

    @staticmethod
    def composition(word: str, pos: int) -> Tuple[str, str, str]:
        return (
            word[:pos],
            word[pos],
            word[pos + 1:],
        )

    def _component(self, kind: str, value: str) -> int:
        key = ComponentKey(kind, value)
        existing = self.components.get(key)

        if existing is not None:
            return existing

        cid = self.next_component_id
        self.next_component_id += 1

        self.components[key] = cid
        self.component_by_id[cid] = key
        return cid

    def component_ids(
        self,
        composition: Tuple[str, str, str],
    ) -> Tuple[int, int, int]:
        prefix, symbol, suffix = composition

        return (
            self._component("prefix", prefix),
            self._component("symbol", symbol),
            self._component("suffix", suffix),
        )

    def learn_position(self, word: str, pos: int) -> FactorizedBinding:
        composition = self.composition(word, pos)
        p, s, x = self.component_ids(composition)

        key = (p, s, x)
        binding = self.bindings.get(key)

        if binding is None:
            binding = FactorizedBinding(
                binding_id=self.next_binding_id,
                prefix_id=p,
                symbol_id=s,
                suffix_id=x,
            )
            self.next_binding_id += 1
            self.bindings[key] = binding

        binding.activations += 1
        binding.strength += 1.0

        return binding

    def train_word(self, word: str) -> None:
        previous: Optional[FactorizedBinding] = None

        for pos in range(len(word)):
            current = self.learn_position(word, pos)

            if previous is not None:
                previous_components = (
                    previous.prefix_id,
                    previous.symbol_id,
                    previous.suffix_id,
                )
                current_components = (
                    current.prefix_id,
                    current.symbol_id,
                    current.suffix_id,
                )

                for src in previous_components:
                    for dst in current_components:
                        self.transitions[(src, dst)] += 1.0

            previous = current

    def train(self, words: Iterable[str]) -> None:
        for word in words:
            self.train_word(word)

    def frozen_observation(
        self,
        word: str,
        pos: int,
    ) -> Dict[str, object]:
        composition = self.composition(word, pos)

        p, s, x = self.component_ids(composition)
        key = (p, s, x)

        binding = self.bindings.get(key)

        return {
            "prefix_id": p,
            "symbol_id": s,
            "suffix_id": x,
            "binding": binding,
            "known_binding": binding is not None,
            "binding_strength": (
                binding.strength if binding is not None else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# Decoupled designer
# ---------------------------------------------------------------------------

class FactorizedDesignerV62:
    """
    The designer receives IDs and learned weights only.

    It never receives the source strings composing those IDs.
    """

    def __init__(self, substrate: FactorizedSubstrateV62) -> None:
        self.substrate = substrate

    def observe(
        self,
        observation: Dict[str, object],
    ) -> Dict[str, object]:
        p = int(observation["prefix_id"])
        s = int(observation["symbol_id"])
        x = int(observation["suffix_id"])

        binding = observation["binding"]

        strength = float(observation["binding_strength"])

        component_ids = (p, s, x)

        transition_mass = 0.0
        for src in component_ids:
            for dst in component_ids:
                transition_mass += self.substrate.transitions.get(
                    (src, dst),
                    0.0,
                )

        return {
            "known_binding": bool(observation["known_binding"]),
            "binding_strength": strength,
            "transition_mass": transition_mass,
            "component_ids": component_ids,
        }

    def decide(
        self,
        observation: Dict[str, object],
    ) -> str:
        """
        Use only learned binding evidence.

        This intentionally does NOT inspect the semantic composition.
        """
        evidence = self.observe(observation)

        if evidence["known_binding"] and evidence["binding_strength"] > 0:
            return "REUSE"

        return "BRANCH"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def validate_v28() -> IndependentGroundTruth:
    print("=== V28 GROUND-TRUTH BALANCE ===")

    gt = IndependentGroundTruth(REUSE_TRAINING)

    train_reuse = []
    train_branch = []
    test_reuse = []
    test_branch = []

    for word in TRAINING:
        for pos in range(len(word)):
            row = (word, pos, word[pos])
            if gt.available(word, pos):
                train_reuse.append(row)
            else:
                train_branch.append(row)

    for word in TEST:
        for pos in range(len(word)):
            row = (word, pos, word[pos])
            if gt.available(word, pos):
                test_reuse.append(row)
            else:
                test_branch.append(row)

    print(
        f"TRAINING positions={len(train_reuse) + len(train_branch)} "
        f"reuse={len(train_reuse)} branch={len(train_branch)}"
    )
    print(
        f"TEST positions={len(test_reuse) + len(test_branch)} "
        f"reuse={len(test_reuse)} branch={len(test_branch)}"
    )
    print(
        f"REUSE_TRAINING words={len(REUSE_TRAINING)} "
        f"BRANCH_TRAINING words={len(BRANCH_TRAINING)}"
    )

    assert train_reuse, "V28 invalid: training has zero REUSE positions"
    assert train_branch, "V28 invalid: training has zero BRANCH positions"
    assert test_reuse, "V28 invalid: test has zero REUSE positions"
    assert test_branch, "V28 invalid: test has zero BRANCH positions"

    print("GROUND TRUTH BALANCE ASSERTIONS: PASS")
    print("=== END V28 GROUND-TRUTH BALANCE ===")
    print()

    return gt


def v62_capacity_report(
    substrate: FactorizedSubstrateV62,
    gt: IndependentGroundTruth,
) -> None:
    reuse_known = 0
    branch_novel = 0
    total_reuse = 0
    total_branch = 0

    for word in TEST:
        for pos in range(len(word)):
            obs = substrate.frozen_observation(word, pos)
            expected_reuse = gt.available(word, pos)

            if expected_reuse:
                total_reuse += 1
                reuse_known += int(obs["known_binding"])
            else:
                total_branch += 1
                branch_novel += int(not obs["known_binding"])

    print("=== V62 FACTORIZED CAPACITY ===")
    print("component_count      :", len(substrate.component_by_id))
    print("binding_count        :", len(substrate.bindings))
    print("transition_count     :", len(substrate.transitions))
    print("reuse_known          :", reuse_known)
    print("reuse_total          :", total_reuse)
    print("branch_novel         :", branch_novel)
    print("branch_total         :", total_branch)
    print(
        "reuse_binding_rate   :",
        reuse_known / max(1, total_reuse),
    )
    print(
        "branch_novelty_rate  :",
        branch_novel / max(1, total_branch),
    )
    print("=== END V62 FACTORIZED CAPACITY ===")
    print()


def v62_designer_eval(
    substrate: FactorizedSubstrateV62,
    gt: IndependentGroundTruth,
) -> None:
    designer = FactorizedDesignerV62(substrate)

    correct = 0
    total = 0

    print("=== V62 DECOUPLED DESIGNER ===")

    for word in TEST:
        for pos in range(len(word)):
            obs = substrate.frozen_observation(word, pos)
            actual = designer.decide(obs)

            expected = (
                "REUSE"
                if gt.available(word, pos)
                else "BRANCH"
            )

            total += 1
            correct += int(actual == expected)

            print(
                f"{word:6s} pos={pos} "
                f"components="
                f"{obs['prefix_id']}/"
                f"{obs['symbol_id']}/"
                f"{obs['suffix_id']} "
                f"binding={obs['binding_strength']:.1f} "
                f"actual={actual:6s} "
                f"expected={expected:6s}"
            )

    print()
    print("correct_positions :", correct)
    print("total_positions   :", total)
    print("accuracy          :", correct / max(1, total))
    print("=== END V62 DECOUPLED DESIGNER ===")
    print()


def v62_factor_reuse_report(
    substrate: FactorizedSubstrateV62,
) -> None:
    print("=== V62 FACTORIZATION ===")

    compositions = len(substrate.bindings)
    prefixes = sum(
        1
        for key in substrate.component_by_id.values()
        if key.kind == "prefix"
    )
    symbols = sum(
        1
        for key in substrate.component_by_id.values()
        if key.kind == "symbol"
    )
    suffixes = sum(
        1
        for key in substrate.component_by_id.values()
        if key.kind == "suffix"
    )

    print("prefix_components :", prefixes)
    print("symbol_components :", symbols)
    print("suffix_components :", suffixes)
    print("composition_bindings :", compositions)

    print()
    print(
        "This ratio is the important comparison against V61's "
        "one-cell-per-composition representation."
    )
    print(
        "component_count / binding_count =",
        len(substrate.component_by_id) / max(1, compositions),
    )

    print("=== END V62 FACTORIZATION ===")
    print()


def main() -> None:
    print("=== V62 FACTORIZED STRUCTURED SUBSTRATE ===")
    print(
        "CAT/CAR/CAN and CAB/CAP/CAG are treated as the same structural "
        "shape with different symbol components."
    )
    print()

    gt = validate_v28()

    substrate = FactorizedSubstrateV62()
    substrate.train(TRAINING)

    print("=== V62 TRAINING ===")
    print("training_positions :", sum(len(w) for w in TRAINING))
    print("components          :", len(substrate.component_by_id))
    print("bindings            :", len(substrate.bindings))
    print()

    v62_factor_reuse_report(substrate)
    v62_capacity_report(substrate, gt)
    v62_designer_eval(substrate, gt)

    print("=== V62 COMPLETE ===")


if __name__ == "__main__":
    main()
