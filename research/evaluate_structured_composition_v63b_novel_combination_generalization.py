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
# V63 NOVEL COMBINATION TEST
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

    assert train_reuse, "V28 invalid: training has zero REUSE positions"
    assert train_branch, "V28 invalid: training has zero BRANCH positions"
    assert test_reuse, "V28 invalid: test has zero REUSE positions"
    assert test_branch, "V28 invalid: test has zero BRANCH positions"

    print("GROUND TRUTH BALANCE ASSERTIONS: PASS")
    print("=== END V28 GROUND-TRUTH BALANCE ===")
    print()

    return gt


def v63_factor_pools():
    prefixes = sorted({
        word[:pos]
        for word in TRAINING
        for pos in range(len(word))
    })
    symbols = sorted({
        word[pos]
        for word in TRAINING
        for pos in range(len(word))
    })
    suffixes = sorted({
        word[pos + 1:]
        for word in TRAINING
        for pos in range(len(word))
    })
    return prefixes, symbols, suffixes


def v63_build_novel_compositions(limit=12):
    """
    Generate exact (prefix, symbol, suffix) triples that:
      1. use only factors already observed during training;
      2. never occurred as an exact triple during training.
    """
    training_compositions = {
        (
            word[:pos],
            word[pos],
            word[pos + 1:],
        )
        for word in TRAINING
        for pos in range(len(word))
    }

    prefixes, symbols, suffixes = v63_factor_pools()
    candidates = []

    for prefix in prefixes:
        for symbol in symbols:
            for suffix in suffixes:
                triple = (prefix, symbol, suffix)

                if triple in training_compositions:
                    continue

                if not prefix and not suffix:
                    continue

                word = prefix + symbol + suffix
                if not word:
                    continue

                candidates.append((triple, word))

    candidates.sort(
        key=lambda row: (
            len(row[0][0]) + len(row[0][2]),
            row[0][0],
            row[0][1],
            row[0][2],
        )
    )

    selected = candidates[:limit]

    assert selected, "V63 invalid: no novel factor combinations found."

    for triple, word in selected:
        assert triple not in training_compositions
        assert word == triple[0] + triple[1] + triple[2]

    return selected


def v63_factor_availability(
    substrate: FactorizedSubstrateV62,
    word: str,
    pos: int,
) -> dict:
    prefix, symbol, suffix = substrate.composition(word, pos)

    prefix_key = ComponentKey("prefix", prefix)
    symbol_key = ComponentKey("symbol", symbol)
    suffix_key = ComponentKey("suffix", suffix)

    prefix_known = prefix_key in substrate.components
    symbol_known = symbol_key in substrate.components
    suffix_known = suffix_key in substrate.components

    exact_known = False

    if prefix_known and symbol_known and suffix_known:
        key = (
            substrate.components[prefix_key],
            substrate.components[symbol_key],
            substrate.components[suffix_key],
        )
        exact_known = key in substrate.bindings

    return {
        "prefix": prefix,
        "symbol": symbol,
        "suffix": suffix,
        "prefix_known": prefix_known,
        "symbol_known": symbol_known,
        "suffix_known": suffix_known,
        "all_factors_known": (
            prefix_known and symbol_known and suffix_known
        ),
        "exact_binding_known": exact_known,
    }


def v63_run_novel_combination_test(
    substrate: FactorizedSubstrateV62,
) -> None:
    generated = v63_build_novel_compositions()

    print("=== V63 NOVEL COMBINATION TEST ===")
    print("generated_positions :", len(generated))

    all_factors_known = 0
    exact_binding_known = 0
    exact_binding_novel = 0

    for triple, word in generated:
        prefix, symbol, suffix = triple
        pos = len(prefix)

        actual = substrate.composition(word, pos)
        assert actual == triple

        evidence = v63_factor_availability(
            substrate,
            word,
            pos,
        )

        all_factors_known += int(evidence["all_factors_known"])
        exact_binding_known += int(
            evidence["exact_binding_known"]
        )
        exact_binding_novel += int(
            not evidence["exact_binding_known"]
        )

        print(
            f"{word:16s} pos={pos:2d} "
            f"composition={actual!r} "
            f"factors_known={evidence['all_factors_known']} "
            f"exact_binding={evidence['exact_binding_known']}"
        )

    print()
    print("all_factors_known   :", all_factors_known)
    print("exact_binding_known :", exact_binding_known)
    print("exact_binding_novel :", exact_binding_novel)

    assert all_factors_known == len(generated), (
        "V63 invalid: not all factors were learned"
    )
    assert exact_binding_known == 0, (
        "V63 invalid: generated test contains an exact training binding"
    )
    assert exact_binding_novel == len(generated)

    print("V63 TEST CONSTRUCTION ASSERTIONS: PASS")
    print("=== END V63 NOVEL COMBINATION TEST ===")
    print()


def v63_generalization_probe(
    substrate: FactorizedSubstrateV62,
) -> None:
    print("=== V63 FACTOR GENERALIZATION PROBE ===")

    for triple, word in v63_build_novel_compositions():
        pos = len(triple[0])
        obs = substrate.frozen_observation(word, pos)

        print(
            f"{word:16s} pos={pos:2d} "
            f"components="
            f"{obs['prefix_id']}/"
            f"{obs['symbol_id']}/"
            f"{obs['suffix_id']} "
            f"binding={obs['binding'] is not None}"
        )

    print("=== END V63 FACTOR GENERALIZATION PROBE ===")
    print()


def main() -> None:
    print("=== V63 FACTORIZED SUBSTRATE — NOVEL COMBINATION GENERALIZATION ===")
    print(
        "Known factors are recombined into exact triples never seen in training."
    )
    print()

    gt = validate_v28()

    substrate = FactorizedSubstrateV62()
    substrate.train(TRAINING)

    print("=== V63 TRAINED SUBSTRATE ===")
    print("components :", len(substrate.components))
    print("bindings   :", len(substrate.bindings))
    print("transitions:", len(substrate.transitions))
    print()

    v63_run_novel_combination_test(substrate)
    v63_generalization_probe(substrate)

    print("=== V63 COMPLETE ===")


if __name__ == "__main__":
    main()
