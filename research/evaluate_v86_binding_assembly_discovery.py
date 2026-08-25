from __future__ import annotations

"""
V81 — BINDING-CELL ASSEMBLY DISCOVERY

The substrate work is now frozen:

    width-1 local factors
        ↓
    reusable binding cells

V81 asks the original designer question directly:

    Can reusable binding cells organize themselves into reusable assemblies
    from the corpus stream, without being handed an assembly list?

Mechanism
---------
1. Each word activates a sequence of width-1 binding cells.
2. Consecutive binding cells strengthen REAL directed assembly synapses.
3. Repeated co-activation reinforces the same synapses.
4. When a binding transition reaches the learned repeat threshold, an
   assembly cell is created automatically.
5. The assembly cell connects to the participating binding cells.
6. Seeing the same transition again reuses the same assembly cell.
7. No word identities, target assembly list, or semantic labels are supplied
   to the assembly learner.

This is intentionally not another REUSE/BRANCH classifier.
The question is graph growth and assembly reuse.

Metrics
-------
    unique binding cells
    binding transition edges
    assembly cells
    assembly reuse
    repeated transition rate
    compression:
        assembly cells / unique observed binding transitions
    replay idempotence
"""

import hashlib
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from simulator import (
        EXCITE,
        Config,
        Network,
    )
except ImportError:
    from .simulator import (
        EXCITE,
        Config,
        Network,
    )


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "dictionary.csv"

TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15

ASSEMBLY_THRESHOLD = 2.0


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def load_dictionary(path: Path) -> list[str]:
    words = []

    for raw in path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        word = raw.strip().lower()

        if word and word.isalpha():
            words.append(word)

    return sorted(set(words))


def stable_rank(word: str) -> str:
    return hashlib.sha256(
        word.encode("utf-8")
    ).hexdigest()


def split_words(words: list[str]):
    ordered = sorted(
        words,
        key=lambda word: (
            stable_rank(word),
            word,
        ),
    )

    n = len(ordered)

    train_end = int(
        n * TRAIN_FRACTION
    )

    validation_end = (
        train_end
        + int(n * VALID_FRACTION)
    )

    return (
        ordered[:train_end],
        ordered[
            train_end:validation_end
        ],
        ordered[validation_end:],
    )


# ---------------------------------------------------------------------------
# Width-1 representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LocalFactors:
    left: str
    symbol: str
    right: str


def local_factors(
    word: str,
    pos: int,
) -> LocalFactors:
    return LocalFactors(
        left=(
            word[pos - 1]
            if pos > 0
            else "^"
        ),
        symbol=word[pos],
        right=(
            word[pos + 1]
            if pos + 1 < len(word)
            else "$"
        ),
    )


# ---------------------------------------------------------------------------
# V81 real graph
# ---------------------------------------------------------------------------

class BindingAssemblyNetwork(Network):
    """
    Real Graph-Topology Network plus:
        factor cells
        binding cells
        binding-transition synapses
        discovered assembly cells
    """

    FACTOR = "v81_factor"
    BINDING = "v81_binding"
    ASSEMBLY = "v81_assembly"

    FACTOR_BINDING = "V81_FACTOR_BINDING"
    BINDING_TRANSITION = "V81_BINDING_TRANSITION"
    ASSEMBLY_MEMBER = "V81_ASSEMBLY_MEMBER"

    def __init__(
        self,
        config: Optional[Config] = None,
    ) -> None:
        super().__init__(config)

        self.factor_by_value: dict[
            tuple[str, str],
            int,
        ] = {}

        self.binding_by_key: dict[
            tuple[int, int, int],
            int,
        ] = {}

        # Directed binding transition key -> real synapse.
        self.binding_transitions: dict[
            tuple[int, int],
            tuple[int, int],
        ] = {}

        # Assembly key is the repeated binding transition pair.
        self.assembly_by_transition: dict[
            tuple[int, int],
            int,
        ] = {}

        self.assembly_activations: Counter[int] = Counter()

        self.binding_activations: Counter[int] = Counter()

        self.transition_exposures: Counter[
            tuple[int, int]
        ] = Counter()

        self.next_assembly_order = 0

    # ------------------------------------------------------------------
    # Factors
    # ------------------------------------------------------------------

    def factor_cell(
        self,
        kind: str,
        value: str,
        learn: bool,
    ) -> int:
        key = (kind, value)

        existing = self.factor_by_value.get(key)

        if existing is not None:
            return existing

        if not learn:
            return -1

        cell_id = self.create_cell(
            self.FACTOR,
            symbol=value,
        )

        self.factor_by_value[key] = cell_id

        return cell_id

    def factorize(
        self,
        word: str,
        pos: int,
        learn: bool,
    ) -> tuple[int, int, int]:
        factors = local_factors(
            word,
            pos,
        )

        return (
            self.factor_cell(
                "left",
                factors.left,
                learn,
            ),
            self.factor_cell(
                "symbol",
                factors.symbol,
                learn,
            ),
            self.factor_cell(
                "right",
                factors.right,
                learn,
            ),
        )

    # ------------------------------------------------------------------
    # Binding cells
    # ------------------------------------------------------------------

    def binding_cell(
        self,
        factors: tuple[int, int, int],
        learn: bool,
    ) -> int:
        existing = self.binding_by_key.get(
            factors
        )

        if existing is not None:
            return existing

        if not learn:
            return -1

        binding_id = self.create_cell(
            self.BINDING,
        )

        self.binding_by_key[
            factors
        ] = binding_id

        for factor_id in factors:
            self.connect(
                factor_id,
                binding_id,
                self.FACTOR_BINDING,
                1.0,
            )

        return binding_id

    # ------------------------------------------------------------------
    # Assembly discovery
    # ------------------------------------------------------------------

    def reinforce_binding_transition(
        self,
        previous_binding: int,
        current_binding: int,
    ) -> float:
        """
        Hebbian transition learning between reusable binding cells.

        No semantic label is needed.
        """
        key = (
            previous_binding,
            current_binding,
        )

        self.transition_exposures[
            key
        ] += 1

        synapse = self.synapses.get(
            key
        )

        if synapse is None:
            synapse = self.connect(
                previous_binding,
                current_binding,
                self.BINDING_TRANSITION,
                1.0,
            )
            self.binding_transitions[
                key
            ] = key

        synapse.weight += 1.0
        synapse.learning += 1.0

        return synapse.weight

    def discover_assembly(
        self,
        previous_binding: int,
        current_binding: int,
    ) -> tuple[str, int]:
        """
        A transition becomes an assembly when it has repeated.

        First observation:
            BUILD TRANSITION

        Second+ observation:
            DISCOVER/REUSE SAME ASSEMBLY CELL
        """
        key = (
            previous_binding,
            current_binding,
        )

        count = self.transition_exposures[
            key
        ]

        assembly_id = self.assembly_by_transition.get(
            key
        )

        if (
            count >= 2
            and assembly_id is None
        ):
            assembly_id = self.create_cell(
                self.ASSEMBLY,
            )

            self.assembly_by_transition[
                key
            ] = assembly_id

            # Assembly membership is a real graph structure.
            self.connect(
                previous_binding,
                assembly_id,
                self.ASSEMBLY_MEMBER,
                1.0,
            )

            self.connect(
                current_binding,
                assembly_id,
                self.ASSEMBLY_MEMBER,
                1.0,
            )

            self.assembly_activations[
                assembly_id
            ] = 1

            return "DISCOVER", assembly_id

        if assembly_id is not None:
            self.assembly_activations[
                assembly_id
            ] += 1

            return "REUSE", assembly_id

        return "LEARN_TRANSITION", -1

    # ------------------------------------------------------------------
    # Stream processing
    # ------------------------------------------------------------------

    def process_word(
        self,
        word: str,
        learn: bool,
    ) -> dict[str, int]:
        bindings = []

        created_bindings = 0
        reused_bindings = 0
        discovered = 0
        reused_assemblies = 0

        for pos in range(len(word)):
            factors = self.factorize(
                word,
                pos,
                learn=learn,
            )

            binding = self.binding_by_key.get(
                factors
            )

            if binding is None:
                binding = self.binding_cell(
                    factors,
                    learn=learn,
                )

                created_bindings += int(
                    learn
                )
            else:
                reused_bindings += 1

            bindings.append(binding)

            if binding >= 0:
                self.binding_activations[
                    binding
                ] += 1

        for previous, current in zip(
            bindings,
            bindings[1:],
        ):
            if previous < 0 or current < 0:
                continue

            if not learn:
                # Frozen readout: do not mutate transition or assembly state.
                continue

            weight = (
                self.reinforce_binding_transition(
                    previous,
                    current,
                )
            )

            if weight >= ASSEMBLY_THRESHOLD:
                action, assembly = (
                    self.discover_assembly(
                        previous,
                        current,
                    )
                )

                if action == "DISCOVER":
                    discovered += 1

                elif action == "REUSE":
                    reused_assemblies += 1

        return {
            "created_bindings": created_bindings,
            "reused_bindings": reused_bindings,
            "discovered_assemblies": discovered,
            "reused_assemblies": reused_assemblies,
        }

    def stream(
        self,
        words: list[str],
        label: str,
        learn: bool,
    ) -> dict[str, int]:
        result = Counter()

        for index, word in enumerate(
            words,
            start=1,
        ):
            output = self.process_word(
                word,
                learn=learn,
            )

            result.update(output)

            if (
                index % 1000 == 0
                or index == len(words)
            ):
                print(
                    f"{label}: "
                    f"{index}/{len(words)} "
                    f"bindings_created={result['created_bindings']} "
                    f"assemblies_discovered={result['discovered_assemblies']} "
                    f"assemblies_reused={result['reused_assemblies']}",
                    flush=True,
                )

        return dict(result)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        return {
            "factor_cells": sum(
                c.kind == self.FACTOR
                for c in self.cells.values()
            ),
            "binding_cells": sum(
                c.kind == self.BINDING
                for c in self.cells.values()
            ),
            "assembly_cells": sum(
                c.kind == self.ASSEMBLY
                for c in self.cells.values()
            ),
            "network_cells": len(self.cells),
            "network_synapses": len(self.synapses),
            "binding_transition_synapses": sum(
                s.kind == self.BINDING_TRANSITION
                for s in self.synapses.values()
            ),
            "assembly_member_synapses": sum(
                s.kind == self.ASSEMBLY_MEMBER
                for s in self.synapses.values()
            ),
            "unique_binding_transitions": len(
                self.binding_transitions
            ),
        }

    def assembly_activation_stats(self) -> dict[str, float]:
        if not self.assembly_activations:
            return {
                "assembly_count": 0.0,
                "mean_activation": 0.0,
                "max_activation": 0.0,
                "reused_assemblies": 0.0,
            }

        values = list(
            self.assembly_activations.values()
        )

        return {
            "assembly_count": float(len(values)),
            "mean_activation": (
                sum(values)
                / len(values)
            ),
            "max_activation": float(
                max(values)
            ),
            "reused_assemblies": float(
                sum(value > 1 for value in values)
            ),
        }


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def main() -> None:
    start = time.perf_counter()

    print(
        "=== V81 WIDTH-1 BINDING ASSEMBLY DISCOVERY ==="
    )
    print(
        "No predefined assembly list."
    )
    print(
        "Assemblies emerge from repeated binding-cell transitions."
    )
    print(
        "corpus:",
        CORPUS_PATH,
    )
    print()

    words = load_dictionary(
        CORPUS_PATH
    )

    train, validation, test = split_words(
        words
    )

    print("corpus_words :", len(words))
    print("train_words  :", len(train))
    print("validation   :", len(validation))
    print("test         :", len(test))
    print()

    network = BindingAssemblyNetwork()

    # Preserve the actual Graph-Topology vocabulary path learning.
    network.train(
        train,
        epochs=1,
    )

    print(
        f"[{time.perf_counter() - start:.2f}s] "
        "real Network training complete"
    )

    # ------------------------------------------------------------------
    # Pass 1 — training substrate + assembly discovery.
    # ------------------------------------------------------------------

    train_result = network.stream(
        train,
        "TRAIN",
        learn=True,
    )

    print()
    print("=== AFTER TRAIN ===")
    print(train_result)
    print(network.counts())
    print(network.assembly_activation_stats())
    print()

    # ------------------------------------------------------------------
    # Pass 2 — validation extends the graph.
    # ------------------------------------------------------------------

    validation_before = network.counts()

    validation_result = network.stream(
        validation,
        "VALIDATION",
        learn=True,
    )

    validation_after = network.counts()

    print()
    print("=== VALIDATION ASSEMBLY GROWTH ===")
    print(validation_result)
    print(
        "new_assemblies :",
        validation_after["assembly_cells"]
        - validation_before["assembly_cells"],
    )
    print(
        "new_transitions:",
        validation_after[
            "unique_binding_transitions"
        ]
        - validation_before[
            "unique_binding_transitions"
        ],
    )
    print()

    # ------------------------------------------------------------------
    # Pass 3 — validation replay.
    #
    # Important lifecycle detail:
    # an assembly is discovered on the SECOND exposure of a binding
    # transition. Therefore the first replay can legitimately discover
    # assemblies for transitions that occurred only once during validation.
    #
    # We test convergence instead:
    #   replay 1 -> may discover second-exposure assemblies
    #   replay 2 -> must discover ZERO additional assemblies
    # ------------------------------------------------------------------

    replay1_before = network.counts()

    replay1_result = network.stream(
        validation,
        "VALIDATION_REPLAY_1",
        learn=True,
    )

    replay1_after = network.counts()

    print()
    print("=== VALIDATION REPLAY 1 ===")
    print(replay1_result)
    print(
        "new_assemblies :",
        replay1_after["assembly_cells"]
        - replay1_before["assembly_cells"],
    )

    replay2_before = network.counts()

    replay2_result = network.stream(
        validation,
        "VALIDATION_REPLAY_2",
        learn=True,
    )

    replay2_after = network.counts()

    print()
    print("=== VALIDATION REPLAY 2 ===")
    print(replay2_result)
    print(
        "new_assemblies :",
        replay2_after["assembly_cells"]
        - replay2_before["assembly_cells"],
    )

    assert (
        replay2_after["assembly_cells"]
        == replay2_before["assembly_cells"]
    )

    print(
        "ASSEMBLY CONVERGENCE / REPLAY IDEMPOTENCE: PASS"
    )
    print()

    # ------------------------------------------------------------------
    # Pass 4 — frozen test observation.
    #
    # Test does NOT mutate the graph. We only ask how many already-known
    # binding transitions / assemblies occur in genuinely unseen words.
    # ------------------------------------------------------------------

    frozen_test = network.stream(
        test,
        "TEST_FROZEN",
        learn=False,
    )

    print()
    print("=== FROZEN TEST ===")
    print(frozen_test)
    print()

    # ------------------------------------------------------------------
    # Final topology accounting.
    # ------------------------------------------------------------------

    print("=== V81 FINAL GRAPH ===")

    for key, value in network.counts().items():
        print(
            f"{key:32s}: {value}"
        )

    for key, value in network.assembly_activation_stats().items():
        print(
            f"{key:32s}: {value}"
        )

    print()
    print(
        "binding_transition_compression :",
        network.counts()[
            "assembly_cells"
        ]
        / max(
            1,
            network.counts()[
                "unique_binding_transitions"
            ],
        ),
    )

    print(
        "elapsed_seconds :",
        f"{time.perf_counter() - start:.2f}",
    )

    print(
        "=== V81 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
