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


# ---------------------------------------------------------------------------
# V83 — RECURSIVE ASSEMBLY CLOSURE
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Unit:
    """
    A reusable graph unit.

    At level 0 this is a binding cell.
    At later levels this is an assembly cell.

    The discovery operator does not care which level produced the unit.
    """
    unit_id: int
    level: int


def collect_binding_units(
    network: BindingAssemblyNetwork,
    words: list[str],
) -> list[int]:
    units = set()

    for word in words:
        sequence = []

        for pos in range(len(word)):
            factors = network.factorize(
                word,
                pos,
                learn=False,
            )

            binding = network.binding_by_key.get(
                factors
            )

            if binding is not None:
                sequence.append(binding)

        units.update(sequence)

    return sorted(units)


def discover_recursive_level(
    network: BindingAssemblyNetwork,
    units: list[int],
    level: int,
    min_occurrences: int = 2,
) -> tuple[list[int], dict[str, int]]:
    """
    Generic discovery operator.

    Input:
        an ordered stream of reusable units

    Operation:
        count adjacent unit transitions
        repeated transition -> one reusable assembly unit

    Output:
        newly discovered assembly units

    This function does not know whether `units` are bindings or assemblies.
    The same operator is reused at every level.
    """
    if len(units) < 2:
        return [], {
            "candidate_transitions": 0,
            "recurring_transitions": 0,
            "new_assemblies": 0,
            "reused_assemblies": 0,
        }

    transitions = Counter(
        zip(
            units,
            units[1:],
        )
    )

    candidate_count = len(transitions)

    recurring = {
        transition: count
        for transition, count in transitions.items()
        if count >= min_occurrences
    }

    new_units = []
    reused = 0

    for transition, count in sorted(
        recurring.items()
    ):
        assembly = network.assembly_by_transition.get(
            transition
        )

        if assembly is None:
            assembly = network.create_cell(
                f"v83_assembly_level_{level}"
            )

            network.assembly_by_transition[
                transition
            ] = assembly

            previous, current = transition

            network.connect(
                previous,
                assembly,
                f"V83_ASSEMBLY_MEMBER_L{level}",
                1.0,
            )

            network.connect(
                current,
                assembly,
                f"V83_ASSEMBLY_MEMBER_L{level}",
                1.0,
            )

            new_units.append(
                assembly
            )

        else:
            reused += count

    return new_units, {
        "candidate_transitions": candidate_count,
        "recurring_transitions": len(recurring),
        "new_assemblies": len(new_units),
        "reused_assemblies": reused,
    }


def stream_units(
    network: BindingAssemblyNetwork,
    words: list[str],
    previous_units_by_binding: dict[int, list[int]],
    assembly_maps: dict[int, dict[int, list[int]]],
) -> None:
    """
    Build reusable unit streams for every word.

    Level 0 = binding IDs.

    For each higher level, an assembly unit replaces each adjacent pair of
    lower-level units whenever that assembly exists.

    This allows discovered assemblies to become inputs to the same operator.
    """
    # Store level-0 unit sequences.
    for word in words:
        binding_sequence = []

        for pos in range(len(word)):
            factors = network.factorize(
                word,
                pos,
                learn=False,
            )

            binding = network.binding_by_key.get(
                factors
            )

            if binding is not None:
                binding_sequence.append(binding)

        previous_units_by_binding[
            hash(word)
        ] = binding_sequence


def compress_sequence_once(
    network: BindingAssemblyNetwork,
    sequence: list[int],
) -> list[int]:
    """
    Apply the same learned assembly dictionary once.

    For adjacent units A,B:
        known assembly(A,B) -> assembly unit
        otherwise keep A

    The final singleton unit is carried forward.
    """
    if len(sequence) < 2:
        return sequence[:]

    result = []
    i = 0

    while i < len(sequence):
        if i + 1 < len(sequence):
            key = (
                sequence[i],
                sequence[i + 1],
            )

            assembly = (
                network.assembly_by_transition.get(
                    key
                )
            )

            if assembly is not None:
                result.append(assembly)
                i += 2
                continue

        result.append(sequence[i])
        i += 1

    return result


def run_recursive_closure(
    network: BindingAssemblyNetwork,
    words: list[str],
    max_levels: int = 12,
) -> list[dict[str, int]]:
    """
    Run the SAME discovery/compression operator repeatedly.

    Stop when a level creates zero new assemblies.

    No code path changes between levels.
    """
    # Build initial unit streams.
    streams: list[list[int]] = []

    for word in words:
        sequence = []

        for pos in range(len(word)):
            factors = network.factorize(
                word,
                pos,
                learn=False,
            )

            binding = network.binding_by_key.get(
                factors
            )

            if binding is not None:
                sequence.append(binding)

        streams.append(sequence)

    results = []

    for level in range(1, max_levels + 1):
        all_sequences = []

        for sequence in streams:
            if len(sequence) >= 2:
                all_sequences.extend(
                    zip(
                        sequence,
                        sequence[1:],
                    )
                )

        transition_counts = Counter(
            all_sequences
        )

        recurring = {
            key: count
            for key, count
            in transition_counts.items()
            if count >= 2
        }

        new_units = []

        for transition in sorted(recurring):
            assembly = (
                network.assembly_by_transition.get(
                    transition
                )
            )

            if assembly is None:
                assembly = network.create_cell(
                    f"v83_assembly_level_{level}"
                )

                network.assembly_by_transition[
                    transition
                ] = assembly

                previous, current = transition

                network.connect(
                    previous,
                    assembly,
                    f"V83_ASSEMBLY_MEMBER_L{level}",
                    1.0,
                )

                network.connect(
                    current,
                    assembly,
                    f"V83_ASSEMBLY_MEMBER_L{level}",
                    1.0,
                )

                new_units.append(
                    assembly
                )

        # Now feed the newly available assembly dictionary back through the
        # SAME compression operator.
        compressed_streams = []

        for sequence in streams:
            compressed = compress_sequence_once(
                network,
                sequence,
            )

            compressed_streams.append(
                compressed
            )

        old_total = sum(
            len(sequence)
            for sequence in streams
        )

        new_total = sum(
            len(sequence)
            for sequence in compressed_streams
        )

        results.append(
            {
                "level": level,
                "input_units": old_total,
                "output_units": new_total,
                "candidate_transitions": len(
                    transition_counts
                ),
                "recurring_transitions": len(
                    recurring
                ),
                "new_assemblies": len(
                    new_units
                ),
                "assembly_cells_total": sum(
                    cell.kind.startswith(
                        "v83_assembly_level_"
                    )
                    for cell in network.cells.values()
                ),
            }
        )

        streams = compressed_streams

        print(
            f"level={level:2d} "
            f"input_units={old_total:7d} "
            f"output_units={new_total:7d} "
            f"candidates={len(transition_counts):7d} "
            f"recurring={len(recurring):7d} "
            f"new_assemblies={len(new_units):7d}"
        )

        if not new_units:
            break

    return results


def main() -> None:
    total_start = time.perf_counter()

    print(
        "=== V83 RECURSIVE ASSEMBLY CLOSURE ==="
    )
    print(
        "One operator, repeatedly applied to its own discovered outputs."
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

    network.train(
        train,
        epochs=1,
    )

    print(
        f"[{time.perf_counter() - total_start:.2f}s] "
        "real Network training complete"
    )

    network.stream(
        train,
        "TRAIN_BINDINGS",
        learn=True,
    )

    print()
    print(
        "binding_cells :",
        network.counts()["binding_cells"],
    )
    print()

    # The complete stream is intentionally included. The question is whether
    # recursion remains productive on real corpus structure.
    evaluation_words = (
        validation + test
    )

    results = run_recursive_closure(
        network,
        evaluation_words,
        max_levels=12,
    )

    print()
    print(
        "=== V83 CLOSURE SUMMARY ==="
    )

    for row in results:
        print(
            f"level={row['level']:2d} "
            f"input={row['input_units']:7d} "
            f"output={row['output_units']:7d} "
            f"candidates={row['candidate_transitions']:7d} "
            f"recurring={row['recurring_transitions']:7d} "
            f"new={row['new_assemblies']:7d} "
            f"total_assemblies={row['assembly_cells_total']:7d}"
        )

    print()

    if results:
        final = results[-1]
        print(
            "fixed_point_reached :",
            final["new_assemblies"] == 0,
        )
        print(
            "final_level         :",
            final["level"],
        )
        print(
            "total_assemblies    :",
            final["assembly_cells_total"],
        )

    print(
        "elapsed_seconds :",
        f"{time.perf_counter() - total_start:.2f}",
    )

    print(
        "=== V83 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
