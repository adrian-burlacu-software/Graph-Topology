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
# V82 — MULTI-BINDING ASSEMBLY DEPTH / COMPRESSION
# ---------------------------------------------------------------------------

ASSEMBLY_DEPTHS = (2, 3, 4)


def binding_sequence_for_word(
    network: BindingAssemblyNetwork,
    word: str,
) -> list[int]:
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

    return sequence


def count_subsequences(
    words: list[str],
    network: BindingAssemblyNetwork,
    depth: int,
) -> Counter[tuple[int, ...]]:
    counts = Counter()

    for word in words:
        bindings = binding_sequence_for_word(
            network,
            word,
        )

        if len(bindings) < depth:
            continue

        for start in range(
            0,
            len(bindings) - depth + 1,
        ):
            seq = tuple(
                bindings[start:start + depth]
            )
            counts[seq] += 1

    return counts


def discover_depth_assemblies(
    network: BindingAssemblyNetwork,
    sequences: Counter[tuple[int, ...]],
    depth: int,
    min_occurrences: int = 2,
) -> dict[str, int]:
    """
    Discover one real assembly cell for each recurring binding sequence.

    This is intentionally separate from V81's pairwise assembly cache.
    A depth-D assembly represents D binding cells, so compression can finally
    become < 1 when multiple occurrences reuse the same multi-binding pattern.
    """
    discovered = 0
    reused = 0

    registry: dict[tuple[int, ...], int] = {}

    for sequence, count in sequences.items():
        if count < min_occurrences:
            continue

        existing = registry.get(sequence)

        if existing is None:
            assembly_id = network.create_cell(
                f"v82_assembly_d{depth}"
            )
            registry[sequence] = assembly_id

            for binding_id in sequence:
                network.connect(
                    binding_id,
                    assembly_id,
                    "V82_ASSEMBLY_MEMBER",
                    1.0,
                )

            discovered += 1
        else:
            reused += count - 1

    total_recurrent_occurrences = sum(
        count
        for count in sequences.values()
        if count >= min_occurrences
    )

    return {
        "candidate_sequences": len(sequences),
        "recurring_sequences": sum(
            count >= min_occurrences
            for count in sequences.values()
        ),
        "discovered_assemblies": discovered,
        "reused_occurrences": reused,
        "recurrent_occurrences": (
            total_recurrent_occurrences
        ),
    }


def run_depth(
    network: BindingAssemblyNetwork,
    words: list[str],
    depth: int,
) -> dict[str, float]:
    sequences = count_subsequences(
        words,
        network,
        depth,
    )

    stats = discover_depth_assemblies(
        network,
        sequences,
        depth,
    )

    recurring = stats[
        "recurring_sequences"
    ]
    assemblies = stats[
        "discovered_assemblies"
    ]

    # Number of recurring sequences represented by assembly cells.
    # With one assembly per recurring sequence this is currently 1.0, but the
    # useful signal is whether the SAME sequence is repeatedly reused.
    compression = (
        assemblies
        / max(1, recurring)
    )

    return {
        "depth": float(depth),
        "candidate_sequences": float(
            stats["candidate_sequences"]
        ),
        "recurring_sequences": float(
            recurring
        ),
        "discovered_assemblies": float(
            assemblies
        ),
        "reused_occurrences": float(
            stats["reused_occurrences"]
        ),
        "compression": compression,
        "max_sequence_frequency": float(
            max(
                sequences.values(),
                default=0,
            )
        ),
    }


def main() -> None:
    total_start = time.perf_counter()

    print(
        "=== V82 MULTI-BINDING ASSEMBLY DEPTH ==="
    )
    print(
        "Question: do recurring sequences of reusable binding cells "
        "form compact multi-binding assemblies?"
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

    # Freeze the width-1 binding substrate first.
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

    # The depth experiment is intentionally run on validation + test rather
    # than inventing a tiny synthetic corpus.
    evaluation_words = (
        validation + test
    )

    results = []

    for depth in ASSEMBLY_DEPTHS:
        result = run_depth(
            network,
            evaluation_words,
            depth,
        )
        results.append(result)

        print(
            f"depth={depth} "
            f"candidates={result['candidate_sequences']:.0f} "
            f"recurring={result['recurring_sequences']:.0f} "
            f"assemblies={result['discovered_assemblies']:.0f} "
            f"reused={result['reused_occurrences']:.0f} "
            f"max_freq={result['max_sequence_frequency']:.0f}"
        )

    print()
    print(
        "=== V82 DEPTH SUMMARY ==="
    )
    print(
        "depth | "
        "candidate_sequences | "
        "recurring_sequences | "
        "assemblies | "
        "reused_occurrences | "
        "max_frequency"
    )

    for row in results:
        print(
            f"{int(row['depth']):5d} | "
            f"{int(row['candidate_sequences']):19d} | "
            f"{int(row['recurring_sequences']):19d} | "
            f"{int(row['discovered_assemblies']):10d} | "
            f"{int(row['reused_occurrences']):19d} | "
            f"{int(row['max_sequence_frequency']):13d}"
        )

    print()
    print(
        "Interpretation:"
    )
    print(
        "  recurring_sequences > 0 means the corpus contains reusable "
        "multi-binding motifs at that depth."
    )
    print(
        "  max_frequency > 1 confirms genuine repeated sequence structure."
    )
    print(
        "  The important next signal is whether longer depths still recur, "
        "rather than merely having pairwise transitions."
    )

    print()
    print(
        "elapsed_seconds :",
        f"{time.perf_counter() - total_start:.2f}",
    )
    print(
        "=== V82 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
