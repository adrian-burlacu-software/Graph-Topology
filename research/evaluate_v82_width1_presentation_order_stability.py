from __future__ import annotations

"""
V78 — WIDTH-1 BINDING PLASTICITY

No pair-support threshold.
No semantic COMPOSE classifier.
No handcrafted positive/negative test set.

Question:
    Can the real graph grow reusable width-1 binding cells as the corpus is
    streamed, with repeated local structures reinforcing/reusing the same
    binding rather than allocating new cells?

Representation:
    left-neighbor | symbol | right-neighbor

Lifecycle:
    first observation of local triple -> create binding cell
    later observation                  -> REUSE same binding cell + reinforce
    factor cells                       -> shared vocabulary

Evaluation:
    * how many local triples are reused across words?
    * how many binding cells are needed?
    * how much binding compression exists?
    * does a second pass create zero new binding cells?
    * do binding activations increase on repeated exposure?
"""

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from simulator import (
        BRANCH,
        REUSE,
        Config,
        Network,
    )
except ImportError:
    from .simulator import (
        BRANCH,
        REUSE,
        Config,
        Network,
    )


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "dictionary.csv"

TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15


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
    train_end = int(n * TRAIN_FRACTION)
    validation_end = (
        train_end
        + int(n * VALID_FRACTION)
    )

    return (
        ordered[:train_end],
        ordered[train_end:validation_end],
        ordered[validation_end:],
    )


# ---------------------------------------------------------------------------
# Width-1 factors
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
# Real graph implementation
# ---------------------------------------------------------------------------

class PlasticWidth1Network(Network):
    V_FACTOR = "v78_factor"
    V_BINDING = "v78_binding"

    V_FACTOR_EDGE = "V78_FACTOR_BINDING"

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

        # Activation count is intentionally graph-local state.
        self.binding_activations: dict[
            int,
            int,
        ] = {}

        # How many times each local triple was exposed.
        self.triple_exposure: dict[
            tuple[int, int, int],
            int,
        ] = {}

    # ------------------------------------------------------------------
    # Factor vocabulary
    # ------------------------------------------------------------------

    def _factor(
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
            self.V_FACTOR,
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
            self._factor(
                "left",
                factors.left,
                learn,
            ),
            self._factor(
                "symbol",
                factors.symbol,
                learn,
            ),
            self._factor(
                "right",
                factors.right,
                learn,
            ),
        )

    # ------------------------------------------------------------------
    # Binding cells
    # ------------------------------------------------------------------

    def exact_binding(
        self,
        factors: tuple[int, int, int],
    ) -> Optional[int]:
        if min(factors) < 0:
            return None

        return self.binding_by_key.get(
            factors
        )

    def _create_binding(
        self,
        factors: tuple[int, int, int],
    ) -> int:
        existing = self.binding_by_key.get(
            factors
        )

        if existing is not None:
            return existing

        binding_id = self.create_cell(
            self.V_BINDING
        )

        self.binding_by_key[
            factors
        ] = binding_id

        self.binding_activations[
            binding_id
        ] = 0

        for factor_id in factors:
            self.connect(
                factor_id,
                binding_id,
                self.V_FACTOR_EDGE,
                1.0,
            )

        return binding_id

    def observe_binding(
        self,
        factors: tuple[int, int, int],
    ) -> tuple[str, int]:
        """
        Pure graph growth rule.

        First exposure:
            BRANCH -> create a binding cell

        Repeated exposure:
            REUSE -> reinforce the existing binding

        There is no threshold or external semantic classifier.
        """
        if min(factors) < 0:
            raise ValueError(
                "Cannot learn binding with unknown factors."
            )

        binding_id = self.exact_binding(
            factors
        )

        if binding_id is None:
            binding_id = self._create_binding(
                factors
            )
            action = BRANCH
        else:
            action = REUSE

        self.binding_activations[
            binding_id
        ] += 1

        self.triple_exposure[
            factors
        ] = (
            self.triple_exposure.get(
                factors,
                0,
            )
            + 1
        )

        # Reinforce every factor -> binding synapse.
        for factor_id in factors:
            synapse = self.synapses.get(
                (
                    factor_id,
                    binding_id,
                )
            )

            if synapse is not None:
                synapse.weight += 1.0
                synapse.learning += 1.0

        return action, binding_id

    # ------------------------------------------------------------------
    # Corpus passes
    # ------------------------------------------------------------------

    def train_word(self, word: str) -> dict[str, int]:
        created = 0
        reused = 0

        for pos in range(len(word)):
            factors = self.factorize(
                word,
                pos,
                learn=True,
            )

            action, _binding = self.observe_binding(
                factors
            )

            if action == BRANCH:
                created += 1
            else:
                reused += 1

        return {
            "created": created,
            "reused": reused,
        }

    def train_words(
        self,
        words: list[str],
        label: str,
    ) -> dict[str, int]:
        created = 0
        reused = 0

        for index, word in enumerate(
            words,
            start=1,
        ):
            result = self.train_word(word)

            created += result["created"]
            reused += result["reused"]

            if (
                index % 1000 == 0
                or index == len(words)
            ):
                print(
                    f"{label}: "
                    f"{index}/{len(words)} "
                    f"created={created} "
                    f"reused={reused}",
                    flush=True,
                )

        return {
            "created": created,
            "reused": reused,
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        return {
            "factor_cells": sum(
                1
                for cell in self.cells.values()
                if cell.kind == self.V_FACTOR
            ),
            "binding_cells": sum(
                1
                for cell in self.cells.values()
                if cell.kind == self.V_BINDING
            ),
            "network_cells": len(self.cells),
            "network_synapses": len(self.synapses),
        }

    def binding_activation_stats(self) -> dict[str, float]:
        if not self.binding_activations:
            return {
                "bindings": 0.0,
                "mean": 0.0,
                "max": 0.0,
                "reused_bindings": 0.0,
            }

        values = list(
            self.binding_activations.values()
        )

        return {
            "bindings": float(len(values)),
            "mean": (
                sum(values)
                / len(values)
            ),
            "max": float(max(values)),
            "reused_bindings": float(
                sum(
                    value > 1
                    for value in values
                )
            ),
        }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# V79 — PRESENTATION ORDER STABILITY
# ---------------------------------------------------------------------------

ORDER_SEEDS = (0, 1, 2, 3, 4)


def deterministic_permutation(
    words: list[str],
    seed: int,
) -> list[str]:
    """
    Deterministic pseudo-random ordering without external dependencies.
    """
    ranked = []

    for word in words:
        digest = hashlib.sha256(
            f"{seed}:{word}".encode("utf-8")
        ).hexdigest()

        ranked.append(
            (digest, word)
        )

    ranked.sort()

    return [
        word
        for _, word in ranked
    ]


def run_order(
    train: list[str],
    validation: list[str],
    seed: int,
) -> dict[str, float]:
    network = PlasticWidth1Network()

    # Existing graph learning.
    network.train(
        train,
        epochs=1,
    )

    ordered_train = deterministic_permutation(
        train,
        seed,
    )

    train_result = network.train_words(
        ordered_train,
        f"ORDER-{seed} TRAIN",
    )

    train_counts = network.counts()
    train_stats = network.binding_activation_stats()

    # Validation is streamed after training, using the SAME order policy.
    ordered_validation = deterministic_permutation(
        validation,
        seed + 1000,
    )

    validation_result = network.train_words(
        ordered_validation,
        f"ORDER-{seed} VALIDATION",
    )

    validation_counts = network.counts()
    validation_stats = network.binding_activation_stats()

    # Replay the exact same validation order. It must add no bindings.
    replay_before = network.counts()

    replay_result = network.train_words(
        ordered_validation,
        f"ORDER-{seed} REPLAY",
    )

    replay_after = network.counts()

    assert replay_result["created"] == 0
    assert (
        replay_after["binding_cells"]
        == replay_before["binding_cells"]
    )

    return {
        "seed": float(seed),
        "train_created": float(
            train_result["created"]
        ),
        "train_reused": float(
            train_result["reused"]
        ),
        "train_bindings": float(
            train_counts["binding_cells"]
        ),
        "validation_created": float(
            validation_result["created"]
        ),
        "validation_reused": float(
            validation_result["reused"]
        ),
        "final_bindings": float(
            validation_counts["binding_cells"]
        ),
        "factor_cells": float(
            validation_counts["factor_cells"]
        ),
        "mean_activation": float(
            validation_stats["mean"]
        ),
        "max_activation": float(
            validation_stats["max"]
        ),
        "reused_bindings": float(
            validation_stats["reused_bindings"]
        ),
    }


def summarize_orders(
    rows: list[dict[str, float]],
) -> None:
    print("=== V79 ORDER STABILITY SUMMARY ===")

    keys = (
        "train_bindings",
        "validation_created",
        "validation_reused",
        "final_bindings",
        "factor_cells",
        "mean_activation",
        "max_activation",
        "reused_bindings",
    )

    for key in keys:
        values = [
            row[key]
            for row in rows
        ]

        minimum = min(values)
        maximum = max(values)
        mean = sum(values) / len(values)

        print(
            f"{key:24s} "
            f"min={minimum:10.3f} "
            f"max={maximum:10.3f} "
            f"mean={mean:10.3f} "
            f"range={maximum - minimum:10.3f}"
        )

    # Strong invariants:
    # 1. factor vocabulary should not depend on presentation order.
    factor_counts = {
        row["factor_cells"]
        for row in rows
    }

    assert len(factor_counts) == 1

    # 2. replay idempotence was checked inside each run.
    # 3. All runs should produce the same number of unique binding types when
    #    they have seen exactly the same train + validation word sets.
    final_binding_counts = {
        row["final_bindings"]
        for row in rows
    }

    assert len(final_binding_counts) == 1

    print(
        "factor_count_order_invariant : PASS"
    )
    print(
        "binding_count_order_invariant: PASS"
    )

    print("=== END V79 ORDER STABILITY SUMMARY ===")
    print()


def main() -> None:
    total_start = time.perf_counter()

    print("=== V79 WIDTH-1 PRESENTATION ORDER STABILITY ===")
    print(
        "Question: does the learned local topology depend materially "
        "on corpus presentation order?"
    )
    print()

    words = load_dictionary(
        CORPUS_PATH
    )

    train, validation, _test = split_words(
        words
    )

    print(
        "corpus_words :",
        len(words),
    )
    print(
        "train_words  :",
        len(train),
    )
    print(
        "validation   :",
        len(validation),
    )
    print()

    rows = []

    for seed in ORDER_SEEDS:
        print()
        print(
            f"================ ORDER SEED {seed} ================"
        )

        start = time.perf_counter()

        result = run_order(
            train,
            validation,
            seed,
        )

        rows.append(result)

        print()
        print(
            f"ORDER {seed} RESULT"
        )

        for key, value in result.items():
            print(
                f"{key:24s}: {value}"
            )

        print(
            f"elapsed={time.perf_counter() - start:.2f}s"
        )

    print()
    summarize_orders(rows)

    print(
        "total_seconds :",
        f"{time.perf_counter() - total_start:.2f}",
    )
    print(
        "=== V79 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
