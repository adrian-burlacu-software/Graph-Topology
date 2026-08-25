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

def main() -> None:
    start = time.perf_counter()

    print(
        "=== V78 WIDTH-1 BINDING PLASTICITY ==="
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

    network = PlasticWidth1Network()

    # Real Graph-Topology learning remains in place.
    network.train(
        train,
        epochs=1,
    )

    print(
        f"[{time.perf_counter() - start:.2f}s] "
        "real Network training complete"
    )

    # ------------------------------------------------------------------
    # Pass 1: build reusable binding graph from TRAIN.
    # ------------------------------------------------------------------

    first = network.train_words(
        train,
        "TRAIN",
    )

    after_train = network.counts()

    print()
    print("=== AFTER TRAIN ===")
    print(after_train)
    print("train_created :", first["created"])
    print("train_reused  :", first["reused"])
    print(
        "binding_stats :",
        network.binding_activation_stats(),
    )
    print()

    # ------------------------------------------------------------------
    # Pass 2: validation grows the same graph.
    #
    # This is intentionally online rather than frozen. The question is:
    # do new local structures get created only once and then reused?
    # ------------------------------------------------------------------

    validation_before = network.counts()

    validation_result = network.train_words(
        validation,
        "VALIDATION",
    )

    validation_after = network.counts()

    print()
    print("=== VALIDATION GROWTH ===")
    print(
        "created :",
        validation_result["created"],
    )
    print(
        "reused  :",
        validation_result["reused"],
    )
    print(
        "new_binding_cells :",
        validation_after["binding_cells"]
        - validation_before["binding_cells"],
    )
    print(
        "binding_stats :",
        network.binding_activation_stats(),
    )
    print()

    # ------------------------------------------------------------------
    # Pass 3: replay VALIDATION.
    #
    # The crucial plasticity test:
    # same words / same local structures must now create ZERO new bindings.
    # ------------------------------------------------------------------

    before_replay = network.counts()

    replay_result = network.train_words(
        validation,
        "VALIDATION_REPLAY",
    )

    after_replay = network.counts()

    print()
    print("=== VALIDATION REPLAY ===")
    print(
        "created :",
        replay_result["created"],
    )
    print(
        "reused  :",
        replay_result["reused"],
    )
    print(
        "new_binding_cells :",
        after_replay["binding_cells"]
        - before_replay["binding_cells"],
    )

    assert replay_result["created"] == 0
    assert (
        after_replay["binding_cells"]
        == before_replay["binding_cells"]
    )

    print("REPLAY IDEMPOTENCE: PASS")
    print()

    # ------------------------------------------------------------------
    # Pass 4: TEST remains an online stream in this experiment.
    #
    # We report how much genuinely new local structure the corpus introduces.
    # ------------------------------------------------------------------

    test_before = network.counts()

    test_result = network.train_words(
        test,
        "TEST",
    )

    test_after = network.counts()

    print()
    print("=== TEST GROWTH ===")
    print(
        "created :",
        test_result["created"],
    )
    print(
        "reused  :",
        test_result["reused"],
    )
    print(
        "new_binding_cells :",
        test_after["binding_cells"]
        - test_before["binding_cells"],
    )
    print()

    # ------------------------------------------------------------------
    # Final accounting.
    # ------------------------------------------------------------------

    print("=== V78 FINAL GRAPH ===")

    for key, value in network.counts().items():
        print(
            f"{key:20s}: {value}"
        )

    stats = network.binding_activation_stats()

    for key, value in stats.items():
        print(
            f"{key:20s}: {value}"
        )

    total_positions = sum(
        len(word)
        for word in words
    )

    print()
    print(
        "binding_cells / corpus_positions :",
        network.binding_cells
        / max(1, total_positions)
        if hasattr(network, "binding_cells")
        else (
            network.counts()["binding_cells"]
            / max(1, total_positions)
        ),
    )

    print()
    print(
        "elapsed_seconds :",
        f"{time.perf_counter() - start:.2f}",
    )

    print(
        "=== V78 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
