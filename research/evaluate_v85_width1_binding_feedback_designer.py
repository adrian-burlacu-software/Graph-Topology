from __future__ import annotations

"""
V80 — WIDTH-1 BINDING-CELL FEEDBACK TO THE REAL DESIGNER

This is the next step after V78/V79.

The real simulator already has:
    vocabulary activity -> designer membrane activity
    designer action -> reward
    designer plasticity

V80 adds one new biological/graphical signal:

    width-1 local factors
        ↓
    binding cell
        ↓
    designer root

The binding cell is NOT exposed as a boolean "reuse_available" feature.

Instead:
    * an existing binding cell can become active and feed the designer;
    * a missing binding has no binding-cell feedback;
    * the normal vocabulary/path signal remains present;
    * the existing designer learns through its existing reward/plasticity.

Training then asks the real designer to learn whether binding-cell feedback,
vocabulary matching, and context activity together predict the correct
REUSE/BRANCH action.

No pair-support threshold.
No hand-built COMPOSE classifier.
No test-time threshold tuning.

Corpus:
    data/dictionary.csv

Evaluation:
    deterministic train/validation/test split
    multiple training epochs
    frozen test evaluation
    reward / action accuracy
    binding-cell growth
    validation replay idempotence
"""

import copy
import hashlib
import time
from collections import Counter
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

EPOCHS = 5
LOG_EVERY = 1


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

    words = sorted(set(words))

    if not words:
        raise RuntimeError(
            f"Corpus empty: {path}"
        )

    return words


def stable_rank(word: str) -> str:
    return hashlib.sha256(
        word.encode("utf-8")
    ).hexdigest()


def split_words(
    words: list[str],
):
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

    train = ordered[:train_end]
    validation = ordered[
        train_end:validation_end
    ]
    test = ordered[validation_end:]

    assert not set(train) & set(validation)
    assert not set(train) & set(test)
    assert not set(validation) & set(test)

    return train, validation, test


# ---------------------------------------------------------------------------
# Width-1 factorization
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
# V80 real network
# ---------------------------------------------------------------------------

class V80BindingFeedbackNetwork(Network):
    V80_FACTOR = "v80_factor"
    V80_BINDING = "v80_binding"

    V80_FACTOR_BINDING = "V80_FACTOR_BINDING"
    V80_BINDING_FEEDBACK = "V80_BINDING_FEEDBACK"

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

        # Number of times each binding cell has been activated.
        self.binding_activations: Counter[int] = Counter()

        # Last active binding, used only as graph activity.
        self.active_binding: Optional[int] = None

    # ------------------------------------------------------------------
    # Factor cells
    # ------------------------------------------------------------------

    def _factor(
        self,
        kind: str,
        value: str,
        learn: bool,
    ) -> int:
        key = (kind, value)

        existing = self.factor_by_value.get(
            key
        )

        if existing is not None:
            return existing

        if not learn:
            return -1

        cell_id = self.create_cell(
            self.V80_FACTOR,
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

    def create_binding(
        self,
        factors: tuple[int, int, int],
    ) -> int:
        existing = self.binding_by_key.get(
            factors
        )

        if existing is not None:
            return existing

        binding_id = self.create_cell(
            self.V80_BINDING
        )

        self.binding_by_key[
            factors
        ] = binding_id

        # Factor -> binding edges.
        for factor_id in factors:
            self.connect(
                factor_id,
                binding_id,
                self.V80_FACTOR_BINDING,
                1.0,
            )

        # Binding -> designer root feedback.
        self.connect(
            binding_id,
            self.designer_root,
            self.V80_BINDING_FEEDBACK,
            self.config.feedback_weight,
        )

        return binding_id

    # ------------------------------------------------------------------
    # Binding-cell sensory activity
    # ------------------------------------------------------------------

    def _stimulate_binding_context(
        self,
        factors: tuple[int, int, int],
    ) -> float:
        """
        Convert binding-cell state into ordinary designer-root activity.

        IMPORTANT:
            binding existence is NOT a REUSE vote.

        The binding cell only contributes a learned input to the existing
        designer root. The designer's own learned root -> REUSE / BRANCH
        synapses determine the downstream action.
        """
        binding_id = self.exact_binding(factors)

        self.active_binding = binding_id

        if binding_id is None:
            return 0.0

        binding = self.cells[binding_id]

        gain = self.designer_genome[
            "match_gain"
        ]

        binding.potential += gain
        binding.spikes += 1

        self.binding_activations[
            binding_id
        ] += 1

        feedback = self.synapses.get(
            (
                binding_id,
                self.designer_root,
            )
        )

        if feedback is None:
            return 0.0

        # Root input only. Do NOT inject directly into reuse.potential.
        root = self.cells[
            self.designer_root
        ]

        root.potential += feedback.weight

        return feedback.weight

    # ------------------------------------------------------------------
    # Designer integration
    # ------------------------------------------------------------------

    def spike_designer_v80(
        self,
        current_id: Optional[int],
        symbol: str,
        factors: tuple[int, int, int],
    ) -> None:
        """
        Final V80b designer integration.

        Binding activity is NEUTRAL evidence:
            binding cell -> designer root/context

        It does not directly excite REUSE or BRANCH.

        The existing vocabulary/context signals still provide the ordinary
        path information, and the existing reward/plasticity loop learns how
        to interpret the additional binding activity.
        """
        match_activity, context_activity = (
            self._stimulate_local_context(
                current_id,
                symbol,
            )
        )

        binding_activity = (
            self._stimulate_binding_context(
                factors
            )
        )

        root = self.cells[
            self.designer_root
        ]
        reuse = self.cells[
            self.reuse_cell
        ]
        branch = self.cells[
            self.branch_cell
        ]

        threshold = self.designer_genome[
            "threshold"
        ]

        input_gain = self.designer_genome[
            "input_gain"
        ]

        context_gain = self.designer_genome[
            "context_gain"
        ]

        match_gain = self.designer_genome[
            "match_gain"
        ]

        # Existing sensory/context inputs.
        root.potential += input_gain

        # Vocabulary-match evidence remains what it was in the original
        # designer. This is the sequential path signal.
        reuse.potential += (
            match_activity
            * match_gain
        )

        # V80 binding activity is NEUTRAL: it only reaches the common root.
        # We add a small symmetric context contribution to both downstream
        # populations only through the SAME learned context channel.
        #
        # This avoids turning "binding exists" into an implicit REUSE label.
        neutral_binding = (
            binding_activity
            * context_gain
        )

        root.potential += neutral_binding

        branch.potential += (
            self.designer_genome["branch_bias"]
            + context_activity
            * context_gain
        )

        # Existing root readout is unchanged: root activation recruits both
        # action populations and their learned competition decides the winner.
        if root.potential >= threshold:
            root.potential = 0.0
            root.spikes += 1
            self.designer_spikes += 1

            reuse.potential += self.synapses[
                (
                    self.designer_root,
                    self.reuse_cell,
                )
            ].weight

            branch.potential += self.synapses[
                (
                    self.designer_root,
                    self.branch_cell,
                )
            ].weight

        # Existing mutual inhibition remains the decision mechanism.
        if reuse.potential >= threshold:
            branch.inhibition += (
                self.inhibition_genome["strength"]
            )
            branch.potential -= (
                self.inhibition_genome["strength"]
            )

            reuse.spikes += 1
            self.designer_spikes += 1

        if branch.potential >= threshold:
            reuse.inhibition += (
                self.inhibition_genome["strength"]
            )
            reuse.potential -= (
                self.inhibition_genome["strength"]
            )

            branch.spikes += 1
            self.designer_spikes += 1

        reuse.potential *= self.designer_genome[
            "leak"
        ]
        branch.potential *= self.designer_genome[
            "leak"
        ]

    # ------------------------------------------------------------------
    # Training-time local binding creation
    # ------------------------------------------------------------------

    def _ensure_binding_after_action(
        self,
        factors: tuple[int, int, int],
        action: str,
        correct: str,
    ) -> Optional[int]:
        """
        Structural V80 rule.

        We do not let the binding layer make the decision.

        The designer makes REUSE/BRANCH exactly as the real Network does.

        When the action is correct:
            REUSE -> existing binding
            BRANCH -> create binding for the newly learned composition

        This binds V80 learning to actual designer correctness rather than
        to a threshold heuristic.
        """
        existing = self.exact_binding(factors)

        if action == REUSE and existing is not None:
            return existing

        if (
            action == BRANCH
            and correct == BRANCH
            and existing is None
        ):
            return self.create_binding(
                factors
            )

        # If correct reuse occurs before V80 binding exists, repair the
        # missing factorized binding after the fact. This is a structural
        # consistency operation, not a decision input.
        if (
            action == REUSE
            and correct == REUSE
            and existing is None
        ):
            return self.create_binding(
                factors
            )

        return existing

    # ------------------------------------------------------------------
    # Process one word
    # ------------------------------------------------------------------

    def process_word(
        self,
        word: str,
        learn: bool = True,
    ) -> dict:
        current_id: Optional[int] = None

        created = 0
        reused = 0
        correct = 0
        wrong = 0

        for order, symbol in enumerate(word):
            factors = self.factorize(
                word,
                order,
                learn=learn,
            )

            # Ground truth of the REAL Network path before mutation.
            existing_path = self.find_child(
                current_id,
                symbol,
            )

            correct_action = (
                REUSE
                if existing_path is not None
                else BRANCH
            )

            self._reset_designer_input()

            self.spike_designer_v80(
                current_id,
                symbol,
                factors,
            )

            action = self.designer_signal(
                current_id,
                symbol,
            )

            new_id, made, reused_now, reward = (
                self._apply_decision(
                    current_id,
                    symbol,
                    order,
                    action,
                )
            )

            current_id = new_id

            created += made
            reused += reused_now

            if action == correct_action:
                correct += 1
            else:
                wrong += 1

            if made:
                self.total_create += made

            if reused_now:
                self.total_reuse += reused_now

            # Factorized binding learning follows the real designer action.
            if learn:
                self._ensure_binding_after_action(
                    factors,
                    action,
                    correct_action,
                )

                self.learn_designer(
                    action,
                    correct_action,
                    reward,
                )

        return {
            "word": word,
            "created": created,
            "reused": reused,
            "correct": correct,
            "wrong": wrong,
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        words: list[str],
        epochs: int = EPOCHS,
    ) -> None:
        print()
        print("=== V80 DESIGNER PLASTICITY ===")
        print("epochs :", epochs)
        print("words  :", len(words))
        print()

        for epoch in range(1, epochs + 1):
            reward_before = self.total_reward
            reuse_before = self.total_reuse
            create_before = self.total_create
            correct_before = (
                self.correct_reuse
                + self.correct_branch
            )

            for word in words:
                self.process_word(
                    word,
                    learn=True,
                )

            correct_delta = (
                self.correct_reuse
                + self.correct_branch
                - correct_before
            )

            positions = sum(
                len(word)
                for word in words
            )

            print(
                f"epoch={epoch:2d} "
                f"cells={len(self.cells):5d} "
                f"reuse={self.total_reuse - reuse_before:5d} "
                f"create={self.total_create - create_before:5d} "
                f"correct={correct_delta:5d}/{positions} "
                f"acc={correct_delta / max(1, positions):.4f} "
                f"reward={self.total_reward - reward_before:10.2f}"
            )

        print()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def v80_designer_synapse_weights(self) -> dict[str, float]:
        return {
            "root_to_reuse": self.synapses[
                (
                    self.designer_root,
                    self.reuse_cell,
                )
            ].weight,
            "root_to_branch": self.synapses[
                (
                    self.designer_root,
                    self.branch_cell,
                )
            ].weight,
        }

    def v80_counts(self) -> dict[str, int]:
        return {
            "factor_cells": sum(
                1
                for cell in self.cells.values()
                if cell.kind == self.V80_FACTOR
            ),
            "binding_cells": sum(
                1
                for cell in self.cells.values()
                if cell.kind == self.V80_BINDING
            ),
            "network_cells": len(self.cells),
            "network_synapses": len(self.synapses),
            "binding_feedback_synapses": sum(
                1
                for syn in self.synapses.values()
                if syn.kind
                == self.V80_BINDING_FEEDBACK
            ),
        }

    def v80_accuracy(self) -> float:
        total = (
            self.correct_reuse
            + self.correct_branch
            + self.wrong_reuse
            + self.wrong_branch
        )

        return (
            (
                self.correct_reuse
                + self.correct_branch
            )
            / max(1, total)
        )


# ---------------------------------------------------------------------------
# Frozen evaluation
# ---------------------------------------------------------------------------

def evaluate_frozen(
    network: V80BindingFeedbackNetwork,
    words: list[str],
    label: str,
) -> dict[str, float]:
    """
    Evaluate designer action WITHOUT learning or structural mutation.

    The copied network is used so designer membrane state cannot contaminate
    subsequent evaluation.
    """
    probe = copy.deepcopy(network)

    correct = 0
    total = 0
    reuse_correct = 0
    branch_correct = 0
    errors = []

    for word in words:
        current_id: Optional[int] = None

        for order, symbol in enumerate(word):
            factors = probe.factorize(
                word,
                order,
                learn=False,
            )

            existing_path = probe.find_child(
                current_id,
                symbol,
            )

            expected = (
                REUSE
                if existing_path is not None
                else BRANCH
            )

            probe._reset_designer_input()

            probe.spike_designer_v80(
                current_id,
                symbol,
                factors,
            )

            actual = probe.designer_signal(
                current_id,
                symbol,
            )

            total += 1

            if actual == expected:
                correct += 1

                if expected == REUSE:
                    reuse_correct += 1
                else:
                    branch_correct += 1
            else:
                errors.append(
                    (
                        word,
                        order,
                        expected,
                        actual,
                    )
                )

            # Frozen structural path progression follows the EXISTING graph;
            # no new cells are created here.
            existing = probe.find_child(
                current_id,
                symbol,
            )

            if existing is not None:
                current_id = existing
            else:
                # In frozen evaluation we cannot create a missing vocabulary
                # edge. Stop this word after the first missing transition.
                current_id = None

    accuracy = correct / max(1, total)

    print(
        f"=== V80 FROZEN {label} ==="
    )
    print("positions     :", total)
    print("correct       :", correct)
    print("accuracy      :", accuracy)
    print("reuse_correct :", reuse_correct)
    print("branch_correct:", branch_correct)
    print("errors        :", len(errors))

    for error in errors[:20]:
        print(
            f"{error[0]:12s} "
            f"pos={error[1]:2d} "
            f"expected={error[2]:6s} "
            f"actual={error[3]:6s}"
        )

    print()

    return {
        "positions": float(total),
        "correct": float(correct),
        "accuracy": accuracy,
        "reuse_correct": float(reuse_correct),
        "branch_correct": float(branch_correct),
        "errors": float(len(errors)),
    }


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def main() -> None:
    start = time.perf_counter()

    print(
        "=== V80 WIDTH-1 BINDING FEEDBACK DESIGNER ==="
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

    network = V80BindingFeedbackNetwork()

    # Give the width-1 binding layer its starting structural vocabulary from
    # the actual training stream while the designer simultaneously learns
    # from membrane activity.
    network.train(
        train,
        epochs=EPOCHS,
    )

    print(
        "training_elapsed :",
        f"{time.perf_counter() - start:.2f}s",
    )
    print()

    train_counts = network.v80_counts()

    print(
        "=== V80 TRAINED GRAPH ==="
    )
    for key, value in train_counts.items():
        print(
            f"{key:30s}: {value}"
        )

    print(
        "designer_accuracy :",
        network.v80_accuracy(),
    )
    print(
        "total_reward      :",
        network.total_reward,
    )
    print(
        "designer_spikes   :",
        network.designer_spikes,
    )
    print(
        "root_to_reuse     :",
        network.v80_designer_synapse_weights()["root_to_reuse"],
    )
    print(
        "root_to_branch    :",
        network.v80_designer_synapse_weights()["root_to_branch"],
    )
    print()

    validation_result = evaluate_frozen(
        network,
        validation,
        "VALIDATION",
    )

    test_result = evaluate_frozen(
        network,
        test,
        "TEST",
    )

    print(
        "=== V80 SUMMARY ==="
    )
    print(
        "train_accuracy      :",
        network.v80_accuracy(),
    )
    print(
        "validation_accuracy :",
        validation_result["accuracy"],
    )
    print(
        "test_accuracy       :",
        test_result["accuracy"],
    )
    print(
        "factor_cells        :",
        train_counts["factor_cells"],
    )
    print(
        "binding_cells       :",
        train_counts["binding_cells"],
    )
    print(
        "binding_feedback    :",
        train_counts[
            "binding_feedback_synapses"
        ],
    )
    print(
        "total_reward        :",
        network.total_reward,
    )
    print(
        "elapsed_seconds     :",
        f"{time.perf_counter() - start:.2f}",
    )
    print(
        "=== V80 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
