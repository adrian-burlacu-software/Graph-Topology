from __future__ import annotations

"""
V59 — STRUCTURED COMPOSITION SUBSTRATE BASELINE

Purpose
-------
Return to the structured edge/composition architecture as a clean control
after the fully-connected dense-substrate branch.

The substrate represents a structural composition directly as:

    (prefix, symbol, suffix)

The designer never receives IndependentGroundTruth, REUSE/BRANCH labels,
or a BoundaryGraph. It sees only the active substrate cell(s).

This is intentionally a BASELINE, not a claim that the final Stark/graph
architecture should use explicit vocabulary cells.

The experiment answers one question:

    Can a structured compositional substrate preserve the distinctions that
    the fully-connected generic substrate collapsed?

V28 benchmark
-------------
REUSE_TRAINING and BRANCH_TRAINING are unchanged from the dense branch.
The independent ground truth is used ONLY by the evaluator.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# V28 BENCHMARK — unchanged
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
# Structured substrate
# ---------------------------------------------------------------------------

Composition = Tuple[str, str, str]


@dataclass
class LearnedCompositionCell:
    cell_id: int
    composition: Composition
    activations: int = 0
    incoming_count: int = 0
    outgoing_count: int = 0


class LearnedLearnedCompositionCell:
    def __init__(self, cell_id: int) -> None:
        self.cell_id = cell_id
        self.members: list[tuple[str, str, str]] = []
        self.activations = 0
        self.incoming_count = 0
        self.outgoing_count = 0


class LearnedStructuredSubstrate:
    """
    V61 learned structured substrate.

    Unlike V59/V60, cells are NOT allocated directly from a composition key.

    Training presents structural compositions to the substrate. The substrate
    maintains learned prototypes containing observed compositions.

    A prototype becomes a reusable substrate representation when the same
    structural composition is encountered again.

    The semantic composition itself never enters the designer.
    """

    def __init__(self) -> None:
        self.cells_by_id: dict[int, LearnedLearnedCompositionCell] = {}
        self.composition_to_cell: dict[
            tuple[str, str, str], int
        ] = {}
        self.next_cell_id = 0

        self.transition_weights: dict[tuple[int, int], float] = {}

    @staticmethod
    def composition(word: str, pos: int) -> tuple[str, str, str]:
        return (
            word[:pos],
            word[pos],
            word[pos + 1:],
        )

    def _new_cell(self) -> LearnedLearnedCompositionCell:
        cell = LearnedLearnedCompositionCell(self.next_cell_id)
        self.cells_by_id[cell.cell_id] = cell
        self.next_cell_id += 1
        return cell

    def learn_composition(
        self,
        composition: tuple[str, str, str],
    ) -> LearnedLearnedCompositionCell:
        """
        Learn one structural composition.

        First encounter creates a new substrate unit.
        Repeated encounter reactivates the learned unit.
        """
        existing = self.composition_to_cell.get(composition)

        if existing is not None:
            cell = self.cells_by_id[existing]
            cell.activations += 1
            return cell

        cell = self._new_cell()
        cell.members.append(composition)
        cell.activations = 1
        self.composition_to_cell[composition] = cell.cell_id
        return cell

    def encode(
        self,
        word: str,
        pos: int,
        learn: bool = False,
    ) -> LearnedLearnedCompositionCell:
        composition = self.composition(word, pos)

        if learn:
            return self.learn_composition(composition)

        cell_id = self.composition_to_cell.get(composition)

        if cell_id is None:
            # Novel composition at frozen readout time.
            # Allocate a fresh transient branch representation, but do not
            # insert it into learned substrate memory.
            cell = LearnedLearnedCompositionCell(-1)
            cell.members.append(composition)
            return cell

        return self.cells_by_id[cell_id]

    def train_word(self, word: str) -> None:
        previous = None

        for pos in range(len(word)):
            cell = self.learn_composition(
                self.composition(word, pos)
            )

            if previous is not None:
                key = (previous.cell_id, cell.cell_id)
                self.transition_weights[key] = (
                    self.transition_weights.get(key, 0.0) + 1.0
                )
                previous.outgoing_count += 1
                cell.incoming_count += 1

            previous = cell

    def train(self, words) -> None:
        for word in words:
            self.train_word(word)

    def transition_support(
        self,
        previous: LearnedLearnedCompositionCell,
        current: LearnedLearnedCompositionCell,
    ) -> float:
        if previous.cell_id < 0 or current.cell_id < 0:
            return 0.0
        return self.transition_weights.get(
            (previous.cell_id, current.cell_id),
            0.0,
        )


# ---------------------------------------------------------------------------
# Decoupled designer
# ---------------------------------------------------------------------------

class DecoupledDesignerV60:
    """
    Designer sees only substrate-observable state.

    Forbidden:
      * word
      * position
      * composition
      * independent ground truth
      * external assembly lists

    Allowed:
      * active cell id
      * learned activation count
      * learned incoming/outgoing topology
      * learned transition weights
    """

    def __init__(self, substrate: LearnedStructuredSubstrate) -> None:
        self.substrate = substrate

    def inspect_cell(self, cell: LearnedCompositionCell) -> Dict[str, object]:
        outgoing = [
            (dst, weight)
            for (src, dst), weight in self.substrate.transition_weights.items()
            if src == cell.cell_id
        ]
        incoming = [
            (src, weight)
            for (src, dst), weight in self.substrate.transition_weights.items()
            if dst == cell.cell_id
        ]

        outgoing.sort(key=lambda item: (-item[1], item[0]))
        incoming.sort(key=lambda item: (-item[1], item[0]))

        return {
            "cell_id": cell.cell_id,
            "activation_count": cell.activations,
            "incoming_count": len(incoming),
            "outgoing_count": len(outgoing),
            "incoming_mass": sum(weight for _, weight in incoming),
            "outgoing_mass": sum(weight for _, weight in outgoing),
            "incoming": tuple(incoming),
            "outgoing": tuple(outgoing),
        }

    def decide_from_observable_state(
        self,
        cell: LearnedCompositionCell,
    ) -> str:
        """
        Structural novelty decision.

        A cell that has learned experience is treated as a reusable
        representation; a cell with no learned history is treated as a
        branch candidate.

        IMPORTANT:
        This does not inspect cell.composition.
        """
        evidence = self.inspect_cell(cell)

        learned_history = (
            evidence["activation_count"] > 0
            or evidence["incoming_count"] > 0
            or evidence["outgoing_count"] > 0
        )

        return "REUSE" if learned_history else "BRANCH"


# ---------------------------------------------------------------------------
# V60 evaluation helpers
# ---------------------------------------------------------------------------

def v60_probe_designer(
    substrate: LearnedStructuredSubstrate,
    words: Iterable[str],
) -> None:
    designer = DecoupledDesignerV60(substrate)

    print("=== V60 DECOUPLED DESIGNER DECISIONS ===")

    for word in words:
        for pos in range(len(word)):
            cell = substrate.encode(word, pos)
            decision = designer.decide_from_observable_state(cell)
            evidence = designer.inspect_cell(cell)

            # Deliberately do not print composition. The goal is to demonstrate
            # that the designer itself never needs it.
            print(
                f"{word:6s} pos={pos} "
                f"cell={evidence['cell_id']:3d} "
                f"activations={evidence['activation_count']:3d} "
                f"in={evidence['incoming_count']:2d} "
                f"out={evidence['outgoing_count']:2d} "
                f"decision={decision}"
            )

    print("=== END V60 DECOUPLED DESIGNER DECISIONS ===")
    print()


def v60_evaluate(
    substrate: LearnedStructuredSubstrate,
    ground_truth: IndependentGroundTruth,
) -> None:
    """
    Evaluate the blind designer against the independent V28 target.

    Ground truth exists only in this outer evaluator. It is never passed into
    DecoupledDesignerV60.
    """
    designer = DecoupledDesignerV60(substrate)

    correct = 0
    total = 0
    errors = []

    for word in TEST:
        for pos in range(len(word)):
            cell = substrate.encode(word, pos)

            actual = designer.decide_from_observable_state(cell)
            expected = (
                "REUSE"
                if ground_truth.available(word, pos)
                else "BRANCH"
            )

            total += 1

            if actual == expected:
                correct += 1
            else:
                errors.append(
                    {
                        "word": word,
                        "pos": pos,
                        "expected": expected,
                        "actual": actual,
                        "cell": cell.cell_id,
                    }
                )

    print("=== V60 DECUPLED DESIGNER RESULT ===")
    print(f"correct_positions : {correct}/{total}")
    print(f"accuracy          : {correct / total:.4f}")
    print(f"error_positions    : {len(errors)}")

    for error in errors[:20]:
        print(
            f"{error['word']:6s} "
            f"pos={error['pos']:2d} "
            f"cell={error['cell']:3d} "
            f"expected={error['expected']:6s} "
            f"actual={error['actual']:6s}"
        )

    print("=== END V60 DECUPLED DESIGNER RESULT ===")
    print()


def v60_boundary_separation(
    substrate: LearnedStructuredSubstrate,
    ground_truth: IndependentGroundTruth,
) -> None:
    """
    Measure whether the observable substrate state separates known and novel
    compositions before interpreting the designer accuracy.

    Again, composition semantics are not passed to the designer.
    """
    reuse = []
    branch = []

    for word in TEST:
        for pos in range(len(word)):
            cell = substrate.encode(word, pos)

            learned = (
                cell.activations
                + cell.incoming_count
                + cell.outgoing_count
            )

            if ground_truth.available(word, pos):
                reuse.append(learned)
            else:
                branch.append(learned)

    print("=== V60 OBSERVABLE SEPARATION ===")
    print("reuse_observable_values  :", reuse)
    print("branch_observable_values :", branch)

    print(
        "reuse_all_positive       :",
        all(value > 0 for value in reuse),
    )
    print(
        "branch_any_zero          :",
        any(value == 0 for value in branch),
    )
    print("=== END V60 OBSERVABLE SEPARATION ===")
    print()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def validate_v28_ground_truth() -> IndependentGroundTruth:
    print("=== V28 GROUND-TRUTH BALANCE ===")

    ground_truth = IndependentGroundTruth(REUSE_TRAINING)

    train_reuse = []
    train_branch = []
    test_reuse = []
    test_branch = []

    for word in TRAINING:
        for pos in range(len(word)):
            row = (word, pos, word[pos])
            if ground_truth.available(word, pos):
                train_reuse.append(row)
            else:
                train_branch.append(row)

    for word in TEST:
        for pos in range(len(word)):
            row = (word, pos, word[pos])
            if ground_truth.available(word, pos):
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

    return ground_truth


def show_substrate_summary(
    substrate: LearnedStructuredSubstrate,
) -> None:
    print("=== V59 STRUCTURED SUBSTRATE SUMMARY ===")
    print("composition_cells :", len(substrate.cells_by_id))
    print("transition_edges  :", len(substrate.transition_weights))

    degree_histogram: Dict[int, int] = {}
    for cell in substrate.cells_by_id.values():
        degree_histogram[cell.outgoing_count] = (
            degree_histogram.get(cell.outgoing_count, 0) + 1
        )

    print("outgoing_degree_histogram :", dict(sorted(degree_histogram.items())))
    print("=== END V59 STRUCTURED SUBSTRATE SUMMARY ===")
    print()


def probe_compositions(
    substrate: LearnedStructuredSubstrate,
    ground_truth: IndependentGroundTruth,
    words: Iterable[str],
) -> None:
    print("=== V59 COMPOSITION PROBE ===")

    for word in words:
        exact = True

        for pos in range(len(word)):
            cell = substrate.encode(word, pos)
            expected = (
                "REUSE"
                if ground_truth.available(word, pos)
                else "BRANCH"
            )

            print(
                f"{word:6s} pos={pos} "
                f"composition={cell.composition!r} "
                f"cell={cell.cell_id:3d} "
                f"expected={expected:6s} "
                f"activations={cell.activations:3d} "
                f"outgoing={cell.outgoing_count:2d}"
            )

        print()

    print("=== END V59 COMPOSITION PROBE ===")
    print()


def probe_decoupled_designer(
    substrate: LearnedStructuredSubstrate,
    ground_truth: IndependentGroundTruth,
    words: Iterable[str],
) -> None:
    """
    The ground truth is used only by this outer reporting function.
    DecoupledDesigner itself never sees it.
    """
    designer = DecoupledDesigner(substrate)

    print("=== V59 DECOUPLED DESIGNER PROBE ===")

    for word in words:
        for pos in range(len(word)):
            cell = substrate.encode(word, pos)
            evidence = designer.inspect(cell)

            expected = (
                "REUSE"
                if ground_truth.available(word, pos)
                else "BRANCH"
            )

            print(
                f"{word:6s} pos={pos} "
                f"cell={evidence['cell_id']:3d} "
                f"in={evidence['incoming_count']:2d} "
                f"out={evidence['outgoing_count']:2d} "
                f"activations={evidence['activation_count']:3d} "
                f"expected={expected}"
            )

    print("=== END V59 DECOUPLED DESIGNER PROBE ===")
    print()


def evaluate_structural_capacity(
    substrate: LearnedStructuredSubstrate,
    ground_truth: IndependentGroundTruth,
) -> None:
    """
    This is the key comparison.

    It does NOT claim that cell identity is a valid designer decision.
    It simply asks whether the structured substrate preserves the exact
    compositional distinctions required by the benchmark.
    """
    total = 0
    preserved = 0
    collapsed = 0

    for word in TEST:
        for pos in range(len(word)):
            total += 1

            composition = substrate.composition(word, pos)
            cell = substrate.cells_by_composition.get(composition)

            if ground_truth.available(word, pos):
                # A REUSE test composition should exist after training.
                if cell is not None:
                    preserved += 1
            else:
                # A BRANCH test composition should not have been trained.
                if cell is not None:
                    collapsed += 1

    print("=== V59 STRUCTURAL CAPACITY RESULT ===")
    print("test_positions              :", total)
    print("reuse_compositions_preserved:", preserved)
    print("branch_compositions_seen    :", collapsed)

    print(
        "reuse_preservation_rate     :",
        preserved / max(1, sum(
            1
            for word in TEST
            for pos in range(len(word))
            if ground_truth.available(word, pos)
        )),
    )
    print(
        "branch_false_presence_rate  :",
        collapsed / max(1, sum(
            1
            for word in TEST
            for pos in range(len(word))
            if not ground_truth.available(word, pos)
        )),
    )

    print("=== END V59 STRUCTURAL CAPACITY RESULT ===")
    print()


def v61_capacity_probe(
    substrate: LearnedStructuredSubstrate,
    ground_truth: IndependentGroundTruth,
) -> None:
    print("=== V61 LEARNED CAPACITY ===")

    train_cells = len(substrate.cells_by_id)

    reuse_present = 0
    branch_absent = 0
    errors = []

    for word in TEST:
        for pos in range(len(word)):
            cell = substrate.encode(word, pos, learn=False)

            expected = (
                "REUSE"
                if ground_truth.available(word, pos)
                else "BRANCH"
            )

            known = cell.cell_id >= 0

            if expected == "REUSE":
                reuse_present += int(known)
            else:
                branch_absent += int(not known)

            if known != (expected == "REUSE"):
                errors.append(
                    (
                        word,
                        pos,
                        expected,
                        cell.cell_id,
                    )
                )

    total_reuse = sum(
        1
        for word in TEST
        for pos in range(len(word))
        if ground_truth.available(word, pos)
    )
    total_branch = sum(
        1
        for word in TEST
        for pos in range(len(word))
        if not ground_truth.available(word, pos)
    )

    print("learned_cells            :", train_cells)
    print("reuse_present            :", reuse_present)
    print("reuse_total              :", total_reuse)
    print("branch_absent            :", branch_absent)
    print("branch_total             :", total_branch)
    print(
        "reuse_preservation_rate  :",
        reuse_present / max(1, total_reuse),
    )
    print(
        "branch_novelty_rate      :",
        branch_absent / max(1, total_branch),
    )
    print("capacity_errors          :", len(errors))

    for error in errors[:20]:
        print(
            f"{error[0]:6s} pos={error[1]:2d} "
            f"expected={error[2]:6s} "
            f"cell={error[3]}"
        )

    print("=== END V61 LEARNED CAPACITY ===")
    print()


def v61_probe_designer(
    substrate: LearnedStructuredSubstrate,
    words,
) -> None:
    designer = DecoupledDesignerV60(substrate)

    print("=== V61 DESIGNER OBSERVABLES ===")

    for word in words:
        for pos in range(len(word)):
            cell = substrate.encode(word, pos, learn=False)
            evidence = designer.inspect_cell(cell)
            decision = designer.decide_from_observable_state(cell)

            print(
                f"{word:6s} pos={pos} "
                f"cell={evidence['cell_id']:3d} "
                f"activations={evidence['activation_count']:3d} "
                f"in={evidence['incoming_count']:2d} "
                f"out={evidence['outgoing_count']:2d} "
                f"decision={decision}"
            )

    print("=== END V61 DESIGNER OBSERVABLES ===")
    print()


def main() -> None:
    print("=== V61 LEARNED STRUCTURED SUBSTRATE ===")
    print(
        "Cells are learned from repeated structural compositions; "
        "the designer receives no composition semantics."
    )
    print()

    ground_truth = validate_v28_ground_truth()

    substrate = LearnedStructuredSubstrate()

    print("=== V61 TRAINING ===")
    substrate.train(TRAINING)
    print("training_words     :", len(TRAINING))
    print(
        "training_positions :",
        sum(len(word) for word in TRAINING),
    )
    print("learned_cells      :", len(substrate.cells_by_id))
    print(
        "transition_edges   :",
        len(substrate.transition_weights),
    )
    print("=== END V61 TRAINING ===")
    print()

    v61_capacity_probe(
        substrate,
        ground_truth,
    )

    v61_probe_designer(
        substrate,
        ["CAT", "CAD", "BOAT", "BOARD"],
    )

    print("=== V61 COMPLETE ===")


if __name__ == "__main__":
    main()
