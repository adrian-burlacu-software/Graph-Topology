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
class CompositionCell:
    cell_id: int
    composition: Composition
    activations: int = 0
    incoming_count: int = 0
    outgoing_count: int = 0


class StructuredCompositionSubstrate:
    """
    Structured baseline.

    A cell is allocated per observed structural composition
        (prefix, symbol, suffix).

    Importantly:
      * the evaluator's ground truth is not supplied here;
      * REUSE / BRANCH labels are not supplied here;
      * the designer gets only the active cell identity;
      * training includes both REUSE_TRAINING and BRANCH_TRAINING.

    This deliberately exposes the compositional substrate's representational
    capacity so we can compare it against the fully-connected generic pool.
    """

    def __init__(self) -> None:
        self.cells_by_composition: Dict[Composition, CompositionCell] = {}
        self.cells_by_id: Dict[int, CompositionCell] = {}
        self.next_cell_id = 0

        # Transition topology between structural compositions.
        self.transition_weights: Dict[Tuple[int, int], float] = {}

    @staticmethod
    def composition(word: str, pos: int) -> Composition:
        prefix = word[:pos]
        symbol = word[pos]
        suffix = word[pos + 1:]
        return prefix, symbol, suffix

    def _get_or_create(self, composition: Composition) -> CompositionCell:
        cell = self.cells_by_composition.get(composition)
        if cell is not None:
            return cell

        cell = CompositionCell(
            cell_id=self.next_cell_id,
            composition=composition,
        )
        self.next_cell_id += 1

        self.cells_by_composition[composition] = cell
        self.cells_by_id[cell.cell_id] = cell
        return cell

    def encode(self, word: str, pos: int) -> CompositionCell:
        """
        Frozen readout.

        No mutable membrane state, no RNG, and no learning occur here.
        """
        composition = self.composition(word, pos)
        return self._get_or_create(composition)

    def train_word(self, word: str) -> None:
        previous: Optional[CompositionCell] = None

        for pos in range(len(word)):
            cell = self._get_or_create(self.composition(word, pos))
            cell.activations += 1

            if previous is not None:
                key = (previous.cell_id, cell.cell_id)
                self.transition_weights[key] = (
                    self.transition_weights.get(key, 0.0) + 1.0
                )
                previous.outgoing_count += 1
                cell.incoming_count += 1

            previous = cell

    def train(self, words: Iterable[str]) -> None:
        for word in words:
            self.train_word(word)

    def transition_support(
        self,
        previous: CompositionCell,
        current: CompositionCell,
    ) -> float:
        return self.transition_weights.get(
            (previous.cell_id, current.cell_id),
            0.0,
        )


# ---------------------------------------------------------------------------
# Decoupled designer
# ---------------------------------------------------------------------------

class DecoupledDesigner:
    """
    Minimal designer.

    It receives ONLY a substrate cell and its learned structural topology.

    It does not receive:
      * IndependentGroundTruth
      * the word
      * the symbol
      * REUSE/BRANCH labels
      * an external list of known compositions

    For this baseline, the designer emits structural evidence rather than
    pretending that structural identity itself is proof of REUSE.
    """

    def __init__(self, substrate: StructuredCompositionSubstrate) -> None:
        self.substrate = substrate

    def inspect(self, cell: CompositionCell) -> Dict[str, object]:
        outgoing = [
            (dst, weight)
            for (src, dst), weight in self.substrate.transition_weights.items()
            if src == cell.cell_id
        ]

        outgoing.sort(key=lambda item: (-item[1], item[0]))

        return {
            "cell_id": cell.cell_id,
            "activation_count": cell.activations,
            "incoming_count": cell.incoming_count,
            "outgoing_count": cell.outgoing_count,
            "outgoing": tuple(outgoing),
        }


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
    substrate: StructuredCompositionSubstrate,
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
    substrate: StructuredCompositionSubstrate,
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
    substrate: StructuredCompositionSubstrate,
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
    substrate: StructuredCompositionSubstrate,
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


def main() -> None:
    print("=== V59 STRUCTURED COMPOSITION BASELINE ===")
    print(
        "Control question: does the structured substrate preserve "
        "compositional distinctions that the dense generic pool collapsed?"
    )
    print()

    ground_truth = validate_v28_ground_truth()

    substrate = StructuredCompositionSubstrate()

    print("=== V59 TRAINING ===")
    substrate.train(TRAINING)
    print("training_words       :", len(TRAINING))
    print(
        "training_positions   :",
        sum(len(word) for word in TRAINING),
    )
    print()

    show_substrate_summary(substrate)

    # Small representative probe.
    probe_words = [
        "CAT",
        "CAD",
        "BOAT",
        "BOARD",
    ]

    probe_compositions(
        substrate,
        ground_truth,
        probe_words,
    )

    probe_decoupled_designer(
        substrate,
        ground_truth,
        probe_words,
    )

    evaluate_structural_capacity(
        substrate,
        ground_truth,
    )

    print("=== V59 COMPLETE ===")


if __name__ == "__main__":
    main()
