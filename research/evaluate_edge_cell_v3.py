from __future__ import annotations

from copy import deepcopy

from genome import GENOME
from evaluate_dual_vocabulary_v6 import DualVocabularyV6


TRAINING = [
    "CAT", "CAR", "CAN", "CARD", "CART", "DOG", "DOT", "BAT",
]

TEST = [
    "CAT", "CAR", "CAN", "CARD", "CART",
    "CAD", "COD", "COT", "BAD", "BAR", "BARD", "BAN",
    "DART", "DAT", "BOT", "BOAT", "CARTD",
    "COARD", "BAND", "BOARD",
]


class AssociativeEdgeCell:
    """
    A small stateful associative neuron.

    The cell is activated by coincident endpoint activity. Its learned
    strength is persistent; membrane potential is transient.

    It does not know whether its output is "reuse" or "branch".
    It only emits activity when its learned association is sufficiently
    strong and the two endpoints are simultaneously active.
    """

    def __init__(
        self,
        learning_rate=0.25,
        decay=0.90,
        threshold=0.75,
    ):
        self.strength = 0.0
        self.potential = 0.0
        self.spikes = 0

        self.learning_rate = learning_rate
        self.decay = decay
        self.threshold = threshold

    def reset(self):
        self.potential = 0.0

    def stimulate(self, coincidence: float, learn: bool):
        # Transient membrane state.
        self.potential *= self.decay
        self.potential += coincidence * self.strength

        if learn and coincidence > 0.0:
            # Hebbian-style association strengthening.
            self.strength = min(
                1.0,
                self.strength + self.learning_rate * coincidence,
            )

        if self.potential >= self.threshold:
            self.spikes += 1
            self.potential = 0.0
            return True

        return False


class EdgeCellV2(DualVocabularyV6):
    """
    EdgeCell V2.

    V1 represented an edge as a persistent scalar lookup.

    V2 turns each learned edge into a real stateful associative cell:

        prefix activity ----\
                              > associative edge cell --> designer
        suffix activity ----/

    The designer never receives BoundaryGraph.has() and never receives
    an "available" Boolean.

    Ground truth is consulted only by the frozen evaluator.
    """

    def __init__(self, genome):
        super().__init__(genome)

        dg = self.net.designer_genome
        pg = self.net.plasticity_genome

        self.edge_learning_rate = max(
            0.25,
            pg["weight_learning_rate"],
        )
        self.edge_decay = dg["leak"]
        self.edge_threshold = min(
            0.75,
            dg["threshold"],
        )

        self.edge_cells = {}

    def edge_key(self, word, pos):
        prefix_node = self.prefix.lookup(word[:pos])
        suffix_node = self.suffix.lookup(word[pos + 1:])
        symbol = word[pos]

        if prefix_node is None or suffix_node is None:
            return None

        return (
            prefix_node,
            symbol,
            suffix_node,
        )

    def ensure_edge_cell(self, key):
        if key not in self.edge_cells:
            self.edge_cells[key] = AssociativeEdgeCell(
                learning_rate=self.edge_learning_rate,
                decay=self.edge_decay,
                threshold=self.edge_threshold,
            )

        return self.edge_cells[key]

    def stimulate_edge(self, word, pos, learn=False):
        """
        Drive the associative edge cell from simultaneous endpoint
        activity.

        If the endpoint pair has never been learned, there is no cell
        and therefore no edge activity.
        """
        key = self.edge_key(word, pos)

        if key is None:
            return False

        cell = self.edge_cells.get(key)

        if cell is None:
            return False

        # Both directional memories have to participate in the
        # coincidence. The edge cell does not inspect the boundary graph.
        prefix_node = self.prefix.lookup(word[:pos])
        suffix_node = self.suffix.lookup(word[pos + 1:])

        coincidence = 1.0 if (
            prefix_node is not None
            and suffix_node is not None
        ) else 0.0

        return cell.stimulate(
            coincidence,
            learn=learn,
        )

    def learn_edge(self, word, pos):
        key = self.edge_key(word, pos)

        if key is None:
            return

        cell = self.ensure_edge_cell(key)

        # Establish the structural edge and reinforce its associative
        # neuron. BoundaryGraph is storage here, not designer input.
        self.boundaries.add(
            key[0],
            key[1],
            key[2],
        )

        cell.stimulate(
            coincidence=1.0,
            learn=True,
        )

        # Training should create an immediately useful association.
        # Repeated exposure then asymptotically strengthens it.
        cell.potential = 0.0

    def designer_from_edge_cell(self, word, pos, learn=False):
        """
        The designer sees only the spike from the associative edge cell.

        There is no BoundaryGraph.has() call here.
        """
        n = self.net
        dg = n.designer_genome

        self.reset_designer_transient_state()
        n._reset_designer_input()

        edge_spike = self.stimulate_edge(
            word,
            pos,
            learn=learn,
        )

        root = n.cells[n.designer_root]
        reuse = n.cells[n.reuse_cell]
        branch = n.cells[n.branch_cell]

        root.potential += dg["input_gain"]

        if edge_spike:
            reuse.potential += dg["match_gain"]
        else:
            branch.potential += dg["branch_bias"]

            # Negative evidence is a transient competition signal.
            reuse.inhibition += n.inhibition_genome["strength"]
            reuse.potential -= n.inhibition_genome["strength"]

        if root.potential >= dg["threshold"]:
            root.potential = 0.0
            root.spikes += 1
            n.designer_spikes += 1

            reuse.potential += n.synapses[
                (n.designer_root, n.reuse_cell)
            ].weight

            branch.potential += n.synapses[
                (n.designer_root, n.branch_cell)
            ].weight

        threshold = dg["threshold"]

        if reuse.potential >= threshold:
            branch.inhibition += n.inhibition_genome["strength"]
            branch.potential -= n.inhibition_genome["strength"]
            reuse.spikes += 1
            n.designer_spikes += 1

        if branch.potential >= threshold:
            reuse.inhibition += n.inhibition_genome["strength"]
            reuse.potential -= n.inhibition_genome["strength"]
            branch.spikes += 1
            n.designer_spikes += 1

        return n.designer_signal(None, "")

    def train_edges(self, words, epochs=5):
        print("=== EDGECELL V3 TRAINING ===")
        print()

        for epoch in range(1, epochs + 1):
            reuse = 0
            branch = 0
            spikes = 0

            for word in words:
                # First create directional structural paths.
                for pos in range(len(word)):
                    self.prefix.ensure_path(word[:pos])
                    self.suffix.ensure_path(word[pos + 1:])

                # Then expose each learned composition to its edge cell.
                for pos in range(len(word)):
                    self.learn_edge(word, pos)

                    action = self.designer_from_edge_cell(
                        word,
                        pos,
                        learn=False,
                    )

                    if action == "REUSE":
                        reuse += 1
                    else:
                        branch += 1

                    key = self.edge_key(word, pos)
                    if key is not None:
                        cell = self.edge_cells.get(key)
                        if cell is not None:
                            spikes += cell.spikes

            print(
                f"epoch={epoch:3d} "
                f"reuse={reuse:3d} "
                f"branch={branch:3d} "
                f"edge_cells={len(self.edge_cells):3d} "
                f"links={len(self.boundaries.links):3d} "
                f"edge_spikes={spikes:3d}"
            )

    def evaluate_frozen(self, words):
        print()
        print("=== EDGECELL V3 FROZEN TEST ===")

        links_before = len(self.boundaries.links)
        prefix_before = self.prefix.next_id
        suffix_before = self.suffix.next_id
        edge_before = len(self.edge_cells)

        exact_words = 0
        correct_positions = 0
        total_positions = 0

        for word in words:
            expected_reuse = 0
            expected_branch = 0
            actual_reuse = 0
            actual_branch = 0
            exact = True

            for pos in range(len(word)):
                # Ground truth is used ONLY for evaluation.
                expected = self.available(word, pos)

                if expected:
                    expected_reuse += 1
                else:
                    expected_branch += 1

                action = self.designer_from_edge_cell(
                    word,
                    pos,
                    learn=False,
                )

                if action == "REUSE":
                    actual_reuse += 1
                else:
                    actual_branch += 1

                wanted = "REUSE" if expected else "BRANCH"

                if action == wanted:
                    correct_positions += 1
                else:
                    exact = False

                total_positions += 1

            if exact:
                exact_words += 1

            print(
                f"{word:6s} "
                f"expected_reuse={expected_reuse:2d} "
                f"expected_branch={expected_branch:2d} "
                f"designer_reuse={actual_reuse:2d} "
                f"designer_branch={actual_branch:2d} "
                f"exact={exact}"
            )

        print()
        print("=== GENERALIZATION ===")
        print(f"test_words        : {len(words)}")
        print(f"exact_words       : {exact_words}/{len(words)}")
        print(
            f"correct_positions : "
            f"{correct_positions}/{total_positions}"
        )
        print(
            f"accuracy          : "
            f"{correct_positions / total_positions:.4f}"
        )

        print()
        print("=== FROZEN INVARIANTS ===")
        print(f"boundary_links_before : {links_before}")
        print(f"boundary_links_after  : {len(self.boundaries.links)}")
        print(f"prefix_nodes_before   : {prefix_before}")
        print(f"prefix_nodes_after    : {self.prefix.next_id}")
        print(f"suffix_nodes_before   : {suffix_before}")
        print(f"suffix_nodes_after    : {self.suffix.next_id}")
        print(f"edge_cells_before     : {edge_before}")
        print(f"edge_cells_after      : {len(self.edge_cells)}")

    def print_edge_cells(self):
        print()
        print("=== ASSOCIATIVE EDGE CELLS V3 ===")

        for key, cell in sorted(self.edge_cells.items()):
            prefix_node, symbol, suffix_node = key

            print(
                f"({prefix_node}, {symbol}, {suffix_node}) "
                f"strength={cell.strength:.4f} "
                f"potential={cell.potential:.4f} "
                f"spikes={cell.spikes} "
                f"threshold={cell.threshold:.4f}"
            )


def run():
    net = EdgeCellV2(deepcopy(GENOME))

    print("=== EDGECELL V3 ===")
    print()
    print("Two directional memories + real associative edge neurons + designer.")
    print("The designer receives edge spikes, not BoundaryGraph.has().")
    print("Edge learning is direct bounded accumulation; threshold is 0.75.")
    print()

    net.train_edges(TRAINING, epochs=5)
    net.evaluate_frozen(TEST)
    net.print_edge_cells()


if __name__ == "__main__":
    run()
