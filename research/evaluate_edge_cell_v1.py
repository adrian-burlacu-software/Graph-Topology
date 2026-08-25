from __future__ import annotations

from collections import defaultdict
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


class EdgeCellV1(DualVocabularyV6):
    """
    Edge-cell associative memory.

    The two directional vocabularies provide the endpoints of a local
    composition. The edge memory binds those endpoints together.

        prefix node ----\
                         > edge cell ---> designer
        suffix node ----/

    The designer never receives BoundaryGraph.has().

    The BoundaryGraph remains the long-term structural ground truth and
    is used only to record learned composition and evaluate correctness.
    """

    def __init__(self, genome):
        super().__init__(genome)

        # One associative cell per learned boundary relation.
        # Key: (prefix_node, boundary_symbol, suffix_node)
        self.edge_cells = {}
        self.edge_activity = defaultdict(float)

    def ensure_edge_cell(self, prefix_node, symbol, suffix_node):
        key = (prefix_node, symbol, suffix_node)

        if key not in self.edge_cells:
            self.edge_cells[key] = {
                "strength": 0.0,
                "spikes": 0,
            }

        return self.edge_cells[key]

    def learn_edge(self, word, pos):
        prefix_node = self.prefix.lookup(word[:pos])
        suffix_node = self.suffix.lookup(word[pos + 1:])
        symbol = word[pos]

        if prefix_node is None or suffix_node is None:
            return

        cell = self.ensure_edge_cell(
            prefix_node,
            symbol,
            suffix_node,
        )

        cell["strength"] = min(
            1.0,
            cell["strength"] + 0.5,
        )

        self.boundaries.add(
            prefix_node,
            symbol,
            suffix_node,
        )

    def edge_signal(self, word, pos):
        """
        Produce associative activity from the actual endpoint pair.

        No boolean availability is passed to the designer.
        """
        prefix_node = self.prefix.lookup(word[:pos])
        suffix_node = self.suffix.lookup(word[pos + 1:])
        symbol = word[pos]

        if prefix_node is None or suffix_node is None:
            return 0.0

        key = (prefix_node, symbol, suffix_node)
        cell = self.edge_cells.get(key)

        if cell is None:
            return 0.0

        return cell["strength"]

    def designer_from_edge_activity(self, word, pos, learn=False):
        """
        Feed the designer only associative edge activity.

        The exact boundary answer is NOT passed in.
        """
        n = self.net
        dg = n.designer_genome

        self.reset_designer_transient_state()
        n._reset_designer_input()

        edge = self.edge_signal(word, pos)

        root = n.cells[n.designer_root]
        reuse = n.cells[n.reuse_cell]
        branch = n.cells[n.branch_cell]

        root.potential += dg["input_gain"]

        if edge > 0.0:
            reuse.potential += (
                dg["match_gain"] * edge
            )
        else:
            branch.potential += dg["branch_bias"]

            # Explicit negative evidence remains local to the decision.
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
        print("=== EDGE-CELL TRAINING ===")
        print()

        for epoch in range(1, epochs + 1):
            reuse = 0
            branch = 0

            for word in words:
                # First establish directional vocabulary paths.
                for pos in range(len(word)):
                    self.prefix.ensure_path(word[:pos])
                    self.suffix.ensure_path(word[pos + 1:])

                # Then bind each pair through an edge cell.
                for pos in range(len(word)):
                    self.learn_edge(word, pos)

                    action = self.designer_from_edge_activity(
                        word,
                        pos,
                        learn=False,
                    )

                    if action == "REUSE":
                        reuse += 1
                    else:
                        branch += 1

            print(
                f"epoch={epoch:3d} "
                f"reuse={reuse:3d} "
                f"branch={branch:3d} "
                f"edge_cells={len(self.edge_cells):3d} "
                f"links={len(self.boundaries.links):3d}"
            )

    def evaluate_frozen(self, words):
        print()
        print("=== EDGE-CELL FROZEN TEST ===")

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
                # Ground truth is used ONLY for scoring.
                expected = self.available(word, pos)

                if expected:
                    expected_reuse += 1
                else:
                    expected_branch += 1

                action = self.designer_from_edge_activity(
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


def run():
    net = EdgeCellV1(deepcopy(GENOME))

    print("=== EDGE-CELL V1 ===")
    print()
    print("Two directional memories + associative edge cells + one designer.")
    print("Designer receives edge activity, not BoundaryGraph.has().")
    print()

    net.train_edges(TRAINING, epochs=5)
    net.evaluate_frozen(TEST)

    print()
    print("=== EDGE MEMORY ===")

    for key, cell in sorted(net.edge_cells.items()):
        prefix_node, symbol, suffix_node = key
        print(
            f"({prefix_node}, {symbol}, {suffix_node}) "
            f"strength={cell['strength']:.3f} "
            f"spikes={cell['spikes']}"
        )


if __name__ == "__main__":
    run()
