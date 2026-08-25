from __future__ import annotations

import random
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


class DenseCell:
    """Generic neuron in an initially dense substrate."""

    def __init__(self, threshold=0.75, leak=0.9):
        self.potential = 0.0
        self.threshold = threshold
        self.leak = leak
        self.spikes = 0

    def reset(self):
        self.potential = 0.0

    def stimulate(self, amount):
        self.potential = self.potential * self.leak + amount
        if self.potential >= self.threshold:
            self.potential = 0.0
            self.spikes += 1
            return True
        return False


class DensePlasticSubstrateV1(DualVocabularyV6):
    """
    Control experiment:

    Start with a fully connected pool of generic associative cells.
    No explicit (prefix, symbol, suffix) edge-cell allocation is used.

    Each candidate cell can receive activity from every structural
    context. Plasticity determines which cells become useful.

    This is deliberately NOT claimed to be a biological model. It is a
    controlled substrate experiment: dense possibility space versus the
    structured edge-cell architecture.
    """

    def __init__(self, genome, cell_count=32, seed=29):
        super().__init__(genome)

        random.seed(seed)

        dg = self.net.designer_genome
        pg = self.net.plasticity_genome

        self.cell_count = cell_count
        self.cells = [
            DenseCell(
                threshold=min(0.75, dg["threshold"]),
                leak=dg["leak"],
            )
            for _ in range(cell_count)
        ]

        # Fully connected potential matrix.
        # Every structural context can address every generic cell.
        self.weights = {}
        for i in range(cell_count):
            for j in range(cell_count):
                if i != j:
                    self.weights[(i, j)] = 0.05

        self.learning_rate = max(
            0.05,
            pg["weight_learning_rate"],
        )

        self.active_edges = set()

    def context_vector(self, word, pos):
        """
        Encode the two directional structural contexts separately.

        The dense substrate receives distinct prefix and suffix drives.
        Their coincidence, not either context alone, controls plasticity.
        """
        prefix_node = self.prefix.lookup(word[:pos])
        suffix_node = self.suffix.lookup(word[pos + 1:])

        return (
            prefix_node,
            word[pos],
            suffix_node,
        )

    def context_hash(self, context):
        """
        Deterministic projection of arbitrary structural context into the
        dense cell pool. This is deliberately many-to-many: every cell
        remains a potential participant.
        """
        prefix_node, symbol, suffix_node = context

        p = 0 if prefix_node is None else prefix_node
        s = 0 if suffix_node is None else suffix_node

        value = (
            (p + 1) * 73856093
            ^ (ord(symbol) * 19349663)
            ^ ((s + 1) * 83492791)
        )

        return value

    def activate_substrate(self, word, pos, learn=False):
        """
        Two-channel coincidence learning.

        Prefix and suffix are separate input populations. A generic cell
        can become strongly connected only when both channels are active
        in the same local event.

        No BoundaryGraph lookup and no explicit edge-cell allocation.
        """
        context = self.context_vector(word, pos)
        prefix_node, symbol, suffix_node = context

        h_prefix = (
            0
            if prefix_node is None
            else (prefix_node + 1) * 73856093
        )

        h_suffix = (
            0
            if suffix_node is None
            else (suffix_node + 1) * 83492791
        )

        h_symbol = ord(symbol) * 19349663

        prefix_active = prefix_node is not None
        suffix_active = suffix_node is not None

        fired = []

        for i, cell in enumerate(self.cells):
            # Distinct projections from the two structural memories.
            p_phase = (
                ((h_prefix ^ (i * 2654435761)) & 0xFFFF)
                / 65535.0
            )

            s_phase = (
                ((h_suffix ^ (i * 2246822519)) & 0xFFFF)
                / 65535.0
            )

            symbol_phase = (
                ((h_symbol ^ (i * 3266489917)) & 0xFFFF)
                / 65535.0
            )

            prefix_drive = (
                0.12 + 0.10 * p_phase
                if prefix_active else 0.0
            )

            suffix_drive = (
                0.12 + 0.10 * s_phase
                if suffix_active else 0.0
            )

            # Additive membrane integration is deliberately weaker than
            # multiplicative coincidence. The latter is the plasticity
            # gate.
            total_drive = (
                prefix_drive
                + suffix_drive
                + 0.04 * symbol_phase
            )

            if cell.stimulate(total_drive):
                fired.append(i)

            if learn and prefix_active and suffix_active:
                coincidence = prefix_drive * suffix_drive

                if coincidence > 0.0:
                    # Coincidence selectively strengthens the active
                    # pathway. The product makes both sides necessary.
                    for j in range(self.cell_count):
                        if j == i:
                            continue

                        key = (i, j)

                        self.weights[key] = min(
                            1.0,
                            self.weights[key]
                            + self.learning_rate * coincidence,
                        )

        return fired

    def train_dense(self, words, epochs=5):
        print("=== DENSE SUBSTRATE TRAINING ===")
        print()
        print(
            f"cells={self.cell_count} "
            f"potential_connections={len(self.weights)}"
        )
        print()

        for epoch in range(1, epochs + 1):
            active = 0

            for word in words:
                for pos in range(len(word)):
                    fired = self.activate_substrate(
                        word,
                        pos,
                        learn=True,
                    )
                    active += len(fired)

            strong = sum(
                1
                for weight in self.weights.values()
                if weight >= 0.50
            )

            print(
                f"epoch={epoch:3d} "
                f"active_spikes={active:4d} "
                f"strong_connections={strong:4d}"
            )

    def designer_from_dense_activity(self, word, pos):
        """
        Collapse distributed substrate activity into the existing
        designer. No exact boundary availability is supplied.
        """
        n = self.net
        dg = n.designer_genome

        self.reset_designer_transient_state()
        n._reset_designer_input()

        fired = self.activate_substrate(
            word,
            pos,
            learn=False,
        )

        # Structural evidence is distributed activity. The designer still
        # receives no exact-boundary Boolean.
        activity = min(
            1.0,
            len(fired) / max(1, self.cell_count),
        )

        root = n.cells[n.designer_root]
        reuse = n.cells[n.reuse_cell]
        branch = n.cells[n.branch_cell]

        root.potential += dg["input_gain"]

        if activity > 0.20:
            reuse.potential += (
                dg["match_gain"] * activity
            )
        else:
            branch.potential += dg["branch_bias"]

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

    def prune_competitive(self, max_outgoing=4, minimum_strength=0.10):
        """
        Competitive magnitude pruning.

        For each source cell, retain only its strongest outgoing
        connections. This converts the dense substrate into an emergent
        sparse graph without prescribing which cells should connect.

        Ties are broken deterministically by destination id.
        """
        kept = {}

        for i in range(self.cell_count):
            outgoing = [
                (j, weight)
                for (src, j), weight in self.weights.items()
                if src == i and weight >= minimum_strength
            ]

            outgoing.sort(
                key=lambda item: (-item[1], item[0])
            )

            for j, weight in outgoing[:max_outgoing]:
                kept[(i, j)] = weight

        # Rebuild the potential matrix. Pruned edges are gone rather than
        # merely marked inactive, so the resulting topology is explicit.
        self.weights = kept

        return len(kept)

    def evaluate_frozen(self, words):
        print()
        print("=== DENSE SUBSTRATE FROZEN TEST ===")

        exact_words = 0
        correct = 0
        total = 0

        strong_before = sum(
            1 for w in self.weights.values() if w >= 0.50
        )

        for word in words:
            exact = True
            reuse = 0
            branch = 0

            for pos in range(len(word)):
                expected = self.available(word, pos)

                action = self.designer_from_dense_activity(
                    word,
                    pos,
                )

                if action == "REUSE":
                    reuse += 1
                else:
                    branch += 1

                wanted = "REUSE" if expected else "BRANCH"

                if action == wanted:
                    correct += 1
                else:
                    exact = False

                total += 1

            if exact:
                exact_words += 1

            print(
                f"{word:6s} "
                f"designer_reuse={reuse:2d} "
                f"designer_branch={branch:2d} "
                f"exact={exact}"
            )

        strong_after = sum(
            1 for w in self.weights.values() if w >= 0.50
        )

        print()
        print("=== GENERALIZATION ===")
        print(f"test_words        : {len(words)}")
        print(f"exact_words       : {exact_words}/{len(words)}")
        print(f"correct_positions : {correct}/{total}")
        print(f"accuracy          : {correct / total:.4f}")

        print()
        print("=== DENSE TOPOLOGY ===")
        print(f"cells                 : {self.cell_count}")
        print(f"potential_connections : {len(self.weights)}")
        print(f"strong_before_test    : {strong_before}")
        print(f"strong_after_test     : {strong_after}")


def run():
    print("=== DENSE SUBSTRATE V3 - COINCIDENCE ===")
    print()
    print(
        "Fully connected generic substrate with separate prefix/suffix "
        "channels and coincidence-gated plasticity."
    )
    print(
        "No explicit edge-cell allocation and no BoundaryGraph.has() "
        "input to the designer."
    )
    print()

    net = DensePlasticSubstrateV1(
        deepcopy(GENOME),
        cell_count=32,
        seed=29,
    )

    net.train_dense(TRAINING, epochs=5)

    print()
    print("=== COINCIDENCE TOPOLOGY ===")
    strong = sum(
        1
        for weight in net.weights.values()
        if weight >= 0.50
    )
    print("potential_connections :", len(net.weights))
    print("strong_connections    :", strong)

    # No pruning in this experiment. We want the effect of coincidence
    # learning in isolation.
    net.evaluate_frozen(TEST)


if __name__ == "__main__":
    run()
