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

        # V8 diagnostics: prove whether coincidence plasticity is actually
        # being reached and whether receptive weights change.
        self.learning_events = 0
        self.learning_debug = []

        # Distributed receptive fields. Every generic cell has both
        # channels; no cell is assigned to a particular composition.
        self.prefix_weights = [0.10 for _ in range(cell_count)]
        self.suffix_weights = [0.10 for _ in range(cell_count)]

        # Stronger learning is intentional here: V3 demonstrated that the
        # coincidence signal existed, but never became a learned topology.
        self.learning_rate = max(
            0.25,
            self.learning_rate,
        )

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
        dense cell pool.
        """
        prefix_node, symbol, suffix_node = context

        p = 0 if prefix_node is None else prefix_node
        s = 0 if suffix_node is None else suffix_node

        return (
            ((p + 1) * 73856093)
            ^ (ord(symbol) * 19349663)
            ^ ((s + 1) * 83492791)
        )

    def activate_substrate(self, word, pos, learn=False):
        """
        Dense coincidence substrate.

        Prefix and suffix are independent inputs. Coincidence is computed
        first and directly drives plasticity. Spiking is a later consequence
        of the learned receptive fields.

        There are no explicit edge cells and no boundary lookup.
        """
        context = self.context_vector(word, pos)
        prefix_node, symbol, suffix_node = context

        p = 0 if prefix_node is None else prefix_node + 1
        s = 0 if suffix_node is None else suffix_node + 1

        hp = (p * 73856093) ^ (ord(symbol) * 19349663)
        hs = (s * 83492791) ^ (ord(symbol) * 2654435761)

        prefix_active = prefix_node is not None
        suffix_active = suffix_node is not None

        fired = []

        for i, cell in enumerate(self.cells):
            p_phase = (
                ((hp ^ (i * 2246822519)) & 0xFFFF) / 65535.0
            )
            s_phase = (
                ((hs ^ (i * 3266489917)) & 0xFFFF) / 65535.0
            )

            prefix_drive = (
                0.35 + 0.25 * p_phase
                if prefix_active else 0.0
            )
            suffix_drive = (
                0.35 + 0.25 * s_phase
                if suffix_active else 0.0
            )

            coincidence = prefix_drive * suffix_drive

            # IMPORTANT: coincidence learning happens independently of
            # whether this generic neuron spikes.
            if learn and prefix_active and suffix_active:
                before_p = self.prefix_weights[i]
                before_s = self.suffix_weights[i]

                delta = self.learning_rate * coincidence

                self.prefix_weights[i] = min(
                    1.0,
                    self.prefix_weights[i] + delta,
                )
                self.suffix_weights[i] = min(
                    1.0,
                    self.suffix_weights[i] + delta,
                )

                self.learning_events += 1

                # Capture only the first few events so output stays readable.
                if len(self.learning_debug) < 12:
                    self.learning_debug.append({
                        "cell": i,
                        "prefix_active": prefix_active,
                        "suffix_active": suffix_active,
                        "coincidence": coincidence,
                        "learning_rate": self.learning_rate,
                        "delta": delta,
                        "prefix_before": before_p,
                        "prefix_after": self.prefix_weights[i],
                        "suffix_before": before_s,
                        "suffix_after": self.suffix_weights[i],
                    })

            # Learned receptive fields now amplify the same coincidence.
            learned_drive = (
                self.prefix_weights[i]
                * self.suffix_weights[i]
                * coincidence
            )

            drive = (
                0.10 * (prefix_drive + suffix_drive)
                + 0.50 * coincidence
                + learned_drive
            )

            if cell.stimulate(drive):
                fired.append(i)

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
                # Populate the two directional memories as input
                # representations. This is NOT ground truth: the dense
                # substrate still has no boundary/edge lookup.
                for pos in range(len(word)):
                    self.prefix.ensure_path(word[:pos])
                    self.suffix.ensure_path(word[pos + 1:])

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

            learned_strength = sum(
                p * s
                for p, s in zip(
                    self.prefix_weights,
                    self.suffix_weights,
                )
            )

            print(
                f"epoch={epoch:3d} "
                f"active_spikes={active:4d} "
                f"strong_connections={strong:4d} "
                f"receptive_strength={learned_strength:.4f}"
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
        # Activity remains distributed. The learned receptive fields
        # provide the persistent memory trace used by the designer.
        learned_activity = sum(
            self.prefix_weights[i] * self.suffix_weights[i]
            for i in fired
        )

        activity = min(
            1.0,
            learned_activity / max(1, self.cell_count),
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

    def build_ground_truth(self, training_words):
        """
        Build evaluation truth independently from the dense substrate.

        This graph is evaluator-only. It is never attached to the network
        and is never consulted by designer_from_dense_activity().
        """
        truth = set()

        for word in training_words:
            for pos in range(len(word)):
                prefix = word[:pos]
                symbol = word[pos]
                suffix = word[pos + 1:]
                truth.add((prefix, symbol, suffix))

        return truth

    def evaluate_frozen(self, words, ground_truth):
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
                expected = (
                    word[:pos],
                    word[pos],
                    word[pos + 1:],
                ) in ground_truth

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
    print("=== DENSE SUBSTRATE V9 - COINCIDENCE DIAGNOSTIC ===")
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

    print()
    print("=== LEARNING PATH DIAGNOSTICS ===")
    print(f"learning_events : {net.learning_events}")

    if not net.learning_debug:
        print("NO LEARNING EVENTS RECORDED")
    else:
        for event in net.learning_debug:
            print(
                "cell={cell} "
                "prefix_active={prefix_active} "
                "suffix_active={suffix_active} "
                "coincidence={coincidence:.6f} "
                "lr={learning_rate:.6f} "
                "delta={delta:.6f} "
                "prefix={prefix_before:.6f}->{prefix_after:.6f} "
                "suffix={suffix_before:.6f}->{suffix_after:.6f}".format(
                    **event
                )
            )

    print()
    print("=== COINCIDENCE TOPOLOGY ===")
    initial_receptive_strength = 32 * 0.1 * 0.1
    final_receptive_strength = sum(
        p * s
        for p, s in zip(
            net.prefix_weights,
            net.suffix_weights,
        )
    )

    print(
        "initial_receptive_strength : "
        f"{initial_receptive_strength:.4f}"
    )
    print(
        "final_receptive_strength   : "
        f"{final_receptive_strength:.4f}"
    )
    print(
        "receptive_strength_delta   : "
        f"{final_receptive_strength - initial_receptive_strength:.4f}"
    )
    strong = sum(
        1
        for weight in net.weights.values()
        if weight >= 0.50
    )
    print("potential_connections :", len(net.weights))
    print("strong_connections    :", strong)

    # No pruning in this experiment. We want the effect of coincidence
    # learning in isolation.
    # IMPORTANT: evaluator-only truth. This is constructed separately
    # from the dense substrate so an empty BoundaryGraph cannot create
    # false 100% scores.
    ground_truth = net.build_ground_truth(TRAINING)

    print()
    print("=== INDEPENDENT GROUND TRUTH ===")
    print(f"training_compositions : {len(ground_truth)}")

    net.evaluate_frozen(
        TEST,
        ground_truth,
    )


if __name__ == "__main__":
    run()
