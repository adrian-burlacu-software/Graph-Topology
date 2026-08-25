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


class ActivityDrivenV1(DualVocabularyV6):
    """
    V1 removes the evaluator-level "exact boundary exists" signal from
    the designer.

    The two directional vocabularies generate structural activity.
    A boundary candidate is represented by coincident prefix/suffix
    activity for the same boundary symbol.

    The exact boundary graph remains the learned structural memory, but
    it is NOT passed into designer_action as a Boolean answer.
    """

    def boundary_activity(self, word, pos):
        """
        Return local structural activity from the two vocabularies.

        Prefix activity:
            path for word[:pos]

        Suffix activity:
            reverse path for word[pos+1:]

        The designer sees only whether the two paths exist and the
        corresponding node IDs/symbol, not the BoundaryGraph lookup.
        """
        prefix_node = self.prefix.lookup(word[:pos])
        suffix_node = self.suffix.lookup(word[pos + 1:])

        return {
            "prefix": prefix_node,
            "suffix": suffix_node,
            "symbol": word[pos],
            "prefix_active": prefix_node is not None,
            "suffix_active": suffix_node is not None,
        }

    def designer_from_activity(self, activity, learn=False):
        """
        Convert directional vocabulary activity into designer input.

        Importantly, this does NOT call BoundaryGraph.has().
        """
        n = self.net
        dg = n.designer_genome

        self.reset_designer_transient_state()
        n._reset_designer_input()

        root = n.cells[n.designer_root]
        reuse = n.cells[n.reuse_cell]
        branch = n.cells[n.branch_cell]

        root.potential += dg["input_gain"]

        prefix_active = activity["prefix_active"]
        suffix_active = activity["suffix_active"]

        # Both directional memories must be active for the designer to
        # receive the strongest structural coincidence.
        if prefix_active and suffix_active:
            reuse.potential += dg["match_gain"]
        elif prefix_active or suffix_active:
            reuse.potential += dg["context_gain"]
            branch.potential += dg["branch_bias"]
        else:
            branch.potential += dg["branch_bias"]

        # The designer must infer the final competition from activity.
        # There is no exact-edge Boolean here.
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

        # Ask the existing designer for its action without supplying the
        # structural answer.
        return n.designer_signal(None, "")

    def train_activity(self, words, epochs=5):
        print("=== ACTIVITY-DRIVEN TRAINING ===")
        print()

        for epoch in range(1, epochs + 1):
            reuse = 0
            branch = 0

            for word in words:
                for pos in range(len(word)):
                    activity = self.boundary_activity(word, pos)

                    action = self.designer_from_activity(
                        activity,
                        learn=False,
                    )

                    # Structural learning happens independently of the
                    # designer's action.
                    if action == "REUSE":
                        reuse += 1
                    else:
                        branch += 1

                self.learn_structure(word)

            print(
                f"epoch={epoch:3d} "
                f"reuse={reuse:3d} "
                f"branch={branch:3d} "
                f"links={len(self.boundaries.links):3d}"
            )

    def evaluate(self, words):
        print()
        print("=== ACTIVITY-DRIVEN FROZEN TEST ===")

        links_before = len(self.boundaries.links)
        prefix_before = self.prefix.next_id
        suffix_before = self.suffix.next_id

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
                activity = self.boundary_activity(word, pos)

                # This is ONLY for measuring ground truth.
                # It is never passed to the designer.
                expected = self.available(word, pos)

                if expected:
                    expected_reuse += 1
                else:
                    expected_branch += 1

                action = self.designer_from_activity(
                    activity,
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


def run():
    genome = deepcopy(GENOME)

    net = ActivityDrivenV1(genome)

    print("=== ACTIVITY-DRIVEN DUAL VOCABULARY V1 ===")
    print()
    print("Designer does NOT receive BoundaryGraph.has().")
    print("Designer receives only directional vocabulary activity.")
    print()

    net.train_activity(TRAINING, epochs=5)
    net.evaluate(TEST)


if __name__ == "__main__":
    run()
