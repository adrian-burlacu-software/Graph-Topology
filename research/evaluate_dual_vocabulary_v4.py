from __future__ import annotations

from copy import deepcopy
from collections import defaultdict

from simulator import Network, Config, REUSE, BRANCH
from genome import GENOME


TRAINING = [
    "CAT", "CAR", "CAN", "CARD", "CART", "DOG", "DOT", "BAT",
]

TEST = [
    "CAT", "CAR", "CAN", "CARD", "CART",
    "CAD", "COD", "COT", "BAD", "BAR", "BARD", "BAN",
    "DART", "DAT", "BOT", "BOAT", "CARTD",
    "COARD", "BAND", "BOARD",
]


class Trie:
    """Directional structural memory. IDs, not strings, define nodes."""

    def __init__(self, reverse=False):
        self.reverse = reverse
        self.next_id = 0
        self.children = {0: {}}
        self.parent = {0: None}
        self.symbol = {0: None}

    def _seq(self, text):
        return reversed(text) if self.reverse else text

    def ensure(self, node, symbol):
        child = self.children[node].get(symbol)
        if child is not None:
            return child, False
        self.next_id += 1
        child = self.next_id
        self.children[node][symbol] = child
        self.children[child] = {}
        self.parent[child] = node
        self.symbol[child] = symbol
        return child, True

    def lookup(self, text):
        node = 0
        for ch in self._seq(text):
            node = self.children[node].get(ch)
            if node is None:
                return None
        return node

    def ensure_path(self, text):
        node = 0
        for ch in self._seq(text):
            node, _ = self.ensure(node, ch)
        return node


class BoundaryGraph:
    """
    Exact reusable composition:

        prefix_node -- boundary_symbol --> suffix_node

    No complete suffix strings are stored here.
    """

    def __init__(self):
        self.links = defaultdict(int)

    def add(self, prefix_node, symbol, suffix_node):
        self.links[(prefix_node, symbol, suffix_node)] += 1

    def has(self, prefix_node, symbol, suffix_node):
        return (prefix_node, symbol, suffix_node) in self.links


class DualVocabularyV4:
    """
    One designer, two vocabularies, exact structural availability.

    Important distinction:

        STRUCTURAL AVAILABILITY
            = exact boundary link exists

        DESIGNER ACTION
            = what the learned designer chooses

    The evaluator reports both independently.

    During frozen testing absolutely nothing is added to either vocabulary
    or to the boundary graph.
    """

    def __init__(self, genome):
        self.genome = deepcopy(genome)
        self.net = Network(Config(genome=deepcopy(genome)))

        self.prefix = Trie(reverse=False)
        self.suffix = Trie(reverse=True)
        self.boundaries = BoundaryGraph()

        self.action_reuse = 0
        self.action_branch = 0
        self.correct_reuse = 0
        self.correct_branch = 0
        self.wrong_reuse = 0
        self.wrong_branch = 0
        self.total_reward = 0.0

    # ------------------------------------------------------------
    # Exact structural state
    # ------------------------------------------------------------

    def ids_for_position(self, word, pos, create=False):
        left = word[:pos]
        right = word[pos + 1:]

        if create:
            p = self.prefix.ensure_path(left)
            s = self.suffix.ensure_path(right)
        else:
            p = self.prefix.lookup(left)
            s = self.suffix.lookup(right)

        return p, word[pos], s

    def available(self, word, pos):
        p, symbol, s = self.ids_for_position(word, pos, create=False)
        if p is None or s is None:
            return False
        return self.boundaries.has(p, symbol, s)

    def learn_structure(self, word):
        for pos, symbol in enumerate(word):
            p = self.prefix.ensure_path(word[:pos])
            s = self.suffix.ensure_path(word[pos + 1:])
            self.boundaries.add(p, symbol, s)

    # ------------------------------------------------------------
    # Designer
    # ------------------------------------------------------------

    def designer_action(self, available, learn):
        n = self.net
        dg = n.designer_genome

        n._reset_designer_input()

        root = n.cells[n.designer_root]
        reuse = n.cells[n.reuse_cell]
        branch = n.cells[n.branch_cell]

        # Structural availability is an input signal, NOT the answer.
        # Exact links strongly excite reuse; their absence excites branch.
        root.potential += dg["input_gain"]

        if available:
            reuse.potential += dg["match_gain"]
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

        action = n.designer_signal(None, "")

        correct = REUSE if available else BRANCH

        if learn:
            if action == REUSE:
                reward = (
                    n.config.reward_correct_reuse
                    if correct == REUSE
                    else n.config.reward_wrong_reuse
                )
                self.action_reuse += 1
                if correct == REUSE:
                    self.correct_reuse += 1
                else:
                    self.wrong_reuse += 1
            else:
                reward = (
                    n.config.reward_correct_branch + n.config.branch_cost
                    if correct == BRANCH
                    else n.config.reward_wrong_branch + n.config.branch_cost
                )
                self.action_branch += 1
                if correct == BRANCH:
                    self.correct_branch += 1
                else:
                    self.wrong_branch += 1

            self.total_reward += reward
            n.learn_designer(action, correct, reward)

        return action

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    def train(self, words, epochs=5):
        print("=== DUAL VOCABULARY V4 ===")
        print("exact boundary availability + single designer")
        print()

        for epoch in range(1, epochs + 1):
            before_r = self.action_reuse
            before_b = self.action_branch
            before_reward = self.total_reward

            for word in words:
                # The word is presented one character at a time.
                # The structural graph is updated only after the word.
                for pos in range(len(word)):
                    self.designer_action(
                        self.available(word, pos),
                        learn=True,
                    )
                self.learn_structure(word)

            print(
                f"epoch={epoch:3d} "
                f"reuse={self.action_reuse-before_r:3d} "
                f"branch={self.action_branch-before_b:3d} "
                f"reward={self.total_reward-before_reward:8.2f} "
                f"links={len(self.boundaries.links):3d}"
            )

    # ------------------------------------------------------------
    # Frozen test
    # ------------------------------------------------------------

    def evaluate_frozen(self, words):
        print()
        print("=== FROZEN NOVEL TEST ===")

        links_before = len(self.boundaries.links)
        prefix_before = self.prefix.next_id
        suffix_before = self.suffix.next_id

        exact_words = 0
        exact_positions = 0
        total_positions = 0
        available_reuse = 0
        unavailable_branch = 0
        designer_reuse_when_available = 0
        designer_branch_when_unavailable = 0
        designer_wrong = 0

        for word in words:
            expected_reuse = 0
            expected_branch = 0
            actual_reuse = 0
            actual_branch = 0

            for pos in range(len(word)):
                available = self.available(word, pos)
                action = self.designer_action(available, learn=False)

                total_positions += 1

                if available:
                    expected_reuse += 1
                    available_reuse += 1
                    if action == REUSE:
                        actual_reuse += 1
                        designer_reuse_when_available += 1
                    else:
                        actual_branch += 1
                        designer_wrong += 1
                else:
                    expected_branch += 1
                    unavailable_branch += 1
                    if action == BRANCH:
                        actual_branch += 1
                        designer_branch_when_unavailable += 1
                    else:
                        actual_reuse += 1
                        designer_wrong += 1

            exact = (
                actual_reuse == expected_reuse
                and actual_branch == expected_branch
            )

            if exact:
                exact_words += 1

            exact_positions += (
                expected_reuse == actual_reuse
                and expected_branch == actual_branch
            )

            print(
                f"{word:6s} "
                f"available_reuse={expected_reuse:2d} "
                f"available_branch={expected_branch:2d} "
                f"designer_reuse={actual_reuse:2d} "
                f"designer_branch={actual_branch:2d} "
                f"exact={exact}"
            )

        print()
        print("=== GENERALIZATION ===")
        print(f"test_words                    : {len(words)}")
        print(f"exact_words                   : {exact_words}/{len(words)}")
        print(f"test_positions                : {total_positions}")
        print(f"available_reuse_positions     : {available_reuse}")
        print(f"unavailable_branch_positions  : {unavailable_branch}")
        print(f"designer_reuse_when_available : {designer_reuse_when_available}")
        print(f"designer_branch_when_unavail  : {designer_branch_when_unavailable}")
        print(f"designer_wrong_positions      : {designer_wrong}")

        print()
        print("=== FROZEN INVARIANTS ===")
        print(f"boundary_links_before         : {links_before}")
        print(f"boundary_links_after          : {len(self.boundaries.links)}")
        print(f"prefix_nodes_before           : {prefix_before}")
        print(f"prefix_nodes_after            : {self.prefix.next_id}")
        print(f"suffix_nodes_before           : {suffix_before}")
        print(f"suffix_nodes_after            : {self.suffix.next_id}")

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    def summary(self):
        n = self.net

        print()
        print("=== LEARNED NETWORK ===")
        print(f"designer_spikes     : {n.designer_spikes}")
        print(f"total_reward        : {self.total_reward:.2f}")
        print(f"action_reuse        : {self.action_reuse}")
        print(f"action_branch       : {self.action_branch}")
        print(f"correct_reuse       : {self.correct_reuse}")
        print(f"correct_branch      : {self.correct_branch}")
        print(f"wrong_reuse         : {self.wrong_reuse}")
        print(f"wrong_branch        : {self.wrong_branch}")

        print()
        print("=== DESIGNER OUTPUT ===")
        print(
            f"reuse_output        : "
            f"{n.cells[n.reuse_cell].potential:.4f}"
        )
        print(
            f"branch_output       : "
            f"{n.cells[n.branch_cell].potential:.4f}"
        )

        print()
        print("=== STRUCTURE ===")
        print(f"prefix_nodes        : {self.prefix.next_id + 1}")
        print(f"suffix_nodes        : {self.suffix.next_id + 1}")
        print(f"boundary_links      : {len(self.boundaries.links)}")


def run():
    net = DualVocabularyV4(GENOME)

    print("=== TRAINING ===")
    net.train(TRAINING, epochs=5)

    print()
    print("=== FREEZE ===")
    print("No designer learning or structural mutation after training.")

    net.evaluate_frozen(TEST)
    net.summary()


if __name__ == "__main__":
    run()
