from __future__ import annotations

from copy import deepcopy
from collections import defaultdict

from simulator import Network, Config, REUSE, BRANCH
from genome import GENOME


TRAINING = [
    "CAT",
    "CAR",
    "CAN",
    "CARD",
    "CART",
    "DOG",
    "DOT",
    "BAT",
]

TEST = [
    "CAT",
    "CAR",
    "CAN",
    "CARD",
    "CART",
    "CAD",
    "COD",
    "COT",
    "BAD",
    "BAR",
    "BARD",
    "BAN",
    "DART",
    "DAT",
    "BOT",
    "BOAT",
    "CARTD",
    "COARD",
    "BAND",
    "BOARD",
]


class DirectionalTrie:
    """Structural memory only. No decision logic lives here."""

    def __init__(self, reverse: bool = False):
        self.reverse = reverse
        self.next_id = 0
        self.children: dict[int, dict[str, int]] = {0: {}}
        self.parent: dict[int, int | None] = {0: None}
        self.symbol: dict[int, str | None] = {0: None}

    def ensure(self, node: int, symbol: str) -> tuple[int, bool]:
        existing = self.children[node].get(symbol)
        if existing is not None:
            return existing, False

        self.next_id += 1
        cid = self.next_id
        self.children[node][symbol] = cid
        self.children[cid] = {}
        self.parent[cid] = node
        self.symbol[cid] = symbol
        return cid, True

    def walk(self, symbols: str) -> int:
        node = 0
        seq = reversed(symbols) if self.reverse else symbols
        for ch in seq:
            node, _ = self.ensure(node, ch)
        return node

    def lookup(self, symbols: str) -> int | None:
        node = 0
        seq = reversed(symbols) if self.reverse else symbols
        for ch in seq:
            node = self.children[node].get(ch)
            if node is None:
                return None
        return node

    def has_edge(self, symbols: str, symbol: str) -> bool:
        node = self.lookup(symbols)
        if node is None:
            return False
        return symbol in self.children[node]


class BoundaryGraph:
    """
    The actual reusable unit is:

        (prefix_node, boundary_symbol, suffix_node)

    Prefix and suffix nodes are IDs in two independent directional
    vocabularies. The full remaining word is NEVER stored as the key.
    """

    def __init__(self):
        self.links: dict[tuple[int, str, int], int] = defaultdict(int)

    def add(self, prefix_id: int, boundary: str, suffix_id: int) -> None:
        self.links[(prefix_id, boundary, suffix_id)] += 1

    def has(self, prefix_id: int, boundary: str, suffix_id: int) -> bool:
        return (prefix_id, boundary, suffix_id) in self.links

    def count(self) -> int:
        return len(self.links)


class DualVocabularyV3:
    """
    One real simulator designer + two directional structural memories.

    PREFIX:
        remembers what has already been traversed.

    SUFFIX:
        remembers what can follow, stored in reverse.

    BOUNDARY:
        connects one concrete prefix node to one concrete suffix node
        through exactly one boundary symbol.

    The novel-test phase is strictly read-only.
    """

    def __init__(self, genome: dict):
        self.genome = deepcopy(genome)

        # Use the repo's actual designer implementation.
        self.network = Network(Config(genome=deepcopy(genome)))

        self.prefix = DirectionalTrie(reverse=False)
        self.suffix = DirectionalTrie(reverse=True)
        self.boundary = BoundaryGraph()

        self.total_reward = 0.0
        self.total_reuse = 0
        self.total_create = 0
        self.correct_reuse = 0
        self.correct_branch = 0
        self.wrong_reuse = 0
        self.wrong_branch = 0
        self.action_reuse = 0
        self.action_branch = 0

        self.frozen = False

    # ------------------------------------------------------------
    # Structural representation
    # ------------------------------------------------------------

    def prefix_id(self, word: str, pos: int, create: bool) -> int | None:
        context = word[:pos]
        if create:
            return self.prefix.walk(context)
        return self.prefix.lookup(context)

    def suffix_id(self, word: str, pos: int, create: bool) -> int | None:
        context = word[pos + 1:]
        if create:
            return self.suffix.walk(context)
        return self.suffix.lookup(context)

    def boundary_key(
        self,
        word: str,
        pos: int,
        create: bool,
    ) -> tuple[int | None, str, int | None]:
        return (
            self.prefix_id(word, pos, create),
            word[pos],
            self.suffix_id(word, pos, create),
        )

    def exact_available(self, word: str, pos: int) -> bool:
        p, ch, s = self.boundary_key(word, pos, create=False)
        if p is None or s is None:
            return False
        return self.boundary.has(p, ch, s)

    def partial_activity(self, word: str, pos: int) -> tuple[float, float, float]:
        """
        Returns:
            exact_activity
            left_activity
            right_activity

        These are structural activities, not a reuse boolean.
        """

        p, ch, s = self.boundary_key(word, pos, create=False)

        left = 0.0
        right = 0.0
        exact = 0.0

        if p is not None:
            left = 1.0 if self.prefix.has_edge(word[:pos], ch) else 0.0

        if s is not None:
            # Suffix trie is reversed, so the boundary's outgoing side
            # is represented by the first symbol of the reversed suffix.
            right = 1.0 if self.suffix.lookup(word[pos + 1:]) is not None else 0.0

        if p is not None and s is not None and self.boundary.has(p, ch, s):
            exact = 1.0

        return exact, left, right

    def learn_structure(self, word: str) -> None:
        """
        Add structural nodes and exact boundary edges.

        This is the only mutation path used during training.
        """

        for pos, ch in enumerate(word):
            p = self.prefix.walk(word[:pos])
            s = self.suffix.walk(word[pos + 1:])
            self.boundary.add(p, ch, s)

    # ------------------------------------------------------------
    # Single designer
    # ------------------------------------------------------------

    def decide(self, word: str, pos: int, learn: bool) -> tuple[str, bool]:
        exact, left, right = self.partial_activity(word, pos)

        n = self.network
        n._reset_designer_input()

        # Feed the SAME designer population used by the simulator.
        root = n.cells[n.designer_root]
        reuse = n.cells[n.reuse_cell]
        branch = n.cells[n.branch_cell]

        dg = n.designer_genome

        root.potential += dg["input_gain"]

        # Exact left/right agreement produces the strongest activity.
        # Partial agreement is weaker and deliberately ambiguous.
        structural_match = exact * 2.0 + (left + right) * 0.25

        reuse.potential += structural_match * dg["match_gain"]
        branch.potential += (
            dg["branch_bias"]
            + (2.0 - structural_match) * dg["context_gain"]
        )

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

        correct = REUSE if exact else BRANCH

        if learn:
            if action == REUSE:
                reward = (
                    n.config.reward_correct_reuse
                    if correct == REUSE
                    else n.config.reward_wrong_reuse
                )
            else:
                reward = (
                    n.config.reward_correct_branch + n.config.branch_cost
                    if correct == BRANCH
                    else n.config.reward_wrong_branch + n.config.branch_cost
                )

            n.learn_designer(action, correct, reward)

            self.total_reward += reward

            if action == REUSE:
                self.action_reuse += 1
                if correct == REUSE:
                    self.correct_reuse += 1
                else:
                    self.wrong_reuse += 1
            else:
                self.action_branch += 1
                if correct == BRANCH:
                    self.correct_branch += 1
                else:
                    self.wrong_branch += 1

        return action, bool(exact)

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    def train(self, words: list[str], epochs: int = 5) -> None:
        print("=== DUAL VOCABULARY V3 ===")
        print("boundary-keyed structural composition")
        print()

        for epoch in range(1, epochs + 1):
            rb = self.total_reward
            ru = self.action_reuse
            br = self.action_branch

            for word in words:
                for pos in range(len(word)):
                    self.decide(word, pos, learn=True)

                # Structure is learned after the designer has observed
                # the word. This prevents same-word later positions from
                # seeing future boundary entries.
                self.learn_structure(word)

            print(
                f"epoch={epoch:3d} "
                f"reuse={self.action_reuse - ru:3d} "
                f"branch={self.action_branch - br:3d} "
                f"reward={self.total_reward - rb:8.2f} "
                f"links={self.boundary.count():3d}"
            )

    def freeze(self) -> None:
        self.frozen = True

    # ------------------------------------------------------------
    # Frozen evaluation
    # ------------------------------------------------------------

    def evaluate(self, words: list[str]) -> None:
        print()
        print("=== FROZEN NOVEL TEST ===")

        exact_words = 0
        test_reuse = 0
        test_create = 0
        expected_reuse = 0
        expected_create = 0

        links_before = self.boundary.count()
        prefix_before = self.prefix.next_id
        suffix_before = self.suffix.next_id

        for word in words:
            actual_reuse = 0
            actual_create = 0
            expected_r = 0
            expected_c = 0

            for pos in range(len(word)):
                available = self.exact_available(word, pos)
                if available:
                    expected_r += 1
                else:
                    expected_c += 1

                before_r = self.network.action_reuse
                before_b = self.network.action_branch

                action, _ = self.decide(word, pos, learn=False)

                # No structural mutation is allowed here.
                if action == REUSE:
                    actual_reuse += 1
                else:
                    actual_create += 1

                _ = before_r, before_b

            # Exact evaluation is action vs structural expectation.
            ok = (
                actual_reuse == expected_r
                and actual_create == expected_c
            )

            if ok:
                exact_words += 1

            test_reuse += actual_reuse
            test_create += actual_create
            expected_reuse += expected_r
            expected_create += expected_c

            print(
                f"{word:6s} "
                f"reuse={actual_reuse:2d} "
                f"create={actual_create:2d} "
                f"expected_reuse={expected_r:2d} "
                f"expected_create={expected_c:2d} "
                f"exact={ok}"
            )

        print()
        print("=== GENERALIZATION ===")
        print(f"training_words       : {len(TRAINING)}")
        print(f"test_words           : {len(words)}")
        print(f"exact_words          : {exact_words}/{len(words)}")
        print(f"test_reuse           : {test_reuse}")
        print(f"test_create          : {test_create}")
        print(f"expected_reuse       : {expected_reuse}")
        print(f"expected_create      : {expected_create}")

        print()
        print("=== FROZEN INVARIANTS ===")
        print(
            f"boundary_links_before: {links_before}"
        )
        print(
            f"boundary_links_after : {self.boundary.count()}"
        )
        print(
            f"prefix_nodes_before  : {prefix_before}"
        )
        print(
            f"prefix_nodes_after   : {self.prefix.next_id}"
        )
        print(
            f"suffix_nodes_before  : {suffix_before}"
        )
        print(
            f"suffix_nodes_after   : {self.suffix.next_id}"
        )

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    def print_summary(self) -> None:
        n = self.network

        print()
        print("=== DESIGNER ===")
        print(f"designer_spikes     : {n.designer_spikes}")
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
        print(f"boundary_links      : {self.boundary.count()}")

        print()
        print("=== LEARNING ===")
        print(f"total_reward        : {self.total_reward:.2f}")
        print(f"action_reuse        : {self.action_reuse}")
        print(f"action_branch       : {self.action_branch}")
        print(f"correct_reuse       : {self.correct_reuse}")
        print(f"correct_branch      : {self.correct_branch}")
        print(f"wrong_reuse         : {self.wrong_reuse}")
        print(f"wrong_branch        : {self.wrong_branch}")

    def dump_boundary_links(self) -> None:
        print()
        print("=== BOUNDARY LINKS ===")

        for (p, ch, s), count in sorted(
            self.boundary.links.items(),
            key=lambda item: item[0],
        ):
            print(
                f"prefix={p:3d} "
                f"boundary={ch} "
                f"suffix={s:3d} "
                f"count={count}"
            )


def run() -> None:
    net = DualVocabularyV3(GENOME)

    print("=== TRAINING ===")
    net.train(TRAINING, epochs=5)

    print()
    print("=== FREEZE ===")
    net.freeze()
    print("No structural or designer learning after training.")

    net.evaluate(TEST)
    net.print_summary()
    net.dump_boundary_links()


if __name__ == "__main__":
    run()
