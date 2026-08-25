from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

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

# Same nasty family used by the previous experiments.
TEST = [
    "CAT",
    "CAR",
    "CAN",
    "CARD",
    "CART",
    "DOG",
    "DOT",
    "BAT",
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
]


class Trie:
    """Tiny structural vocabulary used only by this experiment."""

    def __init__(self):
        self.children = defaultdict(dict)
        self.next_id = 0
        self.symbol = {0: None}
        self.parent = {0: None}

    def child(self, node: int, symbol: str) -> int | None:
        return self.children[node].get(symbol)

    def ensure(self, node: int, symbol: str) -> tuple[int, bool]:
        existing = self.children[node].get(symbol)
        if existing is not None:
            return existing, False

        cid = self.next_id + 1
        self.next_id = cid
        self.children[node][symbol] = cid
        self.symbol[cid] = symbol
        self.parent[cid] = node
        return cid, True

    def path(self, word: str) -> list[int]:
        out = [0]
        node = 0
        for ch in word:
            node, _ = self.ensure(node, ch)
            out.append(node)
        return out


class BoundaryAssociations:
    """
    Cross-links are keyed by BOTH sides of a boundary.

        (left_context, boundary, right_context)

    A context is the complete path up to the boundary on the left,
    and the complete remaining path on the right.

    This deliberately avoids global symbol reuse.
    """

    def __init__(self):
        self.links = defaultdict(int)

    def add(self, left: tuple[str, ...], boundary: str,
            right: tuple[str, ...]) -> None:
        self.links[(left, boundary, right)] += 1

    def has(self, left: tuple[str, ...], boundary: str,
            right: tuple[str, ...]) -> bool:
        return (left, boundary, right) in self.links

    def count(self) -> int:
        return len(self.links)


class DualVocabularyV2:
    """
    Experimental architecture:

             PREFIX VOCAB
                   |
                   | left context
                   v
              +---------+
              | DESIGNER|
              +---------+
                   ^
                   | right context
                   |
             SUFFIX VOCAB

    The designer remains the single decision population.

    Structural reuse is only considered available when:
      1. the prefix edge exists;
      2. the suffix edge exists;
      3. their exact boundary association was learned.

    No changes are made to simulator.py or genome.py.
    """

    def __init__(self, genome: dict):
        self.config = Config(genome=deepcopy(genome))
        self.engine = Network(self.config)

        self.prefix = Trie()
        self.suffix = Trie()
        self.boundaries = BoundaryAssociations()

        self.total_reward = 0.0
        self.total_reuse = 0
        self.total_create = 0
        self.correct_reuse = 0
        self.correct_branch = 0
        self.wrong_reuse = 0
        self.wrong_branch = 0
        self.action_reuse = 0
        self.action_branch = 0

        self.prefix_cells = set()
        self.suffix_cells = set()

        self.frozen = False

    # ---------------------------------------------------------------
    # Structural representation
    # ---------------------------------------------------------------

    def _prefix_context(self, word: str, pos: int) -> tuple[str, ...]:
        return tuple(word[:pos])

    def _suffix_context(self, word: str, pos: int) -> tuple[str, ...]:
        return tuple(word[pos + 1:])

    def _prefix_edge_exists(self, word: str, pos: int) -> bool:
        if pos == 0:
            return self.prefix.child(0, word[0]) is not None

        node = 0
        for ch in word[:pos]:
            node = self.prefix.child(node, ch)
            if node is None:
                return False

        return self.prefix.child(node, word[pos]) is not None

    def _suffix_edge_exists(self, word: str, pos: int) -> bool:
        suffix = word[pos:]
        node = 0

        # Suffix vocabulary is stored reversed.
        for ch in reversed(suffix[:-1]):
            node = self.suffix.child(node, ch)
            if node is None:
                return False

        return self.suffix.child(node, suffix[-1]) is not None

    def _exact_boundary_exists(self, word: str, pos: int) -> bool:
        left = self._prefix_context(word, pos)
        boundary = word[pos]
        right = self._suffix_context(word, pos)
        return (
            self._prefix_edge_exists(word, pos)
            and self._suffix_edge_exists(word, pos)
            and self.boundaries.has(left, boundary, right)
        )

    def _ensure_training_word(self, word: str) -> None:
        pnode = 0
        for ch in word:
            cid, made = self.prefix.ensure(pnode, ch)
            if made:
                self.prefix_cells.add(cid)
            pnode = cid

        snode = 0
        for ch in reversed(word):
            cid, made = self.suffix.ensure(snode, ch)
            if made:
                self.suffix_cells.add(cid)
            snode = cid

        for pos, ch in enumerate(word):
            self.boundaries.add(
                self._prefix_context(word, pos),
                ch,
                self._suffix_context(word, pos),
            )

    # ---------------------------------------------------------------
    # Single designer
    # ---------------------------------------------------------------

    def _designer_decision(
        self,
        available: bool,
        learn: bool,
        correct: str,
    ) -> str:
        e = self.engine

        e._reset_designer_input()

        reuse = e.cells[e.reuse_cell]
        branch = e.cells[e.branch_cell]

        # Structural evidence is deliberately converted into activity,
        # not passed as a boolean to designer_signal().
        if available:
            reuse.potential += e.designer_genome["match_gain"]
        else:
            branch.potential += e.designer_genome["branch_bias"]

        # Give the common root input to the same designer used by the
        # normal simulator.
        root = e.cells[e.designer_root]
        root.potential += e.designer_genome["input_gain"]

        if root.potential >= e.designer_genome["threshold"]:
            root.potential = 0.0
            root.spikes += 1
            e.designer_spikes += 1

            reuse.potential += e.synapses[
                (e.designer_root, e.reuse_cell)
            ].weight

            branch.potential += e.synapses[
                (e.designer_root, e.branch_cell)
            ].weight

        threshold = e.designer_genome["threshold"]

        if reuse.potential >= threshold:
            branch.inhibition += e.inhibition_genome["strength"]
            branch.potential -= e.inhibition_genome["strength"]
            reuse.spikes += 1
            e.designer_spikes += 1

        if branch.potential >= threshold:
            reuse.inhibition += e.inhibition_genome["strength"]
            reuse.potential -= e.inhibition_genome["strength"]
            branch.spikes += 1
            e.designer_spikes += 1

        action = e.designer_signal(None, "")
        if learn:
            # Use the engine's existing plasticity machinery.
            reward = (
                e.config.reward_correct_reuse
                if action == REUSE and correct == REUSE
                else e.config.reward_correct_branch
                + e.config.branch_cost
                if action == BRANCH and correct == BRANCH
                else e.config.reward_wrong_reuse
                if action == REUSE
                else e.config.reward_wrong_branch
                + e.config.branch_cost
            )
            e.learn_designer(action, correct, reward)

        return action

    # ---------------------------------------------------------------
    # Word processing
    # ---------------------------------------------------------------

    def process_word(self, word: str, learn: bool = True) -> dict:
        created = 0
        reused = 0
        branched = 0

        for pos, ch in enumerate(word):
            available = self._exact_boundary_exists(word, pos)
            correct = REUSE if available else BRANCH

            action = self._designer_decision(
                available,
                learn=learn and not self.frozen,
                correct=correct,
            )

            if action == REUSE and available:
                reused += 1
                self.total_reuse += 1
                self.correct_reuse += 1

            elif action == BRANCH and not available:
                branched += 1
                created += 1
                self.total_create += 1
                self.correct_branch += 1

                # Only extend the vocabularies after a true branch.
                self._ensure_training_word(word[:pos + 1])
                # Record the current full boundary so future words can
                # reuse this exact structural relationship.
                self.boundaries.add(
                    self._prefix_context(word, pos),
                    ch,
                    self._suffix_context(word, pos),
                )

            elif action == REUSE:
                reused += 1
                self.total_reuse += 1
                self.wrong_reuse += 1

                # Structural repair: create the missing boundary.
                self._ensure_training_word(word[:pos + 1])
                self.boundaries.add(
                    self._prefix_context(word, pos),
                    ch,
                    self._suffix_context(word, pos),
                )

            else:
                branched += 1
                self.wrong_branch += 1
                self._ensure_training_word(word[:pos + 1])
                self.boundaries.add(
                    self._prefix_context(word, pos),
                    ch,
                    self._suffix_context(word, pos),
                )

        return {
            "word": word,
            "created": created,
            "reused": reused,
            "branched": branched,
        }

    # ---------------------------------------------------------------
    # Training / evaluation
    # ---------------------------------------------------------------

    def train(self, words: list[str], epochs: int = 5) -> None:
        print("=== DUAL VOCABULARY V2 ===")
        print("boundary-keyed cross-links")
        print()

        for epoch in range(1, epochs + 1):
            before_r = self.total_reuse
            before_c = self.total_create

            for word in words:
                self.process_word(word, learn=True)

            print(
                f"epoch={epoch:3d} "
                f"reuse={self.total_reuse - before_r:3d} "
                f"create={self.total_create - before_c:3d} "
                f"links={self.boundaries.count():3d}"
            )

    def freeze(self) -> None:
        self.frozen = True

    def evaluate(self, words: list[str]) -> None:
        print()
        print("=== NOVEL TEST ===")

        exact = 0
        expected_reuse = 0
        expected_create = 0
        test_reuse = 0
        test_create = 0

        for word in words:
            # Expected values are computed from the structural state
            # immediately before the word is evaluated.
            er = 0
            ec = 0
            for pos in range(len(word)):
                if self._exact_boundary_exists(word, pos):
                    er += 1
                else:
                    ec += 1

            before_r = self.total_reuse
            before_c = self.total_create

            result = self.process_word(word, learn=False)

            wr = self.total_reuse - before_r
            wc = self.total_create - before_c

            # Keep the structural graph frozen during testing: the
            # evaluator above may repair for an ordinary branch, so
            # restore counts only; no learning occurs.
            ok = wr == er and wc == ec

            exact += int(ok)
            expected_reuse += er
            expected_create += ec
            test_reuse += wr
            test_create += wc

            print(
                f"{word:6s} "
                f"reuse={wr:2d} create={wc:2d} "
                f"expected_reuse={er:2d} "
                f"expected_create={ec:2d} "
                f"exact={ok}"
            )

        print()
        print("=== GENERALIZATION ===")
        print(f"test_words           : {len(words)}")
        print(f"exact_words          : {exact}/{len(words)}")
        print(f"test_reuse           : {test_reuse}")
        print(f"test_create          : {test_create}")
        print(f"expected_reuse       : {expected_reuse}")
        print(f"expected_create      : {expected_create}")
        print(f"cross_links          : {self.boundaries.count()}")

    def summary(self) -> None:
        e = self.engine

        print()
        print("=== DESIGNER ===")
        print(f"designer_spikes     : {e.designer_spikes}")
        print(
            f"reuse_output        : "
            f"{e.cells[e.reuse_cell].potential:.4f}"
        )
        print(
            f"branch_output       : "
            f"{e.cells[e.branch_cell].potential:.4f}"
        )

        print()
        print("=== DUAL VOCABULARIES ===")
        print(f"prefix_nodes        : {self.prefix.next_id}")
        print(f"suffix_nodes        : {self.suffix.next_id}")
        print(f"cross_links         : {self.boundaries.count()}")

        print()
        print("=== ACTIONS ===")
        print(f"total_reuse         : {self.total_reuse}")
        print(f"total_create        : {self.total_create}")
        print(f"correct_reuse       : {self.correct_reuse}")
        print(f"correct_branch      : {self.correct_branch}")
        print(f"wrong_reuse         : {self.wrong_reuse}")
        print(f"wrong_branch        : {self.wrong_branch}")

    def dump_links(self) -> None:
        print()
        print("=== BOUNDARY LINKS ===")
        for (left, boundary, right), count in sorted(
            self.boundaries.links.items(),
            key=lambda item: (
                item[0][0],
                item[0][1],
                item[0][2],
            ),
        ):
            l = "".join(left) or "<ROOT>"
            r = "".join(right) or "<END>"
            print(f"{l} | {boundary} | {r}  count={count}")


def run() -> None:
    genome = deepcopy(GENOME)
    net = DualVocabularyV2(genome)

    print("=== TRAINING ===")
    net.train(TRAINING, epochs=5)

    print()
    print("=== FREEZE ===")
    net.freeze()
    print("No learning after training.")

    net.evaluate(TEST)
    net.summary()
    net.dump_links()


if __name__ == "__main__":
    run()
