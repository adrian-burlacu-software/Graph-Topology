from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

from simulator import (
    Network,
    Config,
    CREATE,
    REUSE,
    BRANCH,
    EXCITE,
)
from genome import clone_genome


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

# Same family of adversarial tests used in the recent experiments.
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


class DualVocabularyNetwork(Network):
    """
    Experimental architecture:

        one designer
          /       \
       PREFIX   SUFFIX
        trie       trie

    PREFIX stores word structure from left -> right.
    SUFFIX stores word structure from right -> left.

    The two vocabularies do not directly choose each other's edges.
    At each position they provide structural evidence to the same
    designer.  The designer decides whether the current transition
    is reusable.  A successful transition is recorded in both
    directional vocabularies.

    This is intentionally an experimental wrapper. The production
    Network class is untouched.
    """

    def __init__(self, config: Config | None = None):
        super().__init__(config)

        # Remove the normal vocabulary cells if any are ever created by
        # inherited machinery; this experiment creates its own graph.
        self.prefix_roots: dict[str, int] = {}
        self.suffix_roots: dict[str, int] = {}

        # Pair links learned between corresponding forward/reverse states.
        # key = (prefix_node, suffix_node)
        self.dual_links: dict[tuple[int, int], float] = {}

        self.dual_reuse = 0
        self.dual_create = 0
        self.dual_correct = 0
        self.dual_wrong = 0

    # ---------- directional vocabulary primitives ----------

    def _children(self, parent_id: int | None) -> list[int]:
        if parent_id is None:
            return []
        return [
            cid
            for cid in self.cells[parent_id].outgoing
            if self.cells[cid].kind == "vocabulary"
        ]

    def _find_local(self, parent_id: int | None, symbol: str,
                    roots: dict[str, int]) -> int | None:
        if parent_id is None:
            return roots.get(symbol)

        for cid in self._children(parent_id):
            if self.cells[cid].symbol == symbol:
                return cid
        return None

    def _create_local(self, symbol: str, parent_id: int | None,
                      order: int, roots: dict[str, int]) -> int:
        if parent_id is None:
            cid = self.create_cell("vocabulary", symbol, None, order)
            roots[symbol] = cid
        else:
            cid = self.create_cell("vocabulary", symbol, parent_id, order)
            self.connect(parent_id, cid, EXCITE, self.config.excite_weight)

        # Every directional vocabulary cell feeds the SAME designer.
        self.connect(
            cid,
            self.designer_root,
            EXCITE,
            self.config.feedback_weight,
        )
        return cid

    def _path_exists(self, parent_id: int | None, symbol: str,
                     roots: dict[str, int]) -> bool:
        return self._find_local(parent_id, symbol, roots) is not None

    # ---------- dual-context evidence ----------

    def _dual_match_activity(
        self,
        prefix_parent: int | None,
        prefix_symbol: str,
        suffix_parent: int | None,
        suffix_symbol: str,
    ) -> tuple[float, float]:
        """
        Returns:
            match_strength
            context_strength

        A transition is strongly supported only when the directional
        vocabularies agree. One-sided evidence is weaker.
        """
        p = self._find_local(
            prefix_parent, prefix_symbol, self.prefix_roots
        )
        s = self._find_local(
            suffix_parent, suffix_symbol, self.suffix_roots
        )

        p_match = 1.0 if p is not None else 0.0
        s_match = 1.0 if s is not None else 0.0

        # Agreement is the important signal.
        agreement = p_match * s_match
        one_sided = p_match + s_match

        match = (
            agreement * 2.0
            + max(0.0, one_sided - 2.0) * 0.5
        )

        context = (
            (1.0 if prefix_parent is not None else 0.0)
            + (1.0 if suffix_parent is not None else 0.0)
        ) * 0.5

        return match, context

    def _designer_decision(
        self,
        prefix_parent: int | None,
        prefix_symbol: str,
        suffix_parent: int | None,
        suffix_symbol: str,
        learn: bool,
        correct: str,
    ) -> str:
        self._reset_designer_input()

        match, context = self._dual_match_activity(
            prefix_parent,
            prefix_symbol,
            suffix_parent,
            suffix_symbol,
        )

        # Use the real designer dynamics from the current simulator.
        # We inject only the evidence supplied by the two vocabularies.
        root = self.cells[self.designer_root]
        reuse = self.cells[self.reuse_cell]
        branch = self.cells[self.branch_cell]

        root.potential += (
            self.designer_genome["input_gain"]
        )

        reuse.potential += (
            match * self.designer_genome["match_gain"]
        )

        branch.potential += (
            self.designer_genome["branch_bias"]
            + context * self.designer_genome["context_gain"]
        )

        if root.potential >= self.designer_genome["threshold"]:
            root.potential = 0.0
            root.spikes += 1
            self.designer_spikes += 1

            reuse.potential += self.synapses[
                (self.designer_root, self.reuse_cell)
            ].weight

            branch.potential += self.synapses[
                (self.designer_root, self.branch_cell)
            ].weight

        threshold = self.designer_genome["threshold"]

        if reuse.potential >= threshold:
            branch.inhibition += self.inhibition_genome["strength"]
            branch.potential -= self.inhibition_genome["strength"]
            reuse.spikes += 1
            self.designer_spikes += 1

        if branch.potential >= threshold:
            reuse.inhibition += self.inhibition_genome["strength"]
            reuse.potential -= self.inhibition_genome["strength"]
            branch.spikes += 1
            self.designer_spikes += 1

        reuse.potential *= self.designer_genome["leak"]
        branch.potential *= self.designer_genome["leak"]

        action = self.designer_signal(None, "")
        if learn:
            # Reuse the existing plasticity mechanism.
            reward = (
                self.config.reward_correct_reuse
                if action == correct == REUSE
                else self.config.reward_correct_branch + self.config.branch_cost
                if action == correct == BRANCH
                else self.config.reward_wrong_reuse
                if action == REUSE
                else self.config.reward_wrong_branch + self.config.branch_cost
            )
            self.learn_designer(action, correct, reward)

        return action

    # ---------- training / testing ----------

    def process_word(self, word: str, learn: bool = True) -> dict:
        """
        Train both directional vocabularies on the same word.

        Forward side sees:
            C -> A -> T

        Reverse side sees:
            T -> A -> C

        A transition is considered structurally reusable only when the
        corresponding directional context supports it.
        """
        if not word:
            return {
                "word": word,
                "created": 0,
                "reused": 0,
                "branched": 0,
            }

        created = 0
        reused = 0
        branched = 0

        prefix_parent = None
        suffix_parent = None

        n = len(word)

        for i, symbol in enumerate(word):
            reverse_symbol = word[n - 1 - i]

            # At the same position, each side describes the local word
            # context from its own direction.
            existing_prefix = self._find_local(
                prefix_parent,
                symbol,
                self.prefix_roots,
            )
            existing_suffix = self._find_local(
                suffix_parent,
                reverse_symbol,
                self.suffix_roots,
            )

            # Reuse is a joint structural claim.
            exists = (
                existing_prefix is not None
                and existing_suffix is not None
            )
            correct = REUSE if exists else BRANCH

            action = self._designer_decision(
                prefix_parent,
                symbol,
                suffix_parent,
                reverse_symbol,
                learn,
                correct,
            )

            if action == REUSE and exists:
                prefix_id = existing_prefix
                suffix_id = existing_suffix
                reused += 1
                self.total_reuse += 1
                self.dual_reuse += 1
            else:
                # Never create only one half: the representation stays
                # directionally symmetric.
                if existing_prefix is None:
                    prefix_id = self._create_local(
                        symbol,
                        prefix_parent,
                        i,
                        self.prefix_roots,
                    )
                    created += 1
                    self.total_create += 1
                    self.dual_create += 1
                else:
                    prefix_id = existing_prefix

                if existing_suffix is None:
                    suffix_id = self._create_local(
                        reverse_symbol,
                        suffix_parent,
                        n - 1 - i,
                        self.suffix_roots,
                    )
                else:
                    suffix_id = existing_suffix

                if action == BRANCH:
                    branched += 1

            # Record the cross-vocabulary correspondence.
            self.dual_links[(prefix_id, suffix_id)] = (
                self.dual_links.get((prefix_id, suffix_id), 0.0) + 1.0
            )

            prefix_parent = prefix_id
            suffix_parent = suffix_id

        return {
            "word": word,
            "created": created,
            "reused": reused,
            "branched": branched,
        }

    def print_dual_summary(self):
        print()
        print("=== DUAL VOCABULARY ===")
        print(f"prefix_nodes         : {len(self.prefix_roots)} roots")
        print(f"suffix_nodes         : {len(self.suffix_roots)} roots")
        print(f"cross_links          : {len(self.dual_links)}")
        print(f"dual_reuse           : {self.dual_reuse}")
        print(f"dual_create          : {self.dual_create}")

    def print_directional_graph(self, roots: dict[str, int], title: str):
        print()
        print(title)

        visited_global: set[int] = set()

        def walk(root_id: int, prefix: str, path: set[int]):
            if root_id in path:
                print(f"{prefix}[{root_id}] CYCLE")
                return

            if root_id in visited_global:
                print(f"{prefix}[{root_id}] SHARED")
                return

            visited_global.add(root_id)
            cell = self.cells[root_id]
            print(f"{prefix}{cell.symbol} [{cell.id}]")

            next_path = path | {root_id}
            children = self._children(root_id)

            for child_id in children:
                walk(child_id, prefix + "  ", next_path)

        for symbol, root_id in roots.items():
            walk(root_id, "", set())


def expected_counts(training: list[str], word: str) -> tuple[int, int]:
    """
    Expected counts under ordinary prefix-graph semantics:
    each character transition is reusable iff that exact prefix edge
    was already learned.
    """
    prefixes = set()
    for w in training:
        for i in range(1, len(w) + 1):
            prefixes.add(w[:i])

    reused = 0
    created = 0

    for i in range(1, len(word) + 1):
        prefix = word[:i]
        if prefix in prefixes:
            reused += 1
        else:
            created += 1

    return reused, created


def run():
    genome = clone_genome()

    # Preserve the current simulator's defaults; only swap in the
    # experimental dual-vocabulary wrapper.
    net = DualVocabularyNetwork(Config(genome=genome))

    print("=== TRAINING ===")
    print()
    print("=== DUAL-VOCABULARY PLASTICITY ===")
    print("one designer / prefix trie / suffix trie")
    print(f"epochs : 5")
    print(f"words  : {len(TRAINING)}")

    for epoch in range(1, 6):
        rb = net.total_reward
        ru = net.total_reuse
        cr = net.total_create

        for word in TRAINING:
            net.process_word(word, learn=True)

        print(
            f"epoch={epoch:3d} "
            f"reuse={net.total_reuse - ru:3d} "
            f"create={net.total_create - cr:3d} "
            f"reward={net.total_reward - rb:8.2f}"
        )

    # Freeze learning for the generalization test.
    print()
    print("=== FREEZE ===")
    print("No learning after training.")

    net.config.designer_learning_rate = 0.0
    net.config.vocabulary_learning_rate = 0.0

    print()
    print("=== NOVEL TEST ===")

    exact = 0
    test_reuse = 0
    test_create = 0

    for word in TEST:
        expected_reuse, expected_create = expected_counts(
            TRAINING,
            word,
        )

        before_reuse = net.total_reuse
        before_create = net.total_create

        result = net.process_word(word, learn=False)

        actual_reuse = net.total_reuse - before_reuse
        actual_create = net.total_create - before_create

        ok = (
            actual_reuse == expected_reuse
            and actual_create == expected_create
        )
        exact += int(ok)

        test_reuse += actual_reuse
        test_create += actual_create

        print(
            f"{word:6s} "
            f"reuse={actual_reuse:2d} "
            f"create={actual_create:2d} "
            f"expected_reuse={expected_reuse:2d} "
            f"expected_create={expected_create:2d} "
            f"exact={ok}"
        )

    print()
    print("=== GENERALIZATION ===")
    print(f"training_words       : {len(TRAINING)}")
    print(f"test_words           : {len(TEST)}")
    print(f"exact_words          : {exact}/{len(TEST)}")
    print(f"test_reuse           : {test_reuse}")
    print(f"test_create          : {test_create}")

    net.print_dual_summary()
    net.print_summary()

    # Graph dumps are deliberately cycle-safe and ASCII-only.
    net.print_directional_graph(
        net.prefix_roots,
        "=== PREFIX VOCABULARY ===",
    )
    net.print_directional_graph(
        net.suffix_roots,
        "=== SUFFIX VOCABULARY (REVERSED) ===",
    )


if __name__ == "__main__":
    run()
