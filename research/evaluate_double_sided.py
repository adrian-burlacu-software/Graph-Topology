from __future__ import annotations

from copy import deepcopy
from simulator import Network, Config, REUSE, BRANCH
from genome import clone_genome

TRAINING = [
    "CAT", "CAR", "CAN", "CARD", "CART", "DOG", "DOT", "BAT",
]

TEST = [
    "BAT", "BAR", "BOAT", "BOAR", "BOATD",
    "CAB", "COAT", "COAR", "CART", "CARTD",
    "BART", "BARD", "BOARD",
]

def suffix_matches(net: Network, cid: int, suffix: str) -> bool:
    """True only if cid can represent the complete remaining suffix."""
    node = net.cells[cid]
    if not suffix:
        return True
    for ch in suffix:
        nxt = None
        for child_id in node.outgoing:
            child = net.cells[child_id]
            if child.kind == "vocabulary" and child.symbol == ch:
                nxt = child_id
                break
        if nxt is None:
            return False
        node = net.cells[nxt]
    return True

class DoubleSidedV2(Network):
    """
    Experimental v2:
      * forward lookup remains authoritative;
      * if a local edge is missing, search globally for the symbol;
      * a remote candidate is usable only when its complete forward suffix
        matches the remaining word;
      * when a remote candidate is used, splice the current node to it.
    """

    def remote_suffix_candidate(self, symbol: str, suffix: str):
        candidates = [
            c for c in self.vocabulary_cells()
            if c.symbol == symbol
        ]
        candidates.sort(key=lambda c: (c.order, c.id))
        for c in candidates:
            if suffix_matches(self, c.id, suffix):
                return c.id
        return None

    def find_child_for_experiment(self, parent_id, symbol, suffix):
        local = super().find_child(parent_id, symbol)
        if local is not None:
            return local, "LOCAL"

        # Root symbols are still structural. Never remote-splice a root.
        if parent_id is None:
            return None, "NONE"

        remote = self.remote_suffix_candidate(symbol, suffix)
        if remote is not None:
            return remote, "REMOTE"

        return None, "NONE"

    def process_word(self, word: str, learn: bool = True):
        current_id = None
        created = reused = branched = remote_reuses = remote_branches = 0
        diagnostics = []

        for order, symbol in enumerate(word):
            suffix = word[order + 1:]

            existing, mode = self.find_child_for_experiment(
                current_id, symbol, suffix
            )

            # The action target is based on the v2 structural candidate.
            correct = REUSE if existing is not None else BRANCH

            self._reset_designer_input()
            self.spike_designer(current_id, symbol)
            action = self.designer_signal(current_id, symbol)

            # For the experiment, a structurally valid remote candidate is
            # treated as reusable. If absent, branch normally.
            if existing is not None and action == REUSE:
                new_id = existing
                made = 0
                reused_now = 1
                reward = self.config.reward_correct_reuse
                if mode == "REMOTE":
                    remote_reuses += 1

                # Splice only when the remote node is not already local.
                if (
                    current_id is not None
                    and new_id not in self.cells[current_id].outgoing
                ):
                    self.connect(
                        current_id,
                        new_id,
                        kind="EXCITE",
                        weight=self.config.excite_weight,
                    )

            elif existing is None and action == BRANCH:
                new_id = self.create_vocabulary_cell(
                    symbol, current_id, order
                )
                made = 1
                reused_now = 0
                branched += 1
                reward = (
                    self.config.reward_correct_branch
                    + self.config.branch_cost
                )
                remote_branches += 1

            elif existing is not None and action == BRANCH:
                new_id = existing
                made = 0
                reused_now = 0
                branched += 1
                reward = (
                    self.config.reward_wrong_branch
                    + self.config.branch_cost
                )

            else:
                # Wrong reuse: repair structurally with a local cell.
                new_id = self.create_vocabulary_cell(
                    symbol, current_id, order
                )
                made = 1
                reused_now = 0
                reward = self.config.reward_wrong_reuse

            if action == REUSE:
                self.action_reuse += 1
            else:
                self.action_branch += 1

            if made:
                self.total_create += made
                created += made
            if reused_now:
                self.total_reuse += reused_now
                reused += reused_now

            if learn:
                self.learn_designer(action, correct, reward)

            diagnostics.append({
                "pos": order,
                "symbol": symbol,
                "suffix": suffix,
                "mode": mode,
                "candidate": new_id,
                "action": action,
            })
            current_id = new_id

        return {
            "word": word,
            "created": created,
            "reused": reused,
            "branched": branched,
            "remote_reuses": remote_reuses,
            "remote_branches": remote_branches,
            "diagnostics": diagnostics,
        }

def expected_for_word(net, word):
    # Baseline expectation: current graph, before mutation, with ordinary
    # forward edges only. This keeps the benchmark comparable to prior runs.
    reuse = create = 0
    current = None
    for ch in word:
        existing = Network.find_child(net, current, ch)
        if existing is None:
            create += 1
            current = None
        else:
            reuse += 1
            current = existing
    return reuse, create

def run():
    genome = clone_genome()
    net = DoubleSidedV2(Config(genome=genome))

    print("=== DOUBLE-SIDED V2 ===")
    print("suffix-constrained remote reuse + graph splicing")
    print()
    print("=== TRAINING ===")
    net.train(TRAINING, epochs=5)

    net.config.designer_learning_rate = 0.0
    net.config.vocabulary_learning_rate = 0.0

    print()
    print("=== FREEZE ===")
    print("No learning after training.")

    print()
    print("=== NASTY TEST ===")

    exact = 0
    total_reuse = total_create = 0
    before = len(net.cells)

    for word in TEST:
        er, ec = expected_for_word(net, word)
        rb, cb = net.total_reuse, net.total_create

        result = net.process_word(word, learn=False)

        wr = net.total_reuse - rb
        wc = net.total_create - cb
        ok = wr == er and wc == ec
        exact += ok
        total_reuse += wr
        total_create += wc

        print(
            f"{word:6s} "
            f"reuse={wr:2d} create={wc:2d} "
            f"expected_reuse={er:2d} expected_create={ec:2d} "
            f"exact={ok}"
        )
        for d in result["diagnostics"]:
            if d["mode"] == "REMOTE":
                print(
                    f"  remote pos={d['pos']} symbol={d['symbol']} "
                    f"suffix={d['suffix'] or '-'} -> {d['candidate']} "
                    f"action={d['action']}"
                )

    print()
    print("=== GENERALIZATION ===")
    print(f"test_words           : {len(TEST)}")
    print(f"exact_words          : {exact}/{len(TEST)}")
    print(f"test_reuse           : {total_reuse}")
    print(f"test_create          : {total_create}")
    print(f"cells_before_test    : {before}")
    print(f"cells_after_test     : {len(net.cells)}")
    print(f"new_cells            : {len(net.cells) - before}")

    print()
    print("=== LEARNED NETWORK ===")
    net.print_summary()

if __name__ == "__main__":
    run()
