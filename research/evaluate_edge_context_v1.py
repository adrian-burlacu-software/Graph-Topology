from __future__ import annotations

from simulator import Network, Config, REUSE, BRANCH
from genome import clone_genome

TRAINING = [
    "CAT", "CAR", "CAN", "CARD", "CART",
    "DOG", "DOT",
    "BAT",
]

TEST = [
    "COT",
    "COD",
    "COAT",
    "COARD",
    "BART",
    "BARD",
    "BAND",
    "DART",
    "DAT",
    "BOAT",
    "BOATD",
    "BOARD",
    "BOAR",
    "CARTD",
    "CARTDD",
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

class EdgeContextV1(Network):
    """
    Edge-context control experiment.

    A vocabulary node is reusable only when the complete incoming context
    (the symbol path from the root to its parent) matches the current path.
    This prevents a node learned under one prefix from being stolen by
    another prefix merely because its symbol/suffix happens to match.

    The graph may still share nodes when two paths have the same context.
    No remote splice is allowed when the structural context differs.
    """

    def context_signature(self, cid):
        out = []
        seen = set()
        while cid is not None:
            if cid in seen:
                return None
            seen.add(cid)
            cell = self.cells[cid]
            out.append(cell.symbol)
            cid = cell.parent
        out.reverse()
        return tuple(out)

    def edge_context_matches(self, current_id, candidate_id):
        if current_id is None or candidate_id is None:
            return False

        candidate = self.cells[candidate_id]
        if candidate.kind != "vocabulary":
            return False

        candidate_parent = candidate.parent
        if candidate_parent is None:
            return False

        return (
            self.context_signature(current_id)
            == self.context_signature(candidate_parent)
        )

    def remote_candidate(self, current_id, symbol):
        candidates = [
            c for c in self.vocabulary_cells()
            if c.symbol == symbol
            and self.edge_context_matches(current_id, c.id)
        ]
        candidates.sort(key=lambda c: c.id)

        # Even with matching context, never introduce a cycle.
        for candidate in candidates:
            if candidate.id == current_id:
                continue
            if self._reachable(candidate.id, current_id):
                continue
            return candidate.id

        return None

    def _reachable(self, start_id, target_id):
        if start_id is None or target_id is None:
            return False

        seen = set()
        stack = [start_id]

        while stack:
            cid = stack.pop()
            if cid in seen:
                continue
            if cid == target_id:
                return True
            seen.add(cid)
            stack.extend(self.cells[cid].outgoing)

        return False

    def find_child_for_experiment(self, parent_id, symbol):
        local = Network.find_child(self, parent_id, symbol)
        if local is not None:
            return local, "LOCAL"

        remote = self.remote_candidate(parent_id, symbol)
        if remote is not None:
            return remote, "REMOTE"

        return None, "NONE"

    def process_word(self, word: str, learn: bool = True):
        current_id = None
        created = reused = branched = remote_reuses = 0
        diagnostics = []

        for order, symbol in enumerate(word):
            existing, mode = self.find_child_for_experiment(
                current_id, symbol
            )

            correct = REUSE if existing is not None else BRANCH

            self._reset_designer_input()
            self.spike_designer(current_id, symbol)
            action = self.designer_signal(current_id, symbol)

            if existing is not None and action == REUSE:
                new_id = existing
                made = 0
                reused_now = 1
                reward = self.config.reward_correct_reuse

                if mode == "REMOTE":
                    remote_reuses += 1
                    if (
                        current_id is not None
                        and new_id not in self.cells[current_id].outgoing
                        and not self._reachable(new_id, current_id)
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
                new_id = self.create_vocabulary_cell(
                    symbol, current_id, order
                )
                made = 1
                reused_now = 0
                reward = self.config.reward_wrong_reuse

            self.action_reuse += action == REUSE
            self.action_branch += action == BRANCH

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
    net = EdgeContextV1(Config(genome=genome))

    print("=== EDGE-CONTEXT V1 ===")
    print("full incoming-path context required for remote reuse")
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

        exact += int(ok)
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
                    f"-> {d['candidate']} action={d['action']}"
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
