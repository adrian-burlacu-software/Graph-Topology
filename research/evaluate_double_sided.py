from __future__ import annotations

from simulator import Network, Config, REUSE, BRANCH
from genome import clone_genome, genome_summary

TRAINING = [
    "CAT", "CAR", "CAN", "CARD", "CART", "DOG", "DOT", "BAT",
]

TEST = [
    "BAT", "BAR", "BOAT", "BOAR", "BOATD", "CAB", "COAT", "COAR",
    "CART", "CARTD", "BART", "BARD", "BOARD",
]


class DoubleSidedNetwork(Network):
    """Experimental bidirectional vocabulary graph.

    Forward evidence comes from the current parent -> symbol edge.
    Reverse evidence comes from an already-known suffix beginning at the
    current symbol.  When reverse evidence exists, REUSE may attach that
    existing node to the current parent, turning the vocabulary structure
    into a DAG rather than duplicating the suffix.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._remaining = ""
        self.reverse_matches = 0
        self.remote_reuses = 0
        self.remote_branches = 0

    def find_suffix_candidate(self, suffix: str):
        """Return an existing node matching the complete suffix, or None."""
        if len(suffix) < 2:
            return None

        for cell in self.vocabulary_cells():
            if cell.symbol != suffix[0]:
                continue

            current = cell.id
            ok = True
            for symbol in suffix[1:]:
                child = None
                for cid in self.cells[current].outgoing:
                    c = self.cells[cid]
                    if c.kind == "vocabulary" and c.symbol == symbol:
                        child = cid
                        break
                if child is None:
                    ok = False
                    break
                current = child

            if ok:
                return cell.id

        return None

    def reverse_candidate(self, symbol: str):
        return self.find_suffix_candidate(self._remaining)

    def _stimulate_local_context(self, current_id, symbol):
        match_activity, context_activity = super()._stimulate_local_context(
            current_id, symbol
        )

        candidate = self.reverse_candidate(symbol)
        if candidate is not None:
            # Reverse/suffix evidence is another sensory signal from the graph.
            # It is not a privileged boolean; it is activity from an existing
            # downstream path.
            match_activity += 1.0
            self.reverse_matches += 1
            self.cells[candidate].potential += self.designer_genome["match_gain"]

        return match_activity, context_activity

    def _apply_decision(self, current_id, symbol, order, action):
        # Normal local edge wins first.
        existing = self.find_child(current_id, symbol)
        if existing is not None:
            return super()._apply_decision(
                current_id, symbol, order, action
            )

        # No local edge.  If the remaining suffix already exists elsewhere,
        # reuse its starting node and connect the current parent to it.
        candidate = self.reverse_candidate(symbol)
        if candidate is not None and action == REUSE:
            if current_id is not None:
                self.connect(
                    current_id,
                    candidate,
                    "EXCITE",
                    self.config.excite_weight,
                )
            self.remote_reuses += 1
            return (
                candidate,
                0,
                1,
                self.config.reward_correct_reuse,
            )

        if candidate is not None and action == BRANCH:
            # Preserve the existing graph; don't duplicate a known suffix.
            self.remote_branches += 1
            return (
                candidate,
                0,
                0,
                self.config.reward_wrong_branch + self.config.branch_cost,
            )

        return super()._apply_decision(
            current_id, symbol, order, action
        )

    def process_word(self, word: str, learn: bool = True):
        # Keep the base implementation's exact counters/learning behavior,
        # but expose the unconsumed suffix to the reverse sensory system.
        current_id = None
        created = 0
        reused = 0
        branched = 0

        for order, symbol in enumerate(word):
            self._remaining = word[order:]

            existing = self.find_child(current_id, symbol)
            reverse = self.reverse_candidate(symbol)
            correct = REUSE if existing is not None or reverse is not None else BRANCH

            self._reset_designer_input()
            self.spike_designer(current_id, symbol)
            action = self.designer_signal(current_id, symbol)

            new_id, made, reused_now, reward = self._apply_decision(
                current_id, symbol, order, action
            )
            current_id = new_id

            created += made
            reused += reused_now
            if action == BRANCH:
                branched += 1
            if made:
                self.total_create += made
            if reused_now:
                self.total_reuse += reused_now

            if learn:
                self.learn_designer(action, correct, reward)

        self._remaining = ""
        return {
            "word": word,
            "created": created,
            "reused": reused,
            "branched": branched,
        }


def expected_for_word(network, word):
    """Expected behavior under the double-sided graph rule."""
    reuse = 0
    create = 0
    current = None

    for i, symbol in enumerate(word):
        local = network.find_child(current, symbol)
        suffix = word[i:]
        reverse = network.find_suffix_candidate(suffix)

        if local is not None or reverse is not None:
            reuse += 1
            current = local if local is not None else reverse
        else:
            create += 1
            current = None  # expectation only; actual path will create below
            # For expectation, a newly-created node becomes the current node.
            # We don't have an id, so this sentinel is sufficient only until
            # the next symbol; recompute using a virtual path instead below.

    # The simple pass above cannot represent newly-created edges, so use the
    # same structural process on a temporary copy is overkill.  The experiment
    # reports actual behavior; exact comparison uses expected totals supplied
    # by the known target set below.
    return reuse, create


def run():
    genome = clone_genome()
    net = DoubleSidedNetwork(Config(genome=genome))

    print("=== DOUBLE-SIDED EXPERIMENT ===")
    print("genome:", genome_summary(genome))
    print()
    print("=== TRAINING ===")
    net.train(TRAINING, epochs=5)

    net.config.designer_learning_rate = 0.0
    net.config.vocabulary_learning_rate = 0.0

    print()
    print("=== FREEZE ===")
    print("No learning after training.")
    print()
    print("=== NOVEL TEST ===")

    expected = {
        "BAT": (3, 0), "BAR": (2, 1), "BOAT": (3, 1), "BOAR": (3, 1),
        "BOATD": (3, 2), "CAB": (2, 1), "COAT": (3, 1), "COAR": (3, 1),
        "CART": (4, 0), "CARTD": (4, 1), "BART": (3, 1), "BARD": (3, 1),
        "BOARD": (3, 2),
    }

    exact = 0
    total_reuse = 0
    total_create = 0

    for word in TEST:
        rb = net.total_reuse
        cb = net.total_create
        result = net.process_word(word, learn=False)
        reuse = net.total_reuse - rb
        create = net.total_create - cb
        er, ec = expected[word]
        ok = reuse == er and create == ec
        exact += int(ok)
        total_reuse += reuse
        total_create += create
        print(
            f"{word:5s} reuse={reuse:2d} create={create:2d} "
            f"expected_reuse={er:2d} expected_create={ec:2d} exact={ok} "
            f"result={result}"
        )

    print()
    print("=== GENERALIZATION ===")
    print(f"test_words           : {len(TEST)}")
    print(f"test_reuse           : {total_reuse}")
    print(f"test_create          : {total_create}")
    print(f"exact_words          : {exact}/{len(TEST)}")
    print(f"reverse_matches      : {net.reverse_matches}")
    print(f"remote_reuses        : {net.remote_reuses}")
    print(f"remote_branches      : {net.remote_branches}")
    print()
    print("=== LEARNED NETWORK ===")
    net.print_summary()
    net.print_vocabulary_tree()


if __name__ == "__main__":
    run()
