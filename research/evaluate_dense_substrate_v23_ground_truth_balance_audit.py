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
        Encode only the two directional structural contexts.

        No boundary lookup is used.
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
        context = self.context_vector(word, pos)
        h = self.context_hash(context)

        # Dense initial connectivity: every generic cell receives a weak
        # context projection. Only plasticity can make a pathway strong.
        fired = []

        for i, cell in enumerate(self.cells):
            phase = ((h ^ (i * 2654435761)) & 0xFFFF) / 65535.0

            # Weak distributed input.
            input_amount = 0.10 + 0.10 * phase

            if cell.stimulate(input_amount):
                fired.append(i)

        # Local Hebbian-style reinforcement among active cells.
        if learn:
            for i in fired:
                for j in fired:
                    if i != j:
                        key = (i, j)
                        self.weights[key] = min(
                            1.0,
                            self.weights[key] + self.learning_rate,
                        )

        return fired


    def debug_position(self, word, pos, top_k=8):
        """
        Replay the ACTUAL dense substrate computation used by
        activate_substrate(), without learning.

        This diagnostic deliberately does not invent a second activation
        model. It reports the exact context hash, per-cell input, fired cells,
        and learned outgoing connectivity that the real frozen evaluator sees.
        """
        context = self.context_vector(word, pos)
        h = self.context_hash(context)

        rows = []

        # Preserve transient neuron state while measuring this frozen input.
        old_potentials = [cell.potential for cell in self.cells]

        for i, cell in enumerate(self.cells):
            phase = (
                ((h ^ (i * 2654435761)) & 0xFFFF)
                / 65535.0
            )
            input_amount = 0.10 + 0.10 * phase

            predicted_potential = (
                old_potentials[i] * cell.leak
                + input_amount
            )
            predicted_fire = predicted_potential >= cell.threshold

            outgoing = sorted(
                (
                    (target, weight)
                    for (source, target), weight in self.weights.items()
                    if source == i and weight >= 0.50
                ),
                key=lambda item: (-item[1], item[0]),
            )

            rows.append(
                {
                    "cell": i,
                    "input": input_amount,
                    "potential": predicted_potential,
                    "fires": predicted_fire,
                    "outgoing": outgoing[:4],
                }
            )

        # Run the exact frozen activation path once, then restore state.
        fired = self.activate_substrate(word, pos, learn=False)

        for cell, old in zip(self.cells, old_potentials):
            cell.potential = old

        fired_set = set(fired)

        rows.sort(
            key=lambda row: (
                not row["fires"],
                -row["input"],
                row["cell"],
            )
        )

        return context, h, fired_set, rows[:top_k]


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

        # Dense activity magnitude is the designer's structural evidence.
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




    def trace_substrate_input(self, word, pos):
        """Probe the frozen substrate and restore all mutable cell state."""
        snapshot = [
            (
                cell.potential,
                getattr(cell, "activation", None),
                getattr(cell, "spike", None),
            )
            for cell in self.cells
        ]

        context = self.context_vector(word, pos)
        fired = list(self.activate_substrate(word, pos, learn=False))

        for cell, state in zip(self.cells, snapshot):
            cell.potential = state[0]
            if hasattr(cell, "activation") and state[1] is not None:
                cell.activation = state[1]
            if hasattr(cell, "spike") and state[2] is not None:
                cell.spike = state[2]

        return context, fired


    def exact_eval_trace(self, word, pos):
        """Mirror the real designer path and expose its exact substrate evidence."""
        n = self.net
        dg = n.designer_genome

        self.reset_designer_transient_state()
        n._reset_designer_input()

        fired = self.activate_substrate(
            word,
            pos,
            learn=False,
        )

        activity = min(
            1.0,
            len(fired) / max(1, self.cell_count),
        )

        root = n.cells[n.designer_root]
        reuse = n.cells[n.reuse_cell]
        branch = n.cells[n.branch_cell]

        root.potential += dg["input_gain"]

        if activity > 0.20:
            reuse.potential += dg["match_gain"] * activity
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

        return list(fired), activity, action

    def evaluate_frozen(self, words):
        print()
        print("=== DENSE SUBSTRATE FROZEN TEST ===")

        exact_words = 0
        correct = 0
        total = 0
        errors = []

        strong_before = sum(
            1 for w in self.weights.values() if w >= 0.50
        )

        for word in words:
            exact = True
            reuse = 0
            branch = 0

            for pos in range(len(word)):
                expected = self.available(word, pos)

                fired, activity, action = self.exact_eval_trace(
                    word,
                    pos,
                )

                print(
                    f"TRACE exact_eval word={word} pos={pos} "
                    f"symbol={word[pos]} fired={len(fired)} "
                    f"cells={fired} activity={activity:.6f} "
                    f"action={action}"
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
                    errors.append(
                        {
                            "word": word,
                            "pos": pos,
                            "symbol": word[pos],
                            "expected": wanted,
                            "actual": action,
                        }
                    )

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
        print("=== ERROR ANALYSIS ===")
        print(f"error_positions : {len(errors)}")

        if errors:
            for error in errors:
                print(
                    f"{error['word']:6s} "
                    f"pos={error['pos']:2d} "
                    f"symbol={error['symbol']} "
                    f"expected={error['expected']:6s} "
                    f"actual={error['actual']:6s}"
                )
        else:
            print("No incorrect positions.")

        print("=== END ERROR ANALYSIS ===")

        print()
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
    print("=== DENSE SUBSTRATE V23 - GROUND-TRUTH BALANCE AUDIT ===")
    print()
    print(
        "Control experiment: fully connected generic substrate, "
        "plastic effective connectivity."
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

    strong_before_prune = sum(
        1
        for weight in net.weights.values()
        if weight >= 0.50
    )

    print()
    print("=== COMPETITIVE PRUNING ===")
    print(
        "Retaining top 4 outgoing learned connections per source cell."
    )

    remaining = net.prune_competitive(
        max_outgoing=4,
        minimum_strength=0.10,
    )

    strong_after_prune = sum(
        1
        for weight in net.weights.values()
        if weight >= 0.50
    )

    print("strong_before_prune :", strong_before_prune)
    print("connections_after   :", remaining)
    print("strong_after_prune  :", strong_after_prune)

    net.evaluate_frozen(TEST)
    print()
    print("=== V23 GROUND-TRUTH BALANCE AUDIT ===")

    def audit_split(name, words):
        reuse_rows = []
        branch_rows = []

        for word in words:
            for pos in range(len(word)):
                expected = "REUSE" if net.available(word, pos) else "BRANCH"
                row = (word, pos, word[pos], expected)

                if expected == "REUSE":
                    reuse_rows.append(row)
                else:
                    branch_rows.append(row)

        print()
        print(f"--- {name} ---")
        print("words          :", len(words))
        print("positions      :", len(reuse_rows) + len(branch_rows))
        print("reuse_positions:", len(reuse_rows))
        print("branch_positions:", len(branch_rows))

        print("reuse_examples:")
        if reuse_rows:
            for word, pos, symbol, _ in reuse_rows:
                print(f"  {word} pos={pos} symbol={symbol}")
        else:
            print("  NONE")

        print("branch_examples:")
        if branch_rows:
            for word, pos, symbol, _ in branch_rows[:20]:
                print(f"  {word} pos={pos} symbol={symbol}")
            if len(branch_rows) > 20:
                print(f"  ... {len(branch_rows) - 20} more")
        else:
            print("  NONE")

        return reuse_rows, branch_rows

    train_reuse, train_branch = audit_split("TRAINING", TRAINING)
    test_reuse, test_branch = audit_split("TEST", TEST)

    print()
    print("=== BALANCE SUMMARY ===")
    print(
        "TRAIN reuse/branch :",
        len(train_reuse),
        "/",
        len(train_branch),
    )
    print(
        "TEST  reuse/branch :",
        len(test_reuse),
        "/",
        len(test_branch),
    )

    if not test_reuse:
        print("WARNING: TEST CONTAINS ZERO REUSE POSITIONS")
    if not train_reuse:
        print("WARNING: TRAINING CONTAINS ZERO REUSE POSITIONS")

    print("=== END V23 GROUND-TRUTH BALANCE AUDIT ===")

    print()
    print("=== V22 TRAIN-CALIBRATED ACTIVITY THRESHOLD ===")

    def collect_activity(words):
        rows = []
        for word in words:
            for pos in range(len(word)):
                expected = "REUSE" if net.available(word, pos) else "BRANCH"
                context, fired = net.trace_substrate_input(word, pos)
                activity = len(fired) / max(1, net.cell_count)
                rows.append((word, pos, activity, expected))
        return rows

    calibration = collect_activity(TRAINING)
    held_out = collect_activity(TEST)

    # Choose the threshold using calibration data ONLY.
    candidates = sorted(
        {activity for _, _, activity, _ in calibration}
        | {0.0, 1.0}
    )

    best = None
    for threshold in candidates:
        correct = sum(
            ("REUSE" if activity >= threshold else "BRANCH") == expected
            for _, _, activity, expected in calibration
        )
        score = correct / max(1, len(calibration))

        # Prefer the highest calibration accuracy; on ties choose the
        # threshold with the widest conservative branch/reuse separation.
        branch_values = [
            activity for _, _, activity, expected in calibration
            if expected == "BRANCH"
        ]
        reuse_values = [
            activity for _, _, activity, expected in calibration
            if expected == "REUSE"
        ]

        gap = (
            min(reuse_values) - max(branch_values)
            if branch_values and reuse_values
            else float("-inf")
        )

        key = (score, gap, -threshold)
        if best is None or key > best[0]:
            best = (key, threshold, correct)

    _, threshold, calibration_correct = best

    def score_rows(rows):
        correct = 0
        false_reuse = 0
        false_branch = 0

        for _, _, activity, expected in rows:
            predicted = "REUSE" if activity >= threshold else "BRANCH"
            if predicted == expected:
                correct += 1
            elif predicted == "REUSE":
                false_reuse += 1
            else:
                false_branch += 1

        total = len(rows)
        return correct, false_reuse, false_branch, correct / max(1, total)

    cal_correct, cal_fr, cal_fb, cal_acc = score_rows(calibration)
    test_correct, test_fr, test_fb, test_acc = score_rows(held_out)

    branch_cal = [
        activity for _, _, activity, expected in calibration
        if expected == "BRANCH"
    ]
    reuse_cal = [
        activity for _, _, activity, expected in calibration
        if expected == "REUSE"
    ]

    print("calibration_positions :", len(calibration))
    print("held_out_positions    :", len(held_out))
    print("chosen_threshold      :", f"{threshold:.6f}")

    print()
    print("=== CALIBRATION ===")
    print("correct               :", f"{cal_correct}/{len(calibration)}")
    print("accuracy              :", f"{cal_acc:.4f}")
    print("false_reuse            :", cal_fr)
    print("false_branch           :", cal_fb)

    if branch_cal and reuse_cal:
        print("max_branch_activity   :", f"{max(branch_cal):.6f}")
        print("min_reuse_activity    :", f"{min(reuse_cal):.6f}")
        print(
            "calibration_gap       :",
            f"{min(reuse_cal) - max(branch_cal):.6f}",
        )

    print()
    print("=== HELD-OUT TEST ===")
    print("correct               :", f"{test_correct}/{len(held_out)}")
    print("accuracy              :", f"{test_acc:.4f}")
    print("false_reuse           :", test_fr)
    print("false_branch          :", test_fb)
    print("=== END V22 TRAIN-CALIBRATED ACTIVITY THRESHOLD ===")

    print()
    print("=== V21 ACTIVITY THRESHOLD SWEEP ===")

    observations = []

    for word in TEST:
        for pos in range(len(word)):
            context, fired = net.trace_substrate_input(word, pos)
            activity = len(fired) / max(1, net.cell_count)

            # Exact benchmark ground truth from DensePlasticSubstrateV1.
            expected = "REUSE" if net.available(word, pos) else "BRANCH"

            observations.append(
                (word, pos, activity, expected)
            )

    print("observations :", len(observations))

    for threshold_i in range(33):
        threshold = threshold_i / 32.0
        correct = 0
        false_reuse = 0
        false_branch = 0

        for word, pos, activity, expected in observations:
            predicted = "REUSE" if activity >= threshold else "BRANCH"

            if predicted == expected:
                correct += 1
            elif predicted == "REUSE":
                false_reuse += 1
            else:
                false_branch += 1

        print(
            f"threshold={threshold:.6f} "
            f"correct={correct:2d}/{len(observations)} "
            f"accuracy={correct / len(observations):.4f} "
            f"false_reuse={false_reuse:2d} "
            f"false_branch={false_branch:2d}"
        )

    print("=== END V21 ACTIVITY THRESHOLD SWEEP ===")





if __name__ == "__main__":
    run()
