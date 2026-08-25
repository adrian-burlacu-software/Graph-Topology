from __future__ import annotations

import random
from copy import deepcopy

from genome import GENOME
from evaluate_dual_vocabulary_v6 import DualVocabularyV6


# === V28 BALANCED COMPOSITIONAL TRAINING ===
#
# REUSE_TRAINING defines the independent composition memory.
# BRANCH_TRAINING is presented to the dense substrate as negative examples,
# but is NOT inserted into the independent reuse graph.
#
# This distinction is essential: otherwise every position in every training
# word becomes REUSE by definition.
#
# TEST contains both known-composition REUSE examples and non-compositional
# BRANCH examples.

REUSE_TRAINING = [
    "CAT", "CAR", "CAN", "CARD", "CART",
    "CAD", "COD", "COT", "BAD", "BAR",
    "BARD", "BAN", "DART", "DAT", "BOT",
    "BOAT",
]

BRANCH_TRAINING = [
    "CAB", "CAP", "CAG", "COB", "COR",
    "DAB", "DAG", "DAN", "BAT", "BAG",
    "DOA", "DOG", "BOD", "BOR", "CARTB",
]

# The dense substrate sees both positive and negative sequences.
TRAINING = REUSE_TRAINING + BRANCH_TRAINING

TEST_REUSE = [
    "CAT", "CAR", "CAN", "CARD", "CART",
    "CAD", "COD", "COT", "BAD", "BAR",
]

TEST_BRANCH = [
    "CABD", "CAPT", "CAGD", "COBD", "CORD",
    "DABD", "DAGT", "DANT", "BATD", "BAGT",
]

TEST = TEST_REUSE + TEST_BRANCH



class IndependentGroundTruth:
    """
    Ground truth is built independently from the dense substrate.

    A position is REUSE iff the exact (prefix, symbol, suffix) composition
    occurred in TRAINING. The dense substrate never receives this graph.
    """

    def __init__(self, training_words):
        self.prefix = {}
        self.suffix = {}
        self.next_prefix = 0
        self.next_suffix = 0
        self.links = set()

        self._ensure_prefix("")
        self._ensure_suffix("")

        for word in training_words:
            self.learn(word)

    def _ensure_prefix(self, text):
        if text not in self.prefix:
            self.prefix[text] = self.next_prefix
            self.next_prefix += 1
        return self.prefix[text]

    def _ensure_suffix(self, text):
        if text not in self.suffix:
            self.suffix[text] = self.next_suffix
            self.next_suffix += 1
        return self.suffix[text]

    def learn(self, word):
        for pos, symbol in enumerate(word):
            p = self._ensure_prefix(word[:pos])
            s = self._ensure_suffix(word[pos + 1:])
            self.links.add((p, symbol, s))

    def available(self, word, pos):
        p = self.prefix.get(word[:pos])
        s = self.suffix.get(word[pos + 1:])
        if p is None or s is None:
            return False
        return (p, word[pos], s) in self.links


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

        fired_set = set(fired)

        # V33: continuation competition.
        #
        # We do not ask whether the current activation is merely dense, nor
        # whether it overlaps the actual next-symbol activation. Instead,
        # learned outgoing topology votes for candidate continuation cells.
        #
        # Candidate symbols are the symbols represented by the substrate's
        # existing vocabulary cells. No independent ground truth is consulted.

        strong_edges = [
            (src, dst, weight)
            for (src, dst), weight in self.weights.items()
            if src in fired_set and weight >= 0.50
        ]

        votes_by_destination = {}
        for _, dst, weight in strong_edges:
            votes_by_destination[dst] = (
                votes_by_destination.get(dst, 0.0) + weight
            )

        # Recover candidate symbols from the substrate cell labels.
        # Prefer an explicit symbol attribute when available; otherwise use
        # the cell's existing label/name representation.
        symbol_cells = {}
        for cell in self.cells:
            symbol = getattr(cell, "symbol", None)
            if symbol is None:
                symbol = getattr(cell, "label", None)
            if symbol is None:
                symbol = getattr(cell, "value", None)

            if isinstance(symbol, str) and len(symbol) == 1:
                symbol_cells.setdefault(symbol, set()).add(cell.id)

        candidate_scores = {}
        for symbol, cell_ids in symbol_cells.items():
            score = sum(
                votes_by_destination.get(cell_id, 0.0)
                for cell_id in cell_ids
            )
            candidate_scores[symbol] = score

        # The actual next symbol is diagnostic only. It is NOT used to make
        # the decision. At the terminal position there is no continuation
        # candidate, so leave the prediction signal at zero.
        actual_next = (
            word[pos + 1]
            if pos + 1 < len(word)
            else None
        )

        ranked_candidates = sorted(
            candidate_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        best_symbol = (
            ranked_candidates[0][0]
            if ranked_candidates and ranked_candidates[0][1] > 0.0
            else None
        )
        best_score = (
            ranked_candidates[0][1]
            if ranked_candidates
            else 0.0
        )

        second_score = (
            ranked_candidates[1][1]
            if len(ranked_candidates) > 1
            else 0.0
        )

        margin = best_score - second_score

        # A reusable boundary should produce a strong, selective
        # continuation preference. The margin prevents total connectivity
        # from automatically becoming REUSE.
        predictive_threshold = dg.get("predictive_threshold", 0.50)
        margin_threshold = dg.get("prediction_margin_threshold", 0.10)

        predictive_evidence = min(1.0, best_score / max(1.0, len(fired_set)))
        selective_prediction = (
            best_score >= predictive_threshold
            and margin >= margin_threshold
        )

        self.last_dense_trace = {
            "word": word,
            "pos": pos,
            "fired": sorted(fired_set),
            "activity": len(fired_set) / max(1, self.cell_count),
            "actual_next": actual_next,
            "candidate_scores": ranked_candidates[:8],
            "best_symbol": best_symbol,
            "best_score": best_score,
            "second_score": second_score,
            "margin": margin,
            "predictive_evidence": predictive_evidence,
            "selective_prediction": selective_prediction,
            "strong_outgoing": len(strong_edges),
            "learned_mass": sum(weight for _, _, weight in strong_edges),
        }

        root = n.cells[n.designer_root]
        reuse = n.cells[n.reuse_cell]
        branch = n.cells[n.branch_cell]

        root.potential += dg["input_gain"]

        if selective_prediction:
            reuse.potential += (
                dg["match_gain"] * min(1.0, predictive_evidence)
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
                expected = self.ground_truth.available(word, pos)

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
    print("=== DENSE SUBSTRATE V33 - CONTINUATION COMPETITION ===")
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
    # Independent ground truth: no calls to net.learn_structure(), no access
    # from the designer, and no dependence on DensePlasticSubstrateV1 state.
    net.ground_truth = IndependentGroundTruth(REUSE_TRAINING)

    print()
    print("=== V28 GROUND-TRUTH BALANCE ===")

    def gt_rows(words):
        reuse = []
        branch = []
        for word in words:
            for pos in range(len(word)):
                row = (word, pos, word[pos])
                if net.ground_truth.available(word, pos):
                    reuse.append(row)
                else:
                    branch.append(row)
        return reuse, branch

    train_reuse, train_branch = gt_rows(TRAINING)
    test_reuse, test_branch = gt_rows(TEST)

    print(
        f"TRAINING positions={len(train_reuse) + len(train_branch)} "
        f"reuse={len(train_reuse)} branch={len(train_branch)}"
    )
    print(
        f"TEST positions={len(test_reuse) + len(test_branch)} "
        f"reuse={len(test_reuse)} branch={len(test_branch)}"
    )
    print(
        f"REUSE_TRAINING words={len(REUSE_TRAINING)} "
        f"BRANCH_TRAINING words={len(BRANCH_TRAINING)}"
    )

    assert train_reuse, "V28 invalid: training has zero REUSE positions"
    assert train_branch, "V28 invalid: training has zero BRANCH positions"
    assert test_reuse, "V28 invalid: test has zero REUSE positions"
    assert test_branch, "V28 invalid: test has zero BRANCH positions"

    print("GROUND TRUTH BALANCE ASSERTIONS: PASS")
    print("=== END V28 GROUND-TRUTH BALANCE ===")
    print()


    print()
    print("=== V25 INDEPENDENT GROUND TRUTH ===")

    train_reuse = 0
    train_branch = 0
    for word in TRAINING:
        for pos in range(len(word)):
            if net.ground_truth.available(word, pos):
                train_reuse += 1
            else:
                train_branch += 1

    test_reuse = 0
    test_branch = 0
    for word in TEST:
        for pos in range(len(word)):
            if net.ground_truth.available(word, pos):
                test_reuse += 1
            else:
                test_branch += 1

    print(
        f"TRAINING: positions={train_reuse + train_branch} "
        f"reuse={train_reuse} branch={train_branch}"
    )
    print(
        f"TEST: positions={test_reuse + test_branch} "
        f"reuse={test_reuse} branch={test_branch}"
    )

    assert train_reuse > 0, "Ground truth has no REUSE training positions"
    assert test_reuse > 0, "Ground truth has no REUSE test positions"
    assert test_branch > 0, "Ground truth has no BRANCH test positions"

    print("GROUND TRUTH ASSERTIONS: PASS")
    print("=== END V25 INDEPENDENT GROUND TRUTH ===")


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


    print()
    print("=== V30 EXACT FROZEN READOUT AUDIT ===")
    print(
        "The trace below is captured INSIDE "
        "designer_from_dense_activity()."
    )
    print(
        "This is the exact activity the designer uses for its decision."
    )

    def probe_exact_path(label, rows):
        print()
        print(f"--- {label} ---")

        for word, pos, expected in rows:
            action = net.designer_from_dense_activity(word, pos)
            trace = net.last_dense_trace

            print(
                f"{word:6s} pos={pos} symbol={word[pos]} "
                f"expected={expected:6s} actual={action:6s} "
                f"fired={len(trace['fired']):2d}/{net.cell_count} "
                f"activity={trace['activity']:.6f} "
                f"strong_out={trace['strong_outgoing']:3d} "
                f"learned_mass={trace['learned_mass']:.3f} "
                f"cells={trace['fired']}"
            )

    def labelled_rows(words, limit=12):
        rows = []
        for word in words:
            for pos in range(len(word)):
                expected = (
                    "REUSE"
                    if net.ground_truth.available(word, pos)
                    else "BRANCH"
                )
                rows.append((word, pos, expected))
                if len(rows) >= limit:
                    return rows
        return rows

    probe_exact_path(
        "TRAINING MIXED",
        labelled_rows(TRAINING),
    )
    probe_exact_path(
        "HELD-OUT MIXED",
        labelled_rows(TEST),
    )

    # Quantify the key causal question:
    # does the learned topology actually enter the designer decision?
    # The current readout computes its structural evidence from len(fired)
    # only; learned_mass/strong_outgoing are diagnostic measurements.
    print()
    print("=== V32 PREDICTIVE READOUT ===")
    print("decision_signal : learned outgoing topology votes for candidate symbols")
    print("prediction_metric : best-score + winner margin")
    print("=== END V32 PREDICTIVE READOUT ===")
    print()


    print()
    print("=== V33 CONTINUATION COMPETITION AUDIT ===")

    def probe_candidates(words, limit=20):
        count = 0
        for word in words:
            for pos in range(len(word)):
                action = net.designer_from_dense_activity(word, pos)
                trace = net.last_dense_trace
                print(
                    f"{word:6s} pos={pos} actual_next={str(trace['actual_next']):>2s} "
                    f"decision={action:6s} "
                    f"best={str(trace['best_symbol']):>2s} "
                    f"score={trace['best_score']:.3f} "
                    f"second={trace['second_score']:.3f} "
                    f"margin={trace['margin']:.3f} "
                    f"selective={trace['selective_prediction']} "
                    f"candidates={trace['candidate_scores'][:5]}"
                )
                count += 1
                if count >= limit:
                    return

    probe_candidates(TRAINING)
    print("--- HELD-OUT ---")
    probe_candidates(TEST)

    print("=== END V33 CONTINUATION COMPETITION AUDIT ===")
    print()

    net.evaluate_frozen(TEST)


if __name__ == "__main__":
    run()
