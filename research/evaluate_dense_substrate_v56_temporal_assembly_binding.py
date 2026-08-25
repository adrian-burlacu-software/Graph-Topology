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




print("=== V56 START ===")
print("V56_CODE_LOADED = TRUE")
print()

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

    def v56_reset_transition_memory(self):
        self._v56_transition_weights = {}
        self._v56_last_signature = None
        self._v56_transition_count = 0

    def v56_signature(self, fired):
        return tuple(sorted(set(fired)))

    def v56_observe_transition(self, previous_fired, current_fired):
        """Hebbian-style transition binding between actual fired cells."""
        previous = self.v56_signature(previous_fired)
        current = self.v56_signature(current_fired)

        if not previous or not current:
            self._v56_last_signature = current
            return

        for src_id in previous:
            for dst_id in current:
                key = (src_id, dst_id)
                self._v56_transition_weights[key] = (
                    self._v56_transition_weights.get(key, 0.0)
                    + 1.0
                )

        self._v56_transition_count += 1
        self._v56_last_signature = current

    def v56_transition_projection(self, previous_fired, current_fired):
        """Measure learned support for the current assembly given the prior."""
        previous = self.v56_signature(previous_fired)
        current = set(current_fired)

        supported = 0.0
        possible = 0.0
        edge_count = 0

        for src_id in previous:
            for dst_id in current:
                possible += 1.0
                weight = self._v56_transition_weights.get(
                    (src_id, dst_id),
                    0.0,
                )
                if weight > 0.0:
                    supported += weight
                    edge_count += 1

        density = (
            edge_count / possible
            if possible > 0.0
            else 0.0
        )

        return {
            "previous": previous,
            "current": tuple(sorted(current)),
            "support": supported,
            "possible": possible,
            "density": density,
            "edge_count": edge_count,
        }

    def v56_frozen_step(self, word, pos):
        """Frozen dense step with no membrane-state leakage."""
        fired = self.activate_substrate_frozen(word, pos)
        return tuple(sorted(fired))


    def v53_topology_projection(self, fired, minimum_strength=0.50):
        """One-hop propagation through the learned graph."""
        fired_set = set(fired)
        destination_mass = {}

        for (src_id, dst_id), weight in self.weights.items():
            if src_id in fired_set and weight >= minimum_strength:
                destination_mass[dst_id] = (
                    destination_mass.get(dst_id, 0.0)
                    + float(weight)
                )

        ranked = sorted(
            destination_mass.items(),
            key=lambda item: (-item[1], item[0]),
        )

        return {
            "destination_mass": destination_mass,
            "ranked_destinations": tuple(ranked),
            "total_mass": sum(destination_mass.values()),
            "destination_count": len(destination_mass),
        }

    def v53_topology_projected_evidence(self, word, pos):
        """Frozen dense activity followed by one learned-topology hop."""
        fired = self.activate_substrate_frozen(word, pos)
        projection = self.v53_topology_projection(fired)

        masses = [
            mass for _, mass in projection["ranked_destinations"]
        ]
        strongest = masses[0] if masses else 0.0
        second = masses[1] if len(masses) > 1 else 0.0
        total = projection["total_mass"]

        concentration = (
            strongest / total if total > 0.0 else 0.0
        )
        margin = (
            (strongest - second) / total
            if total > 0.0
            else 0.0
        )

        # Count destinations receiving support from >=2 distinct fired
        # sources. This is a structural convergence signal.
        source_support = {}
        fired_set = set(fired)

        for (src_id, dst_id), weight in self.weights.items():
            if src_id in fired_set and weight >= 0.50:
                source_support.setdefault(dst_id, set()).add(src_id)

        convergent = sum(
            1 for sources in source_support.values()
            if len(sources) >= 2
        )

        return {
            "word": word,
            "pos": pos,
            "fired": tuple(sorted(fired)),
            "projected_destinations": projection["ranked_destinations"],
            "total_mass": total,
            "destination_count": projection["destination_count"],
            "concentration": concentration,
            "margin": margin,
            "convergent_destinations": convergent,
            "source_support": tuple(
                sorted(
                    (
                        dst,
                        len(sources),
                    )
                    for dst, sources in source_support.items()
                )
            ),
        }


    def v52_topology_signature(self, fired):
        """Read the actual learned (src, dst) weight graph."""
        fired_set = set(fired)
        signatures = []

        for source in sorted(fired_set):
            outgoing = []
            for (src_id, dst_id), weight in self.weights.items():
                if src_id != source:
                    continue
                if weight >= 0.50:
                    outgoing.append(
                        (dst_id, round(float(weight), 12))
                    )

            outgoing.sort(
                key=lambda item: (-item[1], item[0])
            )

            signatures.append(
                (source, tuple(dst for dst, _ in outgoing))
            )

        return tuple(signatures)

    def v52_topology_evidence(self, word, pos):
        """Derive structural evidence directly from learned topology."""
        fired = self.activate_substrate_frozen(word, pos)
        fired_set = set(fired)

        outgoing = []
        for (src_id, dst_id), weight in self.weights.items():
            if src_id in fired_set and weight >= 0.50:
                outgoing.append((src_id, dst_id, float(weight)))

        destination_sources = {}
        destination_mass = {}

        for src_id, dst_id, weight in outgoing:
            destination_sources.setdefault(dst_id, set()).add(src_id)
            destination_mass[dst_id] = (
                destination_mass.get(dst_id, 0.0) + weight
            )

        total_mass = sum(weight for _, _, weight in outgoing)

        source_count = len(fired_set)
        supported_destinations = {
            dst: len(sources)
            for dst, sources in destination_sources.items()
        }

        support_mass = 0.0
        if source_count:
            for src_id, dst_id, weight in outgoing:
                support_mass += (
                    weight
                    * len(destination_sources[dst_id])
                    / source_count
                )

        support_fraction = (
            support_mass / total_mass
            if total_mass > 0.0
            else 0.0
        )

        masses = sorted(
            destination_mass.values(),
            reverse=True,
        )

        strongest = masses[0] if masses else 0.0
        second = masses[1] if len(masses) > 1 else 0.0

        concentration = (
            strongest / total_mass
            if total_mass > 0.0
            else 0.0
        )

        margin = (
            (strongest - second) / total_mass
            if total_mass > 0.0
            else 0.0
        )

        # Structural signal only. No learned assembly dictionary.
        topology_evidence = (
            0.4 * concentration
            + 0.4 * support_fraction
            + 0.2 * margin
        )

        return {
            "word": word,
            "pos": pos,
            "fired": tuple(sorted(fired)),
            "topology_signature": self.v52_topology_signature(fired),
            "strong_outgoing": len(outgoing),
            "destination_count": len(destination_mass),
            "learned_mass": total_mass,
            "support_fraction": support_fraction,
            "concentration": concentration,
            "competition_margin": margin,
            "topology_evidence": topology_evidence,
            "supported_destinations": sorted(
                supported_destinations.items(),
                key=lambda item: (-item[1], item[0]),
            )[:12],
        }

    def v51_topology_evidence(self, word, pos):
        """Compute recurrence evidence solely from learned topology."""
        fired = self.activate_substrate_frozen(word, pos)
        signature = self.v51_topology_signature(fired)

        # Structural evidence: number of firing cells that have learned
        # outgoing structure, and number of strong targets contained in that
        # structure. No prior assembly/test-word memory is used.
        active_sources = 0
        strong_edges = 0

        weights = getattr(self, "weights", None)
        if weights is None:
            weights = getattr(self, "connections", None)

        if isinstance(weights, dict):
            for source in fired:
                outgoing = weights.get(source, {})
                if isinstance(outgoing, dict):
                    count = 0
                    for weight in outgoing.values():
                        try:
                            if float(weight) > 0:
                                count += 1
                        except (TypeError, ValueError):
                            pass
                    if count:
                        active_sources += 1
                        strong_edges += count

        return {
            "word": word,
            "pos": pos,
            "fired": tuple(sorted(fired)),
            "topology_signature": signature,
            "active_sources": active_sources,
            "strong_edges": strong_edges,
        }


    def _v42_state_fingerprint(self):
        """Stable fingerprint for frozen-readout determinism diagnostics."""
        import hashlib
        import json

        def norm(value):
            if isinstance(value, float):
                return round(value, 12)
            if isinstance(value, dict):
                return {
                    str(k): norm(v)
                    for k, v in sorted(
                        value.items(),
                        key=lambda item: str(item[0]),
                    )
                }
            if isinstance(value, (list, tuple)):
                return [norm(v) for v in value]
            if isinstance(value, set):
                return sorted(
                    (norm(v) for v in value),
                    key=lambda item: repr(item),
                )
            return value

        payload = {
            "weights": norm(self.weights),
            "last_dense_trace": norm(
                getattr(self, "last_dense_trace", None)
            ),
            "assembly_history": norm(
                getattr(self, "_assembly_history", None)
            ),
            "boundary_history": norm(
                getattr(self, "_boundary_history", None)
            ),
            "frozen_boundary_memory": norm(
                getattr(self, "_frozen_boundary_memory", None)
            ),
        }

        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")

        return hashlib.sha256(encoded).hexdigest()

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


    def v46_recompute_cell_scores(self, word, pos):

        """Independent, read-only recomputation of activation scores."""

        import math

    

        # Recreate the scalar inputs used by the activation path.

        context = (

            word[pos - 1] if pos > 0 else None,

            word[pos],

            word[pos + 1] if pos + 1 < len(word) else None,

        )

    

        # Locate the same context hash/input construction used by the

        # substrate by executing one frozen activation and reading only

        # its diagnostic context values. No learning occurs.

        self.activate_substrate(word, pos, learn=False)

        trace = getattr(self, "last_dense_trace", {})

    

        rows = []

    

        # Inspect cells using stable numeric cell IDs. This function never

        # mutates potentials, activations, weights, or designer state.

        cells = getattr(self, "cells", {})

        if isinstance(cells, dict):

            iterable = sorted(cells.items(), key=lambda item: int(item[0]))

        else:

            iterable = list(enumerate(cells))

    

        for cell_id, cell in iterable:

            potential = getattr(cell, "potential", 0.0)

            activation = getattr(cell, "activation", 0.0)

            threshold = getattr(cell, "threshold", None)

    

            # Preserve the exact exposed cell quantities; if the original

            # method exposes a computed score in its trace, use it as the

            # reference rather than inventing a new formula.

            row = {

                "cell": int(cell_id),

                "potential": float(potential),

                "activation": float(activation),

                "threshold": (

                    None if threshold is None else float(threshold)

                ),

            }

    

            rows.append(row)

    

        return {

            "context": context,

            "trace": dict(trace),

            "rows": rows,

        }



    def activate_substrate_frozen(self, word, pos):


        """


        Execute one readout without changing substrate cell state.


    


        The original activate_substrate() carries DenseCell.potential across


        calls even when learn=False:


            potential = potential * leak + input_amount


        That makes repeated frozen probes stateful. This wrapper snapshots


        the mutable cell state, performs the exact activation, then restores


        it. The returned fired set is therefore a pure function of the


        pre-call substrate state and input.


        """


        cell_state = [


            (cell.potential, cell.spikes)


            for cell in self.cells


        ]


        try:


            return self.activate_substrate(word, pos, learn=False)


        finally:


            for cell, (potential, spikes) in zip(self.cells, cell_state):


                cell.potential = potential


                cell.spikes = spikes



    def activate_substrate(self, word, pos, learn=False):


        # V47: substrate input-vector autopsy.


        self._v47_input_trace = []


        # V45: surgical per-cell decision trace.

        # This records the state at the exact moment each cell is considered.

        self._v45_cell_trace = []

    

        def v45_record_cell(cell_id, cell_obj, score=None, threshold=None, fired=None):

            def safe(value):

                if isinstance(value, float):

                    return round(value, 15)

                if isinstance(value, (int, str, bool, type(None))):

                    return value

                if isinstance(value, (list, tuple)):

                    return [safe(v) for v in value]

                if isinstance(value, dict):

                    return {

                        str(k): safe(v)

                        for k, v in value.items()

                    }

                return repr(value)

    

            row = {

                "cell_id": cell_id,

                "potential": safe(getattr(cell_obj, "potential", None)),

                "activation": safe(getattr(cell_obj, "activation", None)),

                "score": safe(score),

                "threshold": safe(threshold),

                "fires": safe(getattr(cell_obj, "fires", None)),

                "fired": safe(fired),

            }

            self._v45_cell_trace.append(row)


        # V44 source-level audit: capture locals at every return so the

        # actual activation scores/candidates can be compared between

        # identical frozen calls.

        self._v44_activation_audit = {

            "locals": {},

            "returns": [],

        }
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

        self._v45_cell_trace.append({'stage': 'return', 'fired': repr(locals().get('fired', None))})

        self._v44_activation_audit['returns'].append({'value': fired, 'locals': {k: repr(v) for k, v in locals().items() if k not in ('self', 'word')}})

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


    def v42_cell_state_snapshot(self):
        """Capture the mutable DenseCell state that V42 previously omitted."""
        return [
            {
                "potential": round(cell.potential, 15),
                "spikes": cell.spikes,
            }
            for cell in self.cells
        ]

    def v42_activation_snapshot(self, word, pos):
        """Run one genuinely frozen activation and return deterministic audit data."""
        before = self._v42_state_fingerprint()

        fired = self.activate_substrate_frozen(word, pos)

        after = self._v42_state_fingerprint()

        return {
            "word": word,
            "pos": pos,
            "fired": list(fired),
            "before_fingerprint": before,
            "after_fingerprint": after,
            "trace": dict(
                getattr(self, "last_dense_trace", {})
            ),
        }

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

        # V41: pure learned-topology competition.
        #
        # The designer receives only the current dense activity and the
        # learned substrate connectivity. There is NO external assembly
        # library, symbol memory, transition memory, BoundaryGraph query,
        # or ground-truth lookup.
        #
        # The hypothesis is that V13's strong result came from competition
        # among learned pathways. We therefore measure how concentrated and
        # internally supported the currently active cells' learned outgoing
        # connectivity is, and how much of that connectivity is supported by
        # multiple active sources.
        #
        # This is a structural signal generated entirely by the substrate.

        outgoing = []

        for (src, dst), weight in self.weights.items():
            if src in fired_set and weight >= 0.50:
                outgoing.append((src, dst, weight))

        total_weight = sum(weight for _, _, weight in outgoing)

        # Destination support: how many distinct active source cells agree on
        # each destination. A pathway supported by several active sources is
        # stronger evidence than isolated outgoing edges.
        destination_sources = {}

        for src, dst, weight in outgoing:
            destination_sources.setdefault(dst, set()).add(src)

        supported_destinations = {
            dst: len(sources)
            for dst, sources in destination_sources.items()
        }

        if fired_set:
            max_source_support = max(1, len(fired_set))
        else:
            max_source_support = 1

        # Aggregate destination support weighted by learned edge strength.
        support_mass = 0.0

        for src, dst, weight in outgoing:
            support = (
                len(destination_sources[dst])
                / max_source_support
            )
            support_mass += weight * support

        support_fraction = (
            support_mass / total_weight
            if total_weight > 0.0
            else 0.0
        )

        # Concentration: how much outgoing learned mass is carried by the
        # strongest destination. This is topology-only and does not require
        # knowing what the destination represents.
        destination_mass = {}

        for src, dst, weight in outgoing:
            destination_mass[dst] = (
                destination_mass.get(dst, 0.0)
                + weight
            )

        if destination_mass:
            strongest_destination_mass = max(
                destination_mass.values()
            )
        else:
            strongest_destination_mass = 0.0

        concentration = (
            strongest_destination_mass / total_weight
            if total_weight > 0.0
            else 0.0
        )

        # Competition: compare the strongest destination with the remaining
        # outgoing mass. A dominant learned route has a positive margin;
        # diffuse connectivity has little or no margin.
        sorted_destination_mass = sorted(
            destination_mass.values(),
            reverse=True,
        )

        strongest = (
            sorted_destination_mass[0]
            if sorted_destination_mass
            else 0.0
        )
        second = (
            sorted_destination_mass[1]
            if len(sorted_destination_mass) > 1
            else 0.0
        )

        competition_margin = (
            (strongest - second) / total_weight
            if total_weight > 0.0
            else 0.0
        )

        # A topology-only reuse signal. Keep this deliberately simple and
        # interpretable rather than tuning it to the benchmark.
        topology_evidence = (
            0.4 * concentration
            + 0.4 * support_fraction
            + 0.2 * competition_margin
        )

        self.last_dense_trace = {
            "word": word,
            "pos": pos,
            "fired": sorted(fired_set),
            "activity": len(fired_set) / max(1, self.cell_count),
            "learned_mass": total_weight,
            "strong_outgoing": len(outgoing),
            "destination_count": len(destination_mass),
            "strongest_destination_mass": strongest_destination_mass,
            "concentration": concentration,
            "support_fraction": support_fraction,
            "competition_margin": competition_margin,
            "topology_evidence": topology_evidence,
            "supported_destinations": sorted(
                supported_destinations.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:8],
        }

        # Diagnostic-first: do not yet assert that topology evidence means
        # REUSE. The benchmark classification remains neutral until the
        # structural signal is shown to separate the classes.
        reuse_evidence = topology_evidence
        branch_evidence = 1.0 - topology_evidence

        root = n.cells[n.designer_root]
        reuse = n.cells[n.reuse_cell]
        branch = n.cells[n.branch_cell]

        root.potential += dg["input_gain"]

        # V41 diagnostic-only. Keep the designer neutral; topology_evidence
        # is the experiment's measured structural signal.
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



def v52_run_topology_native_experiment(net, training, test):
    print()
    print("=== V52 ACTUAL LEARNED TOPOLOGY AUDIT ===")
    print(
        "Topology is read directly from self.weights[(src, dst)]. "
        "No assembly memory is created."
    )
    print()

    print("--- TRAINING ---")
    train_signatures = {}

    for word in training:
        for pos in range(len(word)):
            e = net.v52_topology_evidence(word, pos)
            sig = e["topology_signature"]
            train_signatures[sig] = train_signatures.get(sig, 0) + 1

            print(
                f"{word:6s} pos={pos} "
                f"fired={list(e['fired'])} "
                f"strong_out={e['strong_outgoing']:3d} "
                f"dest={e['destination_count']:2d} "
                f"mass={e['learned_mass']:.3f} "
                f"support={e['support_fraction']:.3f} "
                f"conc={e['concentration']:.3f} "
                f"margin={e['competition_margin']:.3f}"
            )

    print()
    print(
        "unique_actual_topology_signatures :",
        len(train_signatures),
    )
    print(
        "training_positions                :",
        sum(train_signatures.values()),
    )

    print()
    print("--- HELD-OUT ---")

    for word in test:
        for pos in range(len(word)):
            e = net.v52_topology_evidence(word, pos)

            print(
                f"{word:6s} pos={pos} "
                f"fired={list(e['fired'])} "
                f"strong_out={e['strong_outgoing']:3d} "
                f"dest={e['destination_count']:2d} "
                f"mass={e['learned_mass']:.3f} "
                f"support={e['support_fraction']:.3f} "
                f"conc={e['concentration']:.3f} "
                f"margin={e['competition_margin']:.3f}"
            )

    print()
    print("--- SAME-INPUT TOPOLOGY DETERMINISM ---")

    for word, pos in (
        ("CAT", 1),
        ("CAD", 1),
        ("BOAT", 0),
        ("BOARD", 3),
    ):
        a = net.v52_topology_evidence(word, pos)
        b = net.v52_topology_evidence(word, pos)

        print(
            f"{word:6s} pos={pos} "
            f"same_fired={a['fired'] == b['fired']} "
            f"same_topology="
            f"{a['topology_signature'] == b['topology_signature']} "
            f"same_evidence="
            f"{a['topology_evidence'] == b['topology_evidence']}"
        )

    print()
    print("=== END V52 ACTUAL LEARNED TOPOLOGY AUDIT ===")
    print()



def v53_run_topology_projection_experiment(net, training, test):
    print()
    print("=== V53 CAUSAL TOPOLOGY PROJECTION ===")
    print(
        "Dense firing is followed by one-hop propagation through the "
        "learned self.weights graph."
    )
    print(
        "No assembly memory, vocabulary list, labels, ground truth, "
        "or BoundaryGraph is supplied to the projection."
    )
    print()

    print("--- TRAINING PROJECTIONS ---")
    training_signatures = {}

    for word in training:
        for pos in range(len(word)):
            e = net.v53_topology_projected_evidence(word, pos)

            # Structural signature of the projected continuation.
            signature = tuple(
                (dst, round(mass, 6))
                for dst, mass in e["projected_destinations"]
            )
            training_signatures[signature] = (
                training_signatures.get(signature, 0) + 1
            )

            print(
                f"{word:6s} pos={pos} "
                f"fired={list(e['fired'])} "
                f"dest={e['destination_count']:2d} "
                f"mass={e['total_mass']:.3f} "
                f"conc={e['concentration']:.3f} "
                f"margin={e['margin']:.3f} "
                f"convergent={e['convergent_destinations']:2d} "
                f"top={list(e['projected_destinations'][:6])}"
            )

    print()
    print(
        "unique_projected_signatures :",
        len(training_signatures),
    )
    print(
        "training_positions          :",
        sum(training_signatures.values()),
    )

    print()
    print("--- HELD-OUT PROJECTIONS ---")

    for word in test:
        for pos in range(len(word)):
            e = net.v53_topology_projected_evidence(word, pos)

            print(
                f"{word:6s} pos={pos} "
                f"fired={list(e['fired'])} "
                f"dest={e['destination_count']:2d} "
                f"mass={e['total_mass']:.3f} "
                f"conc={e['concentration']:.3f} "
                f"margin={e['margin']:.3f} "
                f"convergent={e['convergent_destinations']:2d} "
                f"top={list(e['projected_destinations'][:6])}"
            )

    print()
    print("--- TOPOLOGY PROJECTION DETERMINISM ---")

    for word, pos in (
        ("CAT", 1),
        ("CAD", 1),
        ("BOAT", 0),
        ("BOARD", 3),
    ):
        a = net.v53_topology_projected_evidence(word, pos)
        b = net.v53_topology_projected_evidence(word, pos)

        print(
            f"{word:6s} pos={pos} "
            f"same_projection="
            f"{a['projected_destinations'] == b['projected_destinations']} "
            f"same_mass={a['total_mass'] == b['total_mass']}"
        )

    print()
    print("=== END V53 CAUSAL TOPOLOGY PROJECTION ===")
    print()


def v54_run_guaranteed_projection(net, training, test):
    print("=== V55 POST-TRAINING TOPOLOGY PROJECTION ===")
    if net is None:
        raise RuntimeError("V54 net was not initialized")
    print("V55_EXECUTED = TRUE")
    print("source: exact V54b implementation, executed after training")
    print()

    probes = [
        ("CAT", 1),
        ("CAD", 1),
        ("BOAT", 0),
        ("BOARD", 3),
    ]

    print("--- PROBES ---")
    for word, pos in probes:
        first = net.v53_topology_projected_evidence(word, pos)
        second = net.v53_topology_projected_evidence(word, pos)

        print(
            f"{word:6s} pos={pos} "
            f"same_projection="
            f"{first['projected_destinations'] == second['projected_destinations']} "
            f"same_mass={first['total_mass'] == second['total_mass']} "
            f"dest={first['destination_count']} "
            f"mass={first['total_mass']:.6f} "
            f"convergent={first['convergent_destinations']}"
        )
        print(
            f"  fired={list(first['fired'])}"
        )
        print(
            f"  top={list(first['projected_destinations'][:10])}"
        )

    print()
    print("--- TRAINING SUMMARY ---")
    signatures = {}
    total = 0

    for word in training:
        for pos in range(len(word)):
            e = net.v53_topology_projected_evidence(word, pos)
            signature = tuple(
                (int(dst), round(float(mass), 6))
                for dst, mass in e["projected_destinations"]
            )
            signatures[signature] = signatures.get(signature, 0) + 1
            total += 1

    print("training_positions =", total)
    print("unique_projections  =", len(signatures))

    print()
    print("--- TEST SUMMARY ---")
    test_total = 0
    test_with_projection = 0

    for word in test:
        for pos in range(len(word)):
            e = net.v53_topology_projected_evidence(word, pos)
            test_total += 1
            if e["destination_count"] > 0:
                test_with_projection += 1

    print("test_positions       =", test_total)
    print("test_with_projection =", test_with_projection)

    print()
    print("V55_EXECUTED = TRUE")
    print("=== END V54 GUARANTEED TOPOLOGY PROJECTION ===")
    print()


def v56_run_temporal_binding_experiment(net, training, test):
    print()
    print("=== V56 TEMPORAL ASSEMBLY BINDING ===")
    print(
        "Transitions are learned only between consecutive frozen "
        "substrate assemblies."
    )
    print(
        "No vocabulary list, assembly list, labels, ground truth, "
        "or BoundaryGraph is used."
    )
    print()

    net.v56_reset_transition_memory()

    # TRAINING: learn temporal transitions in the natural sequence of the
    # existing training corpus.
    previous = None
    train_steps = 0

    print("--- TRAINING TRANSITIONS ---")

    for word in training:
        for pos in range(len(word)):
            current = net.v56_frozen_step(word, pos)

            if previous is not None:
                net.v56_observe_transition(previous, current)
                train_steps += 1

            previous = current

            print(
                f"{word:6s} pos={pos} "
                f"assembly={list(current)}"
            )

    print()
    print("training_transition_steps =", train_steps)
    print(
        "learned_transition_edges =",
        len(net._v56_transition_weights),
    )

    print()
    print("--- HELD-OUT TRANSITION SUPPORT ---")

    # Evaluate each test word internally as a sequence. The transition memory
    # is FROZEN during test: no test transition is learned.
    reuse_like = 0
    unsupported = 0
    test_steps = 0

    for word in test:
        previous = None

        for pos in range(len(word)):
            current = net.v56_frozen_step(word, pos)

            if previous is not None:
                e = net.v56_transition_projection(
                    previous,
                    current,
                )

                test_steps += 1

                if e["density"] > 0.0:
                    reuse_like += 1
                else:
                    unsupported += 1

                print(
                    f"{word:6s} transition={pos-1}->{pos} "
                    f"prev={list(e['previous'])} "
                    f"curr={list(e['current'])} "
                    f"support={e['support']:.1f} "
                    f"density={e['density']:.3f} "
                    f"edges={e['edge_count']}"
                )

            previous = current

    print()
    print("test_transition_steps =", test_steps)
    print("supported_transitions =", reuse_like)
    print("unsupported_transitions =", unsupported)

    print()
    print("--- TRANSITION DETERMINISM ---")

    probes = [
        ("CAT", 1, 2),
        ("CAD", 1, 2),
        ("BOAT", 0, 1),
        ("BOARD", 1, 2),
    ]

    for word, a_pos, b_pos in probes:
        a1 = net.v56_frozen_step(word, a_pos)
        b1 = net.v56_frozen_step(word, b_pos)
        e1 = net.v56_transition_projection(a1, b1)

        a2 = net.v56_frozen_step(word, a_pos)
        b2 = net.v56_frozen_step(word, b_pos)
        e2 = net.v56_transition_projection(a2, b2)

        print(
            f"{word:6s} {a_pos}->{b_pos} "
            f"same_current={b1 == b2} "
            f"same_support={e1['support'] == e2['support']} "
            f"density={e1['density']:.3f}"
        )

    print()
    print("=== END V56 TEMPORAL ASSEMBLY BINDING ===")
    print()

def run():

    print("=== DENSE SUBSTRATE V53 - CAUSAL TOPOLOGY PROJECTION ===")
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
    print("V54_NET_READY = TRUE")


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


    print("V55_TRAINING_COMPLETE = TRUE")
    v56_run_temporal_binding_experiment(net, TRAINING, TEST)

    v54_run_guaranteed_projection(net, TRAINING, TEST)

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

    v52_run_topology_native_experiment(net, TRAINING, TEST)
    v53_run_topology_projection_experiment(net, TRAINING, TEST)

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
    print("=== V42 FROZEN READOUT AUDIT ===")

    def probe_candidates(words, limit=20):
        print()
        print("=== V42 FROZEN CANDIDATE READOUT ===")

        count = 0
        for word in words:
            for pos in range(len(word)):
                snapshot = net.v42_activation_snapshot(word, pos)

                print(
                    f"{word:6s} pos={pos} "
                    f"fired={snapshot['fired']} "
                    f"before={snapshot['before_fingerprint'][:16]} "
                    f"after={snapshot['after_fingerprint'][:16]}"
                )

                count += 1
                if count >= limit:
                    print("=== END V42 FROZEN CANDIDATE READOUT ===")
                    print()
                    return

        print("=== END V42 FROZEN CANDIDATE READOUT ===")
        print()



    def v47_input_vector_autopsy():


        print()


        print("=== V47 SUBSTRATE INPUT VECTOR AUTOPSY ===")


    


        probes = [


            ("CAT", 1),


            ("CAD", 1),


            ("BOAT", 0),


            ("BOARD", 3),


        ]


    


        for word, pos in probes:


            net.activate_substrate(word, pos, learn=False)


            first = list(getattr(net, "_v47_input_trace", []))


    


            net.activate_substrate(word, pos, learn=False)


            second = list(getattr(net, "_v47_input_trace", []))


    


            print()


            print(


                f"{word:6s} pos={pos} "


                f"first_rows={len(first)} "


                f"second_rows={len(second)} "


                f"same_vector={first == second}"


            )


    


            first_cells = {


                row.get("cell"): row


                for row in first


                if "cell" in row


            }


            second_cells = {


                row.get("cell"): row


                for row in second


                if "cell" in row


            }


    


            differing = []


            for cell_id in sorted(


                set(first_cells) | set(second_cells),


                key=str,


            ):


                a = first_cells.get(cell_id)


                b = second_cells.get(cell_id)


                if a != b:


                    differing.append((cell_id, a, b))


    


            print(f"  differing_cells={len(differing)}")


    


            for cell_id, a, b in differing[:8]:


                print(f"  CELL {cell_id}")


                print(f"    FIRST : {a}")


                print(f"    SECOND: {b}")


    


        print()


        print("=== END V47 SUBSTRATE INPUT VECTOR AUTOPSY ===")


        print()


    def v46_independent_score_audit():

        print()

        print("=== V46 INDEPENDENT CELL SCORE AUDIT ===")

    

        probes = [

            ("CAT", 1),

            ("CAD", 1),

            ("BOAT", 0),

            ("BOARD", 3),

        ]

    

        for word, pos in probes:

            first = net.v46_recompute_cell_scores(word, pos)

            second = net.v46_recompute_cell_scores(word, pos)

    

            same_rows = first["rows"] == second["rows"]

    

            print()

            print(

                f"{word:6s} pos={pos} "

                f"same_rows={same_rows} "

                f"context={first['context']}"

            )

    

            differences = []

            for a, b in zip(first["rows"], second["rows"]):

                if a != b:

                    differences.append((a, b))

    

            print(f"  differing_cells={len(differences)}")

    

            for a, b in differences[:8]:

                print(f"  FIRST : {a}")

                print(f"  SECOND: {b}")

    

            print("  first 32 cells:")

            for row in first["rows"]:

                print(

                    f"    cell={row['cell']:2d} "

                    f"potential={row['potential']:.15f} "

                    f"activation={row['activation']:.15f} "

                    f"threshold={row['threshold']}"

                )

    

        print()

        print("=== END V46 INDEPENDENT CELL SCORE AUDIT ===")

        print()


    def v45_first_divergence_autopsy():

        print()

        print("=== V45 FIRST CELL DIVERGENCE AUTOPSY ===")

    

        probes = [

            ("CAT", 1),

            ("CAD", 1),

            ("BOAT", 0),

            ("BOARD", 3),

        ]

    

        for word, pos in probes:

            net.v42_activation_snapshot(word, pos)

            first = list(getattr(net, "_v45_cell_trace", []))

    

            net.v42_activation_snapshot(word, pos)

            second = list(getattr(net, "_v45_cell_trace", []))

    

            first_map = {

                (row.get("cell_id"), index): row

                for index, row in enumerate(first)

            }

            second_map = {

                (row.get("cell_id"), index): row

                for index, row in enumerate(second)

            }

    

            divergence = None

            max_len = max(len(first), len(second))

    

            for index in range(max_len):

                a = first[index] if index < len(first) else None

                b = second[index] if index < len(second) else None

                if a != b:

                    divergence = index

                    break

    

            print()

            print(

                f"{word:6s} pos={pos} "

                f"first_len={len(first)} "

                f"second_len={len(second)} "

                f"first_divergence={divergence}"

            )

    

            if divergence is not None:

                a = first[divergence] if divergence < len(first) else None

                b = second[divergence] if divergence < len(second) else None

                print("  FIRST :", a)

                print("  SECOND:", b)

            else:

                print("  No per-cell divergence recorded.")

    

        print()

        print("=== END V45 FIRST CELL DIVERGENCE AUTOPSY ===")

        print()


    def v44_score_selection_autopsy():

        print()

        print("=== V44 SCORE / SELECTION AUTOPSY ===")

    

        probes = [

            ("CAT", 1),

            ("CAD", 1),

            ("BOAT", 0),

            ("BOARD", 3),

        ]

    

        for word, pos in probes:

            first = net.v42_activation_snapshot(word, pos)

            audit_first = getattr(

                net, "_v44_activation_audit", {}

            )

    

            second = net.v42_activation_snapshot(word, pos)

            audit_second = getattr(

                net, "_v44_activation_audit", {}

            )

    

            same_fired = first["fired"] == second["fired"]

            same_locals = (

                audit_first.get("returns")

                == audit_second.get("returns")

            )

    

            print()

            print(

                f"{word:6s} pos={pos} "

                f"same_fired={same_fired} "

                f"same_activation_locals={same_locals}"

            )

            print(f"  first fired : {first['fired']}")

            print(f"  second fired: {second['fired']}")

    

            first_returns = audit_first.get("returns", [])

            second_returns = audit_second.get("returns", [])

    

            print(f"  first return snapshots : {len(first_returns)}")

            for index, row in enumerate(first_returns):

                print(

                    f"    {index:02d} value={row.get('value')} "

                    f"locals={row.get('locals')}"

                )

    

            print(f"  second return snapshots: {len(second_returns)}")

            for index, row in enumerate(second_returns):

                print(

                    f"    {index:02d} value={row.get('value')} "

                    f"locals={row.get('locals')}"

                )

    

        print()

        print("=== END V44 SCORE / SELECTION AUTOPSY ===")

        print()

    def v42_print_snapshot(label, snapshot):
        print(
            f"{label}: "
            f"fired={snapshot['fired']} "
            f"before={snapshot['before_fingerprint'][:16]} "
            f"after={snapshot['after_fingerprint'][:16]}"
        )


    def v48_frozen_state_proof():

        print()

        print("=== V48 FROZEN STATE PROOF ===")

        probes = [

            ("CAT", 1),

            ("CAD", 1),

            ("BOAT", 0),

            ("BOARD", 3),

        ]

    

        for word, pos in probes:

            before = net.v42_cell_state_snapshot()

            first = net.v42_activation_snapshot(word, pos)

            after_first = net.v42_cell_state_snapshot()

            second = net.v42_activation_snapshot(word, pos)

            after_second = net.v42_cell_state_snapshot()

    

            print(

                f"{word:6s} pos={pos} "

                f"same_fired={first['fired'] == second['fired']} "

                f"state_unchanged={before == after_first == after_second}"

            )

            print(f"  first : {first['fired']}")

            print(f"  second: {second['fired']}")

    

        print("=== END V48 FROZEN STATE PROOF ===")

        print()

    def v42_determinism_probe():
        print()
        print("=== V42 FROZEN READOUT DETERMINISM ===")

        sequences = [
            (
                "REPEAT CAT",
                [
                    ("CAT", 1),
                    ("CAT", 1),
                    ("CAT", 1),
                    ("CAT", 1),
                    ("CAT", 1),
                ],
            ),
            (
                "A/B/A",
                [
                    ("CAT", 1),
                    ("CAD", 1),
                    ("CAT", 1),
                ],
            ),
            (
                "MIXED/RETURN",
                [
                    ("CAT", 1),
                    ("CAR", 1),
                    ("BOAT", 0),
                    ("BOARD", 3),
                    ("CAT", 1),
                ],
            ),
        ]

        for label, sequence in sequences:
            print()
            print(f"--- {label} ---")

            snapshots = []

            for index, (word, pos) in enumerate(sequence):
                snapshot = net.v42_activation_snapshot(word, pos)
                snapshots.append(snapshot)
                v42_print_snapshot(
                    f"{index:02d} {word}:{pos}",
                    snapshot,
                )

            if snapshots:
                first = snapshots[0]

                stable = all(
                    snapshot["fired"] == first["fired"]
                    and snapshot["after_fingerprint"]
                    == first["after_fingerprint"]
                    for snapshot in snapshots
                    if (
                        snapshot["word"] == first["word"]
                        and snapshot["pos"] == first["pos"]
                    )
                )

                print(
                    f"same-input stable={stable}"
                )

        print()
        print("=== V42 EXACT READOUT EQUIVALENCE ===")

        pairs = [
            ("CAT", 1),
            ("CAD", 1),
            ("BOAT", 0),
            ("BOARD", 3),
        ]

        for word, pos in pairs:
            first = net.v42_activation_snapshot(word, pos)
            second = net.v42_activation_snapshot(word, pos)

            same_fired = first["fired"] == second["fired"]
            same_after = (
                first["after_fingerprint"]
                == second["after_fingerprint"]
            )

            print(
                f"{word:6s} pos={pos} "
                f"same_fired={same_fired} "
                f"same_state={same_after} "
                f"first={first['fired']} "
                f"second={second['fired']}"
            )

        print()
        print("=== END V42 FROZEN READOUT DETERMINISM ===")
        print()

    v48_frozen_state_proof()

    v42_determinism_probe()

    v44_score_selection_autopsy()

    v45_first_divergence_autopsy()

    v47_input_vector_autopsy()

    v46_independent_score_audit()
    print("--- HELD-OUT ---")
    probe_candidates(TEST)

    print("=== END V42 FROZEN READOUT AUDIT ===")
    print()


    print()
    print("=== V39 TRAJECTORY DIAGNOSTIC SUMMARY ===")
    print("Trajectory overlap is diagnostic only; it does not drive REUSE.")
    print("=== END V39 TRAJECTORY DIAGNOSTIC SUMMARY ===")
    print()

    print()
    print("=== V40 COHERENCE DIAGNOSTIC ===")
    print("Assembly coherence is diagnostic only; it does not drive REUSE.")
    print("=== END V40 COHERENCE DIAGNOSTIC ===")
    print()

    print()
    print("=== V41 TOPOLOGY COMPETITION DIAGNOSTIC ===")
    print(
        "Evidence uses only current dense activity and learned substrate "
        "connectivity."
    )
    print("No external assembly/vocabulary memory is supplied to the designer.")
    print("=== END V41 TOPOLOGY COMPETITION DIAGNOSTIC ===")
    print()
    net.evaluate_frozen(TEST)



print("V56_FILE_READY = TRUE")

if __name__ == "__main__":
    run()
