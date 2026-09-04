# V681-owned learning implementation; derived from V680.
"""Versioned matched decision-boundary benchmark for V680.1."""
from __future__ import annotations

from .environment import candidate
from .types import DATASET_VERSION


DECISION_BOUNDARY_VERSION = "v680.1-matched-boundary-1"
CATEGORIES = (
    "immediate_proof", "one_step_useful", "multi_step_useful", "genuine_no_proof",
    "locally_attractive_irrelevant", "contradictory", "redundant", "premature_stop",
    "premature_abstain", "association_trap", "lexical_overlap_trap", "direct_vs_indirect",
)


def _partition(index, samples):
    return "train" if index < samples * .6 else ("validation" if index < samples * .8 else "heldout")


def _spec(category, index, samples):
    partition = _partition(index, samples)
    adversarial = category in {
        "genuine_no_proof", "locally_attractive_irrelevant", "contradictory", "redundant",
        "association_trap", "lexical_overlap_trap",
    }
    split = ("adversarial" if partition != "heldout" else "held_out_adversarial") if adversarial else (
        "ordinary" if partition == "train" else
        ("held_out_lexical" if partition == "validation" else "held_out_structural"))
    entity = f"en:entity_{category}_{partition}_{index}"
    term = f"en:goal_{index % 17}_{partition}"
    bridge = f"en:bridge_{category}_{index}"
    distractors = [
        candidate("related_to", f"en:association_{index}", specificity=.15, lexical_score=.95),
        candidate("has_part", f"en:distractor_{index}", goal_relation_match=1, specificity=1,
                  lexical_score=.25, provenance=.25),
        candidate("is_a", f"en:class_{index}", specificity=1),
    ][:1 + index % 3]
    nodes, proof_edges, initial_proof = {entity: list(distractors)}, [], False
    expected = "abstain"
    if category == "immediate_proof":
        entity = f"en:{term.removeprefix('en:')}_confirmed_{index}"
        nodes = {entity: list(distractors)}
        initial_proof, expected = True, "stop"
    elif category in {"one_step_useful", "premature_abstain", "direct_vs_indirect"}:
        useful = candidate("has_part", term, goal_relation_match=1, target_term_match=1,
                           specificity=1, lexical_score=.45, provenance=1)
        nodes[entity].insert(index % (len(nodes[entity]) + 1), useful)
        nodes[term] = [candidate("related_to", entity, specificity=.15)]
        proof_edges, expected = [[entity, term]], "traverse"
        if category == "direct_vs_indirect":
            nodes[entity].append(candidate("related_to", bridge, specificity=.15, lexical_score=.95,
                                           path_length=3))
            nodes[bridge] = [candidate("has_part", term, goal_relation_match=1,
                                       target_term_match=1, specificity=1, provenance=.5)]
    elif category == "multi_step_useful":
        nodes[entity].insert(1, candidate("has_part", bridge, goal_relation_match=1, target_term_match=.3,
                                          specificity=1, lexical_score=1, provenance=1, path_length=2))
        nodes[bridge] = [candidate("has_part", term, goal_relation_match=1, target_term_match=1,
                                   specificity=1, lexical_score=.45, provenance=1)]
        nodes[term] = [candidate("related_to", bridge, specificity=.15)]
        proof_edges, expected = [[entity, bridge], [bridge, term]], "traverse"
    elif category in {"genuine_no_proof", "locally_attractive_irrelevant", "association_trap",
                      "lexical_overlap_trap", "contradictory", "redundant"}:
        if category == "contradictory":
            nodes[entity][0] = candidate("is_a", term, specificity=1, lexical_score=1, contradiction=1)
        elif category == "redundant":
            nodes[entity][0] = candidate("has_part", f"en:visited_{index}", goal_relation_match=1,
                                          specificity=1, lexical_score=.6, already_visited=1)
        elif category == "lexical_overlap_trap":
            nodes[entity][0] = candidate("at_location", term, specificity=1, lexical_score=1)
        elif category == "locally_attractive_irrelevant":
            nodes[entity][0] = candidate("has_property", term, specificity=1, lexical_score=1)
    elif category == "premature_stop":
        nodes[entity].insert(0, candidate("has_part", bridge, goal_relation_match=1, target_term_match=.3,
                                          specificity=1, lexical_score=1, provenance=1))
        nodes[bridge] = [candidate("has_part", term, goal_relation_match=1, target_term_match=1,
                                   specificity=1, lexical_score=.45, provenance=1)]
        nodes[term] = [candidate("related_to", bridge, specificity=.15)]
        proof_edges, expected = [[entity, bridge], [bridge, term]], "traverse"
    return {
        "episode_id": f"boundary_{category}_{partition}_{index}", "split": split,
        "partition": partition, "category": category, "no_proof": not proof_edges and not initial_proof,
        "matched_triplet": f"{partition}_{index}", "goal": "has_part",
        "terms": [term.removeprefix("en:")], "start": entity, "proof_target": term if proof_edges else None,
        "nodes": nodes, "proof_edges": proof_edges, "initial_proof": initial_proof,
        "expected_initial_action": expected, "dataset_version": DATASET_VERSION,
        "benchmark_version": DECISION_BOUNDARY_VERSION,
    }


def decision_boundary_episodes(samples_per_category=100, partition=None):
    """1200 matched cases by default; each category has train/validation/held-out forms."""
    episodes = [_spec(category, index, samples_per_category)
                for category in CATEGORIES for index in range(samples_per_category)]
    return [episode for episode in episodes if partition is None or episode["partition"] == partition]
