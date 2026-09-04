"""Sequential bounded evidence environment; oracle metadata never enters observations."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import sqlite3

from attention_reward import AttentionRewardOracle
from attention_types import AttentionAction, AttentionActionKind, AttentionObservation, CandidateFeatures


def candidate(relation, target, **values):
    return CandidateFeatures(relation, target, **values)


def benchmark_episodes():
    episodes = [
        {
            "episode_id": "train_dog_tail", "split": "ordinary", "goal": "has_part",
            "terms": ["tail"], "start": "en:dog", "proof_target": "en:tail",
            "nodes": {
                "en:dog": [
                    candidate("related_to", "en:animal", specificity=.15, lexical_score=.95),
                    candidate("has_part", "en:tail", goal_relation_match=1, target_term_match=1,
                              specificity=1, lexical_score=.55, provenance=1),
                    candidate("is_a", "en:mammal", specificity=1, lexical_score=.4),
                ],
                "en:animal": [candidate("related_to", "en:dog", specificity=.15, lexical_score=.2)],
                "en:tail": [candidate("has_part", "en:tail", goal_relation_match=1,
                                      target_term_match=1, specificity=1, provenance=1,
                                      verified=1, direct_proof=1, already_visited=1)],
                "en:mammal": [candidate("has_part", "en:fur", goal_relation_match=1, specificity=1)],
            },
            "proof_edges": [["en:dog", "en:tail"]],
        },
        {
            "episode_id": "train_car_wheel", "split": "ordinary", "goal": "has_part",
            "terms": ["wheel"], "start": "en:car", "proof_target": "en:wheel",
            "nodes": {
                "en:car": [candidate("related_to", "en:road", specificity=.15, lexical_score=.8),
                            candidate("has_part", "en:wheel", goal_relation_match=1, target_term_match=1,
                                      specificity=1, lexical_score=.5, provenance=1)],
                "en:road": [candidate("at_location", "en:car", specificity=1)],
                "en:wheel": [candidate("has_part", "en:wheel", goal_relation_match=1,
                                        target_term_match=1, specificity=1, provenance=1,
                                        verified=1, direct_proof=1, already_visited=1)],
            },
            "proof_edges": [["en:car", "en:wheel"]],
        },
        {
            "episode_id": "train_authority_over_worker", "split": "ordinary", "goal": "has_part",
            "terms": ["tail"], "start": "en:fox", "proof_target": "en:tail",
            "nodes": {
                "en:fox": [
                    candidate("related_to", "en:tail", specificity=.15, lexical_score=1, provenance=.1),
                    candidate("has_part", "en:tail", goal_relation_match=1, target_term_match=1,
                              specificity=1, lexical_score=.3, provenance=1),
                    candidate("has_part", "en:brush", goal_relation_match=1, specificity=1, provenance=1),
                ],
                "en:tail": [candidate("has_part", "en:tail", goal_relation_match=1,
                                       target_term_match=1, specificity=1, provenance=1, verified=1)],
                "en:brush": [candidate("related_to", "en:fox", specificity=.15)],
            },
            "proof_edges": [["en:fox", "en:tail"]],
        },
        {
            "episode_id": "train_direct_over_indirect", "split": "ordinary", "goal": "has_part",
            "terms": ["wheel"], "start": "en:scooter", "proof_target": "en:wheel",
            "nodes": {
                "en:scooter": [
                    candidate("related_to", "en:vehicle", specificity=.15, lexical_score=.9),
                    candidate("has_part", "en:wheel", goal_relation_match=1, target_term_match=1,
                              specificity=1, lexical_score=.4, provenance=1),
                ],
                "en:vehicle": [candidate("has_part", "en:wheel", goal_relation_match=1,
                                          target_term_match=1, specificity=1, provenance=.5)],
                "en:wheel": [candidate("has_part", "en:wheel", goal_relation_match=1,
                                        target_term_match=1, specificity=1, provenance=1, verified=1)],
            },
            "proof_edges": [["en:scooter", "en:wheel"]],
        },
        {
            "episode_id": "held_out_wolf_tail", "split": "held_out_structural",
            "goal": "has_part", "terms": ["tail"], "start": "en:wolf", "proof_target": "en:tail",
            "nodes": {
                "en:wolf": [candidate("related_to", "en:pack", specificity=.15, lexical_score=.95),
                            candidate("has_part", "en:tail", goal_relation_match=1, target_term_match=1,
                                      specificity=1, lexical_score=.45, provenance=1)],
                "en:pack": [candidate("related_to", "en:wolf", specificity=.15)],
                "en:tail": [candidate("has_part", "en:tail", goal_relation_match=1,
                                       target_term_match=1, specificity=1, provenance=1,
                                       verified=1, direct_proof=1, already_visited=1)],
            },
            "proof_edges": [["en:wolf", "en:tail"]],
        },
        {
            "episode_id": "held_out_bicycle_wheel", "split": "held_out_structural",
            "goal": "has_part", "terms": ["wheel"], "start": "en:bicycle", "proof_target": "en:wheel",
            "nodes": {
                "en:bicycle": [candidate("related_to", "en:vehicle", specificity=.15, lexical_score=.9),
                               candidate("has_part", "en:wheel", goal_relation_match=1, target_term_match=1,
                                         specificity=1, lexical_score=.4, provenance=1)],
                "en:vehicle": [candidate("related_to", "en:road", specificity=.15)],
                "en:wheel": [candidate("has_part", "en:wheel", goal_relation_match=1,
                                         target_term_match=1, specificity=1, provenance=1,
                                         verified=1, direct_proof=1, already_visited=1)],
            },
            "proof_edges": [["en:bicycle", "en:wheel"]],
        },
        {
            "episode_id": "adversarial_no_proof", "split": "adversarial", "goal": "has_part",
            "terms": ["wing"], "start": "en:dog", "proof_target": None,
            "nodes": {
                "en:dog": [
                    candidate("related_to", "en:wing", specificity=.15, lexical_score=1),
                    candidate("has_part", "en:tail", goal_relation_match=1, specificity=1, lexical_score=.8),
                    candidate("is_a", "en:bird", specificity=1, lexical_score=.7, contradiction=1),
                    candidate("has_part", "en:fur", goal_relation_match=1, specificity=1, lexical_score=.2),
                    candidate("has_property", "en:winged", specificity=1, lexical_score=1),
                ],
                "en:wing": [candidate("related_to", "en:feather", specificity=.15)],
                "en:tail": [candidate("has_part", "en:tail", goal_relation_match=1, specificity=1)],
                "en:bird": [candidate("has_part", "en:wing", goal_relation_match=1, specificity=1)],
                "en:fur": [candidate("has_part", "en:fur", goal_relation_match=1, specificity=1)],
                "en:winged": [candidate("has_property", "en:winged", specificity=1)],
            },
            "proof_edges": [],
        },
    ]
    return episodes + adversarial_benchmark_episodes()


def _adversarial_episode(category, partition, index, valid_stop=False):
    """Create disjoint graph shapes; proof membership remains environment-private."""
    entities = {
        "train": [("en:otter", "en:fin"), ("en:badger", "en:claw")],
        "validation": [("en:lynx", "en:whisker"), ("en:yak", "en:horn")],
        "heldout": [("en:gecko", "en:scale"), ("en:orca", "en:flipper")],
    }[partition]
    source, term = entities[index % len(entities)]
    split = "held_out_adversarial" if partition == "heldout" else "adversarial"
    goal = "has_part"
    trap_relation = "related_to" if category == "related_to_association" else "has_part"
    candidates = [
        candidate(trap_relation, term, specificity=.15 if trap_relation == "related_to" else 1,
                  lexical_score=1, provenance=.1),
        candidate("has_part", f"en:distractor_{category}_{index}", goal_relation_match=1,
                  specificity=1, lexical_score=.35, provenance=.4),
        candidate("is_a", f"en:branch_{category}_{index}", specificity=1,
                  contradiction=float(category == "contradiction")),
    ]
    if category == "wrong_relation":
        candidates[0] = candidate("at_location", term, specificity=1, lexical_score=1)
    elif category == "wrong_subject":
        candidates[0] = candidate("has_part", f"en:other_{term.removeprefix('en:')}",
                                  goal_relation_match=1, specificity=1, lexical_score=.7)
    elif category == "invalid_longer_path":
        candidates.append(candidate("related_to", f"en:dead_end_{index}", specificity=.15, lexical_score=.8,
                                    path_length=3))
    elif category == "redundant_path":
        candidates[0] = candidate("has_part", f"en:visited_{index}", goal_relation_match=1,
                                  specificity=1, lexical_score=.5, already_visited=1)
    elif category == "worker_only":
        candidates[0] = candidate("has_part", term, goal_relation_match=1, target_term_match=1,
                                  specificity=1, lexical_score=.8, provenance=.05)
    elif category == "plausible_unsupported":
        candidates.append(candidate("has_property", f"en:likely_{index}", specificity=1, lexical_score=.7))
    elif category == "direct_vs_indirect":
        candidates.append(candidate("related_to", f"en:route_{index}", specificity=.15, lexical_score=.9))
    target = f"en:proof_{term.removeprefix('en:')}" if valid_stop else None
    if valid_stop:
        candidates[1] = candidate("has_part", target, goal_relation_match=1, target_term_match=1,
                                  specificity=1, lexical_score=.5, provenance=1)
    nodes = {
        source: candidates,
        term: [candidate("related_to", f"en:dead_end_{index}", specificity=.15)],
        candidates[1].target: [candidate("related_to", source, specificity=.15)],
        candidates[2].target: [candidate("related_to", source, specificity=.15)],
    }
    if valid_stop:
        nodes[target] = [candidate("has_part", target, goal_relation_match=1, target_term_match=1,
                                   specificity=1, provenance=1, verified=1)]
    return {
        "episode_id": f"{partition}_{category}_{index}", "split": split, "partition": partition,
        "category": category, "no_proof": not valid_stop, "goal": goal,
        "terms": [term.removeprefix("en:")], "start": source, "proof_target": target,
        "nodes": nodes, "proof_edges": [[source, target]] if target else [],
    }


def adversarial_benchmark_episodes():
    """Train/validation/held-out adversarial graphs, with separate no-proof partitions."""
    categories = (
        "no_valid_proof", "lexical_target_trap", "wrong_subject", "wrong_relation",
        "related_to_association", "plausible_unsupported", "contradiction",
        "invalid_longer_path", "worker_only", "direct_vs_indirect", "redundant_path",
        "valid_abstain", "valid_stop",
    )
    episodes = []
    for partition in ("train", "validation", "heldout"):
        for index, category in enumerate(categories):
            episodes.append(_adversarial_episode(category, partition, index,
                                                  valid_stop=category == "valid_stop"))
    return episodes


def no_proof_episodes(partition=None):
    episodes = [episode for episode in adversarial_benchmark_episodes() if episode["no_proof"]]
    return [episode for episode in episodes if episode["partition"] == partition] if partition else episodes


def episodes_from_database(database, limit=32):
    """Create one-step frozen-graph episodes; no database truth enters policy input."""
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        facts = connection.execute(
            "SELECT subject,relation,object FROM edges ORDER BY subject,relation,object LIMIT ?",
            (int(limit),),
        ).fetchall()
        output = []
        for index, (subject, relation, target) in enumerate(facts):
            rows = connection.execute(
                "SELECT relation,object FROM edges WHERE subject=? ORDER BY relation,object LIMIT 8",
                (subject,),
            ).fetchall()
            nodes = {str(subject): [
                candidate(
                    edge_relation, edge_target,
                    goal_relation_match=float(edge_relation == relation),
                    target_term_match=float(edge_target == target),
                    specificity=.15 if edge_relation == "related_to" else 1.0,
                    lexical_score=float(edge_target == target),
                    provenance=float(edge_relation == relation),
                )
                for edge_relation, edge_target in rows
            ], str(target): [
                candidate(str(relation), str(target), goal_relation_match=1,
                          target_term_match=1, specificity=1, provenance=1,
                          verified=1, direct_proof=1, already_visited=1)
            ]}
            output.append({
                "episode_id": f"graph_{index}", "split": "ordinary", "goal": str(relation),
                "terms": [str(target).removeprefix("en:")], "start": str(subject),
                "proof_target": str(target), "nodes": nodes,
                "proof_edges": [[str(subject), str(target)]],
            })
        return output
    finally:
        connection.close()


class AttentionEnv:
    def __init__(self, episode, budget=4, oracle=None):
        self.spec = deepcopy(episode)
        self.budget = int(budget)
        self.oracle = oracle or AttentionRewardOracle()
        self.state = None
        self.done = False

    def _observation(self, node, step, remaining, visited_nodes, visited_relations, history, relation_activation, candidate_activation):
        candidates = [
            replace(item, verified=0.0, direct_proof=0.0,
                    already_visited=float(item.target in visited_nodes),
                    relation_activation=relation_activation.get(item.relation, 0.0),
                    candidate_activation=candidate_activation.get(item.target, 0.0))
            for item in self.spec["nodes"].get(node, [])
        ]
        return AttentionObservation(
            goal_relation=self.spec["goal"], goal_terms=list(self.spec["terms"]),
            current_focus=node, current_node=node, relation_features={},
            candidate_features=list(candidates), relation_activation=dict(relation_activation),
            candidate_activation=dict(candidate_activation), visited_nodes=list(visited_nodes),
            visited_relations=list(visited_relations), attention_history=list(history),
            step=step, remaining_budget=remaining,
        )

    def reset(self):
        self.done = False
        self.valid_proof_seen = False
        self.state = self._observation(self.spec["start"], 0, self.budget, [], [], [], {}, {})
        return self.state

    def available_actions(self):
        if self.done or self.state is None:
            return []
        return [*(AttentionAction(AttentionActionKind.TRAVERSE, index)
                  for index in range(len(self.state.candidate_features))),
                AttentionAction(AttentionActionKind.STOP), AttentionAction(AttentionActionKind.ABSTAIN)]

    def step(self, action):
        if action not in self.available_actions():
            raise ValueError("invalid or terminal attention action")
        state = self.state
        candidate_item = None
        if action.kind is AttentionActionKind.TRAVERSE:
            candidate_item = state.candidate_features[action.candidate_id]
            node = candidate_item.target
            visited_nodes = state.visited_nodes + [node]
            visited_relations = state.visited_relations + [candidate_item.relation]
            relation_activation = dict(state.relation_activation)
            candidate_activation = dict(state.candidate_activation)
            relation_activation[candidate_item.relation] = relation_activation.get(candidate_item.relation, 0.0) + .5
            candidate_activation[node] = candidate_activation.get(node, 0.0) + .5
            history = state.attention_history + [action.as_dict()]
            self.state = self._observation(node, state.step + 1, state.remaining_budget - 1,
                                           visited_nodes, visited_relations, history,
                                           relation_activation, candidate_activation)
            valid_edge = [state.current_node, node] in self.spec["proof_edges"]
            self.valid_proof_seen = self.valid_proof_seen or valid_edge
            if state.remaining_budget <= 1:
                outcome = "budget_exhausted"
            else:
                outcome = "continue"
        elif action.kind is AttentionActionKind.STOP:
            valid_edge = False
            outcome = "verified" if self.valid_proof_seen else "unsupported_stop"
        else:
            valid_edge = False
            outcome = "no_verified_evidence" if self.spec["proof_target"] is None else "false_abstain"
        self.done = outcome != "continue"
        oracle = {"terminal_outcome": outcome, "valid_proof_edge": valid_edge,
                  "already_visited": bool(candidate_item and candidate_item.already_visited)}
        reward = self.oracle.reward(action, oracle)
        return self.state, reward, self.done, oracle
