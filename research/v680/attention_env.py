"""Immutable bounded semantic-evidence environment with oracle-only evaluation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import sqlite3

from attention_reward import AttentionRewardOracle
from attention_types import AttentionAction, AttentionActionKind, AttentionObservation


def _candidate(relation, target, **values):
    from attention_types import CandidateFeatures
    return CandidateFeatures(relation, target, **values)


def benchmark_episodes():
    return [
        {
            "episode_id": "ordinary_dog_tail", "split": "ordinary", "goal": "has_part",
            "terms": ["tail"], "start": "en:dog", "proof_target": "en:tail",
            "candidates": [
                _candidate("related_to", "en:animal", specificity=.15, lexical_score=.8),
                _candidate("has_part", "en:tail", goal_relation_match=1, target_term_match=1,
                           specificity=1, lexical_score=.6, provenance=1, verified=1, direct_proof=1),
            ],
        },
        {
            "episode_id": "ordinary_bicycle_wheel", "split": "held_out_structural",
            "goal": "has_part", "terms": ["wheel"], "start": "en:bicycle", "proof_target": "en:wheel",
            "candidates": [
                _candidate("related_to", "en:vehicle", specificity=.15, lexical_score=.8),
                _candidate("has_part", "en:wheel", goal_relation_match=1, target_term_match=1,
                           specificity=1, provenance=1, verified=1, direct_proof=1),
            ],
        },
        {
            "episode_id": "adversarial_no_proof", "split": "adversarial", "goal": "has_part",
            "terms": ["wing"], "start": "en:dog", "proof_target": None,
            "candidates": [
                _candidate("related_to", "en:wing", specificity=.15, lexical_score=1),
                _candidate("has_part", "en:tail", goal_relation_match=1, specificity=1, lexical_score=.8),
                _candidate("is_a", "en:bird", specificity=1, lexical_score=.7, contradiction=1),
                _candidate("has_part", "en:fur", goal_relation_match=1, specificity=1, lexical_score=.2),
                _candidate("has_property", "en:winged", specificity=1, lexical_score=1),
            ],
        },
    ]


def episodes_from_database(database, limit=32):
    """Derive direct-proof bounded attention episodes from a frozen graph SQLite DB."""
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        facts = connection.execute(
            """SELECT subject,relation,object FROM edges
               WHERE subject IS NOT NULL AND relation IS NOT NULL AND object IS NOT NULL
               ORDER BY subject,relation,object LIMIT ?""",
            (int(limit),),
        ).fetchall()
        episodes = []
        for index, (subject, relation, target) in enumerate(facts):
            neighbors = connection.execute(
                """SELECT relation,object FROM edges WHERE subject=?
                   ORDER BY relation,object LIMIT 8""", (subject,)
            ).fetchall()
            candidates = [
                _candidate(
                    edge_relation, edge_target,
                    goal_relation_match=float(edge_relation == relation),
                    target_term_match=float(edge_target == target),
                    specificity=.15 if edge_relation == "related_to" else 1.0,
                    lexical_score=float(edge_target == target),
                    provenance=float(edge_relation == relation and edge_target == target),
                    verified=float(edge_relation == relation and edge_target == target),
                    direct_proof=float(edge_relation == relation and edge_target == target),
                )
                for edge_relation, edge_target in neighbors
            ]
            if not any(candidate.verified for candidate in candidates):
                continue
            episodes.append({
                "episode_id": f"graph_{index}_{subject}_{relation}_{target}",
                "split": "ordinary", "goal": relation,
                "terms": [str(target).removeprefix("en:")], "start": subject,
                "proof_target": target, "candidates": candidates,
            })
        return episodes
    finally:
        connection.close()


class AttentionEnv:
    def __init__(self, episode, budget=3, oracle=None):
        self.spec = deepcopy(episode)
        self.budget = int(budget)
        self.oracle = oracle or AttentionRewardOracle()
        self.state = None
        self.done = False

    def reset(self):
        self.done = False
        self.state = AttentionObservation(
            goal_relation=self.spec["goal"], goal_terms=self.spec["terms"],
            current_focus=self.spec["start"], current_node=self.spec["start"],
            relation_features={}, candidate_features=list(self.spec["candidates"]),
            relation_activation={}, candidate_activation={}, visited_nodes=[],
            visited_relations=[], step=0, remaining_budget=self.budget,
        )
        return self.state

    def available_actions(self):
        if self.done or self.state is None:
            return []
        return [
            *(AttentionAction(AttentionActionKind.TRAVERSE, index)
              for index in range(len(self.state.candidate_features))),
            AttentionAction(AttentionActionKind.STOP),
            AttentionAction(AttentionActionKind.ABSTAIN),
        ]

    def step(self, action):
        if self.done or action not in self.available_actions():
            raise ValueError("invalid or terminal attention action")
        candidate = None
        proof = False
        if action.kind is AttentionActionKind.TRAVERSE:
            candidate = self.state.candidate_features[action.candidate_id]
            proof = bool(candidate.verified)
            self.state.visited_nodes.append(candidate.target)
            self.state.visited_relations.append(candidate.relation)
            self.state.current_focus = candidate.target
            self.state.current_node = candidate.target
            self.state.relation_activation[candidate.relation] = (
                self.state.relation_activation.get(candidate.relation, 0.0) + .5
            )
            self.state.candidate_activation[candidate.target] = (
                self.state.candidate_activation.get(candidate.target, 0.0) + .5
            )
            self.state.candidate_features = [
                replace(item, already_visited=float(item.target == candidate.target))
                for item in self.state.candidate_features
            ]
        self.state.step += 1
        self.state.remaining_budget -= 1
        if action.kind is AttentionActionKind.ABSTAIN:
            outcome = "no_verified_evidence" if self.spec["proof_target"] is None else "false_abstain"
        elif action.kind is AttentionActionKind.STOP:
            outcome = "verified" if any(item.verified for item in self.state.candidate_features) else "unsupported_stop"
        elif proof:
            # Observation now contains verified, inspected evidence; the teacher
            # must explicitly choose STOP at the next sequential attention state.
            outcome = "continue"
        elif self.state.remaining_budget <= 0:
            outcome = "budget_exhausted"
        else:
            outcome = "continue"
        self.done = outcome != "continue"
        transition = {
            "terminal_outcome": outcome, "valid_proof_edge": proof,
            "already_visited": bool(candidate and candidate.already_visited),
        }
        reward = self.oracle.reward(action, transition)
        return self.state, reward, self.done, transition
