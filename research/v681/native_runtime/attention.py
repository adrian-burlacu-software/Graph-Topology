# V681-owned runtime implementation; derived from V679.
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import re


@dataclass
class AttentionState:
    goal: str | None = None
    current_focus: str | None = None
    visited_nodes: set[str] = field(default_factory=set)
    visited_relations: set[str] = field(default_factory=set)
    relation_activation: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    candidate_activation: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    decay: float = 0.65
    step: int = 0
    history: list[dict] = field(default_factory=list)

    def begin_turn(self, goal, focus):
        self.relation_activation = self._decayed(self.relation_activation)
        self.candidate_activation = self._decayed(self.candidate_activation)
        self.goal = str(goal or "")
        self.current_focus = str(focus or "")
        self.step += 1
        self._record("begin_turn", goal=self.goal, focus=self.current_focus)

    def focus_hypothesis(self, goal, focus):
        self.goal = str(goal or "")
        self.current_focus = str(focus or "")
        self._record(
            "focus_hypothesis", goal=self.goal, focus=self.current_focus
        )

    def visit(self, node, relation=None, target=None):
        self.current_focus = str(node)
        self.visited_nodes.add(str(node))
        if relation:
            self.visited_relations.add(str(relation))
        self.step += 1
        self._record(
            "visit",
            node=str(node),
            relation=str(relation or ""),
            target=str(target or ""),
        )

    def reinforce(self, relation, candidate, value, reason):
        if relation:
            key = str(relation)
            self.relation_activation[key] = (
                self.relation_activation.get(key, 0.0) + float(value)
            )
        if candidate:
            key = str(candidate)
            self.candidate_activation[key] = (
                self.candidate_activation.get(key, 0.0) + float(value)
            )
        self._record(
            "reinforce",
            relation=str(relation or ""),
            candidate=str(candidate or ""),
            value=round(float(value), 6),
            reason=reason,
        )

    def snapshot(self):
        return {
            "goal": self.goal,
            "current_focus": self.current_focus,
            "visited_nodes": sorted(self.visited_nodes),
            "visited_relations": sorted(self.visited_relations),
            "relation_activation": self._sorted_activation(
                self.relation_activation
            ),
            "candidate_activation": self._sorted_activation(
                self.candidate_activation
            ),
            "decay": self.decay,
            "step": self.step,
            "history": list(self.history),
        }

    def _record(self, event, **payload):
        self.history.append({"step": self.step, "event": event, **payload})
        self.history = self.history[-128:]

    def _decayed(self, values):
        return {
            key: value * self.decay
            for key, value in values.items()
            if value * self.decay >= 0.0001
        }

    @staticmethod
    def _sorted_activation(values):
        return {
            key: round(value, 6)
            for key, value in sorted(
                values.items(), key=lambda item: (-item[1], item[0])
            )
        }


class HandCodedAttentionPolicy:
    """The initial replaceable policy; all fixed scoring lives here."""

    @staticmethod
    def relation_specificity(relation):
        return 0.15 if str(relation) == "related_to" else 1.0

    def score_hypothesis(self, features):
        return (
            2.0 * features["lexical"]
            + features["specificity"]
            + features["relation_activation"]
        )

    def score_traversal(self, features):
        return (
            3.0 * features["goal_relation_match"]
            + features["relation_activation"]
            + 0.5 * features["target_term_match"]
            + 0.25 * features["specificity"]
            + 0.25 * features["candidate_activation"]
        )

    def score_arbitration(self, features):
        return (
            4.0 * features["support"]
            - 2.0 * features["contradiction"]
            + 1.5 * features["specificity"]
            + features["provenance"]
            + 0.5 * features["lexical"]
        )


class DistilledAttentionPolicy(HandCodedAttentionPolicy):
    """A deterministic learned-policy seam fitted from teacher policy traces."""

    def __init__(self, relation_bias=None):
        self.relation_bias = dict(relation_bias or {})

    @classmethod
    def fit(cls, teacher_examples):
        totals = defaultdict(float)
        counts = defaultdict(int)
        for example in teacher_examples:
            relation = str(example.get("relation", ""))
            target_score = float(example.get("teacher_score", 0.0))
            baseline = float(example.get("baseline_score", 0.0))
            totals[relation] += target_score - baseline
            counts[relation] += 1
        return cls({
            relation: totals[relation] / counts[relation]
            for relation in totals
            if counts[relation]
        })

    def score_traversal(self, features):
        return super().score_traversal(features) + self.relation_bias.get(
            str(features["relation"]), 0.0
        )

    def save(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {"policy": type(self).__name__, "relation_bias": self.relation_bias},
                handle,
                ensure_ascii=False,
                sort_keys=True,
            )

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(payload.get("relation_bias", {}))


class EvidenceArbitrator:
    def __init__(self, policy):
        self.policy = policy

    def decide(self, ranked):
        candidates = []
        for index, (_, hypothesis, result) in enumerate(ranked):
            verified = bool(result.get("success", False))
            direct = bool(result.get("direct_proof", False))
            target_terms = list(hypothesis.evidence.get("target_terms", []) or [])
            features = {
                "support": float(verified),
                "contradiction": (
                    1.0
                    if hypothesis.evidence.get("argument_unverified", False)
                    else (0.25 if not verified else 0.0)
                ),
                "specificity": min(
                    1.0,
                    self.policy.relation_specificity(hypothesis.relation)
                    + (0.15 if target_terms and verified else 0.0),
                ),
                "provenance": 1.0 if direct else (0.8 if verified else 0.0),
                "lexical": min(1.0, float(hypothesis.lexical_score)),
            }
            candidates.append({
                "candidate_index": index,
                "subject": hypothesis.subject,
                "relation": hypothesis.relation,
                "intent": hypothesis.intent,
                "path": list(result.get("path", [])),
                "target": result.get("target"),
                "verified": verified,
                "features": {key: round(value, 6) for key, value in features.items()},
                "arbitration_score": round(
                    self.policy.score_arbitration(features), 6
                ),
            })
        candidates.sort(
            key=lambda item: (
                -item["arbitration_score"],
                -int(item["verified"]),
                item["relation"],
                item["subject"] or "",
            )
        )
        if not candidates or not candidates[0]["verified"]:
            return {
                "decision_type": "semantic_evidence_arbitration",
                "selected_candidate_index": None,
                "outcome": "abstain",
                "reason": "no_verified_evidence",
                "candidates": candidates,
            }
        return {
            "decision_type": "semantic_evidence_arbitration",
            "selected_candidate_index": candidates[0]["candidate_index"],
            "outcome": "verified",
            "reason": "verified_evidence_available",
            "candidates": candidates,
        }


class AttentionController:
    """Coordinates temporal state, replaceable scoring policy, and arbitration."""

    def __init__(self, policy=None, state=None, max_trace_targets=256):
        self.policy = policy or HandCodedAttentionPolicy()
        self.state = state or AttentionState()
        self.arbitrator = EvidenceArbitrator(self.policy)
        self.max_trace_targets = max(1, int(max_trace_targets))
        self.traversal_targets = []
        self.policy_examples = []

    def begin_turn(self, focus):
        self.state.begin_turn("semantic_turn", focus)

    def begin_hypothesis(self, hypothesis):
        self.state.focus_hypothesis(hypothesis.relation, hypothesis.subject)

    def prioritize_hypotheses(self, hypotheses):
        scored = []
        for hypothesis in hypotheses:
            features = {
                "lexical": min(1.0, float(hypothesis.lexical_score)),
                "specificity": self.policy.relation_specificity(
                    hypothesis.relation
                ),
                "relation_activation": self.state.relation_activation.get(
                    hypothesis.relation, 0.0
                ),
            }
            scored.append((self.policy.score_hypothesis(features), hypothesis))
        return [
            hypothesis
            for _, hypothesis in sorted(
                scored,
                key=lambda item: (
                    -item[0],
                    item[1].relation,
                    item[1].subject or "",
                ),
            )
        ]

    def record_visit(self, node):
        self.state.visit(node)

    def record_direct_proof(self, hypothesis, target):
        if target is None:
            return
        self.state.visit(hypothesis.subject, hypothesis.relation, target)
        self.state.reinforce(
            hypothesis.relation, target, 1.0, "direct_graph_proof"
        )

    def record_traversal_target(self, relation, target):
        self.state.visit(self.state.current_focus or "", relation, target)
        self.state.reinforce(
            relation, target, 0.1, "selected_traversal_target"
        )

    def select_traversal_targets(self, hypothesis, prefix, edges):
        query_terms = set(re.findall(
            r"[a-z0-9]+",
            " ".join(str(value) for value in hypothesis.evidence.get(
                "target_terms", []
            )).lower(),
        ))
        targets = []
        for edge in edges:
            features = {
                "relation": edge.relation,
                "goal_relation_match": float(edge.relation == hypothesis.relation),
                "relation_activation": self.state.relation_activation.get(
                    edge.relation, 0.0
                ),
                "candidate_activation": self.state.candidate_activation.get(
                    edge.object, 0.0
                ),
                "target_term_match": float(bool(query_terms) and bool(
                    query_terms & set(re.findall(
                        r"[a-z0-9]+", str(edge.object).lower()
                    ))
                )),
                "specificity": self.policy.relation_specificity(edge.relation),
            }
            score = self.policy.score_traversal(features)
            self.policy_examples.append({
                "relation": edge.relation,
                "teacher_score": score,
                "baseline_score": HandCodedAttentionPolicy().score_traversal(
                    features
                ),
            })
            targets.append({
                "from": prefix[-1] if prefix else hypothesis.subject,
                "path_prefix": list(prefix),
                "relation": edge.relation,
                "target": edge.object,
                "features": {
                    key: round(value, 6)
                    for key, value in features.items()
                    if key != "relation"
                },
                "score": round(score, 6),
            })
        targets.sort(key=lambda item: (-item["score"], item["relation"], item["target"]))
        if len(self.traversal_targets) < self.max_trace_targets:
            self.traversal_targets.extend(
                targets[:self.max_trace_targets - len(self.traversal_targets)]
            )
        score_by_key = {
            (item["relation"], item["target"]): item["score"] for item in targets
        }
        return sorted(
            edges,
            key=lambda edge: (
                -score_by_key[(edge.relation, edge.object)],
                edge.relation,
                edge.object,
            ),
        )

    def record_selected_path(self, hypothesis, result):
        if result.get("success"):
            target = result.get("target")
            self.state.reinforce(
                hypothesis.relation,
                target,
                1.0 if result.get("direct_proof") else 0.5,
                "verified_path",
            )

    def arbitrate(self, ranked):
        decision = self.arbitrator.decide(ranked)
        if decision["outcome"] == "abstain":
            self.state._record("abstain", reason=decision["reason"])
        return decision

    def trace(self):
        return {
            "state": self.state.snapshot(),
            "traversal_targets": self.traversal_targets,
            "policy": type(self.policy).__name__,
            "policy_model_version": getattr(self.policy, "version", "fallback"),
            "distillation_examples": list(self.policy_examples[-256:]),
        }
