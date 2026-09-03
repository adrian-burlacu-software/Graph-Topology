from __future__ import annotations

import re


class AttentionController:
    """Ranks bounded graph evidence and records why a semantic choice won."""

    def __init__(self, max_trace_targets=256):
        self.max_trace_targets = max(1, int(max_trace_targets))
        self.traversal_targets = []

    @staticmethod
    def _relation_specificity(relation):
        return 0.15 if str(relation) == "related_to" else 1.0

    def prioritize_hypotheses(self, hypotheses):
        return sorted(
            hypotheses,
            key=lambda item: (
                -float(item.lexical_score),
                -self._relation_specificity(item.relation),
                item.relation,
                item.subject or "",
            ),
        )

    def select_traversal_targets(self, hypothesis, prefix, edges, relation_scores):
        query_terms = set(
            re.findall(
                r"[a-z0-9]+",
                " ".join(
                    str(value)
                    for value in hypothesis.evidence.get("target_terms", [])
                ).lower(),
            )
        )
        targets = []
        for edge in edges:
            relation_score = float(relation_scores.get(edge.relation, 0.0))
            exact_relation = float(edge.relation == hypothesis.relation)
            target_score = float(
                bool(query_terms)
                and bool(
                    query_terms
                    & set(re.findall(r"[a-z0-9]+", str(edge.object).lower()))
                )
            )
            specificity = self._relation_specificity(edge.relation)
            score = (
                3.0 * exact_relation
                + relation_score
                + 0.5 * target_score
                + 0.25 * specificity
            )
            targets.append(
                {
                    "from": prefix[-1] if prefix else hypothesis.subject,
                    "path_prefix": list(prefix),
                    "relation": edge.relation,
                    "target": edge.object,
                    "score": round(score, 6),
                    "reasons": {
                        "goal_relation_match": bool(exact_relation),
                        "prior_attention": round(relation_score, 6),
                        "target_term_match": bool(target_score),
                        "specificity": round(specificity, 6),
                    },
                }
            )
        targets.sort(
            key=lambda item: (
                -item["score"],
                item["relation"],
                item["target"],
            )
        )
        if len(self.traversal_targets) < self.max_trace_targets:
            self.traversal_targets.extend(
                targets[: self.max_trace_targets - len(self.traversal_targets)]
            )
        by_key = {(item["relation"], item["target"]): item["score"] for item in targets}
        return sorted(
            edges,
            key=lambda edge: (
                -by_key[(edge.relation, edge.object)],
                edge.relation,
                edge.object,
            ),
        )

    def arbitrate(self, ranked):
        candidates = []
        for index, (_, hypothesis, result) in enumerate(ranked):
            verified = bool(result.get("success", False))
            direct = bool(result.get("direct_proof", False))
            target_terms = list(hypothesis.evidence.get("target_terms", []) or [])
            missing_argument = bool(hypothesis.evidence.get("argument_unverified", False))
            support = 1.0 if verified else 0.0
            contradiction = 1.0 if missing_argument else (0.25 if not verified else 0.0)
            specificity = self._relation_specificity(hypothesis.relation)
            if target_terms and verified:
                specificity = min(1.0, specificity + 0.15)
            provenance = 1.0 if direct else (0.8 if verified else 0.0)
            score = (
                4.0 * support
                - 2.0 * contradiction
                + 1.5 * specificity
                + provenance
                + 0.5 * min(1.0, float(hypothesis.lexical_score))
            )
            candidates.append(
                {
                    "candidate_index": index,
                    "subject": hypothesis.subject,
                    "relation": hypothesis.relation,
                    "intent": hypothesis.intent,
                    "path": list(result.get("path", [])),
                    "target": result.get("target"),
                    "verified": verified,
                    "components": {
                        "support": support,
                        "contradiction": contradiction,
                        "specificity": round(specificity, 6),
                        "provenance": provenance,
                        "lexical": round(float(hypothesis.lexical_score), 6),
                    },
                    "arbitration_score": round(score, 6),
                }
            )
        candidates.sort(
            key=lambda item: (
                -item["arbitration_score"],
                -int(item["verified"]),
                item["relation"],
                item["subject"] or "",
            )
        )
        selected_index = candidates[0]["candidate_index"] if candidates else None
        return {
            "decision_type": "semantic_evidence_arbitration",
            "selected_candidate_index": selected_index,
            "outcome": "verified" if candidates and candidates[0]["verified"] else "unverified",
            "candidates": candidates,
        }
