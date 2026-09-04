"""Frozen symbolic V679-style teacher expressed as a bounded action policy."""
from __future__ import annotations

import math

from attention_types import AttentionAction, AttentionActionKind


def softmax(logits, temperature=2.0):
    scaled = [value / max(float(temperature), 1e-6) for value in logits]
    ceiling = max(scaled)
    weights = [math.exp(value - ceiling) for value in scaled]
    total = sum(weights)
    return [weight / total for weight in weights]


class V679AttentionTeacher:
    """Frozen, inspectable teacher; it never receives episode oracle metadata."""

    def __init__(self, temperature=2.0):
        self.temperature = float(temperature)

    def score_candidates(self, state, candidates=None):
        candidates = candidates if candidates is not None else state.candidate_features
        logits = []
        for candidate in candidates:
            logits.append(
                3.0 * candidate.goal_relation_match
                + .5 * candidate.target_term_match
                + .25 * candidate.specificity
                + .5 * candidate.lexical_score
                + .5 * candidate.provenance
                + candidate.relation_activation
                + .25 * candidate.candidate_activation
                - 2.0 * candidate.contradiction
                - 2.0 * candidate.already_visited
            )
        # The current focus is observable, while proof-path membership remains oracle-only.
        proof_focus = any(term in state.current_focus for term in state.goal_terms)
        logits.append(5.0 if proof_focus else -1.0)
        logits.append(4.0 if not proof_focus else -2.0)
        return logits

    def select_action(self, state, candidates=None, deterministic=False):
        candidates = candidates if candidates is not None else state.candidate_features
        logits = self.score_candidates(state, candidates)
        probabilities = softmax(logits, self.temperature)
        selected = max(range(len(logits)), key=lambda index: logits[index])
        if selected < len(candidates):
            action = AttentionAction(AttentionActionKind.TRAVERSE, selected)
            outcome = "continue"
        elif selected == len(candidates):
            action = AttentionAction(AttentionActionKind.STOP)
            outcome = "stop"
        else:
            action = AttentionAction(AttentionActionKind.ABSTAIN)
            outcome = "abstain"
        return {
            "logits": logits, "probabilities": probabilities, "selected_action": selected,
            "action": action, "outcome": outcome,
        }

    def value(self, state):
        return max(self.score_candidates(state), default=0.0)
