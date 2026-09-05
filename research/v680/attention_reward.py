"""Oracle-only rewards; oracle fields are deliberately absent from observations."""
from __future__ import annotations


class AttentionRewardOracle:
    def reward(self, action, transition):
        if transition["terminal_outcome"] == "verified":
            return 10.0
        if transition["terminal_outcome"] == "no_verified_evidence":
            return 4.0 if action.kind.value == "abstain" else -10.0
        if transition.get("valid_proof_edge"):
            return 3.0
        if transition.get("already_visited"):
            return -0.5
        return -1.0
