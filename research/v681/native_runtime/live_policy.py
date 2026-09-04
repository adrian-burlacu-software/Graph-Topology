"""Safe bridge from a promoted V681 neural artifact to observable live scoring."""
from __future__ import annotations

import threading

from .attention import HandCodedAttentionPolicy


class PolicyProvider:
    def __init__(self, path=""):
        self._path, self._lock = str(path), threading.Lock()

    @property
    def path(self):
        with self._lock:
            return self._path

    def set(self, path):
        with self._lock:
            self._path = str(path)


class ReloadingAttentionPolicy(HandCodedAttentionPolicy):
    """Atomically adopts a promoted checkpoint between scoring calls."""
    def __init__(self, provider):
        self.provider, self.loaded_path, self.loaded = provider, "", None

    @property
    def version(self):
        return self.provider.path or "fallback"

    def score_traversal(self, features):
        return self._policy().score_traversal(features)

    def score_arbitration(self, features):
        return self._policy().score_arbitration(features)

    def _policy(self):
        path = self.provider.path
        if path and path != self.loaded_path:
            self.loaded, self.loaded_path = PromotedAttentionPolicy(path), path
        return self.loaded or HandCodedAttentionPolicy()


class PromotedAttentionPolicy(HandCodedAttentionPolicy):
    """Uses only observable candidate/state features and preserves verified-only abstention."""
    def __init__(self, checkpoint):
        from ..native_learning.evaluate import load_student
        self.model = load_student(checkpoint)
        self.version = str(checkpoint)

    def score_traversal(self, features):
        return self._candidate_logit(features)

    def score_arbitration(self, features):
        return self._candidate_logit({**features, "relation": "", "goal_relation_match": features["specificity"],
                                      "target_term_match": features["lexical"], "relation_activation": 0.0,
                                      "candidate_activation": 0.0, "path_length": 1, "verified": features["support"],
                                      "direct_proof": features["provenance"], "already_visited": 0.0})

    def _candidate_logit(self, value):
        from ..native_learning.types import AttentionObservation, CandidateFeatures
        candidate = CandidateFeatures(
            relation=str(value.get("relation", "")), target=str(value.get("target", "")),
            path_length=int(value.get("path_length", 1)), goal_relation_match=float(value.get("goal_relation_match", 0)),
            target_term_match=float(value.get("target_term_match", 0)), specificity=float(value.get("specificity", 0)),
            lexical_score=float(value.get("lexical_score", value.get("lexical", 0))),
            relation_activation=float(value.get("relation_activation", 0)),
            candidate_activation=float(value.get("candidate_activation", 0)),
            provenance=float(value.get("provenance", 0)), verified=float(value.get("verified", value.get("support", 0))),
            contradiction=float(value.get("contradiction", 0)), direct_proof=float(value.get("direct_proof", 0)),
        )
        state = AttentionObservation("", [], "", "", {}, [candidate], {}, {}, [], [], [], 0, 1)
        return float(self.model.select_action(state, deterministic=True)["logits"][0])
