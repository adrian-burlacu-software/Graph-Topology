
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class RevisionDecision:
    action:str
    reason:str
    confidence:float


class RegimeModelReviser:
    TRANSITIONS={
        "rule_change":(
            "rule_conditioned",
            "pre_change_rule",
        ),
        "counterfactual":(
            "counterfactual",
            "factual_projection",
        ),
        "sequence_binding":(
            "ordered_binding",
            "memory_only",
        ),
        "interference":(
            "relevant_binding",
            "competitive_binding",
        ),
        "planning":(
            "planning",
            "memory_only",
        ),
        "delayed_memory":(
            "delayed_memory",
        ),
    }

    def alternatives(self,task,regime):
        return self.TRANSITIONS.get(
            task,
            ("memory_only",),
        )

    def decide(self,model,current_regime,recent_failures):
        if model is None:
            return RevisionDecision(
                "generate",
                "no_model",
                1.0,
            )

        if model.regime!=current_regime:
            return RevisionDecision(
                "revise",
                "regime_transition_requires_new_version",
                min(1.0,0.65+0.05*recent_failures),
            )

        if recent_failures>=2:
            return RevisionDecision(
                "revise",
                "same_regime_repeated_failure",
                0.85,
            )

        if recent_failures==1:
            return RevisionDecision(
                "test",
                "single_failure",
                0.60,
            )

        return RevisionDecision(
            "reuse",
            "stable_model",
            model.score,
        )
