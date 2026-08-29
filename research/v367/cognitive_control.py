
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CognitiveState:
    uncertainty: float
    model_uncertainty: float
    state_uncertainty: float
    conflict: float
    novelty: float
    goal_gap: float
    resource_fraction: float


@dataclass(frozen=True)
class CognitiveAction:
    name: str
    score: float
    reason: str


class EndogenousCognitiveController:
    ORDER=(
        "retrieve","generate","simulate",
        "intervene","revise","execute","stop",
    )

    def choose(self,state,available):
        scores=[]
        for a in available:
            if a=="retrieve":
                score=1.0*(1-state.model_uncertainty)
                reason="retrieve_stable_model"
            elif a=="generate":
                score=1.2*state.novelty+0.9*state.model_uncertainty
                reason="generate_novel_model"
            elif a=="simulate":
                score=0.7*state.goal_gap+0.5*state.model_uncertainty
                reason="predict_candidate_consequence"
            elif a=="intervene":
                score=2.0*state.conflict+1.5*state.model_uncertainty
                reason="discriminate_competing_hypotheses"
            elif a=="revise":
                score=1.7*state.conflict+1.2*state.model_uncertainty
                reason="revise_inconsistent_model"
            elif a=="execute":
                score=1.6*(1-max(state.uncertainty,state.model_uncertainty))
                reason="execute_resolved_model"
            else:
                score=1.2*(1-max(state.uncertainty,state.model_uncertainty))-0.7*state.goal_gap
                reason="stop_when_resolved"
            scores.append(CognitiveAction(a,score,reason))
        return max(
            scores,
            key=lambda x:(x.score,-self.ORDER.index(x.name))
        )
