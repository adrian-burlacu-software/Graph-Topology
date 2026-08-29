
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ModelEvidence:
    kind: str
    strength: float
    reason: str
    regime: int


@dataclass(frozen=True)
class SemanticModel:
    model_id: str
    task_family: str
    signature: Tuple[str,...]
    transformation: str
    regime: int
    alpha: float
    beta: float
    version: int
    evidence: Tuple[ModelEvidence,...]
    parent_id: str | None = None
    active: bool = True

    @property
    def confidence(self):
        return self.alpha / max(1e-9, self.alpha+self.beta)

    @property
    def reliability(self):
        wins=sum(
            e.strength for e in self.evidence
            if e.kind=="success"
        )
        losses=sum(
            e.strength for e in self.evidence
            if e.kind=="failure"
        )
        return wins/max(1e-9,wins+losses)

    @property
    def score(self):
        regime_support=sum(
            e.strength for e in self.evidence
            if e.regime==self.regime
        )
        regime_conflict=sum(
            e.strength for e in self.evidence
            if e.regime!=self.regime and e.kind=="failure"
        )
        return (
            0.40*self.confidence
            +0.35*self.reliability
            +0.15*min(1.0,regime_support/3.0)
            -0.25*min(1.0,regime_conflict/3.0)
            +0.10*int(self.active)
        )


class PersistentModelLifecycle:
    def __init__(
        self,
        stale_threshold=2,
        retire_threshold=0.25,
    ):
        self.models: Dict[str,SemanticModel]={}
        self.history=[]
        self.stale_threshold=stale_threshold
        self.retire_threshold=retire_threshold

    def _key(self,task,signature,transformation,regime):
        return (
            f"{task}|{','.join(sorted(signature))}|"
            f"{transformation}|r{regime}"
        )

    def candidates(self,task,signature,regime):
        return tuple(sorted(
            (
                m for m in self.models.values()
                if m.active
                and m.task_family==task
                and set(m.signature)==set(signature)
            ),
            key=lambda m:(
                int(m.regime==regime),
                m.score,
                m.version,
            ),
            reverse=True,
        ))

    def ensure_model(
        self,
        task,
        signature,
        transformation,
        regime,
        prior=0.5,
        reason="creation",
        parent_id=None,
    ):
        key=self._key(
            task,signature,transformation,regime
        )
        existing=self.models.get(key)
        if existing is not None:
            if not existing.active:
                reactivated=SemanticModel(
                    **{
                        **existing.__dict__,
                        "active":True,
                        "version":existing.version+1,
                        "evidence":existing.evidence + (
                            ModelEvidence(
                                "reactivation",
                                1.0,
                                reason,
                                regime,
                            ),
                        ),
                    }
                )
                self.models[key]=reactivated
                self.history.append(
                    ("reactivate",key,regime,existing.model_id)
                )
                return reactivated
            return existing

        model=SemanticModel(
            model_id=key,
            task_family=task,
            signature=tuple(sorted(signature)),
            transformation=transformation,
            regime=regime,
            alpha=1.0+2.0*prior,
            beta=1.0+2.0*(1.0-prior),
            version=(
                self.models[parent_id].version+1
                if parent_id in self.models
                else 1
            ),
            evidence=(
                ModelEvidence(
                    "creation",
                    1.0,
                    reason,
                    regime,
                ),
            ),
            parent_id=parent_id,
        )
        self.models[key]=model
        self.history.append(
            ("create",key,regime,parent_id)
        )
        return model

    def record_outcome(
        self,
        model,
        success,
        regime,
        reason,
    ):
        if model is None:
            return None

        evidence=ModelEvidence(
            "success" if success else "failure",
            1.0,
            reason,
            regime,
        )

        updated=SemanticModel(
            model_id=model.model_id,
            task_family=model.task_family,
            signature=model.signature,
            transformation=model.transformation,
            regime=model.regime,
            alpha=model.alpha+(1.0 if success else 0.0),
            beta=model.beta+(0.0 if success else 1.0),
            version=model.version,
            evidence=model.evidence+(evidence,),
            parent_id=model.parent_id,
            active=model.active,
        )

        self.models[model.model_id]=updated
        self.history.append(
            (
                "outcome",
                model.model_id,
                regime,
                bool(success),
                updated.active,
            )
        )
        return self._stale_check(
            updated,
            regime,
        )

    def _stale_check(self,model,current_regime):
        recent_failures=0
        cross_regime_failures=0

        for e in reversed(model.evidence):
            if e.kind=="success":
                break
            if e.kind=="failure":
                recent_failures+=1
                if e.regime!=model.regime:
                    cross_regime_failures+=1

        if (
            model.active
            and current_regime!=model.regime
            and cross_regime_failures>=self.stale_threshold
        ):
            retired=SemanticModel(
                **{
                    **model.__dict__,
                    "active":False,
                }
            )
            self.models[model.model_id]=retired
            self.history.append(
                (
                    "retire",
                    model.model_id,
                    current_regime,
                    "cross_regime_failure",
                )
            )
            return retired

        if (
            model.active
            and model.confidence<self.retire_threshold
            and recent_failures>=self.stale_threshold
        ):
            retired=SemanticModel(
                **{
                    **model.__dict__,
                    "active":False,
                }
            )
            self.models[model.model_id]=retired
            self.history.append(
                (
                    "retire",
                    model.model_id,
                    current_regime,
                    "low_confidence",
                )
            )
            return retired

        return model

    def revise(
        self,
        stale,
        new_transformation,
        new_regime,
        reason,
    ):
        if stale is not None and stale.active:
            self.models[stale.model_id]=SemanticModel(
                **{
                    **stale.__dict__,
                    "active":False,
                }
            )
            self.history.append(
                (
                    "retire",
                    stale.model_id,
                    new_regime,
                    "revision_parent",
                )
            )

        revised=self.ensure_model(
            stale.task_family,
            stale.signature,
            new_transformation,
            new_regime,
            prior=0.60,
            reason=reason,
            parent_id=stale.model_id,
        )
        self.history.append(
            (
                "revise",
                stale.model_id,
                revised.model_id,
                new_regime,
            )
        )
        return revised

    def snapshot(self):
        return tuple(
            sorted(
                self.models.values(),
                key=lambda m:m.model_id,
            )
        )

    def stats(self):
        return {
            "total_models":len(self.models),
            "active_models":sum(
                int(m.active) for m in self.models.values()
            ),
            "retired_models":sum(
                int(not m.active) for m in self.models.values()
            ),
            "revisions":sum(
                int(x[0]=="revise") for x in self.history
            ),
            "retirements":sum(
                int(x[0]=="retire") for x in self.history
            ),
            "creations":sum(
                int(x[0]=="create") for x in self.history
            ),
            "outcomes":sum(
                int(x[0]=="outcome") for x in self.history
            ),
        }


def task_signature(episode):
    base = {
        "delayed_memory":(
            "persistent_memory",
        ),
        "sequence_binding":(
            "persistent_memory",
            "primary_evidence",
            "secondary_evidence",
            "tertiary_evidence",
            "ordered_relation",
        ),
        "interference":(
            "persistent_memory",
            "primary_evidence",
            "competitive_evidence",
        ),
        "planning":(
            "persistent_memory",
            "primary_evidence",
            "secondary_evidence",
            "tertiary_evidence",
            "ordered_relation",
        ),
        "rule_change":(
            "persistent_memory",
            "primary_evidence",
            "secondary_evidence",
            "regime_indicator",
        ),
        "counterfactual":(
            "persistent_memory",
            "primary_evidence",
            "counterfactual_world",
            "counterfactual_control",
            f"cf_mode_{int(episode.counterfactual_bit)}",
            f"regime_{int(episode.rule_version)}",
        ),
    }.get(
        episode.task,
        ("persistent_memory",),
    )
    return base
