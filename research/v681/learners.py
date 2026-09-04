"""V681 learner registry and orchestration over the one explicit V680 adapter."""
from __future__ import annotations

from dataclasses import dataclass

from .experience import ExperienceSource
from .trajectory import AttentionTrajectoryAdapter, OutcomeTransitionAdapter, SequentialTransitionAdapter


@dataclass(frozen=True)
class LearnerDescriptor:
    learner_type: str
    accepted_sources: tuple
    required_capabilities: tuple
    training_mode: str
    artifact_type: str


REGISTRY = {
    "attention_distillation": LearnerDescriptor("attention_distillation",
        (ExperienceSource.DAGGER, ExperienceSource.CHAT_SEQUENTIAL, ExperienceSource.SYNTHETIC_CHAT),
        ("sequential",), "imitation", "attention_checkpoint"),
    "attention_dagger": LearnerDescriptor("attention_dagger", (ExperienceSource.DAGGER,),
        ("sequential",), "bootstrap", "attention_checkpoint"),
    "jepa_auxiliary": LearnerDescriptor("jepa_auxiliary",
        (ExperienceSource.DAGGER, ExperienceSource.CHAT_SEQUENTIAL, ExperienceSource.SYNTHETIC_CHAT),
        ("sequential",), "predictive_auxiliary", "jepa_checkpoint"),
    "semantic_worker": LearnerDescriptor("semantic_worker", (ExperienceSource.OFFLINE_WORKER,),
        ("knowledge_only",), "knowledge_evidence", "knowledge_manifest"),
}


def capability_report(experiences, descriptor, sources):
    allowed = set(sources)
    matching = [item for item in experiences if item.source in allowed]
    compatible = [item for item in matching if item.sequence_capability in descriptor.required_capabilities]
    if compatible:
        return {"supported": True, "records": len(compatible), "reason": ""}
    capability = sorted({item.sequence_capability for item in matching})
    return {"supported": False, "records": 0,
            "reason": "no compatible %s experience; found capabilities: %s" %
                      ("/".join(descriptor.required_capabilities), ",".join(capability) or "none")}


class AttentionDistillationLearner:
    descriptor = REGISTRY["attention_distillation"]
    def prepare(self, experiences, sources, **filters):
        return AttentionTrajectoryAdapter().extract(experiences, sources=sources, **filters)
    def train(self, adapter, records_path, checkpoint_path, epochs, seed):
        return adapter.train_attention(records_path, checkpoint_path, epochs, seed)
    def evaluate(self, adapter, records_path, checkpoint_path, output_path):
        return adapter.evaluate_attention(records_path, checkpoint_path, output_path)


class JEPAAuxiliaryLearner:
    descriptor = REGISTRY["jepa_auxiliary"]
    def prepare(self, experiences, sources, **filters):
        return SequentialTransitionAdapter().extract(experiences, sources=sources, **filters)
    def train(self, adapter, records_path, checkpoint_path, output_path, epochs, seed):
        return adapter.train_jepa(records_path, checkpoint_path, output_path, epochs, seed)


class OutcomeLearnerInterface:
    """Future-RL-ready transitions only; this interface deliberately has no optimizer."""
    def prepare(self, experiences):
        return OutcomeTransitionAdapter().extract(experiences)
