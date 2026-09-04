"""Small orchestration adapters over existing V680 learners; no replacement algorithms."""
from __future__ import annotations

import sys
from pathlib import Path

from experience import ExperienceSource, ExperienceStore

V680 = Path(__file__).resolve().parents[1] / "v680"
if str(V680) not in sys.path:
    sys.path.insert(0, str(V680))


def attention_records(store, sources=(ExperienceSource.DAGGER,)):
    episodes = {}
    for item in store.load(training_only=True):
        if item.source not in sources or "attention_step" not in item.payload:
            continue
        step = item.payload["attention_step"]
        episodes.setdefault(item.episode_id, {"episode_id": item.episode_id, "split": step["split"],
                                               "partition": step.get("partition", ""), "trajectory": []})["trajectory"].append(step)
    return list(episodes.values())


class AttentionDistillationLearner:
    learner_type = "attention_distillation"
    def collect(self, store): return attention_records(store)
    def prepare(self, store): return self.collect(store)
    def train(self, records, **configuration):
        from attention_distill import train_distillation
        return train_distillation(records, **configuration)
    def evaluate(self, records, model, **configuration):
        from attention_evaluate import evaluate
        return evaluate(records, model, **configuration)
    def save(self, model, path):
        import torch
        torch.save({"model": model.state_dict(), "learner_type": self.learner_type}, path)
    def load(self, path): return path


class JEPAAuxiliaryLearner:
    learner_type = "jepa_auxiliary"
    def collect(self, store):
        return attention_records(store)
    def prepare(self, store): return self.collect(store)
    def train(self, records, **configuration):
        from attention_jepa import train_jepa
        return train_jepa(records, **configuration)
    def evaluate(self, records, model, **configuration):
        from attention_jepa import evaluate_jepa
        return evaluate_jepa(records, model)
    def save(self, model, path):
        import torch
        torch.save({"model": model.state_dict(), "learner_type": self.learner_type}, path)
    def load(self, path): return path


class DaggerBootstrapLearner:
    learner_type = "attention_dagger_bootstrap"
    def collect(self, store): return attention_records(store, (ExperienceSource.DAGGER,))
    def prepare(self, store): return self.collect(store)
    def train(self, records, **configuration):
        from attention_dagger import run_dagger
        return run_dagger(records, **configuration)
    def evaluate(self, *args, **kwargs): return {}
    def save(self, artifact, path): return path
    def load(self, path): return path


class SemanticWorkerLearner:
    learner_type = "semantic_worker"
    def collect(self, store): return store.load(source=ExperienceSource.OFFLINE_WORKER)
    def prepare(self, store): return self.collect(store)
    def train(self, records, **configuration): return {"evidence_records": len(records), **configuration}
    def evaluate(self, *args, **kwargs): return {}
    def save(self, artifact, path): return path
    def load(self, path): return path
