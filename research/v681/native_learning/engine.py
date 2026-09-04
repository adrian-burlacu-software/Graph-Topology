"""V681-native learning entry points derived from the stable V680 implementation."""
from __future__ import annotations

import json
from pathlib import Path

SOURCE_LEARNING_VERSION = "v680.1-frozen-attention-jepa-engine"
V681_LEARNING_VERSION = "v681.5-native-learning-1"


class NativeLearningEngine:
    """Runs V681-owned learning functions directly; no subprocess/version dependency."""
    def generate_teacher_records(self, output_path, samples_per_category, transitions=False):
        from .dataset import collect_jepa_transition_episodes, collect_teacher_episodes, write_jsonl
        from .benchmark import decision_boundary_episodes
        source = decision_boundary_episodes(samples_per_category)
        records = collect_jepa_transition_episodes(source) if transitions else collect_teacher_episodes(source)
        write_jsonl(output_path, records)

    def run_dagger(self, records_path, checkpoint_dir, rounds, epochs, seed):
        from .dagger import run_dagger
        from .dataset import write_jsonl
        _, aggregate, _ = run_dagger(records_path, rounds=rounds, epochs=epochs, seed=seed,
                                     checkpoint_dir=checkpoint_dir)
        path = Path(checkpoint_dir) / f"dagger_aggregate_round_{int(rounds) - 1}.jsonl"
        if not path.is_file():
            write_jsonl(path, aggregate)
        return path

    def train_attention(self, records_path, checkpoint_path, epochs, seed):
        import torch
        from .dataset import read_jsonl
        from .distill import metadata, train_distillation
        model, optimizer = train_distillation(read_jsonl(records_path), epochs=epochs, seed=seed)
        checkpoint_path = Path(checkpoint_path); checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "hidden_size": model.hidden_size, "jepa_dim": model.jepa_dim,
                    "metadata": metadata(records_path, seed, epochs=epochs, learning_rate=1e-3,
                                         temperature=2.0, lambda_soft=1.0, lambda_rank=.2,
                                         lambda_hard=1.0, use_jepa=False, jepa_checkpoint="",
                                         class_balance=True)}, checkpoint_path)

    def evaluate_attention(self, records_path, checkpoint_path, output_path):
        from .dataset import read_jsonl
        from .evaluate import evaluate, load_student
        result = evaluate(read_jsonl(records_path), load_student(checkpoint_path))
        Path(output_path).write_text(json.dumps(result, indent=2, sort_keys=True))
        return result

    def train_jepa(self, records_path, checkpoint_path, output_path, epochs, seed):
        import torch
        from .dataset import read_jsonl
        from .jepa import AttentionJEPA, evaluate_jepa, train_jepa
        records = read_jsonl(records_path)
        model, optimizer = train_jepa(records, epochs=epochs, seed=seed, model=AttentionJEPA())
        result = evaluate_jepa(records, model)
        checkpoint_path = Path(checkpoint_path); checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "seed": seed,
                    "configuration": {"representation_dim": model.representation_dim,
                                      "hidden_size": model.hidden_size,
                                      "target_momentum": model.target_momentum}}, checkpoint_path)
        Path(output_path).write_text(json.dumps(result, indent=2, sort_keys=True))
        return result
