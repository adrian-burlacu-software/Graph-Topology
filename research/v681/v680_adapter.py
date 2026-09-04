"""The only V681 boundary to the frozen V680 attention/JEPA engine."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

V680_ENGINE_VERSION = "v680.1-frozen-attention-jepa-engine"


class V680EngineAdapter:
    def __init__(self, engine_root=None):
        self.root = Path(engine_root) if engine_root else Path(__file__).resolve().parents[1] / "v680"
        self._validate()

    def _validate(self):
        required = ("attention_dataset.py", "attention_distill.py", "attention_jepa.py", "attention_evaluate.py")
        missing = [str(self.root / name) for name in required if not (self.root / name).is_file()]
        if missing:
            raise FileNotFoundError("V681 requires the frozen V680 engine at %s; missing: %s" %
                                    (self.root, ", ".join(missing)))

    def _run(self, script, *arguments):
        command = [sys.executable, str(self.root / script), *map(str, arguments)]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError("frozen V680 engine command failed (%s): %s" %
                               (script, result.stderr.strip() or result.stdout.strip()))

    def train_attention(self, records_path, checkpoint_path, epochs, seed):
        self._run("attention_distill.py", "--dataset", records_path, "--checkpoint", checkpoint_path,
                  "--epochs", epochs, "--seed", seed)

    def generate_teacher_records(self, output_path, samples_per_category, transitions=False):
        arguments = ["--decision-boundary", "--samples-per-category", samples_per_category, "--output", output_path]
        if transitions:
            arguments.append("--jepa-transitions")
        self._run("attention_dataset.py", *arguments)

    def evaluate_attention(self, records_path, checkpoint_path, output_path):
        self._run("attention_evaluate.py", "--dataset", records_path, "--checkpoint", checkpoint_path,
                  "--output", output_path)
        return json.loads(Path(output_path).read_text())

    def train_jepa(self, records_path, checkpoint_path, output_path, epochs, seed):
        self._run("attention_jepa.py", "--dataset", records_path, "--checkpoint", checkpoint_path,
                  "--output", output_path, "--epochs", epochs, "--seed", seed)
        return json.loads(Path(output_path).read_text())

    def run_dagger(self, records_path, checkpoint_dir, rounds, epochs, seed):
        """Run frozen V680 DAgGER; V681 imports its completed aggregate as experience."""
        self._run("attention_dagger.py", "--dataset", records_path, "--checkpoint-dir", checkpoint_dir,
                  "--rounds", rounds, "--epochs", epochs, "--seed", seed)
        aggregate = Path(checkpoint_dir) / f"dagger_aggregate_round_{int(rounds) - 1}.jsonl"
        if not aggregate.is_file():
            raise RuntimeError(f"frozen V680 DAgGER did not write expected aggregate: {aggregate}")
        return aggregate
