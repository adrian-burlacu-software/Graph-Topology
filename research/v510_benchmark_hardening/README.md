# V510 — benchmark hardening + reference normalization

V510 preserves the V509 cognitive bridge as the baseline and adds two focused
changes:

1. **Reference normalization** — yes/no questions such as `is it red?` now
   normalize color/size/shape adjectives into typed property targets.
2. **Benchmark hardening** — the original 36 V509 regression cases are kept,
   and a deterministic V510 adversarial suite exercises reference resolution,
   topic switching, state updates, counts, operators, state arithmetic,
   unknowns, and mixed-entity scope.

The architecture remains architecture-first:

```text
perception
→ goal
→ typed target
→ episodic state
→ deterministic logic
→ targeted evidence
→ semantic decision
→ cognitive frame
→ language realization
```

No new learned component is introduced in V510. This release is intended to
validate the deterministic teacher/reference architecture before distilling
teacher traces into the attention controller.

## V510 changes

### Reference normalization

```text
The dog is red.
Is it red?
```

now yields a typed target equivalent to:

```text
kind=property
subject=dog
attribute=color
value=red
```

The same normalization exists for common size and shape adjectives.

### Regression preservation

The original V509 36-case benchmark is unchanged and is run first as a
regression suite.

### Adversarial suite

`benchmark_v510_cases.py` contains deterministic semantic-boundary tests. The
suite intentionally does not add large new operator coverage; its purpose is
to expose where the architecture fails.

## Run V510 benchmark

PowerShell:

```powershell
python .\research\v510_benchmark_hardening\run_v510.py `
  --memory ".\results\full_semantic_memory.sqlite" `
  --freeze-knowledge `
  --verbose `
  --output ".\research\v510_benchmark_hardening\v510_benchmark_results.json"
```

## Run tests

```powershell
python .\research\v510_benchmark_hardening\test_v510.py
```

## Run chat

V510 keeps the same chat surface as V509. Use the existing architecture CLI or
copy `assistant_cli.py` into this release directory if you want a self-contained
CLI entry point.

## Expected research sequence

```text
V509 baseline
   ↓
V510 hardening
   ↓
freeze benchmark
   ↓
teacher traces
   ↓
attention-controller distillation
   ↓
learned cognition ablation
```

Do not optimize the adversarial suite away. Failures are intended diagnostic
signals for the next architectural change.
