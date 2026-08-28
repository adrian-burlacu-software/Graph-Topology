
# V270 — Memory Read-Path Benchmark

V270 is V269 with a hardened preflight.

## Fixed

V269 failed before model execution because `model.py` used:

```python
F.normalize(...)
```

without importing:

```python
import torch.nn.functional as F
```

V270 adds that import.

The preflight now checks the model source for this dependency **before importing
the model**, so this exact class of bug is caught immediately.

## Read-path ladder

```text
baseline_graph
latent_workspace
direct_read
normalized_read
gated_read
protected_read
```

The candidate mechanisms are:

```text
direct_read:
    explicit workspace → controller projection

normalized_read:
    normalize workspace before reading

gated_read:
    learned adaptive memory-read gate

protected_read:
    gated persistent workspace + explicit read path
```

## Diagnostics

The benchmark reports:

```text
normal
freeze
zero
workspace swap
workspace read sensitivity
per-timestep workspace separation
per-timestep action-logit separation
```

## Preflight

```powershell
python .\research\v270\preflight.py --pairs-per-horizon 24
```

The source dependency check happens before `model.py` is imported.

## Benchmark

```powershell
python .\research\v270\run_memory_diagnostic.py --pairs-per-horizon 24 --epochs 10 --batch-size 2 --parallelism 2
```

Do not proceed if preflight is not green.
