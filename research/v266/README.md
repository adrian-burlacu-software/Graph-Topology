
# V266 — Memory Fault-Isolation Benchmark

V265 showed:

```text
latent workspace forms a pair-distinct state at t=1
the state persists partially to t=3/4
the decoder's action sensitivity decays much faster
```

V266 tests the likely causes in **one run** instead of fixing them serially.

## Hypotheses tested

### A. State overwrite / decay

Intervention:

```text
freeze
```

Keep the first memory state after its initial write instead of allowing later
state updates.

Interpretation:

```text
freeze succeeds, normal fails
    → later state updates are overwriting useful memory
```

### B. Decoder blindness

Intervention:

```text
workspace swap
```

Inject A's carried workspace into B's terminal decision and vice versa.

Interpretation:

```text
swap changes decision
    → controller actually reads workspace

swap does not change decision
    → memory may exist but is not causally decoded
```

### C. State ablation

Intervention:

```text
zero
```

Erase workspace immediately before the terminal decision.

This gives the direct causal state-use effect.

### D. Credit assignment

Training condition:

```text
aux=1.0
```

adds a temporary intermediate memory objective that asks the model to preserve
the remembered terminal action through the latent workspace.

Interpretation:

```text
auxiliary supervision restores long-horizon memory
    → original failure was primarily credit assignment
```

This auxiliary condition is diagnostic, not a new architectural claim.

### E. Pure retention

Trace every timestep:

```text
working A-B distance
action-logit A-B distance
retention relative to t=1
```

This distinguishes:

```text
state formation
state decay
decoder decay
```

## Cells

```text
baseline_graph       × H1/H2/H3/H4
latent_workspace     × H1/H2/H3/H4
latent_workspace+aux × H2/H3/H4
```

15 cells total, parallelism 2.

This is intentionally still much smaller than the earlier architectural surveys.

## Run

```powershell
python .\research\v266\run_memory_diagnostic.py --pairs-per-horizon 24 --epochs 10 --batch-size 2 --parallelism 2
```

The summary JSON contains all intervention and timestep diagnostics.

## How to interpret the result

The important patterns are:

```text
normal fails
freeze succeeds
    → overwrite/decay

normal fails
zero == normal
swap has no effect
    → workspace is not causally used

normal fails
aux succeeds
    → credit assignment

workspace separates at t=1
workspace remains separated
decoder logits remain separated
normal still fails
    → terminal controller/training mapping is the bottleneck

workspace separation itself collapses
    → state dynamics are the bottleneck
```

This benchmark is intended to identify the bottleneck before another architecture
redesign is attempted.
