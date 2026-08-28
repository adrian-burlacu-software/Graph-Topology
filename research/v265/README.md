
# V265 — Isolated Memory Persistence Diagnostic

V263 gave the first convincing short-horizon memory signal:

```text
latent_workspace
horizon=2
```

but the signal disappeared at horizon 4.

V264 does not add another architecture matrix. It diagnoses where the memory
mechanism breaks.

## Four diagnostic axes

### 1. Horizon sweep

```text
H1
H2
H3
H4
```

H1 is the direct-observation control: the instruction is still visible at the
decision.

H2-H4 require retention after the instruction disappears.

### 2. Workspace separation trace

For each paired A/B case, report:

```text
||working_A(t) - working_B(t)||
```

at every timestep.

This tells us whether the two memories are actually being maintained in the
latent workspace.

### 3. Decision sensitivity trace

Also report:

```text
max |action_logits_A(t) - action_logits_B(t)|
```

at every timestep.

This distinguishes:

```text
memory exists
```

from:

```text
decoder actually uses the memory
```

### 4. Terminal ablation

Compare:

```text
normal workspace
zero workspace
```

and report:

```text
accuracy drop
```

This is the causal state-use check.

## The useful diagnostic patterns

### State formation failure

```text
working delta ≈ 0 immediately
```

The architecture never writes distinguishable memory.

### State decay

```text
working delta > 0 at t=1
↓
working delta → 0 by t=3/4
```

The memory is formed but not persistent.

### Decoder failure

```text
working delta stays large
but
action-logit delta ≈ 0
```

The state survives but is not being used by the controller.

### Successful persistent memory

```text
working delta stays nonzero
action-logit delta stays nonzero
normal > zero_mem
paired success > baseline
```

## Run

Preflight:

```powershell
python .\research\v264\preflight.py --pairs-per-horizon 24
```

Diagnostic benchmark:

```powershell
python .\research\v264\run_memory_diagnostic.py --pairs-per-horizon 24 --epochs 10 --batch-size 2 --parallelism 2
```

The four horizons and two architectures produce only eight cells.

The summary JSON includes the full per-timestep diagnostics.


## V265 bug fix

V264 accidentally reused `zero` as both the zero-workspace tensor and the
ablation-result list. V265 separates them into `zero_workspace` and
`zero_accuracy`.
