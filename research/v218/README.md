
# V218 — True Iterative State Experiments

V218 fixes the recurrence definition and the V215 training bug.

The previous V215 failure was caused by:

```python
outs.index(o)
```

where `o` contains tensors. Python list equality then attempts tensor boolean comparisons, producing:

```text
RuntimeError: Boolean value of Tensor with more than one value is ambiguous
```

V218 removes that entirely and uses:

```python
for step_index, (out, target) in enumerate(...):
```

## Three controlled experiment families

### A — Transformer depth

One static final cognitive decision:

```text
graph
  ↓
Transformer × depth
  ↓
attention
  ↓
action
```

Runs:

```text
d2
d4
d6
d8
```

This isolates computational depth.

### B — True iterative state

The model processes a sequence of teacher states and carries a learned working state:

```text
state₀
  ↓
Transformer
  ↓
attention
  ↓
action
  ↓
working_state₁
  ↓
state₁
  ↓
Transformer
  ↓
attention
  ↓
action
  ↓
working_state₂
  ↓
...
```

The recurrent object is the cognitive state, NOT repeated Transformer weights.

### C — Both

Deep Transformer processing at every cognitive step plus persistent working state:

```text
state₀
  ↓
Transformer × 8
  ↓
attention/action
  ↓
working_state₁
  ↓
state₁
  ↓
Transformer × 8
  ↓
attention/action
  ↓
working_state₂
  ↓
...
```

Runs:

```text
d8 + 2 steps
d8 + 4 steps
d8 + 6 steps
d8 + 8 steps
```

## Important correction

The `steps` value in B/C is the experiment's recurrence budget. The deterministic teacher trajectory determines the number of supervised cognitive states available in each example.

This keeps the trajectory itself identical across experiments and avoids silently fabricating state transitions.

## Run

From repository root:

```powershell
python .\research\v218\run_all.py
```

If this archive is extracted as `research\v218`, that command runs the complete 12-experiment matrix.

Quick smoke test:

```powershell
python .\research\v218\run_all.py --samples 50 --epochs 2
```

Default matrix:

```text
Depth:
  d2
  d4
  d6
  d8

Iterative:
  2
  4
  6
  8

Both:
  d8 + 2
  d8 + 4
  d8 + 6
  d8 + 8
```

Results:

```text
results/v218/
```

Summary:

```text
results/v218/v218_summary.json
```

## What the experiment answers

The matrix separates:

1. Does more Transformer computation help?
2. Does persistent iterative cognitive state help?
3. Does combining Transformer depth with iterative state help?

Hard attention F1 is also reported to ensure that downstream decisions are actually being made through the attention bottleneck.


## V218 fixes

1. **CUDA/CPU metric bug fixed.**
   Attention targets are created as CPU tensors by the dataset. `metrics_update`
   now explicitly moves the target mask to the prediction device before doing
   boolean operations.

2. **Iterative step budget is now actually enforced.**
   `--steps 2`, `--steps 4`, etc. are passed into the iterative forward pass.
   The model processes at most that many teacher states; it never invents
   additional states.

3. The previous `outs.index(o)` tensor-comparison bug remains fixed.


## V218 fix

The iterative training function now receives `steps` explicitly. The previous
V217 archive referenced `steps` inside `run_iterative_epoch()` without adding
it to that function's parameters, causing:

```text
NameError: name 'steps' is not defined
```

V218 passes the requested recurrence budget through:

```text
run_experiment
    -> run_iterative_epoch(..., steps, ...)
        -> model.forward_iterative(..., max_steps=steps)
```

This is now compiled/validated before packaging.
