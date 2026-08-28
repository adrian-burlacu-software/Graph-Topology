
# V277 — Long-Horizon Memory Improvement Experiment

V275's survey showed a strong H2 causal memory result for
`protected_read_progress`, but H3/H4 still lost terminal decision sensitivity.

V277 tests three likely improvements together:

```text
1. slow memory lane
2. persistent progress anchor
3. progress-conditioned memory read gain
```

## Candidate architectures

```text
baseline_graph
protected_read_progress
slow_memory
progress_anchor
progress_read_gain
slow_progress_anchor_read
```

Exactly:

```text
6 architectures × 4 horizons = 24 cells
```

No auxiliary duplicate runs.

## Predictions

```text
slow_memory wins H3/H4
    → fast workspace overwrite/dilution was the bottleneck

progress_anchor wins H3/H4
    → task-position context helps preserve/use memory

progress_read_gain wins H3/H4
    → memory access needs to become stronger/adaptive later

slow_progress_anchor_read wins
    → the mechanisms interact

none wins
    → stop changing memory persistence plumbing and revisit
       the semantic state/readout design
```

## Diagnostics

The trace contains:

```text
fast workspace A/B separation
slow-memory A/B separation
progress-memory A/B separation
action-logit A/B separation
```

along with:

```text
normal
freeze
zero workspace
workspace swap
read sensitivity
```

## Run

Preflight:

```powershell
python .\research\v277\preflight.py --pairs-per-horizon 24
```

Then:

```powershell
python .\research\v277\run_memory_diagnostic.py --pairs-per-horizon 24 --epochs 10 --batch-size 2 --parallelism 2
```

The survey asserts:

```text
total_cells == 24
```

before launching any workers.
