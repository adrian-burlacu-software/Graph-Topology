
# V284 — Strongest-Credit Follow-up

V283 gave the clearest H3 evidence so far:

```text
eligibility_trace
    H3 normal=1.0
    H3 pair discrimination=1.0
    H3 workspace drop=1.0

transition_supervision
    H3 normal=1.0
    H3 pair discrimination=1.0
    H3 workspace drop=1.0
```

H4 remained difficult.

V284 therefore focuses on the strongest discovered mechanism instead of
adding another memory subsystem.

## Survey

Exactly:

```text
6 architectures × H2/H4 = 12 cells
```

Conditions:

```text
baseline_graph
protected_read_progress
transition_supervision
eligibility_trace
eligibility_transition
adaptive_eligibility
```

## Predictions

### eligibility_transition

Combines the two strongest mechanisms.

Prediction:

```text
eligibility_transition > both parents at H4
```

If it wins, the H4 bottleneck is likely jointly temporal credit + transition
dynamics.

### adaptive_eligibility

Uses a horizon-aware credit schedule rather than a fixed 0.80 decay.

Prediction:

```text
adaptive_eligibility > eligibility_trace at H4
```

If it wins, the fixed eligibility decay was too aggressive for longer horizons.

## Why only H2/H4?

H2 is the learned-memory sanity check.

H4 is the stress case.

H1/H3 are dropped to halve the survey size, as requested.

## Run

Regression:

```powershell
python .\research\v284\credit_attack_regression.py
```

Preflight:

```powershell
python .\research\v284\preflight.py --pairs-per-horizon 24
```

Full survey:

```powershell
python .\research\v284\run_memory_diagnostic.py --pairs-per-horizon 24 --epochs 10 --batch-size 2 --parallelism 2
```

Exactly 12 cells.

## What would make V284 decisive?

```text
eligibility_transition wins
    → combine the two mechanisms into the next architecture

adaptive_eligibility wins
    → tune/learn temporal credit rather than adding storage

both fail at H4
    → stop changing the credit objective and make the graph itself
      part of the controller/readout
```
