
# V285 — Decisive Memory→Decision Bridge

V284 showed that eligibility + transition supervision is the strongest
long-horizon substrate so far, but H4 remained weak.

V285 tests the hypothesis that the remaining failure is now the **interface
between preserved memory and the terminal decision**.

## Predictions

### bridge

Adds a dedicated:

```text
workspace + goal + progress
        ↓
memory bridge
        ↓
terminal action logits
```

If bridge alone improves H4, the terminal readout was the remaining bottleneck.

### bridge_eligibility_transition

Combines:

```text
eligibility credit
+
transition supervision
+
explicit terminal memory bridge
```

This is the primary candidate.

Prediction:

```text
H4 > eligibility_transition
```

### bridge_query_eligibility_transition

Adds the older explicit query path on top of the bridge and strongest credit
mechanism.

Prediction:

```text
if this wins:
    redundant direct memory access is useful at terminal time

if bridge_eligibility_transition wins:
    the simpler explicit bridge is sufficient
```

## Survey

H2/H4 only:

```text
6 architectures × 2 horizons = 12 cells
```

```text
baseline_graph
protected_read_progress
eligibility_trace
transition_supervision
bridge_eligibility_transition
bridge_query_eligibility_transition
```

## Why this should be decisive

We have already demonstrated that:

```text
memory can be formed
memory can remain causally discriminative
credit assignment can improve H3/H4
```

The remaining experiment is therefore the direct test of whether that preserved
information can be routed into the terminal decision reliably.

## Run

Regression:

```powershell
python .\research\v285\bridge_regression.py
```

Preflight:

```powershell
python .\research\v285\preflight.py --pairs-per-horizon 24
```

Survey:

```powershell
python .\research\v285\run_memory_diagnostic.py --pairs-per-horizon 24 --epochs 10 --batch-size 2 --parallelism 2
```

Exactly 12 cells.
