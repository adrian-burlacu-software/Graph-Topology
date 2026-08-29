
# V300 — Frozen Core + Credit Recombination

V299 selected a promising credit family:

```text
global persistent signal
+
long eligibility trace
```

V300 freezes the graph-native cognitive core and compares the two strongest
operating regimes from that family.

## Frozen core

```text
persistent memory
transform dynamics
memory readout
binding planner
```

## Credit variants

```text
slow_high_signal
long_trace
```

## Why this experiment matters

We are now asking:

```text
which credit regime best integrates with the discovered graph-native core?
```

rather than searching everything again.

## Smoke

```powershell
python .\research\v300\validate.py
python .\research\v300\experiment.py --train-seeds 3 --eval-seeds 3 --episodes 8 --horizon 7
```

## Full

```powershell
python .\research\v300\experiment.py --train-seeds 12 --eval-seeds 12 --episodes 16 --horizon 7
```

The script reports the number of episode runs, state-transition steps and wall
time so an unexpectedly fast result is visible rather than hidden.

The transformer remains excluded.
