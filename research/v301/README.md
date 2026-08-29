
# V301 — Richer Graph-Native Cognition

V300 established that the current frozen cognitive core plus global/eligibility
credit is computationally cheap and easy to iterate.

V301 therefore changes the pressure, not the architecture.

## The benchmark is richer

```text
delayed_memory
sequence_binding
interference
rule_change
planning
counterfactual
```

The environment adds:

```text
multiple delayed cues
multiple distractor routes
ordered plan structure
phase-dependent rule changes
counterfactual control structure
larger nuisance subgraphs
longer horizons
```

## Frozen architecture

```text
persistent memory
+
transform dynamics
+
memory readout
+
binding planner
```

## Credit family

```text
global_fast
global_persistent
long_eligibility
adaptive_baseline
error_accumulator
```

## Why V301 exists

The earlier benchmark could be solved almost immediately.

V301 asks whether the discovered architecture survives:

```text
longer temporal dependency
+
interference
+
rule changes
+
multiple operations
```

without giving the cognitive modules task metadata.

## Smoke

```powershell
python .\research\v301\validate.py
python .\research\v301\search.py --seeds 3 --episodes 8 --horizon 9
```

## Full

Do not run this automatically.

```powershell
python .\research\v301\search.py --seeds 12 --episodes 16 --horizon 9
```

The benchmark intentionally remains cheap. Wall time is reported so that we
can see when we eventually cross from toy-state exploration into genuinely
expensive graph computation.
