
# V260 — Causal Input-Verified Architecture Benchmark

V260 fixes the core V259 problem: a few supposed causal variables were present
only as dataset metadata or were not guaranteed to reach the model.

## Hard rule

A causal claim is not allowed into the experiment unless the preflight proves
that changing the claimed variable changes an actual model decision input.

Checks:

```text
P1 memory:
    hidden instruction changes initial graph

P2 action context:
    previous action changes at the actual previous-action timestep

P3 attention:
    goal focus changes the model decision

P4 progress:
    stage is an actual graph token

P5 action + progress:
    stage + previous action are actual inputs
```

## Run preflight

```powershell
python .\research\v260\preflight.py --pairs-per-probe 12
```

Expected:

```text
P1_memory_input_wiring: PASS
P2_previous_action_wiring: PASS
P3_goal_focus_wiring: PASS
P4_stage_input_wiring: PASS
P5_previous_action_wiring: PASS
TASK / CAUSAL INPUT PREFLIGHT: PASS
```

The preflight deliberately fails if an asserted causal field is invisible to
the model.

## Run benchmark

```powershell
python .\research\v260\run_benchmark.py --pairs-per-probe 12 --epochs 2 --batch-size 8 --parallelism 2
```

The benchmark runs two architecture/horizon cells concurrently.

## Readout

Each cell reports:

```text
P1
P2
P3
P4
P5
zero_mem
```

and paired counterfactual rates.

A pair only counts as solved when both members are correct.

The goal is to test the predicted architectural contrasts directly, not to
reward generic task accuracy.

## Prediction source

See:

```text
ARCHITECTURAL_PREDICTIONS_MATRIX.md
```
