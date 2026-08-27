
# V219 — Closed-Loop Cognitive Experiment

V219 tests whether the learned cognitive architecture can operate without the
teacher supplying the next graph state.

Training is still teacher-forced for stability:

```text
state_0 -> action_0
state_1 -> action_1
state_2 -> action_2
```

But after training, autonomous evaluation starts with only `state_0`:

```text
state_0
  ↓
Transformer
  ↓
attention controller
  ↓
predicted action_0
  ↓
REAL STATE TRANSITION
  ↓
model-generated state_1
  ↓
Transformer
  ↓
attention controller
  ↓
predicted action_1
  ↓
REAL STATE TRANSITION
  ↓
...
```

No teacher state is injected after the initial state.

## Primary new metrics

`autonomous_final_action`
: final action accuracy when the model generates its own states.

`autonomous_exact_trajectory`
: percentage of cases where the entire predicted action trajectory exactly
matches the teacher trajectory.

`autonomous_early_stop`
: fraction of rollouts that terminate before the requested step budget.

The teacher-forced metrics remain in the summary so the gap between learned
mapping and autonomous operation is visible.

## Matrix

```text
depth:      d2 d4 d6 d8
iterative:  2  4  6  8
both:       d8+2 d8+4 d8+6 d8+8
```

## Run

```powershell
python .\research\v219\run_all.py
```

Smoke test:

```powershell
python .\research\v219\run_all.py --samples 50 --epochs 2
```

Results:

```text
results/v219/
```

Summary:

```text
results/v219/v219_summary.json
```

The important result is not another tiny improvement in teacher-forced loss.
It is the difference between:

```text
teacher_action
```

and:

```text
autonomous_final_action
```

If the first is high and the second collapses, the architecture has learned the
teacher trajectory but not a stable self-generated cognitive loop.

If autonomous performance remains high, that is a much stronger result.
