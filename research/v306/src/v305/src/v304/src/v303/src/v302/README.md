
# V302 — Hypothesis + Revision Architecture

V301 showed three hard failures:

```text
interference      ≈ chance
rule_change       collapses after the rule changes
counterfactual    ≈ chance
```

More credit tuning did not solve those.

V302 therefore adds a distinct cognitive subsystem:

```text
persistent memory
+
transform dynamics
+
binding planner
+
hypothesis
+
confidence
+
revision
```

## Hypothesis loop

```text
current observations
      ↓
current hypothesis
      ↓
decision
      ↓
delayed outcome
      ↓
contradiction detection
      ↓
confidence update
      ↓
hypothesis revision
      ↓
future decisions
```

The key difference from V301 is that learning no longer simply accumulates a
global “flip” signal. The system maintains an explicit hypothesis and can
revise it when contradictory evidence arrives.

## Smoke

```powershell
python .\research\v302\validation.py
python .\research\v302\evaluation.py --seeds 3 --episodes 8 --horizon 9
```

## Full

```powershell
python .\research\v302\evaluation.py --seeds 12 --episodes 16 --horizon 9
```

## What to look for

The primary target is:

```text
rule_change
```

with:

```text
second_half > first_half
```

The secondary targets are:

```text
interference
counterfactual
```

If rule change improves substantially while the other two stay near chance,
the next step is selective representation/control rather than more hypothesis
tuning.

The transformer remains excluded.
