
# V304 — Goal-Conditioned Competitive Working Set

V303 showed that generic distractor suppression was invisible.

V304 therefore attacks the actual remaining hypothesis:

```text
the problem is not "remove irrelevant nodes"
the problem is "bind a working representation to the current goal"
```

## Expected failure modes

This experiment explicitly targets:

```text
goal drift
binding collisions
stale working state
distractor competition
working-set persistence
```

## Architecture

```text
persistent memory
        ↓
transform dynamics
        ↓
goal-conditioned competition
        ↓
memory readout
        ↓
binding planner
        ↓
hypothesis/revision
```

## Competition variants

```text
goal_weak
goal_balanced
goal_strong
goal_persistent
```

The selector maintains a temporary competitive working-set representation
instead of applying a global distractor filter.

## Smoke

```powershell
python .\research\v304\validate.py
python .\research\v304\search.py --seeds 4 --episodes 8 --horizon 9
```

## Full

```powershell
python .\research\v304\search.py --seeds 12 --episodes 16 --horizon 9
```

Primary target:

```text
interference
```

The secondary check is that rule change and counterfactual performance do not
collapse.

The transformer remains excluded.
