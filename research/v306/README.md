
# V306 — Beyond Compositional Binding

V305 showed explicit candidate generation + winner binding barely improved
interference.

V306 therefore broadens the hypothesis space rather than tuning binding.

## Hypotheses

```text
role_separation
    keep typed roles separate until decision time

relational_messages
    aggregate evidence through graph relations

iterative_settling
    repeatedly settle competing internal states

episodic_chunking
    reuse repeated state/context combinations

counterfactual_sim
    compare actual and alternative internal states

active_query
    allocate computation toward the most useful cue
```

## Frozen pieces

```text
persistent memory
transform dynamics
hypothesis/revision controller
```

Only the cognitive overlay changes.

## Smoke

```powershell
python .\research\v306\validation.py
python .\research\v306\search.py --seeds 4 --episodes 8 --horizon 9
```

## Full

```powershell
python .\research\v306\search.py --seeds 12 --episodes 16 --horizon 9
```

The primary metric remains the interference task.

A useful hypothesis should move interference materially without destroying
rule-change and counterfactual performance.
