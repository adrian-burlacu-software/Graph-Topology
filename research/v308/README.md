
# V308 — Preserve Alternate States During Settling

V307 found a valuable composition:

```text
active query
+
iterative settling
```

but counterfactual performance collapsed.

V308 changes settling so it does not immediately destroy the alternative.

## Architecture

```text
active query
      ↓
construct branches
      ├── actual state
      └── alternate state
              ↓
       independent settling
              ↓
        preserve both
              ↓
         recombination
              ↓
       hypothesis/revision
              ↓
             action
```

## Configurations

```text
dual_balanced
dual_contrastive
dual_blend
dual_deep
```

## Primary question

Does preserving an alternate internal state recover counterfactual reasoning
while retaining V307's interference gains?

## Smoke

```powershell
python .\research\v308\validate.py
python .\research\v308\search.py --seeds 4 --episodes 8 --horizon 9
```

## Full

```powershell
python .\research\v308\search.py --seeds 12 --episodes 16 --horizon 9
```

The important comparison is:

```text
interference
counterfactual
rule_change
overall
```

The transformer remains excluded.
