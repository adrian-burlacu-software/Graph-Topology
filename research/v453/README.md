
# V453 — epistemic-gap curiosity for million-node graphs

The curiosity stage previously appeared to hang at:

```text
probing 5 targets
```

because the target selector repeatedly queried SQLite for every node/family.
With >1M nodes this can take an effectively unbounded amount of time.

V453 changes curiosity selection to bounded, set-based graph analysis:

```text
top 3,000 assistant-relevant nodes
        ↓
one aggregate relation query
        ↓
one attempted-target query
        ↓
in-memory epistemic-gap scoring
        ↓
target
```

It specifically prioritizes missing:

```text
required information
clarification
failure / fallback
action
confirmation
state / memory
result
goal
alternative
computer context
conversation context
```

The probe phase reports its own duration:

```text
curiosity target scan time=...s targets=5
```

so the system cannot silently disappear into target selection again.

## Diagnostic run

```powershell
python .\research\v453\v453_epistemic_curiosity.py `
  --max-concepts 100 `
  --rounds 2 `
  --status-every 10 `
  --teacher-probe 5
```

## Full run

```powershell
python .\research\v453\v453_epistemic_curiosity.py
```

## Existing-net curiosity

```powershell
python .\research\v453\v453_epistemic_curiosity.py `
  --curiosity-only `
  --max-concepts 5000 `
  --rounds 5
```

The persistent semantic memory remains:

```text
C:\Users\adria\Desktop\dev\Graph-Topology\results\assistant_semantic_net.sqlite
```
