
# V410 — Structural realization

V410 addresses the remaining realization-fidelity problem.

The generator is now **constituent-owned** rather than a flat dependency
walker. It explicitly realizes:

```text
NP
VP
PP / oblique phrase
relative / embedded clause
coordination
auxiliary chain
negation
```

It learns morphological realization and dependency ordering from the GUM
training split.

Function words such as:

```text
det
case
aux
cc
neg
```

belong to their owning constituent and are not independently emitted.

The cognitive architecture remains the semantic grounding substrate.

Generated text is judged independently by spaCy.

## Smoke

```powershell
python .\research\v410\v410_structural_realization.py --smoke
```

## Run

```powershell
python .\research\v410\v410_structural_realization.py `
  .\data\UD_GUM `
  --conceptnet .\data\conceptnet_compact.db
```

Controlled:

```powershell
python .\research\v410\v410_structural_realization.py `
  .\data\UD_GUM `
  --conceptnet .\data\conceptnet_compact.db `
  --max-cases 100 `
  --progress-every 25
```

Results:

```text
.\results\v410_structural_realization.json
```

The report also contains realization traces for the first ten cases.
