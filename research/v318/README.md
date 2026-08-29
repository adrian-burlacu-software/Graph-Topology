
# V318 — Explicit Rule Schemas / Causal Explanations

V317 showed that binary rule induction was too impoverished.

V318 represents rules as explicit schemas:

```text
selector + operator
```

Selectors:

```text
memory
cue1
cue2
cue3
memory+c1
memory+c2
memory+c3
```

Operators:

```text
identity
invert
```

Total:

```text
14 candidate causal explanations
```

## Induction

```text
structured working memory
        ↓
execute every candidate schema
        ↓
score predictive consistency
        ↓
simplicity tie-break
        ↓
pending challenger
        ↓
commit explanation
```

## Rule change

The explicit rule-change marker starts a fresh induction phase:

```text
old schema
   ↓
regime marker
   ↓
fresh evidence window
   ↓
new schema
```

Old knowledge is retained as the previously committed explanation; old evidence
does not permanently dominate the new rule.

## Smoke

```powershell
python .\research\v318\validate.py
python .\research\v318\search.py --seeds 4 --episodes 8 --horizon 9
```

## Full

```powershell
python .\research\v318\search.py --seeds 12 --episodes 16 --horizon 9
```


Verified: schema competition uses simplicity to resolve equivalent causal explanations.
