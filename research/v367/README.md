
# V367 — Verified Novel Operator Induction

V367 closes the remaining seam:

```text
induced schema
    ↓
executable operator candidates
    ↓
intervention-based verification
    ↓
posterior confidence + margin
    ↓
persistent semantic model
    ↓
reuse on subsequent episodes
```

The benchmark contains a genuinely novel R2 rule:

```text
R2: output = initial_bit + cue1
```

which can produce integer value 2. The architecture must discover and verify
an executable operator rather than falling back to the environment implementation.

## Smoke

```powershell
python .\research\v367\validate.py
python .\research\v367\evaluate.py --seeds 4
```

## Full

```powershell
python .\research\v367\validate.py
python .\research\v367\evaluate.py --seeds 12
```
