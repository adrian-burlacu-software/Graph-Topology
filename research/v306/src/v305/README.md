
# V305 — Competitive Hypothesis Binding

V303 showed generic relevance filtering had no effect.

V304 showed goal-conditioned suppression/competition still had no effect.

V305 therefore changes the mechanism itself:

```text
generate candidate interpretations
        ↓
score candidates against the current graph
        ↓
retain a top-k working set
        ↓
bind the winner
        ↓
apply hypothesis/revision
```

This explicitly targets:

```text
binding collisions
goal drift
competing interpretations
working-set formation
```

Smoke:

```powershell
python .\research\v305\validate.py
python .\research\v305\search.py --seeds 4 --episodes 8 --horizon 9
```

Full:

```powershell
python .\research\v305\search.py --seeds 12 --episodes 16 --horizon 9
```
