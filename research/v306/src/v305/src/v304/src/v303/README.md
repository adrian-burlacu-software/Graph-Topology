
# V303 — Selective Representation

V301 identified interference as the remaining major failure.

V302 added hypothesis/revision and solved rule change and counterfactual
reasoning, leaving interference near chance.

V303 therefore adds a graph-native selection layer.

## Architecture

```text
persistent memory
        ↓
transform dynamics
        ↓
selective representation
        ↓
memory readout
        ↓
binding planner
        ↓
hypothesis / revision
```

## Selective mechanism

The selector maintains persistent relevance values and suppresses nodes that
are structurally irrelevant.

It does NOT:

```text
delete graph topology
inspect answer labels
inspect task names
inspect hidden rule
```

## Search space

```text
weak filter
balanced filter
strong filter
persistent filter
```

## Smoke

```powershell
python .\research\v303\validate.py
python .\research\v303\search.py --seeds 4 --episodes 8 --horizon 9
```

## Full

```powershell
python .\research\v303\search.py --seeds 12 --episodes 16 --horizon 9
```

The primary metric is the interference task. Overall accuracy matters only
after interference improves substantially without destroying the already
working rule-change and counterfactual abilities.

The transformer remains excluded.
