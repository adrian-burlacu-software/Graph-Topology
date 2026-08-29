
# V299 — Global + Long Eligibility Credit

V297 established that credit assignment is real.

V298 established:

```text
global persistence
+
eligibility
```

is a promising combination.

V299 searches the interaction between those two mechanisms.

## Frozen core

```text
persistent memory
transform dynamics
memory readout
binding planner
```

## Hybrid credit path

```text
delayed prediction error
        ↓
long eligibility trace
        ↓
persistent global signal
        ↓
next graph state
        ↓
next decision
```

## Search dimensions

```text
trace decay
signal decay
injection threshold
```

There are only 12 configurations.

## Smoke

```powershell
python .\research\v299\validate.py
python .\research\v299\search.py --train-seeds 3 --eval-seeds 3 --episodes 8 --horizon 7 --topk 12
```

## Full

```powershell
python .\research\v299\search.py --train-seeds 12 --eval-seeds 12 --episodes 16 --horizon 7 --topk 12
```

## What matters

The winning mechanism should ideally improve both:

```text
absolute held-out accuracy
```

and:

```text
online learning gain
```

If a long eligibility trace improves adaptation but hurts final accuracy, tune
the global signal persistence and injection threshold rather than abandoning
the mechanism.

After V299, reinsert the best hybrid credit mechanism into the broader
graph-native architecture search.

The transformer remains excluded.
