
# V298 — Combined Global Reward + Eligibility Architecture

V297 identified two leading credit hypotheses:

```text
global reward
    best absolute held-out performance

eligibility
    strongest temporal learning behavior
```

V298 combines them instead of choosing one blindly.

## Frozen cognitive core

```text
persistent memory
+
transform dynamics
+
memory readout
+
binding planner
```

## Combined credit architecture

```text
delayed outcome
      ↓
terminal prediction error
      ↓
eligibility trace
      ↓
global learning signal
      ↓
persistent cognitive state
      ↓
next decision
```

A baseline can additionally subtract a running prediction-error baseline.

## Configurations

```text
global_fast
eligibility_balanced
eligibility_long
surprise_eligibility
global_persistent
```

This is a focused hyperparameter search, not another broad architecture
search.

## What each hypothesis tests

```text
global_fast
    can fast global feedback learn immediately?

eligibility_balanced
    does temporal smoothing improve robust learning?

eligibility_long
    does longer credit retention help?

surprise_eligibility
    does baseline-subtracted error improve credit quality?

global_persistent
    does a more persistent global signal help?
```

## Smoke

```powershell
python .\research\v298\validation.py
python .\research\v298\search.py --train-seeds 3 --eval-seeds 3 --episodes 8 --horizon 7
```

## Full

```powershell
python .\research\v298\search.py --train-seeds 12 --eval-seeds 12 --episodes 16 --horizon 7
```

## Promotion rule

Promote a combined credit architecture when it shows:

```text
high held-out accuracy
+
positive online learning gain
+
stable performance across fresh sequences
```

Then reinsert that credit mechanism into the wider graph-native architecture
search.

The transformer remains excluded.
