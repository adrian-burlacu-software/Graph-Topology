
# V297 — Credit Architecture Search

V296 showed:

```text
persistent memory
+
transform dynamics
+
binding planner
```

is a promising frozen core, but the credit slot remained weak.

V297 therefore freezes the core and searches the *kind* of credit architecture.

## Candidate improvements

```text
global_reward
    global neuromodulatory-style signal

eligibility
    temporal eligibility trace

local_path
    structural/path-specific credit

td_error
    prediction-error driven temporal difference

replay
    short experience replay

contextual
    context-conditioned credit

advantage
    baseline-subtracted credit
```

## Why this is different

Credit is no longer just a score printed after an action.

Every credit mechanism has:

```text
inject_state()
modify_decision()
feedback()
```

The learned state is injected into the next graph before decision time.

Therefore the causal path is:

```text
action
  ↓
delayed outcome
  ↓
credit update
  ↓
persistent credit state
  ↓
next graph
  ↓
next decision
```

## Frozen core

```text
memory   = persistent
dynamics = transform
readout  = memory
planner  = binding
```

## Fast smoke

```powershell
python .\research\v297\validation.py
python .\research\v297\search.py --train-seeds 3 --eval-seeds 3 --episodes 8 --horizon 7
```

## Full search

```powershell
python .\research\v297\search.py --train-seeds 12 --eval-seeds 12 --episodes 16 --horizon 7
```

## Interpretation

The important metric is not just total accuracy.

Look for:

```text
higher second-half accuracy
+
larger online learning gain
+
generalization to fresh sequence-local rules
```

The strongest candidate becomes the credit mechanism for the next full
architecture recombination.

The transformer remains excluded.
