
# V296 — Focused Graph-Native Recombination

V294 identified a graph-native core:

```text
persistent memory
+
transforming dynamics
+
state readout
+
binding/planning
```

V295 isolated credit assignment and found:

```text
immediate       strong
eligibility     strongest learning gain in the focused test
```

V296 freezes the core and only searches the remaining uncertain slots.

## Fixed core

```text
memory = persistent
dynamics = transform
```

## Search

```text
4 readouts
×
3 planners
×
2 credit mechanisms
=
24 architectures
```

Credit candidates:

```text
immediate
eligibility
```

Readouts:

```text
memory
relational
integrative
state
```

Planners:

```text
binding
control
rollout
```

## Why this experiment exists

This is no longer a broad architecture search.

It is a **recombination test**:

```text
Does the discovered credit mechanism improve the discovered graph-native core?
```

The search ranks held-out performance first, then longitudinal learning gain,
then causal synergy.

Top candidates are causally probed by disabling:

```text
readout
planner
credit
```

## Fast smoke

```powershell
python .\research\v296\validate.py
python .\research\v296\recombine.py --train-seeds 2 --eval-seeds 2 --episodes 6 --horizon 7 --topk 8
```

## Full focused search

```powershell
python .\research\v296\recombine.py --train-seeds 6 --eval-seeds 6 --episodes 10 --horizon 7 --topk 12
```

## Interpretation

The result we want is not merely:

```text
higher accuracy
```

We want:

```text
eligibility beats immediate
+
learning gain increases
+
credit ablation hurts
+
readout/planner still contribute
```

If eligibility wins, it becomes the provisional credit mechanism for the next
full graph-native architecture search.

Only after that should the transformer return as a candidate implementation of
one discovered module.


## Smoke validation

The 24-way smoke configuration is:

```text
train seeds: 296, 297
eval seeds: 10296, 10297
episodes/sequence: 6
horizon: 7
```

It has been executed in-process successfully.

A separate subprocess may encounter the host environment's unrelated
spreadsheet-runtime startup hook; that is not part of this experiment.
