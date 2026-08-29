
# V289 — Graph-Native Cognitive Algorithm Search

This temporarily removes the transformer from the loop.

The goal is not to find another neural architecture. The goal is to discover
whether a useful cognitive algorithm can be assembled from small
graph-native mechanisms.

## Search space

Five independently swappable families:

```text
MEMORY
    edge
    activation
    persistent_slot

CREDIT
    none
    immediate
    eligibility
    td

DYNAMICS
    static
    leaky
    persistent
    gated

READOUT
    structural
    activation
    voting

PLANNING
    none
    one_step
    two_step
```

Total:

```text
3 × 4 × 4 × 3 × 3 = 432
```

composable graph-native strategies.

The search is intentionally CPU-cheap.

## Tasks

```text
delayed_recall
interference
composition
counterfactual
```

The tasks are deliberately tiny. They are used to distinguish algorithmic
roles, not to claim general intelligence.

## Search procedure

Stage 1:

```text
432 strategies
× small seed set
× H2/H4
```

Stage 2:

```text
top K
↓
memory ablation
↓
hidden-state swap
↓
causal validation
```

A strategy should not be promoted merely because it gets high task accuracy.

## What makes a candidate interesting?

```text
high normal accuracy
+
large memory-ablation drop
+
high counterfactual swap correctness
```

That combination suggests the graph state is doing causal computational work.

## Why this is different from the transformer work

The transformer is intentionally absent.

Later, after the winning graph-native components are known, the transformer
can be evaluated as a possible implementation of one module:

```text
memory
credit
dynamics
readout
planning
```

It does not get to define the architecture.

## Run

```powershell
python .\research\v289\search.py --seeds 12 --horizons 2,4 --topk 12
```

For an ultra-fast smoke search:

```powershell
python .\research\v289\search.py --seeds 3 --horizons 2,4 --topk 6
```

Analyze:

```powershell
python .\research\v289\analyze.py .\results\v289_strategy_search.json
```

## Next stage

Once the search produces stable graph-native winners, freeze the winners and
build a second-stage search around:

```text
graph-native winner
×
new memory mechanisms
×
new planning mechanisms
×
new credit mechanisms
```

Only then revisit the transformer as a candidate module.
