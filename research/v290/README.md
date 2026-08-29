
# V290 — Graph-Native Cognitive Algorithm Search

V289 proved the need for a modular graph-native search, but its benchmark had
shortcuts.

V290 fixes those shortcuts before doing another architectural ranking.

## Problems fixed

### 1. Direct answer encoding

The answer is no longer represented by one special memory edge.

The core question is:

```text
source --relation A--> middle --relation B--> target
```

The answer depends on whether the full relational composition exists.

### 2. One-hop lookup

A decoy one-hop route can exist.

A candidate must follow the requested relation sequence.

### 3. Semantic node names

Node identifiers are opaque per episode.

Strategies cannot hard-code concepts such as:

```text
MEM_ONE
MEM_ZERO
TOKEN
```

### 4. Task-family leakage

The algorithm receives:

```text
Graph
+
Query
```

It never receives the task family.

The four families are only data-generation mechanisms.

### 5. Unbalanced answers

Positive and negative instances are generated across seeds for every family.

A strategy cannot get a high score by always predicting the same answer.

### 6. Counterfactual shortcut

Negation is encoded structurally as:

```text
query_control
    --mode-->
negation_marker

query_control
    --applies_to-->
target
```

The planner must discover that generic structure. It is never given a
`counterfactual=True` flag.

### 7. Train/eval contamination

Training and evaluation seeds are disjoint.

The search ranking is based on held-out evaluation accuracy first.

### 8. Causal shortcut detection

Top strategies are subjected to:

```text
normal
memory-path ablation
first-hop counterfactual swap
```

A high score with no causal dependence is not considered a strong cognitive
candidate.

## Search space

```text
3 memory
× 5 credit
× 4 dynamics
× 4 readout
× 4 planning
=
960 strategies
```

## Components

Memory:

```text
structural
activated
route
```

Credit:

```text
none
immediate
eligibility
td
path_reinforcement
```

Dynamics:

```text
static
leaky
persistent
gated
```

Readout:

```text
one_hop
two_hop
consistency
voting
```

Planning:

```text
none
compose
control
two_stage
```

The transformer is completely absent.

## Small smoke

```powershell
python .\research\v290\validate.py
```

Then:

```powershell
python .\research\v290\search.py --train-seeds 3 --eval-seeds 3 --topk 12
```

Full:

```powershell
python .\research\v290\search.py --train-seeds 8 --eval-seeds 8 --topk 20
```

Analyze:

```powershell
python .\research\v290\analyze.py .\results\v290_strategy_search.json
```

## What a real winner should show

```text
high held-out accuracy
+
large memory ablation drop
+
high counterfactual swap correctness
+
high swap sensitivity
```

A trivial topology lookup should fail once the benchmark asks for a relation
composition that is absent or decoyed.

## What happens after the winner

Do not immediately reintroduce the transformer.

First:

```text
freeze the best graph-native composition
        ↓
expand the cognitive task family
        ↓
stress test H4/H6/H8
        ↓
test multi-step planning
        ↓
only then evaluate transformers as module replacements
```

The transformer becomes an implementation candidate, not the assumed
cognitive substrate.


## Additional safeguards

V290 also verifies:

```text
no raw task internals in algorithm code
no hard-coded swapped answer in causal validation
```

The causal swap answer is recomputed from the modified graph and generic
control structure.
