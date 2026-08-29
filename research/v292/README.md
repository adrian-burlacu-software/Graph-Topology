
# V292 — Graph-Native Full-Cognition Search

V292 is the first version in this branch where the benchmark is explicitly
designed around the complete cognitive loop.

The transformer is completely removed.

## Cognitive loop

```text
observe
  ↓
encode transient information
  ↓
working / episodic memory
  ↓
maintain state through time
  ↓
receive delayed outcome
  ↓
assign credit
  ↓
retrieve memory
  ↓
bind later cues
  ↓
plan / transform
  ↓
choose action
  ↓
receive feedback
  ↓
update future behavior
```

## The five task families

```text
recall_bind
relational_query
interference
counterfactual
multi_step
```

### Why memory is genuinely required

The initial fact exists only in a transient sensory node.

Immediately after the memory write phase:

```text
initial_fact
    ↓
removed
```

The persistent graph topology is deliberately independent of the hidden bit.

That prevents a structural readout from simply querying the graph for the
answer.

### Delayed credit is genuine

Each seed creates an episode sequence with a latent rule that is not exposed to
the cognitive algorithm.

The rule is only inferable through:

```text
action
→ feedback
→ credit update
→ later episode
```

This is where:

```text
none
immediate
eligibility
path_reinforcement
```

can genuinely differ.

## Search space

```text
3 memory
× 4 credit
× 4 dynamics
× 4 readout
× 4 planners
=
768 strategies
```

## Anti-shortcut contract

Cognitive modules receive:

```text
Graph
+
generic Query
```

They do not receive:

```text
task family
answer
hidden bit
context bit
third cue bit
latent rule
```

## State dynamics

Strategies can choose:

```text
static
leaky
recurrent
selective
```

This lets us test whether continued graph-state transformation helps or harms
long-horizon cognition.

## Readout

Strategies can use:

```text
memory
relational
integrative
credit
```

The relational path is deliberately not answer-coded. It is useful for
structural attribution and interference handling.

## Planning

Strategies can use:

```text
none
bind
control
rollout
```

The planners operate on generic graph cues and control structure.

## What to look for

A strong cognitive composition should show:

```text
high held-out accuracy
+
memory ablation sensitivity
+
online learning gain
+
robustness across task families
```

The particularly interesting signal is:

```text
second-half accuracy > first-half accuracy
```

for strategies that have real credit mechanisms.

That means feedback is changing later behavior.

## Fast smoke

```powershell
python .\research\v292\validation.py
```

Then:

```powershell
python .\research\v292\search.py --train-seeds 2 --eval-seeds 2 --episodes-per-sequence 6 --horizon 6 --topk 12
```

## Full search

```powershell
python .\research\v292\search.py --train-seeds 8 --eval-seeds 8 --episodes-per-sequence 8 --horizon 6 --topk 20
```

Analyze:

```powershell
python .\research\v292\analyze.py .\results\v292_search.json
```

## Next stage

Do not add the transformer yet.

First freeze the best graph-native compositions and test:

```text
H6
H8
H12
more distractors
longer episode sequences
multiple latent rules
compositional rule changes
```

Then, and only then, bring transformer mechanisms back as candidate
implementations of specific discovered functions.
