
# V294 — Fully Discriminated Graph-Native Architecture Search

V293 showed that modular search was ready but the benchmark still allowed
one component to solve too much of the task.

V294 changes the benchmark so each functional area has an explicit necessity
test.

## Cognitive responsibilities

```text
MEMORY
    preserve a transient sensory fact

DYNAMICS
    transform / stabilize that remembered state over time

READOUT
    retrieve internal state and/or structural evidence

PLANNER
    bind retrieved state to later operations and ordered plans

CREDIT
    change persistent internal policy state after delayed feedback
```

## Tasks

```text
memory
binding
dynamics
credit
planning
```

Each task isolates a different cognitive responsibility.

## Anti-shortcut rules

```text
- hidden sensory facts are transient
- persistent topology does not encode the hidden fact
- node names are opaque hashes
- labels are balanced
- distractor paths are always present
- later cues arrive separately
- counterfactual control is graph-encoded
- train and evaluation sequences are disjoint
- cognitive modules cannot inspect task/answer fields
```

## Architecture search

```text
4 memory
× 4 dynamics
× 5 readout
× 4 planner
× 4 credit
=
1280 architectures
```

## Causal architecture test

The top candidates are individually ablated:

```text
memory
dynamics
readout
planner
credit
```

A useful architecture should not merely score well.

It should lose performance when its important module is disabled.

The search reports:

```text
eval accuracy
learning gain
module ablation drops
architecture synergy
```

## Fast smoke

```powershell
python .\research\v294\validation.py
```

Then:

```powershell
python .\research\v294\search.py --train-seeds 2 --eval-seeds 2 --episodes-per-sequence 6 --horizon 7 --topk 8
```

## Full search

Only run the full search after the smoke run shows that the architecture
distribution is actually discriminated and the module ablations are not
universally flat.

```powershell
python .\research\v294\search.py --train-seeds 6 --eval-seeds 6 --episodes-per-sequence 10 --horizon 7 --topk 20
```

Analyze:

```powershell
python .\research\v294\analyze.py .\results\v294_architecture_search.json
```

## Promotion rule

Prefer architectures with:

```text
high held-out accuracy
+
positive learning gain on credit tasks
+
positive memory ablation drop
+
positive dynamics ablation drop
+
positive readout ablation drop
+
positive planner ablation drop
+
positive credit ablation drop
```

Do not promote a strategy if only one mechanism is doing all the work.

## Transformer

Still deliberately absent.

Once the graph-native architecture stabilizes, compare transformer mechanisms
against individual discovered module contracts rather than using a transformer
as the architecture definition.
