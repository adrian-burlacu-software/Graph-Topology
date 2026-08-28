
# V283 — Temporal Credit Attack Map

V282 established the graph-native reference lane. V281/V282 established that
deep supervision materially improves H3 while H4 remains difficult.

V283 attacks the remaining temporal-credit/readout problem directly.

## 24-cell survey

```text
baseline_graph
protected_read_progress
eligibility_trace
td_credit
action_binding
transition_supervision

× H1/H2/H3/H4
= 24 cells
```

`baseline_graph` is the stateless transformer control.

`protected_read_progress` is the validated transformer memory base.

The four experimental attacks are independent hypotheses.

## Attack map

### 1. Eligibility traces — highest priority

```text
memory event @ t
      ↓
eligibility trace
      ↓
terminal reward @ T
      ↓
credit flows back to t
```

Instead of forcing every intermediate state to match the terminal action,
earlier states get exponentially decayed terminal supervision.

Prediction:

```text
eligibility_trace > protected_read_progress
especially H3/H4
```

A win means the missing ingredient is likely temporal credit propagation.

### 2. Temporal-difference credit

Bootstrap a value estimate:

```text
V(t) ← gamma * V(t+1)
```

Prediction:

```text
td_credit > protected_read_progress
```

A win would indicate that the terminal objective is simply too sparse/distant,
and bootstrapped learning is sufficient.

### 3. Action-specific memory binding

Create an explicit gate:

```text
workspace + goal
       ↓
action-relevant memory subspace
       ↓
decision
```

Prediction:

```text
action_binding > protected_read_progress
```

A win means the problem is not only credit; the workspace contains information
that the generic readout fails to bind to the correct action.

### 4. Transition supervision

Train the post-step transition together with the terminal action.

Prediction:

```text
transition_supervision > protected_read_progress
```

A win would implicate the dynamics/readout boundary rather than pure temporal
gradient distance.

## Why contrastive memory is no longer here

The previous run showed that contrastive representation separation can produce
very low training loss without improving H3/H4 terminal behavior. It is not
worth spending another cell budget on it.

## Graph-native reference

The original graph designer remains a separate, non-gating reference:

```text
graph_native_reference.py
```

It uses the actual graph simulator and reports its native REUSE/BRANCH memory
behavior. It cannot make transformer preflight fail.

Run it independently:

```powershell
python .\research\v283\graph_native_reference.py --repeats 12
```

## Regression before CUDA

```powershell
python .\research\v283\credit_attack_regression.py
```

This executes:

```text
eligibility_loss
td_value_loss
transition_supervision_loss
training_objective
```

with a differentiable fake model.

## Preflight

```powershell
python .\research\v283\preflight.py --pairs-per-horizon 24
```

## Full survey

```powershell
python .\research\v283\run_memory_diagnostic.py --pairs-per-horizon 24 --epochs 10 --batch-size 2 --parallelism 2
```

Exactly 24 cells.

## Decision rule after the run

```text
eligibility wins
    → implement real eligibility traces as a first-class state/credit path

TD wins
    → move toward recurrent value learning

action binding wins
    → redesign memory→action interface

transition supervision wins
    → redesign recurrent state dynamics

none wins
    → stop adding generic credit tricks and make the graph state itself
      the teacher / structural controller
```

The graph-native reference tells us whether the original graph has already
solved some of the same problem structurally.
