
# V267 — Protected Persistent Workspace Diagnostic

V266 showed that latent memory is formed, but useful decision sensitivity
decays rapidly with repeated state updates.

V267 tests a concrete architectural explanation:

> the recurrent state update overwrites useful memory even when the memory
> representation itself has not completely disappeared.

## New architecture

```text
gated_workspace
```

State update:

```text
candidate = f(attended, goal, old_state)

retain = sigmoid(
    gate(old_state, candidate, goal)
)

next_state =
    retain * old_state
    + (1 - retain) * candidate
```

The retention gate is initialized toward preserving the old state.

## Comparisons

```text
baseline_graph
latent_workspace
gated_workspace
```

at:

```text
H1
H2
H3
H4
```

plus auxiliary-memory diagnostic runs for H3/H4.

## Diagnostics

### Freeze

Keep the first memory state instead of updating it.

If this helps, later updates are likely overwriting useful memory.

### Zero

Erase the terminal workspace.

A performance drop demonstrates causal state use.

### Swap

Replace the terminal workspace with the paired case's workspace.

This asks whether the controller's decision follows the workspace content.

### Trace

At every timestep:

```text
working A/B separation
action-logit A/B separation
workspace retention
logit retention
```

## Interpretation

```text
gated normal > ungated normal
and
gated workspace retention > latent retention
```

supports the overwrite/retention hypothesis.

If the gated state remains distinct but action logits remain insensitive,
the controller/decoder is the bottleneck.

If gating changes neither state retention nor behavior, overwrite is not the
dominant explanation.

## Run

```powershell
python .\research\v267\run_memory_diagnostic.py --pairs-per-horizon 24 --epochs 10 --batch-size 2 --parallelism 2
```

The output contains the full intervention/trace JSON.
