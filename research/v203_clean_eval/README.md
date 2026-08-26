# V203 — Multi-Action Cognitive Controller

V202 proved that a graph-transformer can learn a closed loop:

```text
working graph -> latent state -> action -> graph mutation
```

but the action distribution was effectively:

```text
BIND
```

for every example.

V203 fixes that **in one comprehensive experiment**.

## What changes

The training environment now produces several graph-derived cognitive stages:

```text
REUSE
    activate an existing long-term concept in working memory

BIND
    create a semantic working-memory edge

COMMIT
    move a stable working-memory structure toward persistent memory

INHIBIT
    suppress a distractor node

BRANCH
    create a temporary branch from an active concept

CREATE
    create a temporary working-memory concept

NOOP
    state already satisfies its local objective
```

Actions are **not semantic relation labels**. Relations such as `IsA` or
`CapableOf` are arguments to `BIND`/`BRANCH`.

The controller therefore has to learn:

```text
current graph state
        |
        +--> action
        |
        +--> source pointer
        +--> target pointer
        +--> relation module
        |
        v
environment mutation
        |
        v
next graph state
```

## Training

V203 will initialize the graph-transformer backbone from the V202 checkpoint
when it exists:

```text
../results/v202_graph_state_cognitive.pt
```

This means we are **fine-tuning the learned graph representation**, not throwing
away the previous run.

Run from `research/`:

```powershell
python .\v203_multi_action_cognitive\train.py --epochs 5 --samples 10000 --batch-size 16
```

For a quick smoke run:

```powershell
python .\v203_multi_action_cognitive\run.py
```

## Evaluation

```powershell
python .\v203_multi_action_cognitive\evaluate.py
```

The single report contains:

```text
overall action accuracy
argument accuracy
relation accuracy
one-step transition success
3-step rollout success
persistent commit success
per-action accuracy
inhibition success
branch/create success
no-op stability
```

Output:

```text
results/v203_multi_action_cognitive.pt
results/v203_multi_action_cognitive_eval.json
```

## Why this is the next architecture step

The important test is no longer:

```text
"What relation is this?"
```

It is:

```text
"What should the cognitive system do next?"
```

The semantic graph supplies the world state.
The working graph supplies the temporary cognitive state.
The transformer supplies learned attention.
The controller supplies learned action selection.
The environment supplies the transition.

No LLM is required for V203.
