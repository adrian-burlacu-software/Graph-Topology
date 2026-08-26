# V209 — Teacher-Distilled Graph Cognitive Controller

V209 is the first graph-model training run that consumes the **cached SmolLM2
teacher trajectories**.

There are no LLM calls during training.

Input:

```text
results/v205r_teacher_dataset.jsonl
```

Architecture:

```text
ConceptNet / cached working-memory state
                |
                v
        Graph Transformer
                |
        +-------+--------+
        |                |
    state latent      goal latent
        |                |
        +-------+--------+
                |
         teacher-distilled
           controller
                |
     +----------+----------+
     |          |          |
   action     pointers   relation
     |          |          |
     +----------+----------+
                |
          next-state latent
```

The teacher supplies:

- explicit goal;
- final accepted action;
- final action arguments;
- resulting working-memory state.

V209 learns from those cached trajectories.

## Important distinction

The LLM is **not** queried here.

The experiment measures whether the graph architecture can absorb the noisy
teacher's behavior and reproduce useful state transitions.

## Outputs

```text
results/v209_teacher_distilled_cognitive.pt
results/v209_teacher_distilled_eval.json
```

## Run from `research/`

```powershell
python .\v209_teacher_distilled_cognitive\train.py --epochs 8 --batch-size 32
```

Then:

```powershell
python .\v209_teacher_distilled_cognitive\evaluate.py
```

For a quick smoke run:

```powershell
python .\v209_teacher_distilled_cognitive\run.py
```

## Initialization

V209 automatically initializes the graph-transformer backbone from:

```text
results/v203_multi_action_cognitive.pt
```

when available.

This is transfer learning:

```text
V203 learned graph/controller representation
        ↓
V209 teacher-distilled fine-tuning
```

Use `--no-v203-init` to train from scratch.

## Main metrics

```text
action_accuracy
source_accuracy
target_accuracy
relation_accuracy
goal_one_step_success
next_state_mse
```

The important architectural test is:

```text
graph state + goal
        ↓
predicted action + arguments
        ↓
actual working-memory mutation
        ↓
goal reached?
```
