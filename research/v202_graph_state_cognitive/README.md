# V202 — Graph-State Cognitive Loop

V202 turns the V201 graph-transformer representation learner into a closed-loop
graph-state learner.

The core loop is:

```text
long-term semantic memory
        |
        v
working graph state_t
        |
        v
graph transformer
        |
        +--> latent state
        |
        +--> designer action
        |
        v
apply action
        |
        v
working graph state_t+1
        |
        v
graph transformer
        |
        v
predict / match next state
```

There is no LLM teacher in V202.

## Self-supervised trajectory

For each ConceptNet edge:

```text
state_t:
    source concept active

target transition:
    target concept becomes active
    source -> target semantic edge becomes active

designer target:
    BIND(source, relation, target)
```

The model learns:

1. relation reconstruction
2. action prediction
3. next-state latent prediction
4. next-state graph reconstruction
5. state consistency between generated and target states

The designer is now a real learned action head rather than a permanently hard-coded
`BIND` output.

## Files

- `graph_state.py` — working-memory graph state and action application.
- `dataset.py` — graph-derived trajectory dataset.
- `model.py` — graph transformer + action head + state transition head.
- `train.py` — self-supervised closed-loop training.
- `evaluate.py` — held-out trajectory evaluation.
- `run.py` — quick smoke run.
- `__init__.py`

## Run from `research/`

Quick smoke test:

```powershell
python .\v202_graph_state_cognitive\run.py
```

Longer training:

```powershell
python .\v202_graph_state_cognitive\train.py --epochs 6 --samples 12000 --batch-size 16
```

Evaluation:

```powershell
python .\v202_graph_state_cognitive\evaluate.py
```

Outputs:

```text
results/v202_graph_state_cognitive.pt
results/v202_graph_state_cognitive_eval.json
```

## Important

The model is trained on graph-derived trajectories. It is not yet a language
system. The next research stage can add a language teacher (for example a 1.7B
model) to supervise parsing or language realization once the graph-state loop
itself is working.
