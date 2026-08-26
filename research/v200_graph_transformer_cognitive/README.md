# V200 — Graph Transformer Cognitive Architecture

This is the architectural pivot from hand-designed attention rules to a **learned,
general graph-transformer mechanism** operating over a semantic memory graph and
a separate working-memory graph.

## Architecture

```text
ConceptNet / learned long-term memory
               |
               v
       Long-term memory
               |
        retrieve a local
        semantic neighborhood
               |
               v
       Working-memory graph
               |
               v
      Graph Transformer
       - node states
       - edge states
       - sparse graph attention
       - residual updates
               |
               v
        latent working state
               |
               v
          Designer head
       REUSE / CREATE / BRANCH
       INHIBIT / BIND / COMMIT
```

The important change is that there are **no hand-authored IsA/CapableOf/etc.
attention rules** in the transformer. Relation identities are input features;
the learned attention mechanism decides which interactions matter.

## What is implemented

### `long_term_memory.py`
Loads a bounded dictionary-centered ConceptNet graph from:

```text
../data/conceptnet_compact.db
```

and exposes local subgraphs.

### `working_graph.py`
Represents a temporary cognitive graph. It supports nodes, typed edges,
activations, provenance, and working-memory mutation.

### `graph_transformer.py`
A sparse graph transformer with:

- learned node embeddings
- learned relation/edge embeddings
- graph-masked multi-head attention
- edge-conditioned message features
- residual connections
- layer normalization
- feed-forward blocks
- graph pooling

It also exposes:
- relation prediction
- edge scoring
- latent state extraction

### `designer.py`
A learned designer head over the working graph latent state.

Actions:

```text
REUSE
CREATE
BRANCH
INHIBIT
BIND
COMMIT
```

The head is intentionally **not** hard-coded to semantic relations.

### `dataset.py`
Builds training examples directly from ConceptNet:

- edge reconstruction
- masked relation prediction
- graph-consistency examples

This is the initial teacher-free learning baseline.

### `train.py`
Trains the graph transformer on ConceptNet-derived local graphs.

### `evaluate.py`
Runs a compact held-out suite and reports:

- relation prediction accuracy
- top-k relation accuracy
- source/target edge recovery
- latent-state consistency
- designer action distribution

### `run.py`
Convenience runner for a quick end-to-end smoke test.

## First run

From `research/`:

```powershell
python .\v200_graph_transformer_cognitive\run.py
```

## Training

For a short CUDA run:

```powershell
python .\v200_graph_transformer_cognitive\train.py
```

Optional:

```powershell
python .\v200_graph_transformer_cognitive\train.py --epochs 10 --samples 20000 --batch-size 64
```

## Evaluation

```powershell
python .\v200_graph_transformer_cognitive\evaluate.py
```

## Dependencies

```text
torch
```

The model is intentionally small enough to iterate locally.

## Next teacher integration

The code does not depend on an LLM teacher yet.

A future teacher can supply:
- semantic state targets
- working-memory transitions
- language -> graph parses
- graph -> language realization targets

without changing the graph-transformer core.

The long-term goal is:

```text
teacher ON:
    stronger LLM supervises latent/graph transitions

teacher OFF:
    graph transformer + designer operates independently
```
