# V682: real semantic graph rule discovery

Run the self-contained experiment from the repository root:

```bash
python -m research.v682.run_v682
```

The command fails loudly unless it can open the repository's focused semantic
database at `data/v673_focused_semantic.sqlite` (or the supplied `--database`
path). It loads every direct `subject, relation, object, source` edge, analyzes
all relation pairs, and empirically certifies composition candidates against a
deterministic held-out path split. No fixture graph, predefined aliases, or
predefined composition rules are used.

It writes `graph_stats.json`, `relations.json`, `relation_rules.json`,
`inferred_facts.json`, `proofs.json`, `evaluation.json`, and `ontology.html`.
The self-contained HTML contains the full real graph data and an interactive
canvas 3D globe with level-of-detail rendering, relation filtering, graph-backed
queries, search, node focus, dragging, rotation, pan, zoom, and direct/inferred
proof inspection.

Run the permanent regression suite with:

```bash
python -m unittest research.v682_ontological_learning.test_v682
```
