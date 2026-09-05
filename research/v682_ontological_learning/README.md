# V682: clean semantic world

Run the self-contained experiment from the repository root:

```bash
python -m research.v682.run_v682
```

The command fails loudly unless it can open the repository's immutable focused
semantic database at `data/v673_focused_semantic.sqlite` (or the supplied
`--database` path). It loads every source edge, normalizes concepts only where
direct structural evidence supports it, retains all relation distinctions, and
bounds dense low-information evidence in the clean projection. It discovers and
certifies composition candidates with held-out precision and lift over the
direct relation base rate. No fixture graph, predefined aliases, or predefined
composition rules are used.

It writes `clean_graph.json`, `concepts.json`, `relations.json`, `rules.json`,
`inferred_facts.json`, `proofs.json`, `evaluation.json`, and
`knowledge_globe.html`. The self-contained globe contains the actual clean
canonical graph, with level-of-detail rendering, relation filtering, graph-backed
queries, search, node focus, dragging, rotation, pan, zoom, provenance, and
direct/inferred proof inspection.

Run the permanent regression suite with:

```bash
python -m unittest research.v682_ontological_learning.test_v682
```
