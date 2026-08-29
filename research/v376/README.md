
# V376 — Fully Grounded Semantic Benchmark

V376 hardens the grounding benchmark before grammar work resumes.

It prevents the main failure modes discovered in V375:

```text
prefix-derived candidate artifacts
unverified ambiguity
stale evidence across revisions
single-demo evaluation
unbounded semantic lookup
```

## Smoke

```powershell
python .\research\v376\fully_grounded_benchmark.py --smoke
```

## Full real ConceptNet benchmark

From the Graph-Topology project root:

```powershell
python .\research\v376\fully_grounded_benchmark.py `
  --conceptnet .\data\conceptnet_compact.db
```

Smaller diagnostic run:

```powershell
python .\research\v376\fully_grounded_benchmark.py `
  --conceptnet .\data\conceptnet_compact.db `
  --limit 10
```

The real run:

1. builds the ConceptNet index once;
2. discovers actual ambiguous surface/candidate pairs from graph structure;
3. requires genuinely asymmetric semantic neighborhoods;
4. derives context directly from those graph edges;
5. tests resolved cases;
6. tests ambiguity/no-context controls;
7. tests belief revision;
8. applies explicit quality gates.

A result of `PASS` means the semantic grounding mechanism passes all benchmark
gates on the sampled real graph.
