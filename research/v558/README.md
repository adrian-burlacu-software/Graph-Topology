# V558 — Semantic Composition + Cognitive Strategy Benchmark

V558 is the next experimental step after V557.

It does not treat every graph path as meaningful. It adds a semantic path
filter and then benchmarks different cognitive search policies.

## What is being tested

The benchmark compares:

- `direct_baseline`
- `semantic_priority`
- `type_guided`
- `relation_guided`
- `adaptive_depth`
- `hybrid_attention`

The benchmark is constructed from actual graph facts. A direct fact `A-R-B`
is selected as an oracle and hidden from the search. The controller must find
an alternative path from A to B.

This tests whether a strategy can exploit the graph structure rather than
simply repeating an already-known direct edge.

## Source safety

The source graph is opened with SQLite read-only mode.

All mined compositions and benchmark results go into the shadow DB:

`results\v558_semantic_cognitive_benchmark.sqlite`

No facts are written back into the source graph.

## Run

```powershell
python .\research\v558\v558_semantic_cognitive_benchmark.py --source ".\results\full_semantic_memory.sqlite" --out ".\results\v558_semantic_cognitive_benchmark.sqlite" --workers 20 --max-hops 4 --per-node 100 --seeds 5000 --max-paths 80 --holdout 300 --budget 80 --meaning-threshold 0.5 --seed 558
```

## Important result

The most important fields are:

- `semantic_filter.unique_meaningful_paths`
- `benchmark.oracle_cases`
- `strategies`
- `winner`

If the search strategies beat the direct baseline and one strategy clearly
wins, we have evidence that the cognitive search policy is doing useful work.

If all strategies are close, the next step is to improve the semantic filter
or benchmark construction before adding more controller complexity.
