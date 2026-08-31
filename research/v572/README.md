# V572.1 — Robust Balanced Multi-Seed Cognitive Benchmark (bugfix)

V572 is a clean rebuild of the V571 benchmark.

It fixes the recurring inventory representation problem by using a typed
`PredicateInfo(predicate, edge_count)` model internally. The V568 artifact is
also normalized independently and validated before the benchmark begins.

The graph is authoritative. V568 profiles are an optional search-prior
artifact and may have partial overlap with the graph's predicates.

The benchmark:

- builds graph-derived compositional supported cases;
- creates verified hard negatives;
- balances the holdout by target relation;
- hides the direct target edge for supported cases;
- evaluates multiple search mechanisms across multiple seeds;
- reports per-relation results;
- reports mean/std/95% CI;
- keeps the source graph SQLite READ-ONLY.

## Recommended run

From the Graph-Topology repo root:

```powershell
python .\research\v572\v572_robust_benchmark.py --database ".\results\v562_kg_composition_audit.sqlite" --v568 ".\results\v568_20_relation_induction.json" --output ".\results\v572_balanced_cognitive_benchmark.json" --workers 20 --top-predicates 20 --sample-each 600 --middle-out-limit 150 --cases-per-predicate 100 --supported-per-relation 20 --negative-per-relation 20 --seeds 5 --seed-start 57200 --budget 80 --per-node 60 --max-depth 3
```

## Faster run

```powershell
python .\research\v572\v572_robust_benchmark.py --database ".\results\v562_kg_composition_audit.sqlite" --v568 ".\results\v568_20_relation_induction.json" --output ".\results\v572_balanced_fast.json" --workers 20 --top-predicates 20 --sample-each 300 --middle-out-limit 100 --cases-per-predicate 60 --supported-per-relation 10 --negative-per-relation 10 --seeds 3 --seed-start 57200 --budget 50 --per-node 40 --max-depth 3
```

## The matrix

```text
bfs
depth
behavior
family
full_hybrid
shuffled_behavior
shuffled_family
shuffled_all
```

The important interpretation rule is:

```text
mechanism is convincing when:
  improvement over BFS
  + survives multiple seeds
  + does not materially increase false proofs
  + per-relation gains are not concentrated in one skewed relation
```

V572 is intended to answer whether V569/V570's observed effects were robust
or artifacts of relation imbalance and implementation details.


## V572.1 bugfix

Removed an unused `control_profiles` construction that mutated a dictionary
while iterating over it, causing:

    RuntimeError: dictionary changed size during iteration

The evaluator already receives the real V568 profiles and seed-specific
shuffle controls separately, so the mutable derived dictionary was redundant.
