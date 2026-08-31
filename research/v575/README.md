# V575.4 — Optimized Multi-Hypothesis Cognitive Controller

V575.4 is the performance rebuild of V575.

The prior version could spend enormous time repeatedly probing a ~45 GB SQLite
graph. This version changes the execution model:

```text
frontier nodes
    ↓
batch indexed subject lookup
    ↓
bounded LRU topology cache
    ↓
compute branch/value features in memory
    ↓
attention quantum
    ↓
batch child topology
    ↓
repeat
```

It also adds hard guards so pathological cases cannot silently consume the
entire run:

```text
--max-probes-per-case
--max-case-seconds
```

and prints periodic:

```text
[CASE x/y] policy rate=... elapsed=... eta=... cache=... batchq=...
```

## Matrix

```text
bfs
depth_branch
multi_depth
multi_depth_branch_value
multi_depth_backtrack
multi_depth_frontier_value
multi_depth_counterfactual
multi_depth_learned_state
full_cognitive_controller
```

`depth_branch` remains the primary validated base.

## Recommended run

```powershell
python .\research\v575\v575_optimized_multi_hypothesis_controller.py --database ".\results\v562_kg_composition_audit.sqlite" --v568 ".\results\v568_20_relation_induction.json" --output ".\results\v575_3_optimized.json" --workers 20 --top-predicates 20 --sample-each 600 --middle-limit 150 --case-cap 100 --supported-per-relation 20 --negative-per-relation 20 --seeds 5 --seed-start 57530 --budget 80 --per-node 60 --max-depth 3 --cache-entries 50000 --progress-every 10 --max-probes-per-case 500 --max-case-seconds 5
```

## Fast / diagnostic run

This keeps the same architecture but reduces the oracle and seeds:

```powershell
python .\research\v575\v575_optimized_multi_hypothesis_controller.py --database ".\results\v562_kg_composition_audit.sqlite" --v568 ".\results\v568_20_relation_induction.json" --output ".\results\v575_3_fast.json" --workers 20 --top-predicates 20 --sample-each 300 --middle-limit 100 --case-cap 60 --supported-per-relation 10 --negative-per-relation 10 --seeds 3 --seed-start 57530 --budget 50 --per-node 40 --max-depth 3 --cache-entries 30000 --progress-every 5 --max-probes-per-case 250 --max-case-seconds 3
```

## Skip the redundant BFS baseline

When the expensive part is the new controller matrix:

```powershell
python .\research\v575\v575_optimized_multi_hypothesis_controller.py --database ".\results\v562_kg_composition_audit.sqlite" --v568 ".\results\v568_20_relation_induction.json" --output ".\results\v575_3_no_bfs.json" --workers 20 --top-predicates 20 --sample-each 600 --middle-limit 150 --case-cap 100 --supported-per-relation 20 --negative-per-relation 20 --seeds 5 --seed-start 57530 --budget 80 --per-node 60 --max-depth 3 --cache-entries 50000 --progress-every 10 --max-probes-per-case 500 --max-case-seconds 5 --skip-bfs
```

## What to watch

The first important performance fields are:

```text
batchq
cache
topology_cache_hits
topology_cache_misses
cases_aborted
```

The first important cognitive fields are:

```text
hypotheses_created
hypothesis_promotions
hypothesis_abandons
backtracks
branch_switches
counterfactual_switches
information_gain
```

A successful result should show that the controller both changes attention
meaningfully and improves supported recovery against `depth_branch` without
creating false proofs.

## V575.4 performance rebuild

The multi-hypothesis controller has a strict no-I/O rule inside edge scoring.
Only the active hypothesis frontier is batch-fetched. Child topology is not
queried candidate-by-candidate.

Use `--only-policy multi_depth` for a fast controller smoke test before the
full matrix.
