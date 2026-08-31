# V573 — Depth Control Compatibility / Improvement Matrix

V572 established the strongest signal so far: adaptive depth control improved
over BFS, while unconstrained behavior/family priors did not.

V573 therefore treats **depth control as the base mechanism** and asks what
is compatible with it.

## Matrix

```text
bfs
depth
depth_branch
depth_hop
depth_budget
depth_gated_behavior
depth_gated_family
depth_gated_hybrid
depth_branch_behavior
depth_branch_family
depth_full
```

### Hypotheses

`depth_branch`
: use observed intermediate-node branching factor to avoid expensive regions.

`depth_hop`
: use the V568 first-hop -> target-relation composition profile to improve
  the depth-controlled search ordering.

`depth_budget`
: depth-aware policy with explicit budget concentration around the selected
  depth.

`depth_gated_behavior`
: behavior priors are allowed to rank edges only after depth selection and
  only when their local composition signal exceeds a threshold.

`depth_gated_family`
: relation-family similarity is allowed only as a high-confidence
  tie-breaker.

`depth_gated_hybrid`
: combine the two gated priors.

`depth_branch_behavior`
: branch-aware depth + gated behavior.

`depth_branch_family`
: branch-aware depth + gated family.

`depth_full`
: depth + branch + target-hop prior + gated behavior + gated family.

## Why this experiment

The previous result does NOT justify piling all the mechanisms together.

The purpose of V573 is to determine whether the useful depth signal can be
extended without reintroducing the noise observed from the global behavior and
family policies.

The source graph is READ-ONLY.

## Recommended run

From the Graph-Topology repo root:

```powershell
python .\research\v573\v573_depth_compatibility_matrix.py --database ".\results\v562_kg_composition_audit.sqlite" --v568 ".\results\v568_20_relation_induction.json" --output ".\results\v573_depth_compatibility_matrix.json" --workers 20 --top-predicates 20 --sample-each 600 --middle-out-limit 150 --case-cap 100 --supported-per-relation 20 --negative-per-relation 20 --seeds 5 --seed-start 57300 --budget 80 --per-node 60 --max-depth 3
```

## Faster run

```powershell
python .\research\v573\v573_depth_compatibility_matrix.py --database ".\results\v562_kg_composition_audit.sqlite" --v568 ".\results\v568_20_relation_induction.json" --output ".\results\v573_depth_compatibility_fast.json" --workers 20 --top-predicates 20 --sample-each 300 --middle-out-limit 100 --case-cap 60 --supported-per-relation 10 --negative-per-relation 10 --seeds 3 --seed-start 57300 --budget 50 --per-node 40 --max-depth 3
```

## Interpretation

The primary comparisons are against `depth`, not just BFS.

A factor is promising when it:

```text
improves recovery
+ does not increase false proofs
+ reduces or preserves mean steps
+ survives multiple seeds
```

A gated prior is especially interesting if the corresponding ungated mechanism
was weak in V570 but the gated version improves depth.

The JSON includes per-target-relation metrics and confidence intervals.
