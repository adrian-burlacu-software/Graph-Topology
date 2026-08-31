# V569.1 — Cognitive Semantic Relation Validation Matrix (bugfix)

V569 asks whether the behavioral relation induction from V568 actually helps
the cognitive architecture search the 50GB YAGO + DBpedia graph.

The source graph is READ-ONLY.

It builds graph-derived compositional cases from:

    S --R1--> M --R2--> O
    S --Rtarget--> O

For the primary `NOVEL_COMPOSITION` benchmark, the direct `S--Rtarget-->O`
edge is hidden from the policy. The policy must find a path to the target
within its search budget.

Hard negatives are constructed from similar local endpoint pools where the
target relation is absent.

Strategies compared:

- direct_visible_control
- bounded_bfs
- relation_frequency
- behavior_attention
- induced_family_attention
- adaptive_depth_attention
- hybrid_cognitive

The V568 behavioral profiles are used only as search priors. Behavioral
similarity is not treated as semantic equivalence.

## Recommended run

From the Graph-Topology repo root:

```powershell
python .\research\v569\v569_cognitive_validation_matrix.py --database ".\results\v562_kg_composition_audit.sqlite" --v568 ".\results\v568_20_relation_induction.json" --output ".\results\v569_cognitive_validation.json" --workers 20 --top-predicates 20 --sample-each 600 --middle-out-limit 150 --cases-per-predicate 80 --negative-ratio 0.75 --holdout-supported 500 --holdout-negative 500 --budget 80 --per-node 60 --max-depth 3 --hide-direct --seed 569
```

## Faster run

```powershell
python .\research\v569\v569_cognitive_validation_matrix.py --database ".\results\v562_kg_composition_audit.sqlite" --v568 ".\results\v568_20_relation_induction.json" --output ".\results\v569_cognitive_validation_fast.json" --workers 20 --top-predicates 20 --sample-each 300 --middle-out-limit 100 --cases-per-predicate 40 --holdout-supported 250 --holdout-negative 250 --budget 40 --per-node 40 --max-depth 3 --hide-direct --seed 569
```

## Control run with direct evidence visible

This is useful as a sanity/control benchmark:

```powershell
python .\research\v569\v569_cognitive_validation_matrix.py --database ".\results\v562_kg_composition_audit.sqlite" --v568 ".\results\v568_20_relation_induction.json" --output ".\results\v569_direct_control.json" --workers 20 --top-predicates 20 --sample-each 400 --middle-out-limit 100 --cases-per-predicate 40 --holdout-supported 250 --holdout-negative 250 --budget 40 --per-node 40 --max-depth 3 --no-hide-direct --seed 569
```

## Output

The JSON contains:

    oracle
    policies
    comparison_vs_bfs
    utility_scores
    winner
    per-relation breakdowns

The most important numbers are:

    supported_recovery
    false_proof_rate
    mean_steps

A policy that merely predicts "supported" often is not considered good:
false proof rate is explicitly penalized in the utility score.


## V569.1 bugfix

The policy dispatcher now passes only compatible arguments to each policy. The `direct_visible_control` policy no longer receives `max_depth`.
