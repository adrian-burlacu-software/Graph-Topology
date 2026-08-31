# V570 — Cognitive Mechanism Ablation Matrix

V570 isolates which mechanism produced V569's improvement.

The SAME graph-derived holdout is used for every policy.

Policies:

```text
bfs
frequency
behavior
family
depth
behavior_family
behavior_depth
family_depth
full_hybrid
```

Controls that deliberately destroy the relation-specific alignment of the
V568 priors:

```text
shuffled_behavior
shuffled_family
shuffled_all
```

The primary benchmark hides the direct confirmation edge for supported
compositional cases.

Metrics:

- accuracy
- supported recovery
- false proof rate
- mean steps
- budget exhaustion
- performance by target relation
- delta versus BFS
- mechanism-level deltas
- shuffle-test degradation

The source graph is opened read-only. No inferred facts are written back.

## Recommended run

From the Graph-Topology repo root:

```powershell
python .\research\v570\v570_cognitive_mechanism_ablation.py --database ".\results\v562_kg_composition_audit.sqlite" --v568 ".\results\v568_20_relation_induction.json" --output ".\results\v570_mechanism_ablation.json" --workers 20 --top-predicates 20 --sample-each 600 --middle-out-limit 150 --cases-per-predicate 60 --negative-ratio 0.75 --holdout-supported 500 --holdout-negative 500 --budget 80 --per-node 60 --max-depth 3 --seed 570
```

## Faster run

```powershell
python .\research\v570\v570_cognitive_mechanism_ablation.py --database ".\results\v562_kg_composition_audit.sqlite" --v568 ".\results\v568_20_relation_induction.json" --output ".\results\v570_mechanism_ablation_fast.json" --workers 20 --top-predicates 20 --sample-each 300 --middle-out-limit 100 --cases-per-predicate 30 --negative-ratio 0.75 --holdout-supported 250 --holdout-negative 250 --budget 40 --per-node 40 --max-depth 3 --seed 570
```

Interpretation:

- If `behavior > family`, behavioral composition priors contribute more.
- If `family > behavior`, induced relation-family similarity contributes more.
- If `full_hybrid > behavior_family`, adaptive depth contributes additional
  value.
- If `behavior_family > max(behavior, family)`, there is interaction/synergy.
- If shuffled controls collapse toward BFS, the V568 relation alignment is
  doing real work rather than merely adding arbitrary ranking noise.

Do not interpret the mechanism winner as universal; it is a result under this
specific graph, oracle construction, budget, and sample.


## V570.1 bugfix

The `frequency` ablation is now a pure global predicate-frequency baseline and no longer receives V568 behavioral composition scores.
