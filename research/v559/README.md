# V559 — Semantic Composition Rule Mining + Cognitive Search

V559 inserts a composition-learning layer between the semantic graph and the
cognitive controller.

It mines actual 2-hop/3-hop graph paths, groups predicate sequences, estimates
which sequences behave like relation-preserving compositions, and then trains
and evaluates several cognitive search strategies.

The source graph is opened read-only.

Outputs:
- shadow SQLite: `results\v559_semantic_composition_cognitive.sqlite`
- report JSON: `results\v559_semantic_composition_cognitive.json`

Run from the Graph-Topology repo root:

```powershell
python .\research\v559\v559_semantic_composition_cognitive.py --source ".\results\full_semantic_memory.sqlite" --shadow ".\results\v559_semantic_composition_cognitive.sqlite" --output ".\results\v559_semantic_composition_cognitive.json" --workers 20 --max-hops 3 --per-node 80 --seeds 5000 --max-paths-per-seed 60 --holdout 500 --budget 80 --min-rule-paths 5 --min-rule-score 2.5 --per-rule-cases 150 --epochs 35 --hidden 96 --lr 0.0015 --top-rules 50 --seed 559
```

Strategies compared:
- direct
- bfs
- greedy_semantic
- relation_family
- learned_edge_state
- learned_rich_state
- learned_hybrid

Important metrics:
- supported_recovery
- false_proof_rate
- overall accuracy

The source graph is never updated with inferred facts.
