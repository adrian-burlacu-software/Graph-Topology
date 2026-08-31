# V596 — Goal-Directed Path-Prior Search

V596 builds directly on V595.

## Experiment

Use the verified path patterns discovered by V595 as a **soft ranking prior** during goal-directed multi-hop proof construction.

The prior:
- is loaded from the V595 JSON result;
- gives partial credit to prefixes of verified path patterns;
- never hard-gates an expansion;
- keeps V595's budget, beam width, depth, and graph access behavior unchanged.

The goal is to improve proof precision without sacrificing the useful multi-hop recovery signal.

## Inputs

- SQLite graph: `v562_kg_composition_audit.sqlite`
- Relation profiles: `v568_20_relation_induction.json`
- V595 results: `v595_goal_directed_path_score.json`

## Run

```powershell
python .\v596_goal_directed_path_prior.py --database .\results\v562_kg_composition_audit.sqlite --v568 .\results\v568_20_relation_induction.json --v595 .\results\v595_goal_directed_path_score.json --output .\results\v596_goal_directed_path_prior.json --workers 20 --seeds 5 --seed-start 59600 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --progress-every 10 --max-probes-per-case 500 --max-case-seconds 2.0 --top-predicates 7 --sample-each 600 --middle-limit 150 --case-cap 100 --supported-per-relation 20 --negative-per-relation 20
```

The graph is read-only.
