# V590 — Fixed Goal-Directed Search

V590 fixes a namespace boundary bug in V589.

The goal-directed evaluator is defined in the V590 runner module, while
`second_profile()` and `target_family()` belong to the embedded controller
module. V589 called them as local globals, producing:

```text
NameError: name 'target_family' is not defined
```

V590 explicitly calls them through `v575.*` and also makes the local
standard-library dependencies (`math`, `random`, `defaultdict`) explicit.

The experiment remains self-contained apart from:
- the SQLite semantic graph;
- the V568 profile JSON.

## Run

```powershell
python .\research\v590\v590_goal_directed_search.py --database ".\results\v562_kg_composition_audit.sqlite" --v568 ".\results\v568_20_relation_induction.json" --output ".\results\v590_goal_directed.json" --workers 20 --seeds 5 --seed-start 59000 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --progress-every 10 --max-probes-per-case 500 --max-case-seconds 2 --top-predicates 7 --sample-each 600 --middle-limit 150 --case-cap 100 --supported-per-relation 20 --negative-per-relation 20
```
