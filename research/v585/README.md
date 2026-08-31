# V585 — Fixed Self-Contained Parallel Cognitive Matrix

V585 fixes the V582 case-schema failure:

`Case.__init__() got an unexpected keyword argument 'relation'`

The oracle uses `relation`; the embedded controller's immutable `Case` uses
`target_relation`. V585 explicitly translates the oracle representation into
the controller representation and constructs the frozen Case directly.

It also:

- keeps the SQLite semantic graph READ-ONLY;
- keeps global parallelism capped by `--workers`;
- keeps one bounded cache per job;
- normalizes the database CLI argument to `Path` before entering the
  controller;
- prints the actual `evaluate_policy` and `Case` signatures before the
  expensive matrix begins;
- contains the controller source, so there is no runtime dependency on V575,
  V576, V577, V578, V579, V580, V581, V582, or V583 files;
- is compile-checked and its embedded Case constructor is instantiated during
  build validation.

## Run

```powershell
python .\research\v585\v585_parallel_matrix.py --database ".\results\v562_kg_composition_audit.sqlite" --v568 ".\results\v568_20_relation_induction.json" --output ".\results\v585_parallel_matrix.json" --workers 20 --seeds 5 --seed-start 58500 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --progress-every 10 --max-probes-per-case 500 --max-case-seconds 2 --top-predicates 7 --sample-each 600 --middle-limit 150 --case-cap 100 --supported-per-relation 20 --negative-per-relation 20
```
