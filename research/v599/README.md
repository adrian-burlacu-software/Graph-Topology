# V599 — Self-Contained Goal-Directed Annealed Prior

V599 tests annealed best-first path priors. It is runtime-self-contained: the relation-transition profiles and verified-path prior are embedded in the Python file. The only external input is the read-only SQLite graph.

## Run

```powershell
python .\research\v205_refinement_teacher\v599_goal_directed_annealed_prior.py `
  --database ".\results\v562_kg_composition_audit.sqlite" `
  --output ".\results\v599_goal_directed_annealed_prior.json" `
  --workers 20 `
  --seeds 5 `
  --seed-start 59900 `
  --budget 80 `
  --per-node 60 `
  --max-depth 3 `
  --cache-entries 12000 `
  --progress-every 10 `
  --max-probes-per-case 500 `
  --max-case-seconds 2.0 `
  --prior-decay 0.65 `
  --top-predicates 7 `
  --sample-each 600 `
  --middle-limit 150 `
  --case-cap 100 `
  --supported-per-relation 20 `
  --negative-per-relation 20
```

No `--v568`, `--v595`, `--v596`, or `--v597` argument is required.
