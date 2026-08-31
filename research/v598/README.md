# V598 — Annealed Best-First Goal-Directed Path Search

V598 is fully self-contained: it embeds its controller dependency and does **not** import V597 or V596 Python source at runtime.

## Experiment

V597 moves the verified-path prior into best-first frontier ordering. V598 keeps that search architecture but **anneals the prior by path depth**:

- hop 1: full verified-path prior strength
- hop 2: prior × `prior_decay`
- hop 3: prior × `prior_decay²`
- composition and target-closure scores remain active at every depth

Default `prior_decay=0.65` tests whether early exploitation plus deeper exploration is better than applying the learned prior at full strength throughout the path.

Every returned proof is verified against the SQLite graph. The graph is opened read-only.

## Inputs

- SQLite graph: `v562_kg_composition_audit.sqlite`
- Relation profiles: `v568_20_relation_induction.json`
- Verified path prior: `v596_goal_directed_path_prior.json`

No previous-version Python file is required.

## Runline

```powershell
python .\\research\\v205_refinement_teacher\\v598_goal_directed_annealed_prior.py `
  --database ".\\results\\v562_kg_composition_audit.sqlite" `
  --v568 ".\\results\\v568_20_relation_induction.json" `
  --v596 ".\\results\\v596_goal_directed_path_prior.json" `
  --output ".\\results\\v598_goal_directed_annealed_prior.json" `
  --workers 20 `
  --seeds 5 `
  --seed-start 59800 `
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
