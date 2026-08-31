# V595 — Goal-Directed Path-Level Relation Composition

V593 established that explicit multi-hop proof construction is a strong
inductive bias:

- 55.72% accuracy
- 33.57% supported recovery
- 20.61% false-proof rate

V594 showed that explicit reverse bridging was too broad and regressed.

V595 keeps V593's bounded multi-hop forward search but changes the scoring
unit:

```text
V593:
    score(r1) + score(r2)

V595:
    score(r1 -> r2 -> target)
```

The controller therefore evaluates relation sequences as complete paths.
A relation can be useful in one composition and useless in another.

Depth 2 is always expanded when a first-hop candidate exists. Depth 3 scores
the complete three-relation sequence. Positive predictions require a
continuous >=2-hop path ending in the requested relation/object and every
edge is independently verified in SQLite.

The run also records the most common relation-path patterns and successful
patterns for later analysis.

## Run

```powershell
python .\research\v595\v595_goal_directed_path_score.py --database ".\results\v562_kg_composition_audit.sqlite" --v568 ".\results\v568_20_relation_induction.json" --output ".\results\v595_goal_directed_path_score.json" --workers 20 --seeds 5 --seed-start 59500 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --progress-every 10 --max-probes-per-case 500 --max-case-seconds 2 --top-predicates 7 --sample-each 600 --middle-limit 150 --case-cap 100 --supported-per-relation 20 --negative-per-relation 20
```

### Watch

```text
d1 > 0
d2 > 0
d3 > 0
verified > 0
```

The JSON additionally contains `top_success_patterns`, which tells us which
relation sequences actually formed verified proofs.

SQLite is read-only. No previous-version Python file is required at runtime.
