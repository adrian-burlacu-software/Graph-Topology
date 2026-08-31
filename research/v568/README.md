# V568-20 — Top 20 Semantic Relation Behavioral Probe

This is the corrected fast version of the relation-induction experiment.

It deliberately profiles only the top 20 predicates by actual edge count.

Crucially, it does NOT scan every row belonging to each predicate. Instead,
each worker takes bounded random rowid windows and accepts matching predicate
rows until its sample is full or the probe ceiling is reached.

The source database is read-only.

## Recommended run

From the Graph-Topology repo root:

```powershell
python .\research\v568_20\v568_top20_relation_induction.py --database ".\results\v562_kg_composition_audit.sqlite" --output ".\results\v568_20_relation_induction.json" --workers 20 --top-predicates 20 --sample-size 600 --window-size 5000 --max-probes 5000 --max-second-edges 150 --max-paths 12000 --neighbors 5 --seed 56820
```

## Faster run

```powershell
python .\research\v568_20\v568_top20_relation_induction.py --database ".\results\v562_kg_composition_audit.sqlite" --output ".\results\v568_20_fast.json" --workers 20 --top-predicates 20 --sample-size 250 --window-size 5000 --max-probes 2500 --max-second-edges 100 --max-paths 5000 --neighbors 5 --seed 56820
```

## What it measures

- actual top-20 predicate inventory
- bounded random predicate samples
- two-hop relation distributions
- endpoint confirmation distributions
- inverse overlap
- intermediate-node degree
- behavioral nearest neighbors
- candidate similar predicate pairs
- predicates worth semantic review

Behavioral similarity is not treated as semantic equivalence. It is a discovery
signal for the cognitive architecture's next validation stage.
