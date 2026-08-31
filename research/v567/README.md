# V567 — Relation-Aware Targeted Composition Discovery

V567 fixes the two issues in V566:

1. Windows worker connections now receive a `Path`, fixing the `as_uri()`
   failure.
2. Target relations are discovered from the actual 50GB graph vocabulary.
   The script no longer assumes that the database contains canonical names
   such as `has_part` or `located_in`.

For example, the current graph visibly contains relations such as:

```text
is_a
schema:location
yago:partOf
schema:gender
schema:nationality
yago:hasFather
yago:creator
...
```

V567 maps only recognized real predicates into conservative semantic families,
then selects the highest-frequency predicates in each family.

The original predicate names are preserved in all composition results.

## Recommended run

From the Graph-Topology repo root:

```powershell
python .\research\v567\v567_relation_aware_composition.py --database ".\results\v562_kg_composition_audit.sqlite" --output ".\results\v567_relation_aware_composition.json" --workers 20 --first-sample 1000 --max-second-edges 200 --max-paths-per-predicate 50000 --in-chunk 300 --min-paths 20 --min-predicate-edges 1000 --max-predicates-per-family 8 --top-rules 50 --seed 567
```

## Faster first pass

```powershell
python .\research\v567\v567_relation_aware_composition.py --database ".\results\v562_kg_composition_audit.sqlite" --output ".\results\v567_relation_aware_fast.json" --workers 20 --first-sample 300 --max-second-edges 100 --max-paths-per-predicate 20000 --in-chunk 300 --min-paths 10 --min-predicate-edges 1000 --max-predicates-per-family 4 --top-rules 30 --seed 567
```

## Structural families only

```powershell
python .\research\v567\v567_relation_aware_composition.py --database ".\results\v562_kg_composition_audit.sqlite" --output ".\results\v567_structural.json" --workers 20 --families is_a has_part part_of contains made_of --first-sample 2000 --max-second-edges 250 --max-paths-per-predicate 75000 --min-paths 20 --min-predicate-edges 1000 --max-predicates-per-family 8 --top-rules 50 --seed 567
```

## Why this version matters

The 50GB graph is heterogeneous. A semantic benchmark must first discover the
actual predicates, then decide which ones belong to the intended semantic
families. Treating `has_part` as a literal relation name when the graph stores
equivalent structure as `yago:partOf` was making the previous benchmark
silently blind.

V567 remains source-graph read-only and does not create inferred facts in the
source.
