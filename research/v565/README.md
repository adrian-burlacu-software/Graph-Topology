# V565 — Targeted Composition Discovery / Coverage

V565 audits the 50GB YAGO + DBpedia graph by targeting the semantic
relations that matter to the cognitive architecture.

It does NOT perform a global edges×edges join.

For every selected first-hop relation, it:
1. samples actual edges for that relation;
2. follows the sampled intermediate nodes through the subject index;
3. discovers observed two-hop predicate sequences;
4. checks whether the original subject and final endpoint are directly related;
5. measures confirmation rate, subject diversity, endpoint diversity, and
source diversity.

Default target relations:

```text
is_a
has
has_part
part_of
contains
made_of
has_property
capable_of
used_for
located_in
causes
```

The source graph is read-only.

## Run

From the Graph-Topology repo root:

```powershell
python .\research\v565\v565_targeted_composition.py --database ".\results\v562_kg_composition_audit.sqlite" --output ".\results\v565_targeted_composition.json" --relations is_a has has_part part_of contains made_of has_property capable_of used_for located_in causes --first-sample 1000 --max-second-edges 200 --min-paths 20 --top-rules 30 --seed 565
```

### Faster first pass

```powershell
python .\research\v565\v565_targeted_composition.py --database ".\results\v562_kg_composition_audit.sqlite" --output ".\results\v565_targeted_composition_fast.json" --first-sample 300 --max-second-edges 100 --min-paths 10 --top-rules 20 --seed 565
```

### Focus only on the structural relations

```powershell
python .\research\v565\v565_targeted_composition.py --database ".\results\v562_kg_composition_audit.sqlite" --output ".\results\v565_structural_composition.json" --relations is_a has_part part_of contains made_of --first-sample 2000 --max-second-edges 250 --min-paths 20 --top-rules 50 --seed 565
```

The important result is the per-relation coverage profile, not only the global
verdict.

A "STRONG" relation here means the graph has repeated direct endpoint
confirmations for multiple observed compositions. It is evidence for learning
a composition rule, not proof that the rule is universally valid.
