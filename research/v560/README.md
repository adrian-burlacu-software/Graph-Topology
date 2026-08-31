# V560 — Composition Data Sufficiency / Coverage Audit

V560 answers the question we need to answer before spending more time on a
neural cognitive controller:

> Does the existing semantic graph contain enough independent, confirmed
> compositional evidence to train the desired relation-preserving rules?

It audits the real `facts` + `concepts` graph and opens the source database
read-only.

It reports:

- raw 2-hop and sampled 3-hop path volume
- candidate predicate-composition rules
- GOLD graph-self-confirmed composition examples
- SILVER candidate compositions lacking direct confirmation
- hard-negative candidates
- subject diversity
- endpoint diversity
- source diversity
- domain diversity
- trainability gates: TRAINABLE / PROMISING / THIN / DATA_POOR

The audit deliberately does not turn an unconfirmed path into a positive fact.

## Run

```powershell
python .\research\v560\v560_composition_data_audit.py --source ".\results\full_semantic_memory.sqlite" --shadow ".\results\v560_composition_data_audit.sqlite" --output ".\results\v560_composition_data_audit.json" --workers 20 --max-hops 3 --per-node 80 --seeds 5000 --paths-per-seed 100 --seeds-3hop 1000 --paths-per-seed-3hop 30 --hard-negatives 1000 --top-rules 50 --top-sequences 50 --seed 560
```

Or:

```powershell
.\research\v560\run_v560.bat
```

### Interpretation

The most important fields are:

- `adequacy.trainable_rules`
- `adequacy.promising_rules`
- `adequacy.recommendation`
- `composition_rules.top_rules`

Do not judge the graph from raw path count alone. A composition with 100,000
paths but only 5 distinct subjects and 1 source is not equivalent to a
composition with 500 confirmed examples across hundreds of concepts and
multiple sources.
