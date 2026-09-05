# V682: compact ontological reasoning

Run the self-contained experiment from the repository root:

```bash
python -m research.v682.run_v682
```

The command builds a compact canonical graph from explicit facts, normalizes
declared semantic relation aliases, derives facts only in memory, and writes
`output/ontology.json`, `inferred_facts.json`, `proofs.json`, `evaluation.json`,
and a self-contained `ontology.html` visualization. The visualization renders
the actual direct and inferred graph edges; selecting an entity focuses its
neighborhood and clicking an inferred edge shows its proof.

Run the permanent regression suite with:

```bash
python -m unittest research.v682_ontological_learning.test_v682
```
