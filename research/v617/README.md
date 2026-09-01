# V617 — Strict Entity Resolution + Anti-Contamination

V617 keeps spaCy frozen and fixes the V616 false-identity path.

The pipeline is:

```text
frozen spaCy
    ↓
graph identifier / entity mention extraction
    ↓
exact DB subject/object resolution
    ↓
canonical graph node OR unresolved
    ↓
contextual relation attention
    ↓
semantic goal
    ↓
direct proof / bounded bridge search
    ↓
learning only after verified identity + proof
```

Approximate graph matches are diagnostics only. They can never become canonical
entities and can never train attention.

DBpedia-style URLs are extracted structurally because spaCy may tokenize them as
ordinary text rather than assigning a named-entity label.

Grammar remains fixed:

```text
grammar training = OFF
```

## Smoke

The smoke dynamically finds a real graph subject and relation, then tests:

```text
1. conversation
2. exact identity + proof
3. context carry-forward
4. intentionally unresolved identity
```

```powershell
python .\research\v617\v617_semantic_chat_gateway.py --database ".\results\v562_kg_composition_audit.sqlite" --output ".\results\v617_smoke.json" --trace-output ".\results\v617_smoke_traces.jsonl" --prior-output ".\results\v617_smoke_prior.json" --memory-output ".\results\v617_memory.json" --spacy-model en_core_web_sm --mode smoke --max-hypotheses 12 --relation-vocabulary 200 --goal-budget 40 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --prior-decay 0.65 --seed 61700 --progress-every 1
```
