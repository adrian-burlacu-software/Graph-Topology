# V612 — Semantic Chat Gateway

V612 fixes the V611 runtime crash:

```text
AttributeError: 'Context' object has no attribute 'learn_relation'
```

The cause was another API drift: V611's learner still called an older
`Context.learn_relation()` method even though the persistent memory model had
moved to:

```text
record_turn()
remember_path()
relation_outcomes
```

V612 aligns the learning adapter with the actual memory contract.

The fix also makes a successful semantic result persist its actual relation
path, so interaction traces become useful semantic memory.

## Smoke

```powershell
python .\research\v612\v612_semantic_chat_gateway.py --database ".\results\v562_kg_composition_audit.sqlite" --output ".\results\v612_smoke.json" --trace-output ".\results\v612_smoke_traces.jsonl" --prior-output ".\results\v612_smoke_prior.json" --memory-output ".\results\v612_memory.json" --spacy-model en_core_web_sm --mode smoke --max-hypotheses 12 --relation-vocabulary 200 --goal-budget 40 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --prior-decay 0.65 --seed 61200 --progress-every 1
```

## Interactive

```powershell
python .\research\v612\v612_semantic_chat_gateway.py --database ".\results\v562_kg_composition_audit.sqlite" --output ".\results\v612_semantic_chat.json" --trace-output ".\results\v612_chat_traces.jsonl" --prior-output ".\results\v612_global_attention_prior.json" --memory-output ".\results\v612_memory.json" --spacy-model en_core_web_sm --mode chat --max-hypotheses 12 --relation-vocabulary 200 --goal-budget 40 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --prior-decay 0.65 --seed 61200 --progress-every 1
```

No previous V-version artifact is required.
