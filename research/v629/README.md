# V629 — Realizer Interface Integrity Fix

V629 repairs the V628 regression where `SemanticResultRealizer` was missing
its `_generate()` implementation even though conversation and unresolved
semantic paths called it.

The existing V628 API is preserved rather than rewritten.

```text
verified graph result
    → grounded realizer

unverified named entity / fact
    → unresolved semantic realizer
    → cannot answer from pretrained knowledge

ordinary conversation / general concept
    → conversational realizer
```

The gateway also reads entity-resolution evidence from the selected hypothesis,
where this runtime stores it.

## Temperatures

```text
grounded realization = 0.15
unresolved semantic  = 0.10
conversation         = 0.15
```

## Chat runline

```powershell
python .\research\v629\v629_semantic_chat_gateway.py --database ".\results\v562_kg_composition_audit.sqlite" --output ".\results\v629_chat.json" --trace-output ".\results\v629_chat_traces.jsonl" --prior-output ".\results\v629_chat_prior.json" --memory-output ".\results\v629_memory.json" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --max-hypotheses 12 --relation-vocabulary 200 --goal-budget 40 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --prior-decay 0.65 --seed 62900 --progress-every 1
```

Use these first:

```text
hello
tell me a joke
who was Albert Einstein?
what is a person?
```
