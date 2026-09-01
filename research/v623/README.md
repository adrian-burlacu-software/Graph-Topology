# V623 — Structured Semantic Fact Realization

V623 fixes V622's over-aggressive token-level output filter.

Instead of checking every generated word, the LLM receives a structured,
closed-world fact packet:

```text
QUESTION
SUBJECT
RELATION
RELATION LABEL
VERIFIED OBJECT
EVIDENCE PHRASES
```

The cognitive runtime remains authoritative:

```text
chat
 ↓
frozen spaCy
 ↓
entity/context
 ↓
cognitive search
 ↓
verified subject/relation/object
 ↓
SmolLM3-3B
 ↓
natural-language realization
```

The realizer is forbidden to invent facts or guess what graph identifiers
represent. A post-generation check requires the verified object and the
relation (or structural relation label) to survive the realization. Otherwise
the runtime uses a deterministic fallback based only on the verified result.

Default local model:

```text
C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B
```

## Smoke

```powershell
python .\research\v623\v623_semantic_chat_gateway.py --database ".\results\v562_kg_composition_audit.sqlite" --output ".\results\v623_smoke.json" --trace-output ".\results\v623_smoke_traces.jsonl" --prior-output ".\results\v623_smoke_prior.json" --memory-output ".\results\v623_memory.json" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode smoke --max-hypotheses 12 --relation-vocabulary 200 --goal-budget 40 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --prior-decay 0.65 --seed 62300 --progress-every 1
```

## Chat

```powershell
python .\research\v623\v623_semantic_chat_gateway.py --database ".\results\v562_kg_composition_audit.sqlite" --output ".\results\v623_chat.json" --trace-output ".\results\v623_chat_traces.jsonl" --prior-output ".\results\v623_chat_prior.json" --memory-output ".\results\v623_memory.json" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --max-hypotheses 12 --relation-vocabulary 200 --goal-budget 40 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --prior-decay 0.65 --seed 62300 --progress-every 1
```
