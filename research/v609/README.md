# V609 — Hypothesis-Driven Semantic Chat Gateway

V609 removes the handwritten conversational exception.

There is no rule like:

```text
hello -> conversation
born -> schema:birthPlace
where -> schema:location
```

Instead, **intent is itself a hypothesis** produced from spaCy structure:

```text
spaCy
  ↓
POS / dependencies / root / entities / question shape
  ↓
intent hypotheses
  ↓
relation hypotheses
  ↓
persistent context + learned memory
  ↓
global conditional attention
  ↓
bounded graph search
  ↓
evidence
  ↓
selected interpretation
```

Conversation-only hypotheses have an empty relation and therefore never get
forced through semantic graph search.

The context layer persists entity mentions, active subject, recent turns,
and lightweight relation outcome memory. Pronouns can therefore resolve from
conversation state without hardcoding named entities.

## Install

```powershell
python -m pip install spacy
python -m spacy download en_core_web_sm
```

## Smoke

```powershell
python .\research\v609\v609_semantic_chat_gateway.py --database ".\results\v562_kg_composition_audit.sqlite" --output ".\results\v609_smoke.json" --trace-output ".\results\v609_smoke_traces.jsonl" --prior-output ".\results\v609_smoke_prior.json" --memory-output ".\results\v609_memory.json" --spacy-model en_core_web_sm --mode smoke --max-hypotheses 12 --relation-vocabulary 200 --goal-budget 40 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --prior-decay 0.65 --progress-every 1
```

## Interactive

```powershell
python .\research\v609\v609_semantic_chat_gateway.py --database ".\results\v562_kg_composition_audit.sqlite" --output ".\results\v609_semantic_chat.json" --trace-output ".\results\v609_chat_traces.jsonl" --prior-output ".\results\v609_global_attention_prior.json" --memory-output ".\results\v609_memory.json" --spacy-model en_core_web_sm --mode chat --max-hypotheses 12 --relation-vocabulary 200 --goal-budget 40 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --prior-decay 0.65 --progress-every 1
```

The console output remains copy/paste friendly and all traces/memory are
persisted.
