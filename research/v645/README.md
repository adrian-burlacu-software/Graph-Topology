# V645 — Live Distillation Runtime Integrity

V645 fixes the V644 failure:

```text
NameError: name 'apply_live_distillation' is not defined
```

The gateway now includes both required local helpers:

```text
distilled_choice()
apply_live_distillation()
```

The live architecture remains:

```text
graph candidate set
    ↓
distilled memory lookup
    ↓ miss
shared SmolLM3 teacher
    ↓
candidate selection
    ↓
SQLite distilled memory
    ↓
cognitive search
    ↓
shared SmolLM3 realization
```

V645 also audits gateway-local function references before release, so a missing
helper such as `apply_live_distillation` is detected during packaging.

The existing full graph is reused:

```text
.\data\v633_full_semantic.sqlite
```

## Run

```powershell
python .\research\v645\v645_semantic_chat_gateway.py --database ".\data\v633_full_semantic.sqlite" --output ".\results\v645_chat.json" --trace-output ".\results\v645_chat_traces.jsonl" --memory-output ".\results\v645_memory.json" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --max-hypotheses 12 --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000 --seed 64500
```
