# V642 — Live Semantic Distillation Runtime Contract Fix

V642 fixes the V641 startup/runtime integration failure:

```text
NameError: name 'choose_best' is not defined
```

The gateway now contains its own `choose_best()` helper and the package performs
a static gateway symbol audit before packaging.

The live semantic distillation architecture remains unchanged:

```text
graph candidates
    ↓
distilled-memory lookup
    ↓ miss
SmolLM3 constrained teacher
    ↓
selected graph candidate
    ↓
persistent SQLite distilled decision
    ↓
cognitive search
    ↓
verified result
    ↓
SmolLM3 realization
```

The V641 sense-aware definition fix is retained.

## Existing graph

No rebuild is required:

```text
.\data\v633_full_semantic.sqlite
```

## Run

```powershell
python .\research\v642\v642_semantic_chat_gateway.py --database ".\data\v633_full_semantic.sqlite" --output ".\results\v642_chat.json" --trace-output ".\results\v642_chat_traces.jsonl" --memory-output ".\results\v642_memory.json" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --max-hypotheses 12 --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000 --seed 64200
```

Test:

```text
What is a dog?
What is a dog?
What can a dog do?
What parts does a dog have?
```
