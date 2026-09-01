# V656 — Semantic Gateway Import Contract Fix

Fixes the V655 missing-import failure:

```text
NameError: name 'structural_question_frame' is not defined
```

The gateway explicitly imports `structural_question_frame` from the versioned
semantic core.

A startup AST symbol audit scans module-wide imports (including local imports)
and direct function calls before the database/model work begins.

No graph rebuild required.

```powershell
python .\research\v656\v656_semantic_chat_gateway.py --database ".\data\v633_full_semantic.sqlite" --output ".\results\v656_chat.json" --trace-output ".\results\v656_chat_traces.jsonl" --memory-output ".\results\v656_memory.json" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --max-hypotheses 12 --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000 --seed 65600
```
