# V653 — Compact Teacher + Exact Grounded Realization

V653 tightens the two LLM boundaries that caused the previous follow-up
failure.

The semantic teacher sees a compact set of graph facts rather than a verbose
representation:

```text
relation | relation meaning | target
```

The model must select one graph-supplied candidate.

The surface realizer receives the exact verified triple:

```text
VERIFIED_SUBJECT
VERIFIED_RELATION
VERIFIED_TARGET
```

and cannot safely return a realization containing another graph entity; such
an output is rejected and replaced by a sentence derived directly from the
verified triple.

The distilled graph-fact namespace is `graph_fact_v653`.

Existing graph; no rebuild required:

```powershell
python .\research\v653\v653_semantic_chat_gateway.py --database ".\data\v633_full_semantic.sqlite" --output ".\results\v653_chat.json" --trace-output ".\results\v653_chat_traces.jsonl" --memory-output ".\results\v653_memory.json" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --max-hypotheses 12 --goal-budget 40 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --seed 65300
```
