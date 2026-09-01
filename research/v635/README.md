# V635 — Full Semantic Chat API Contract Fix

V635 is a surgical repair of the full WordNet + ConceptNet chat runtime.

The V634 failure was:

```text
AttributeError: 'Graph' object has no attribute 'definition'
```

The gateway now has a corresponding `Graph.definition()` implementation.

Resolution behavior remains:

```text
exact normalized concept
    ↓
canonical en:<word>
    ↓
semantic search
    ↓
verified result
    ↓
LLM surface realization
```

`Graph.definition()` first reads a lexical definition stored directly on the
node and then falls back through `has_sense` to a WordNet synset gloss.

No rebuild of the existing full database is required.

## Run

```powershell
python .\research\v635\v635_semantic_chat_gateway.py --database ".\data\v633_full_semantic.sqlite" --output ".\results\v635_full_chat.json" --trace-output ".\results\v635_full_chat_traces.jsonl" --memory-output ".\results\v635_full_memory.json" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --max-hypotheses 12 --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000 --seed 63500
```

First test:

```text
What is a dog?
What is an animal?
What can a dog do?
What parts does a dog have?
```

V635 also performs a static Graph↔gateway API check before packaging and tests
definition lookup against the same full-network schema.
