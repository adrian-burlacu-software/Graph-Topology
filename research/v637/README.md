# V637 — Definition Proof Correctness Fix

V637 fixes a correctness bug in V636 where `definition` was treated as an
ordinary graph relation, allowing nonsense paths such as:

```text
dog -> antonym -> has_sense -> definition
```

For definition requests, V637 now performs:

```text
en:dog
  ↓
Graph.definition()
  ↓
direct lexical definition / WordNet synset gloss
```

Only non-definition relations use cognitive path search.

## Run

```powershell
python .\research\v637\v637_semantic_chat_gateway.py --database ".\data\v633_full_semantic.sqlite" --output ".\results\v637_full_chat.json" --trace-output ".\results\v637_full_chat_traces.jsonl" --memory-output ".\results\v637_full_memory.json" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --max-hypotheses 12 --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000 --seed 63700
```

First tests:

```text
What is a dog?
What is an animal?
What can a dog do?
What parts does a dog have?
```
