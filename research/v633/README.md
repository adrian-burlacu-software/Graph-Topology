# V633 — ALL WordNet + ALL English ConceptNet — FIXED2

This package fixes the V633 builder regression where `get_wordnet()` was
accidentally omitted.

It also includes the previous ConceptNet 5.7 TSV parsing fix and the missing
`json` import.

## Build

```powershell
python -m nltk.downloader wordnet omw-1.4

python .\research\v633\v633_full_semantic_builder.py --conceptnet ".\data\conceptnet-assertions-5.7.0.csv.gz" --output ".\data\v633_full_semantic.sqlite"
```

The scope is:

```text
ALL WordNet synsets
ALL WordNet lemmas
ALL WordNet definitions/relations
ALL English → English ConceptNet 5.7 assertions
```

The ConceptNet assertions file is read as tab-separated data even though its
filename ends in `.csv.gz`.

## Chat

```powershell
python .\research\v633\v633_full_semantic_chat_gateway.py --database ".\data\v633_full_semantic.sqlite" --output ".\results\v633_full_chat.json" --trace-output ".\results\v633_full_chat_traces.jsonl" --memory-output ".\results\v633_full_memory.json" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --max-hypotheses 12 --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000 --seed 63300
```

## Smoke

```powershell
python .\research\v633\v633_full_semantic_chat_gateway.py --database ".\data\v633_full_semantic.sqlite" --output ".\results\v633_full_smoke.json" --trace-output ".\results\v633_full_smoke_traces.jsonl" --memory-output ".\results\v633_full_smoke_memory.json" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode smoke --max-hypotheses 12 --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000 --seed 63300
```
