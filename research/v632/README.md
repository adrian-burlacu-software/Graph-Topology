# V632 — Beginner Semantic Network + Cognitive Chat

V632 replaces the previous giant graph with a compact, controlled semantic
dictionary built from the two sources already used in the project:

```text
NLTK WordNet
C:\Users\adria\Desktop\dev\Graph-Topology\data\conceptnet-assertions-5.7.0.csv.gz
```

## Build

Install:

```powershell
pip install -r .\research\v632\requirements.txt
python -m nltk.downloader wordnet omw-1.4
```

Build 4,000 words:

```powershell
python .\research\v632\v632_semantic_network_builder.py --conceptnet ".\data\conceptnet-assertions-5.7.0.csv.gz" --output ".\data\v632_beginner_semantic.sqlite" --vocab-size 4000
```

The resulting database is independent of the old 50GB graph.

It contains:

```text
3,000–5,000 simple English words
definitions
WordNet taxonomy
WordNet part/whole links
WordNet causal/similarity links
ConceptNet common-sense links
```

Only English single-word concepts inside the controlled vocabulary are kept
from ConceptNet.

## Smoke

```powershell
python .\research\v632\v632_semantic_chat_gateway.py --database ".\data\v632_beginner_semantic.sqlite" --output ".\results\v632_smoke.json" --trace-output ".\results\v632_smoke_traces.jsonl" --memory-output ".\results\v632_smoke_memory.json" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode smoke --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000 --seed 63200
```

## Chat

```powershell
python .\research\v632\v632_semantic_chat_gateway.py --database ".\data\v632_beginner_semantic.sqlite" --output ".\results\v632_chat.json" --trace-output ".\results\v632_chat_traces.jsonl" --memory-output ".\results\v632_memory.json" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --max-hypotheses 12 --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000 --seed 63200
```

## What this is trying to establish

The network should make ordinary questions first-class:

```text
What is a dog?
What is an animal?
What can a dog do?
What parts does a dog have?
What is a house?
What is water?
What is a person?
```

The cognitive controller then operates in a semantic space intentionally
designed for beginner language rather than attempting to mine arbitrary
encyclopedic identifiers.

The LLM remains a surface-realization / conversational layer. It is not the
authority for graph facts.

Every completed turn is appended to JSONL with:

```text
question
answer
route
selected hypothesis
all hypotheses
search result
path
attention
exploration
timing
memory
```
