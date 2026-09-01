# V643 — Live Distillation Actually Used

V643 fixes the two issues exposed by the V642 trace.

## 1. The selected sense is now authoritative

The semantic teacher/memory can select:

```text
wn:synset:dog.n.01
```

V643 passes that exact sense into `Graph.definition()`. The graph cannot fall
back to an unrelated first sense when a distilled sense was selected.

## 2. Repeated semantic answers can bypass the LLM realization

V643 adds:

```text
realized_answers
```

to the semantic SQLite database.

The first verified answer is realized by SmolLM3 and cached against:

```text
question
subject
relation
target
path
```

A repeated semantically identical turn uses:

```text
semantic_answer_cache
```

instead of invoking SmolLM3 again.

This means a warm interaction can evolve toward:

```text
graph
  ↓
distilled semantic memory
  ↓
verified semantic result
  ↓
semantic answer cache
  ↓
answer
```

with no LLM call.

## Run

Use the existing full graph:

```powershell
python .\research\v643\v643_semantic_chat_gateway.py --database ".\data\v633_full_semantic.sqlite" --output ".\results\v643_chat.json" --trace-output ".\results\v643_chat_traces.jsonl" --memory-output ".\results\v643_memory.json" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --max-hypotheses 12 --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000 --seed 64300
```

Try:

```text
What is a dog?
What is a dog?
What is a dog?
What can a dog do?
What can a dog do?
```

Watch for:

```text
sense_source=distilled_memory
mode=grounded
mode=grounded_cache
```

and:

```text
llm_seconds=0
```

on cached repetitions.
