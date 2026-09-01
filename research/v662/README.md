# V662 — Clean Semantic Goal Boundary

V662 enforces a strict interface between language understanding and the raw
semantic graph.

The LLM sees only a compact user-facing goal vocabulary:

```text
definition
type
property
part
capability
location
purpose
cause
association
contrast
```

It does not see raw implementation relations such as:

```text
has_sense
usage_count
etymologically_related_to
dbpedia/...
```

The selected clean goal is then translated internally by the graph adapter
into the raw relation(s) needed for proof.

Example:

```text
user:
    is it brown?

LLM:
    property

internal graph adapter:
    property -> has_property

graph:
    verify has_property(bear,brown)
```

If the graph cannot prove that goal, the answer is not verified. It cannot fall
back to an unrelated raw relation merely because it contains a matching word.

The realization cache is keyed by the clean semantic goal.

No graph rebuild required.

## Run

```powershell
python .\research\v662\v662_semantic_chat_gateway.py --database ".\data\v633_full_semantic.sqlite" --output ".\results\v662_chat.json" --trace-output ".\results\v662_chat_traces.jsonl" --memory-output ".\results\v662_memory.json" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --max-hypotheses 12 --relation-vocabulary 64 --goal-budget 40 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --prior-decay 0.65 --seed 66200 --progress-every 1
```

Test:

```text
What is a bear?
Is it brown?
Is it an animal?
What is a dog?
Can it bark?
What parts does a dog have?
```
