# V664 — Argument-Grounded Semantic Goal Proof

V664 keeps the V662 clean semantic-goal boundary and fixes the remaining failure mode: after the teacher selects a public goal such as `property`, graph proof must satisfy the **requested argument** as well as the subject. The proof/search layer now enforces target grounding during both direct and bounded-path proof, so an unrelated path such as `antonym -> has_property -> aggressive animal` cannot satisfy `property(bear, brown)`.

For:

```text
is it brown?
```

the flow is now:

```text
LLM teacher -> property
argument grounding -> brown
internal adapter -> property -> has_property
graph proof -> has_property(bear, brown)
```

The LLM never receives raw graph relations. The adapter does not contain adjective-specific mappings; it uses the parsed semantic argument to constrain the target of the already-selected goal. If the requested argument cannot be proven, the result stays unverified instead of wandering through unrelated graph edges.

No graph rebuild required.

## Run

```powershell
python .\research\v664\v664_semantic_chat_gateway.py --database ".\data\v633_full_semantic.sqlite" --output ".\results\v664_chat.json" --trace-output ".\results\v664_chat_traces.jsonl" --memory-output ".\results\v664_memory.json" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --max-hypotheses 12 --relation-vocabulary 64 --goal-budget 40 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --prior-decay 0.65 --seed 66300 --progress-every 1
```

## Smoke sequence

```text
What is a bear?
Is it brown?
Is it an animal?
What is a dog?
Can it bark?
What parts does a dog have?
```
