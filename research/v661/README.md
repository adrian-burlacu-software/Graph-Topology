# V661 — Argument-Aware Semantic Relation Teacher

V661 fixes the V660 failure where a graph edge such as:

```text
bear -> etymologically_related_to -> brown
```

could win simply because it contains the same words as the question.

The teacher now receives:

```text
question
spaCy argument structure
subject-specific graph relation candidates
direct question-object matches
observed examples of each relation
```

The teacher is explicitly asked to match the *kind of relationship*, not word
overlap.

No relation-specific `ADJ -> has_property` or `brown -> has_property` rule is
hard-coded.

After relation selection, the graph remains authoritative for target lookup.

No graph rebuild required.

```powershell
python .\research\v661\v661_semantic_chat_gateway.py --database ".\data\v633_full_semantic.sqlite" --output ".\results\v661_chat.json" --trace-output ".\results\v661_chat_traces.jsonl" --memory-output ".\results\v661_memory.json" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --max-hypotheses 12 --relation-vocabulary 64 --goal-budget 40 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --prior-decay 0.65 --seed 66100 --progress-every 1
```

Suggested test:

```text
What is a bear?
Is it brown?
Is it an animal?
What is a dog?
Can it bark?
What color is a dog?
```

Watch:

```text
argument_frame
candidate_relation_details
relation_frame_source
relation_from_frame
fact_source
selected_fact_target
```
