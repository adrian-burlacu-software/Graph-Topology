# V626 — Dual-Mode Semantic Chat Gateway

V626 makes SmolLM3 the actual conversational surface.

The semantic/cognitive runtime remains authoritative for grounded knowledge,
while ordinary conversation is handed to SmolLM3 when no verified graph result
exists.

```text
user
 ↓
frozen spaCy
 ↓
cognitive controller
 ├── verified semantic result
 │       ↓
 │   GROUNDED LLM
 │       ↓
 │   grounded answer
 │
 └── no verified result / conversation
         ↓
     CONVERSATIONAL LLM
         ↓
     normal chat
```

Grounded mode receives:

```text
QUESTION
SUBJECT
RELATION
RELATION LABEL
VERIFIED OBJECT
EVIDENCE PHRASES
```

Conversation mode receives recent conversation plus the user message.

## Chat

```powershell
python .\research\v626\v626_semantic_chat_gateway.py --database ".\results\v562_kg_composition_audit.sqlite" --output ".\results\v626_chat.json" --trace-output ".\results\v626_chat_traces.jsonl" --prior-output ".\results\v626_chat_prior.json" --memory-output ".\results\v626_memory.json" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --max-hypotheses 12 --relation-vocabulary 200 --goal-budget 40 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --prior-decay 0.65 --seed 62600 --progress-every 1
```

Expected:

```text
chat> hello
answer: <SmolLM3 conversational response>

chat> tell me a joke
answer: <SmolLM3 conversational response>

chat> What is the academicDiscipline of ...
answer: <SmolLM3 grounded realization>
```

The trace records:

```text
llm_mode = conversation
```

or:

```text
llm_mode = grounded
```
