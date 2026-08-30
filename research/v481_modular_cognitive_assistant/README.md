
# V481 — Modular Goal-Directed Cognitive Assistant

V481 splits the conversational architecture into explicit modules:

```text
config.py
memory.py
perception.py
goals.py
knowledge.py
participant.py
evaluator.py
planner.py
realizer.py
assistant.py
cli.py
```

## Architectural ownership

The architecture owns:

```text
perception
working conversation memory
goal inference
reference resolution
semantic retrieval
answerable-knowledge filtering
candidate evaluation
candidate selection
```

The LLM has two subordinate roles:

```text
PARTICIPANT
  proposes an interpretation, hypothesis, fact or conversational move

REALIZER
  turns architecture-selected content into natural English
```

The participant is not automatically the answer.

## Conversation loop

```text
USER
 ↓
PERCEPTION
 ↓
WORKING MEMORY
 ↓
GOAL
 ↓
KNOWLEDGE RETRIEVAL
 ↓
LLM PARTICIPANT
 ↓
ARCHITECTURE EVALUATION
 ↓
SELECTED CONTENT
 ↓
LLM REALIZER
 ↓
USER
```

## Knowledge boundary

Classification/metadata edges such as:

```text
universe -> in_domain -> computer_support
```

are not answerable knowledge.

They therefore cannot become a user-facing answer simply because a lexical
node matched the user.

## Goal-directed scoring

Candidates are scored for:

```text
goal satisfaction
context coherence
evidence
progress
naturalness
brevity
```

Brevity is secondary to actually accomplishing the user's goal.

## Runtime memory

Working memory remains active with `--freeze-learning`. That flag prevents
long-term conversational learning/policies from becoming new training signals;
it does not disable the current conversation.

## Default paths

```text
MEMORY:
C:\Users\adria\Desktop\dev\Graph-Topology\results\assistant_semantic_net.sqlite

LLM:
C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM2-1.7B-Instruct

PARSER:
en_core_web_sm
```

## Run

From the Graph-Topology project root:

```powershell
python .\research\v481_modular_cognitive_assistant\cli.py `
  --freeze-learning `
  --teacher ".\llm\SmolLM2-1.7B-Instruct"
```

Diagnostics are on by default.

Use:

```text
--no-trace
```

to suppress planner diagnostics.

No corpus re-ingestion is required.
