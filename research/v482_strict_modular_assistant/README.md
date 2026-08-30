
# V482 — strict modular cognitive loop

V482 addresses the failure shown by V481's decision trace.

## Problems fixed

### 1. Grammar relations are not knowledge

Relations such as:

```text
subject
object
nsubj
aux
modifier
...
```

and classification relations such as:

```text
in_domain
domain
source
provenance
mapping
```

are not answerable semantic knowledge.

They are excluded before candidate generation.

### 2. Conversational intent is ordered by specificity

Examples:

```text
how's it going?
    -> explore_assistant

I want to know about the universe.
    -> request_information

isn't it mostly empty though?
    -> challenge_claim
```

Generic `request` detection no longer overrides these.

### 3. The LLM participant is actually a participant

The LLM is instructed to return:

```text
one proposition / hypothesis / conversational move
```

rather than:

```text
a useful candidate proposition for the architecture...
```

Common meta wrappers are filtered before planning.

### 4. Unsupported graph knowledge cannot beat a participant proposal

A graph candidate receives high evidence only when it contains actual
answerable semantic relations.

Otherwise it is absent from the knowledge candidate pool.

### 5. The architecture still selects the content

```text
USER
 ↓
PERCEPTION
 ↓
WORKING MEMORY
 ↓
GOAL
 ↓
KNOWLEDGE
 ↓
LLM PARTICIPANT
 ↓
ARCHITECTURAL EVALUATION
 ↓
SELECT CONTENT
 ↓
LLM REALIZER
 ↓
USER
```

The participant does not directly become the answer.

## Run

From the Graph-Topology project root:

```powershell
python .\research\v482_strict_modular_assistant\cli.py `
  --freeze-learning `
  --teacher ".\llm\SmolLM2-1.7B-Instruct"
```

Keep learning frozen while validating.

No corpus re-ingestion is required.
