
# V501 — goal-scored cognitive architecture

V501 changes the LLM boundary from:

```text
LLM text -> planner -> reply
```

to:

```text
LLM proposal
    ↓
parse proposal
    ↓
goal contribution
context contribution
evidence grounding
unsupported-claim detection
naturalness
brevity
    ↓
architecture accept/reject
    ↓
planner
    ↓
LLM realizer
```

The participant cannot win merely because its text sounds natural.

For factual goals:

```text
no evidence + new factual claims
    => reject participant proposal
```

For conversational goals, act-fit and contextual contribution can justify a
proposal even without external knowledge.

Internal diagnostics remain visible only as trace output.

## Run

From Graph-Topology root:

```powershell
python .\research\v501_goal_scored_cognition\assistant_cli.py `
  --memory ".\results\full_semantic_memory.sqlite" `
  --teacher ".\llm\SmolLM2-1.7B-Instruct"
```

Architecture without the LLM:

```powershell
python .\research\v501_goal_scored_cognition\assistant_cli.py `
  --memory ".\results\full_semantic_memory.sqlite"
```

Run the goal-scoring regression:

```powershell
python .\research\v501_goal_scored_cognition\test_goal_scoring.py
```

The existing combined memory database can be used directly; this version does
not require re-ingestion.


V502 fixes a stale `participant_content` variable reference left by the participant proposal refactor.


## V503 critical boundary fix

The LLM realizer does not run on every architecture winner.

It runs only when:

```text
winner.source == participant
```

Architecture-owned content from:

```text
state
knowledge
fallback
```

cannot be handed back to a free-form generator.

Participant realizations are independently verified so the realizer cannot add
new concrete entities, quantities, or unsupported facts.

This prevents a rejected participant proposal from reappearing by asking the
realizer to rewrite a fallback such as `Tell me more.`.


### V503 invariant

Only an accepted participant can invoke the LLM realizer. Architecture-owned
fallback/state/knowledge responses are never passed to a generative realizer.


## V504.1

The typed target normalizes adjective size questions such as:

```text
is it big?
```

to:

```text
property(subject, size, value=big)
```

so unrelated evidence such as `infinite` or `very old` cannot answer the
question.
