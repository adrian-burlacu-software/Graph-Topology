
# V459 — multi-participant cognitive assistant

V459 changes the conversational model:

```text
USER
  ↓
architecture perceives
  ↓
architecture asks LLM a simple question
  ↓
LLM PARTICIPANT speaks
  ↓
architecture perceives LLM speech
  ↓
architecture evaluates both participants
  ↓
architecture decides what to say
  ↓
USER
```

The LLM's response is stored as an internal participant turn:

```text
speaker = llm
purpose = internal_consultation
```

It is **not automatically copied to the user**.

The architecture can use the LLM answer as:
- a conversational candidate;
- evidence;
- something to learn from;
- or something to ignore.

## Run with the 1.7B model

```powershell
python .\research\v459\v459_multi_participant_assistant.py `
  --teacher ".\llm\SmolLM2-1.7B-Instruct"
```

## Native-only mode

```powershell
python .\research\v459\v459_multi_participant_assistant.py
```

Persistent memory:

```text
C:\Users\adria\Desktop\dev\Graph-Topology\results\assistant_semantic_net.sqlite
```

The console now makes the internal conversation explicit:

```text
[ARCHITECTURE → LLM] ...
[LLM PARTICIPANT] ...
[ARCHITECTURE ← LLM] ...
[ARCHITECTURE] decision=...
Assistant: ...
```

Commands:

```text
/status
/new
/quit
```
