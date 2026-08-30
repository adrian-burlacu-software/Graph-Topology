
# V460 — natural conversation

V460 adds a dialogue-act layer so ordinary social conversation is not treated
as a semantic dictionary lookup.

Supported first-pass acts:

```text
greeting
thanks
goodbye
affection
question
request
statement
other
```

The LLM remains an internal participant:

```text
USER
  ↓
architecture perceives speech act
  ↓
architecture asks simple English question
  ↓
LLM participant speaks
  ↓
architecture perceives/evaluates LLM speech
  ↓
architecture decides response
  ↓
USER
```

For example:

```text
I like you!
```

becomes:

```text
speech_act = affection

architecture → LLM:
Someone says they like you. What is a natural friendly reply?

LLM:
...

architecture ← LLM:
parsed response

architecture:
social_reply_candidate

assistant:
chosen conversational response
```

Run with SmolLM2:

```powershell
python .\research\v460\v460_natural_conversation.py `
  --teacher ".\llm\SmolLM2-1.7B-Instruct"
```

Native-only:

```powershell
python .\research\v460\v460_natural_conversation.py
```

Persistent memory:

```text
C:\Users\adria\Desktop\dev\Graph-Topology\results\assistant_semantic_net.sqlite
```
