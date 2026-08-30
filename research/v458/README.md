
# V458 — native LLM interaction fix

Fixes the V457 interaction crash:

```text
sqlite3.ProgrammingError:
Error binding parameter 6: type 'dict' is not supported
```

The inherited graph API is:

```python
add_edge(
    con,
    source_node,
    relation,
    target_node,
    source_dataset,
    payload=None,
    weight=1.0,
)
```

The LLM-derived edge now passes the payload before the numeric weight.

Greetings also get a normal conversational query:

```text
Someone says hello to you. What is a natural reply?
```

## Run

```powershell
python .\research\v458\v458_native_llm_interaction_fix.py `
  --teacher ".\llm\SmolLM2-1.7B-Instruct"
```

Or without an LLM:

```powershell
python .\research\v458\v458_native_llm_interaction_fix.py
```

Memory:

```text
C:\Users\adria\Desktop\dev\Graph-Topology\results\assistant_semantic_net.sqlite
```
