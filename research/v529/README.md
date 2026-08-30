# V529 — Instrumented Structured LLM Interface

V529 keeps the V528 cognitive boundary intact while adding visibility into exactly what the language model receives.

## Architecture boundary

```text
user
  ↓
architecture
  ├─ perception
  ├─ goal
  ├─ target
  ├─ state
  ├─ memory/evidence
  └─ deterministic operators
        ↓
structured answer request
        ↓
SmolLM3
        ↓
natural-language response
```

## Debugging

By default the CLI prints:

- `[ARCHITECTURE ANSWER]` — the semantic answer produced by the architecture
- `[ARCHITECTURE EVIDENCE]` — evidence selected for the current target
- `[LLM REQUEST]` — exact structured JSON handed to the LLM interface
- `[LLM PROMPT]` — rendered chat prompt actually passed into `generate()`
- `[LLM RESPONSE]` — raw model output before the response firewall

Use `--no-llm-debug` to hide the large request/prompt blocks while retaining normal traces.

## Knowledge map

Type `/topics` in the CLI to see the topics with the strongest populated semantic coverage. The ranking considers fact volume, relation diversity, source diversity, fact types, and confidence.
