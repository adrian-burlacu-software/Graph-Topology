# V532 — Instrumented Cognitive Knowledge Interface

V532 keeps the cognitive architecture as semantic/control authority and uses SmolLM3 only as a renderer when architecture-owned answer data or grounded evidence exists.

## New debugging tools

- `/topics` ranks strong long-term semantic areas.
- `/know <topic> [depth]` recursively inspects the semantic graph without consulting live conversation state.
- Per-turn trace prints the structured request and model response.
- When the architecture returns `answer=null` with no directly supporting evidence, the LLM is not asked to invent an answer; the user-facing result is `I don't know.`
- When grounded evidence exists but no deterministic operator answers the query, SmolLM3 is instructed to use only that evidence and return exactly `I don't know.` when it is insufficient.

## Run

```powershell
python .\research\v531\assistant_cli.py --memory ".\results\full_semantic_memory.sqlite" --model ".\llm\SmolLM3-3B" --freeze-knowledge --quantization 4bit
```

## V532 evidence boundary
The LLM receives only human-readable semantic evidence selected for the current target. Retrieval scores, datasets, provenance, frequency, and lexical metadata remain architecture/debug-only.
