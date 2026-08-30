# Graph-Topology Cognitive Bridge — V516

V516 fixes the class of conversational failures exposed by V513/V514:

- explicit subjects always override stale conversation context;
- only genuine pronouns inherit the previous explicit subject;
- `I want to know ...` is routed as information-seeking;
- working memory facts are deduplicated;
- live state answers outrank long-term semantic memory;
- semantic memory is converted into typed, human-readable content rather than raw graph rows;
- low-value lexical relations are not allowed to dominate definitions/descriptions;
- generic requests do not inherit an unrelated subject;
- the teacher is not allowed to become an ungrounded factual answer source;
- teacher proposals remain limited to social/conversational/open-generation turns;
- the ingestion pipeline from V514 is carried forward intact.

## Run

From the repository root:

```powershell
python .\research\v516\assistant_cli.py --memory ".\results\full_semantic_memory.sqlite" --teacher ".\llm\SmolLM2-1.7B-Instruct" --freeze-knowledge
```

Without the teacher:

```powershell
python .\research\v516\assistant_cli.py --memory ".\results\full_semantic_memory.sqlite" --freeze-knowledge
```

## Ingestion

The previous ingestion pipeline is intentionally retained in this version.
Run the same ingestion tools from `research/v516/` as needed.

## Smoke test

```powershell
python .\research\v516\test_v516.py
```


## V516 CLI fix

V516 restores the executable module entrypoint so `assistant_cli.py` actually starts the interactive REPL when invoked directly.
