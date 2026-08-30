# V513 — Cognitive Bridge + Preserved Ingestion Pipeline

V513 continues the V512 bridge corrections while preserving the previous ingestion pipeline intact.

## Bridge changes

- explicit subjects override stale conversational context
- generic requests do not inherit stale subjects
- state facts are deduplicated
- existing working state is authoritative when it answers the target
- semantic SQLite evidence is consulted conservatively
- an optional local teacher can provide candidate semantic content
- the architecture remains the semantic authority; teacher output is only a candidate
- `--freeze-knowledge` explicitly freezes knowledge for benchmark/interactive runs
- `/freeze` and `/unfreeze` remain available at runtime
- `/status` exposes teacher, state, target, and freeze state

## Preserved ingestion pipeline

The V509 ingestion stack is carried forward without modification, including:

- `combined_ingest.py`
- `schema.py`
- `ingest_adapters.py`
- ConceptNet / WordNet / VerbNet / UD-GUM sources
- Ubuntu dialogue ingestion
- SGD / MultiWOZ ingestion
- semantic data download/install helpers

The preserved pipeline can therefore still build the SQLite semantic memory used by the bridge.

## Run the ingestion pipeline

From the repository root:

```powershell
python .\research\v513\combined_ingest.py --data-root "." --out ".\results\combined_cognitive_memory.sqlite" --reset --conceptnet ".\data\assertions.csv"
```

Add `--wordnet`, `--verbnet`, or `--ud-gum <path>` exactly as supported by the preserved V509 pipeline when those sources are available.

## Run the assistant with teacher + frozen knowledge

From the repository root:

```powershell
python .\research\v513\assistant_cli.py --memory ".\results\full_semantic_memory.sqlite" --teacher ".\llm\SmolLM2-1.7B-Instruct" --freeze-knowledge
```

## Run without teacher

```powershell
python .\research\v513\assistant_cli.py --memory ".\results\full_semantic_memory.sqlite" --freeze-knowledge
```

## Smoke test

```powershell
python .\research\v513\benchmark_v513.py
```
