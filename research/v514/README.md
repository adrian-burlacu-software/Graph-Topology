# V514 — Cognitive Bridge Repair

V514 repairs the V513 live bridge without changing the underlying semantic ingestion pipeline.

## Changes

- Static semantic-memory retrieval uses the typed `facts`/`concepts` tables only.
- Live/session facts are no longer mixed into long-term-memory retrieval.
- Property queries are rendered as natural answers instead of raw graph syntax.
- Explicitly named entities always override stale conversational subjects.
- Generic action requests such as `help me` do not inherit the previous topic.
- Session state remains deduplicated.
- The teacher remains optional and is used only as a candidate proposer/realizer.
- `--teacher` and `--freeze-knowledge` are explicit CLI controls.
- The previous ingestion pipeline is retained intact: ConceptNet, WordNet/VerbNet/UD-GUM adapters, combined ingestion, schema, download/install helpers, and inspection tools.

## Run

From the repository root:

```powershell
python .\\research\\v514\\assistant_cli.py --memory ".\\results\\full_semantic_memory.sqlite" --teacher ".\\llm\\SmolLM2-1.7B-Instruct" --freeze-knowledge
```

## Ingestion

```powershell
python .\\research\\v514\\combined_ingest.py --help
```
