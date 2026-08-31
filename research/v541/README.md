# V541 — DB-Native Knowledge & Goal Discovery

This release fixes the V540 table-selection bug for the Graph-Topology semantic store.

The long-term knowledge graph uses `facts.subject_id/object_id` joined through `concepts.canonical`; `live_facts` is conversational state and is explicitly excluded from discovery.

The benchmark derives query/goal families from the actual relation inventory. It does not invent a benchmark ontology.

## Run

```powershell
python .\research\v541\graph_knowledge_discovery.py --memory ".\results\full_semantic_memory.sqlite" --direct 500 --indirect 500 --unknown 100 --negative 25 --max-hops 3 --workers 20 --per-node 100 --seed 541 --show 10 --json ".\results\v541_db_native_discovery.json"
```

Force the canonical table if desired:

```powershell
python .\research\v541\graph_knowledge_discovery.py --memory ".\results\full_semantic_memory.sqlite" --table facts --direct 500 --indirect 500 --unknown 100 --negative 25 --max-hops 3 --workers 20 --seed 541
```

## Smoke test

```powershell
python .\research\v541\test_v541.py
```
