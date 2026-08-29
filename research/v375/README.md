
# V375 — Full Real-ConceptNet Grounding Benchmark (Fixed)

This is the corrected V375 full benchmark. It combines:

```text
real ConceptNet
    ↓
indexed semantic memory
    ↓
real ambiguous surface forms
    ↓
graph-derived distinguishing contexts
    ↓
cognitive grounding / belief competition
    ↓
resolution, uncertainty, revision metrics
```

The bug fixed here was an adapter contract mismatch: the real-graph edge type now
provides the `weight` and `provenance` fields expected by the cognitive semantic
memory.

## Smoke

```powershell
python .\research\v375\full_grounding_benchmark.py --smoke
```

## Full real graph

```powershell
python .\research\v375\full_grounding_benchmark.py `
  --conceptnet .\data\conceptnet_compact.db
```

## Smaller real run

```powershell
python .\research\v375\full_grounding_benchmark.py `
  --conceptnet .\data\conceptnet_compact.db `
  --limit 10
```
