
# V377 — Hardened Full Grounding Benchmark (Fixed)

This version fixes the real-graph adapter seam exposed during the first local
full run.

The ConceptNet indexer uses:

```text
rev
```

while the cognitive semantic memory uses:

```text
reverse
```

The case adapter now normalizes both names at the boundary and the benchmark
has an explicit adapter-contract gate.

## Smoke

```powershell
python .\research\v377\fully_grounded_benchmark.py --smoke
```

## Adapter regression

```powershell
python .\research\v377\adapter_regression_test.py
```

## Full real ConceptNet run

From the Graph-Topology project root:

```powershell
python .\research\v377\fully_grounded_benchmark.py `
  --conceptnet .\data\conceptnet_compact.db
```

Smaller real run:

```powershell
python .\research\v377\fully_grounded_benchmark.py `
  --conceptnet .\data\conceptnet_compact.db `
  --limit 10
```

A full PASS now requires the semantic adapter contract as well as the grounding
quality gates.
