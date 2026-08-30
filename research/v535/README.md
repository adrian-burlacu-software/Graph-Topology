# V535 — Instrumented Graph Path Audit

Pure SQLite graph diagnostics. No LLM, no conversation state.

The previous audit performed one SQL query per expanded node. This version uses batched `IN (...)` frontier reads, prints per-depth timing, edge counts, frontier sizes, and optional JSON metrics.

## Run

```powershell
python .\research\v535\graph_path_audit.py --memory ".\results\full_semantic_memory.sqlite" --start people --target hand --depth 4 --workers 1
```

Try 4 read workers:

```powershell
python .\research\v535\graph_path_audit.py --memory ".\results\full_semantic_memory.sqlite" --start people --target hand --depth 4 --workers 4
```

Write a report:

```powershell
python .\research\v535\graph_path_audit.py --memory ".\results\full_semantic_memory.sqlite" --start people --target hand --depth 4 --workers 4 --json ".\results\people_hand_v535.json"
```

The output shows, for each depth:

- frontier size
- edges examined
- SQL time
- total layer time
- semantic vs lexical-inclusive search time

This makes it possible to tell whether SQLite I/O, graph branching, or path depth is the bottleneck.
