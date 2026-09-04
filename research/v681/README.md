# V681.4 unified learning runtime

## Normal operation

```powershell
python -m research.v681.run_v681
```

V681 discovers the repository-relative V679 runtime, compatible focused graph,
local LLM, V680 engine, existing V679 artifacts, and V681 store. It starts the
real V679 chat/worker runtime with V681-owned internal outputs, appends observed
chat and worker experience, batches eligible learning, evaluates candidate
artifacts, and writes `results/v681/v681_runtime_results.json`,
`v681_runtime_report.md`, and `v681_experience_manifest.json`.
It also writes a per-session `v681_session_manifest.json` and the stable
aliases `v681_latest_results.json` and `v681_latest_report.md`.

No path to a chat trace, worker log, dataset, or checkpoint is needed. The
runtime uses `GRAPH_TOPOLOGY_LLM_MODEL` when set, otherwise expects the local
model at `llm/SmolLM3-3B`. If a runtime component is absent, V681 writes its
precise capability failure and continues with the components that are available.

V679 traces are collected through its real trace lifecycle, annotated with the
V681 session ID, and classified as `decision_only` unless V679 supplies an
explicit canonical sequential trajectory. V681 never fabricates steps. Worker
events remain knowledge-only and are never attention labels.

Training is batched: V681 bootstraps with frozen V680 DAgGER when fewer than
eight trainable sequential episodes exist, then trains attention and auxiliary
JEPA candidates. A new candidate is evaluated and preserved with provenance,
but is not automatically promoted: explicit project safety criteria are required.
The current V679 runtime has no compatible sequential observation adapter for a
frozen V680 checkpoint, so V681 records that unavailable feedback capability
rather than falsely applying an incompatible model to chat.

```powershell
# One-shot CI/developer lifecycle: discover, collect, learn, evaluate, report, exit.
python -m research.v681.run_v681 --once

# Discovery/report only; does not start chat or learners.
python -m research.v681.run_v681 --dry-run

# Package test discovery from repository root.
python -m unittest discover -s research\v681 -p "test_*.py"

# V681.3 diagnostic smoke test.
python -m research.v681.run_v681_experiment --output-dir ".\results\v681\selftest" --smoke

# V681.3 diagnostic ingestion inspection.
python -m research.v681.run_v681_experiment --output-dir ".\results\v681\chat_ingest" --chat-traces ".\results\v679_chat_traces.jsonl" --inspect-experience
```

The V681.3 commands are diagnostics only, not the normal workflow.
