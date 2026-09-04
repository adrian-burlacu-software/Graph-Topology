# V681.5 self-contained learning runtime

## Normal operation

```powershell
python -m research.v681.run_v681
```

V681 owns its runtime and learning implementation under `research/v681/`.
The runtime starts V681-native chat and offline workers, appends direct chat and
worker events to its SQLite experience store, batches eligible learning,
evaluates candidates, promotes passing candidates atomically, hot-swaps the
current attention policy, and writes `results/v681/runtime_results.json`,
`runtime_report.md`, `experience_manifest.json`, and `current_model_manifest.json`.
It also writes a per-session `v681_session_manifest.json` and the stable
aliases `v681_latest_results.json` and `v681_latest_report.md`.

No path to a chat trace, worker log, dataset, or checkpoint is needed. The
runtime uses `GRAPH_TOPOLOGY_LLM_MODEL` when set, otherwise expects the local
model at `llm/SmolLM3-3B`. If a runtime component is absent, V681 writes its
precise capability failure and continues with the components that are available.

The native runtime is derived from the stable V679 chat/worker implementation;
the native learning package is derived from the stable V680 attention/DAgGER/JEPA
implementation. Their source versions are recorded in artifacts. V681 has no
runtime import or dynamic loading dependency on either historical package.
Chat records are classified as `decision_only` unless actual sequential
transitions exist. V681 never fabricates steps. Worker events remain
knowledge-only and are never attention labels.

Training is batched: V681 bootstraps native DAgGER when fewer than eight
trainable sequential episodes exist, then trains attention and auxiliary JEPA
candidates. A durable learner cursor prevents duplicate training after restart:
no new eligible experience means no retraining or expensive evaluation. A
new live sequential trajectory is copied once into an immutable train dataset
view; its original `live` record is never changed.
candidate is promoted only after valid held-out evaluation passes the
conservative gate; otherwise the current model remains active. Live model
scoring uses only observable state/candidate features and retains
verified-evidence abstention safety.

```powershell
# One-shot CI/developer lifecycle: discover, collect, learn, evaluate, report, exit.
python -m research.v681.run_v681 --once

# Discovery/report only; does not start chat or learners.
python -m research.v681.run_v681 --dry-run

# Package test discovery from repository root.
python -m unittest discover -s research\v681 -p "test_*.py"

# Native bounded runtime smoke test.
python -m research.v681.run_v681 --smoke
```

The remaining options are diagnostics only, not the normal workflow.
