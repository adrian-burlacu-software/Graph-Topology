# V681.3 unified learning substrate

V681.3 owns canonical experience, storage, source ingestion, manifests, and
orchestration. It reaches V680 only through `v680_adapter.py`, which validates
the fixed `research/v680` engine path. It has no dynamic Python-path imports and
does not modify V679. V679 artifacts are imported afterward. V680 remains the
frozen attention/JEPA engine.

Every record separates `model_view`, `supervision`, and `diagnostics`. Legacy
V679 chat traces are explicitly `decision_only`; sequential synthetic/live chat
and DAgGER use the same `AttentionTrajectoryAdapter`. Worker artifacts are
knowledge-only events, never attention labels. JEPA remains auxiliary.

```powershell
# Package test discovery from repository root.
python -m unittest discover -s research\v681 -p "test_*.py"

# Synthetic self-test: package, store, sequential synthetic chat, DAgGER,
# V680 attention/JEPA adapters, held-out evaluation, and reports.
python -m research.v681.run_v681_experiment --output-dir ".\results\v681\selftest" --smoke

# Real-chat ingestion only: sanitize/import completed V679 traces, write the
# manifest/report/examples, and never initialize V680 or train weights.
python -m research.v681.run_v681_experiment --output-dir ".\results\v681\chat_ingest" --chat-traces ".\results\v679_chat_traces.jsonl" --inspect-experience

# Full integration after inspecting ingestion. Legacy V679 traces are
# decision_only and are excluded from sequential attention learning.
python -m research.v681.run_v681_experiment --output-dir ".\results\v681\full" --chat-traces ".\results\v679_chat_traces.jsonl" --worker-logs ".\results\v679_workers" --epochs 8 --seed 7

# Inspect an existing V681 store.
python -m research.v681.run_v681_experiment --output-dir ".\results\v681\inspect" --inspect-experience

# Materialize reviewed sequential live records into an immutable new train dataset view.
python -m research.v681.run_v681_experiment --output-dir ".\results\v681\materialized" --materialize-live --min-quality verified --epochs 8
```

The experiment writes `v681_learning_integration_results.json`,
`v681_experience_manifest.json`, and `v681_learning_integration_report.md`.
In ingestion-only mode it writes `chat_ingestion_report.json` and
`chat_ingestion_examples.json`, with accepted capability/provenance descriptors
and malformed-record reasons. Successful ingestion does not mean a trace has
sequential attention supervision.
Each trained V680 checkpoint has a companion V681 `<comparison>.provenance.json`
that identifies sources, dataset version, engine version, seed, and configuration.
It explicitly marks source comparisons unsupported when chat/worker data lacks
compatible sequential attention observations; it does not fabricate results.
