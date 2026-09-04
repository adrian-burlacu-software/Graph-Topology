# V681 unified learning substrate

V681.2 owns canonical experience, storage, source ingestion, manifests, and
orchestration. It reaches V680 only through `v680_adapter.py`, which validates
the fixed `research/v680` engine path. It has no dynamic Python-path imports and
does not modify V679. V679 artifacts are imported afterward. V680 remains the
frozen attention/JEPA engine.

Every record separates `model_view`, `supervision`, and `diagnostics`. Legacy
V679 chat traces are explicitly `decision_only`; sequential synthetic/live chat
and DAgGER use the same `AttentionTrajectoryAdapter`. Worker artifacts are
knowledge-only events, never attention labels. JEPA remains auxiliary.

```powershell
# Deterministic V681 wiring check (synthetic chat is explicitly marked).
python -m research.v681.run_v681_experiment --output-dir ".\results\v681\smoke" --epochs 1 --smoke

# Include existing V679 traces and worker batch evidence.
python -m research.v681.run_v681_experiment --output-dir ".\results\v681\full" --chat-traces ".\results\v679_chat_traces.jsonl" --worker-logs ".\results\v679_workers" --epochs 8 --seed 7

# Run V679 normally; V681 imports its completed artifacts after collection.
python -m research.v681.run_v681_experiment --output-dir ".\results\v681\inspect" --inspect-experience

# Materialize reviewed sequential live records into an immutable new train dataset view.
python -m research.v681.run_v681_experiment --output-dir ".\results\v681\materialized" --materialize-live --min-quality verified --epochs 8
```

The experiment writes `v681_learning_integration_results.json`,
`v681_experience_manifest.json`, and `v681_learning_integration_report.md`.
Each trained V680 checkpoint has a companion V681 `<comparison>.provenance.json`
that identifies sources, dataset version, engine version, seed, and configuration.
It explicitly marks source comparisons unsupported when chat/worker data lacks
compatible sequential attention observations; it does not fabricate results.
