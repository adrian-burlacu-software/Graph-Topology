# V681 unified learning substrate

V681 is a self-contained integration package, not a modification to the V679
runtime. It ingests completed V679 JSONL traces and worker logs through explicit
file adapters into one SQLite experience store. V680 remains the frozen
attention/JEPA engine consumed by V681's single orchestration boundary; it is
not imported by V679, workers, or chat. DAgGER traces are bootstrap/imitative
experience; JEPA consumes sequential transitions only as an auxiliary predictive
learner. Neither directly controls attention logits, and chat collection never
updates weights.

```powershell
# Deterministic V681 wiring check (synthetic chat is explicitly marked).
python .\research\v681\run_v681_experiment.py --output-dir ".\results\v681\smoke" --epochs 1 --smoke

# Include existing V679 traces and worker batch evidence.
python .\research\v681\run_v681_experiment.py --output-dir ".\results\v681\run" --chat-traces ".\results\v679_chat_traces.jsonl" --worker-logs ".\results\v679_workers" --epochs 8

# Run V679 normally; V681 imports its completed artifacts after collection.
python .\research\v679\v679_runtime.py --database ".\data\v679_focused_semantic.sqlite" --llm-model "<model>" --trace-output ".\results\v679_chat_traces.jsonl" --worker-log-dir ".\results\v679_workers"
```

The experiment writes `v681_learning_integration_results.json`,
`v681_experience_manifest.json`, and `v681_learning_integration_report.md`.
It explicitly marks source comparisons unsupported when chat/worker data lacks
compatible sequential attention observations; it does not fabricate results.
