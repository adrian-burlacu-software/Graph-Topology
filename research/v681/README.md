# V681 unified learning substrate

V681 is the integration layer, not a replacement graph or attention system.
V679 chat and offline workers optionally append provenance-preserving records to
one SQLite experience store. V680 DAgGER traces are bootstrap/imitative
experience; JEPA consumes sequential transitions only as an auxiliary predictive
learner. Neither directly controls attention logits, and chat collection never
updates weights.

```powershell
# Deterministic V681 wiring check (synthetic chat is explicitly marked).
python .\research\v681\run_v681_experiment.py --output-dir ".\results\v681\smoke" --epochs 1 --smoke

# Include existing V679 traces and worker batch evidence.
python .\research\v681\run_v681_experiment.py --output-dir ".\results\v681\run" --chat-traces ".\results\v679_chat_traces.jsonl" --worker-logs ".\results\v679_workers" --epochs 8

# Have live V679 collection append experience without online training.
python .\research\v679\v679_runtime.py --database ".\data\v679_focused_semantic.sqlite" --llm-model "<model>" --experience-store ".\results\v681\live_experience.sqlite"
```

The experiment writes `v681_learning_integration_results.json`,
`v681_experience_manifest.json`, and `v681_learning_integration_report.md`.
It explicitly marks source comparisons unsupported when chat/worker data lacks
compatible sequential attention observations; it does not fabricate results.
