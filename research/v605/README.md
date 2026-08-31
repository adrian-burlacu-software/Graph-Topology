# V605 — Semantic Graph Cognitive Runtime

V605 fixes the V603 runtime crash:

```text
TypeError: 'tuple' object does not support item assignment
```

Cause:
`Graph.outgoing()` returns a tuple, while `random.shuffle()` mutates its
argument in place.

The affected live-negative-case code now explicitly converts the returned
edges to a list before shuffling.

No other experiment dependency is introduced.

The workload profiler, global conditional attention, BFS runtime, trace
collection, and prior consolidation remain unchanged.

## Run

```powershell
python .\research\v605\v605_semantic_graph_cognitive_runtime.py --database ".\results\v562_kg_composition_audit.sqlite" --output ".\results\v605_semantic_graph_cognitive_runtime.json" --trace-output ".\results\v605_cognitive_traces.jsonl" --prior-output ".\results\v605_global_attention_prior.json" --workers 20 --seed-start 60500 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --max-probes-per-case 500 --prior-decay 0.65 --train-subjects-per-relation 120 --train-traces-per-relation 30 --interactions-per-relation 20 --progress-every 10 --profile-relations 20 --profile-subjects 500 --profile-probes 250
```

The final console summary remains copy/paste friendly.
