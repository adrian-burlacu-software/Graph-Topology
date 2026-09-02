# V672 — provenance, arbitration, and real interval staggering

V672 is the next experimental runtime after V671.

## What changed

### 1. Checkpoints are actually spread across the whole interval

V671 mapped workers to `worker_id` seconds inside the interval. With a 300-second checkpoint interval, all 20 workers therefore synchronized inside the first 20 seconds.

V672 assigns slots across the entire interval:

- 300s mode: workers are about 15s apart
- 60s mode: workers are about 3s apart

The slot is still determined from wall-clock time, so workers do not need a coordinator.

### 2. Evidence has provenance

Semantic records now carry:

- `provenance`: `observed`, `derived`, or `arbitrated`
- `derivation_depth`: 0 for observed, 1+ for derived evidence
- `contradiction_group`
- `promotion_state`: `candidate`, `contested`, or `eligible`

Derived relations are explicitly marked as derived. They are training evidence, not graph truth.

### 3. Shared merge is contribution-based

V671 used `MAX()` for counts/confidence. That hid the difference between:

- two workers independently observing the same thing
- one worker importing the other worker's result
- contradictory evidence

V672 keeps per-worker evidence tables and materializes an aggregate view. Imported shared state is not re-exported as new worker evidence.

### 4. Goal decisions are arbitrated

A goal is grouped by semantic goal frame + candidate set. If different workers select different goals, the shared store records the disagreement as `contested` rather than silently treating whichever row arrived last as truth.

The online goal lookup uses the shared arbitration result when one is available.

### 5. Composition is bounded

The V671 composition lane produced ~55k learned records in the observed run. V672 defaults to:

- fanout = 4
- max derived compositions per batch = 250
- derivation depth = 2

These are instrumentation/training limits, not semantic assumptions.

### 6. Instrumentation

Worker status now exposes:

- last batch duration
- learned items/sec
- sync duration
- sync count
- import/export/merge/conflict counts

`v672_inspect.py` additionally shows:

- provenance/promotion distribution
- decision arbitration
- evidence table sizes
- checkpoint timing and assigned slots
- merge conflict summaries

Each offline worker continues to emit detailed JSONL logs.

## Important interpretation

The first thing to look for is **not** a larger knowledge count.

The useful signals are:

1. `contested` decisions — whether the system is exposing ambiguity instead of hiding it.
2. positive/negative evidence — whether counter-relations actually create useful discrimination.
3. derivation depth — whether composed relations remain bounded and visibly derived.
4. relation transition counts — whether relation composition has stable structure.
5. learned/sec by lane — whether any lane is burning CPU without producing useful structure.
6. checkpoint spacing — in 300s mode, events should be distributed over roughly five minutes rather than clustered into ~20 seconds.

## Run

Use a fresh V672 shared database for this experiment. Do not reuse the V671 shared database because V672 has a new evidence schema.

```powershell
python .\research\v672\v672_runtime.py --database ".\data\v633_full_semantic.sqlite" --output ".\results\v672_chat.json" --trace-output ".\results\v672_chat_traces.jsonl" --memory-output ".\results\v672_memory.json" --worker-log-dir ".\results\v672_workers" --shared-memory ".\results\v672_shared_memory.sqlite" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --max-hypotheses 12 --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000 --checkpoint-seconds 300 --seed 67200 --batch-sleep 0.20
```

For a faster checkpoint-spacing test:

```powershell
python .\research\v672\v672_runtime.py --database ".\data\v633_full_semantic.sqlite" --output ".\results\v672_chat.json" --trace-output ".\results\v672_chat_traces.jsonl" --memory-output ".\results\v672_memory.json" --worker-log-dir ".\results\v672_workers" --shared-memory ".\results\v672_shared_memory_1min.sqlite" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --max-hypotheses 12 --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000 --checkpoint-seconds 60 --seed 67200 --batch-sleep 0.20
```

Inspect after a run:

```powershell
python .\research\v672\v672_inspect.py --shared-memory ".\results\v672_shared_memory.sqlite" --events 40
```
