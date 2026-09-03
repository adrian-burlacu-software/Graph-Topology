# V678 — Grounded Dialogue Reliability

V678 inherits V677 and hardens graph-grounded dialogue: canonical lexical
definitions, plural concept resolution, pronoun carryover, and generic
part/component questions. Definition answers bypass persisted realization
caches so stale answers from a different lexical sense cannot be reused.

V678 includes all preceding experiment baselines. Its candidate-constrained
teacher is for unresolved semantic ambiguity only; teacher selections remain
provenanced evidence and never replace verified graph facts.

V678 mirrors V672 and turns its specialist evidence into a bounded semantic
combination laboratory.

## What changed

### 1. Checkpoints are actually spread across the whole interval

V671 mapped workers to `worker_id` seconds inside the interval. With a 300-second checkpoint interval, all 20 workers therefore synchronized inside the first 20 seconds.

V678 assigns slots across the entire interval:

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

V678 keeps per-worker evidence tables and materializes an aggregate view. Imported shared state is not re-exported as new worker evidence.

### 4. Goal decisions are arbitrated

A goal is grouped by semantic goal frame + candidate set. If different workers select different goals, the shared store records the disagreement as `contested` rather than silently treating whichever row arrived last as truth.

The online goal lookup uses the shared arbitration result when one is available.

### 5. Composition is bounded

The V671 composition lane produced ~55k learned records in the observed run. V678 defaults to:

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

`v678_inspect.py` additionally shows:

- provenance/promotion distribution
- decision arbitration
- evidence table sizes
- checkpoint timing and assigned slots
- merge conflict summaries

Each offline worker continues to emit detailed JSONL logs.

### 7. Relation combinations

The specialist lanes publish their observations to the provenance-aware shared
pool. The composition lane learns bounded two-edge transitions, while the
combination-statistics lane separately records co-occurring relations around an
entity. The inspector reports both, including transition support, confidence,
contributing workers, derivation depth, and an evidence-based candidate status.

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

Use a fresh V678 shared database for this experiment. Do not reuse the V671 shared database because V678 has a new evidence schema.

### Focused semantic graph

Build the small graph before starting the runtime. `--focus-concepts bear dog animal`
retains direct evidence for those concepts (including features such as paws, tail,
and fur) and follows upward `is_a` relations for two hops, retaining concepts such
as canid and organism without expanding into sibling species.

ConceptNet is scanned once for direct evidence and once per requested type-closure
hop. The builder reports its active pass and every 250,000 scanned rows, including
the retained-edge count, pending type frontier, and scan rate.

```powershell
python .\research\v678\v678_semantic_network_builder.py --conceptnet ".\data\conceptnet-assertions-5.7.0.csv.gz" --output ".\data\v678_focused_semantic.sqlite" --focus-concepts bear dog animal --focus-depth 2 --progress-every 250000
```

```powershell
python .\research\v678\v678_runtime.py --database ".\data\v673_focused_semantic.sqlite" --output ".\results\v678_chat.json" --trace-output ".\results\v678_chat_traces.jsonl" --memory-output ".\results\v678_memory.json" --worker-log-dir ".\results\v678_workers" --shared-memory ".\results\v678_shared_memory.sqlite" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --max-hypotheses 12 --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000 --checkpoint-seconds 300 --seed 67800 --batch-sleep 0.20
```

For a faster checkpoint-spacing test:

```powershell
python .\research\v678\v678_runtime.py --database ".\data\v673_focused_semantic.sqlite" --output ".\results\v678_chat.json" --trace-output ".\results\v678_chat_traces.jsonl" --memory-output ".\results\v678_memory.json" --worker-log-dir ".\results\v678_workers" --shared-memory ".\results\v678_shared_memory_1min.sqlite" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --max-hypotheses 12 --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000 --checkpoint-seconds 60 --seed 67800 --batch-sleep 0.20
```

Inspect after a run:

```powershell
python .\research\v678\v678_inspect.py --shared-memory ".\results\v678_shared_memory.sqlite" --events 40
```

For a compact uploadable overnight report (three JSONL records: run totals,
per-worker status, and shared merge/arbitration/combination diagnostics):

```powershell
python .\research\v678\v678_worker_summary.py --worker-log-dir ".\results\v678_workers" --shared-memory ".\results\v678_shared_memory.sqlite" --output ".\results\v678\worker_summary.jsonl"
```

### Focused dialogue benchmark

This runs 32 graph-backed animal/dog/bear questions selected from verified
focused-graph facts. It overwrites the JSONL output and records the expected
fact, answer, pass/fail verdict, and complete internal routing/search/
distillation/realization instrumentation for every case.

```powershell
python .\research\v678\v678_focused_benchmark.py --database ".\data\v673_focused_semantic.sqlite" --output ".\results\v678\benchmark.jsonl" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --max-hypotheses 12 --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000
```

Add equivalent determiner, contraction, and part-subject grammar forms for a
90-case normalization run:

```powershell
python .\research\v678\v678_focused_benchmark.py --database ".\data\v673_focused_semantic.sqlite" --output ".\results\v678\normalization_benchmark.jsonl" --normalization-variants --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --max-hypotheses 12 --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000
```
