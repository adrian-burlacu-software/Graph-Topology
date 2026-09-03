# V679 — Cognitive Attention and Evidence Arbitration

V679 is based on V678. It makes the system decide which bounded, graph-backed
evidence matters for the current turn. It does not merge graph structures.

V679 includes all preceding experiment baselines. Its candidate-constrained
teacher is for unresolved semantic ambiguity only; teacher selections remain
provenanced evidence and never replace verified graph facts.

V679 mirrors V672 and turns its specialist evidence into a bounded semantic
combination laboratory.

## What changed

### Cognitive attention

V678 assigned `0.5` to its primary observed lane, counterrelation mining,
relation interaction statistics, and graph-health sampling. That value was a
placeholder source default, not calibrated confidence. V679 assigns
source-specific values and writes a confidence audit grouped by evidence table,
source, provenance, confidence, record count, and support.

For every semantic turn V679 now:

1. exposes every bounded hypothesis and its candidate path evidence;
2. prioritizes traversal targets from goal compatibility, prior attention,
   target-term matches, and relation specificity;
3. arbitrates each candidate using explicit support, contradiction,
   specificity, provenance, and lexical components; and
4. records the selected semantic decision and all rejected alternatives in the
   chat trace; and
5. carries an `AttentionState` across turns and traversal steps. It decays
   relation and candidate activation, tracks focus and visited graph elements,
   and reinforces verified paths.

The architecture is explicitly `features → attention policy → scores →
arbitration`. `HandCodedAttentionPolicy` is the initial policy;
`DistilledAttentionPolicy` learns relation biases from emitted teacher-policy
traces and can be supplied to chat with `--attention-policy`. Arbitration
remains separate and abstains with `no_verified_evidence` when no candidate is
graph-verified. This produces the explicit answer “I don't know” rather than
selecting an attractive but unsupported candidate.

Only verified graph evidence can produce a grounded answer. Worker evidence
remains provenance-bearing training evidence. Shared evidence aggregation is
unchanged and is not graph-structure merging.

### 1. Checkpoints are actually spread across the whole interval

V671 mapped workers to `worker_id` seconds inside the interval. With a 300-second checkpoint interval, all 20 workers therefore synchronized inside the first 20 seconds.

V679 assigns slots across the entire interval:

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

V679 keeps per-worker evidence tables and materializes an aggregate view. Imported shared state is not re-exported as new worker evidence.

### 4. Goal decisions are arbitrated

A goal is grouped by semantic goal frame + candidate set. If different workers select different goals, the shared store records the disagreement as `contested` rather than silently treating whichever row arrived last as truth.

The online goal lookup uses the shared arbitration result when one is available.

### 5. Composition is bounded

The V671 composition lane produced ~55k learned records in the observed run. V679 defaults to:

- fanout = 4
- max derived compositions per worker run = 2,000
- maximum composition depth = 3 edges
- a 1,000-batch consecutive no-new-result termination streak

These are instrumentation/training limits, not semantic assumptions.

### 6. Instrumentation

Worker status now exposes:

- last batch duration
- learned items/sec
- sync duration
- sync count
- import/export/merge/conflict counts

`v679_inspect.py` additionally shows:

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
3. derivation depth — whether composed relations remain bounded, visibly derived, and include reusable prior paths.
4. relation transition counts — whether relation composition has stable structure.
5. learned/sec by lane — whether any lane is burning CPU without producing useful structure.
6. checkpoint spacing — in 300s mode, events should be distributed over roughly five minutes rather than clustered into ~20 seconds.

## Run

Use a fresh V679 shared database for this experiment. Do not reuse a V678
shared database because the V679 confidence audit expects source-specific
evidence.

### Focused semantic graph

Build the small graph before starting the runtime. `--focus-concepts bear dog animal`
retains direct evidence for those concepts (including features such as paws, tail,
and fur) and follows upward `is_a` relations for two hops, retaining concepts such
as canid and organism without expanding into sibling species.

ConceptNet is scanned once for direct evidence and once per requested type-closure
hop. The builder reports its active pass and every 250,000 scanned rows, including
the retained-edge count, pending type frontier, and scan rate.

```powershell
python .\research\v679\v679_semantic_network_builder.py --conceptnet ".\data\conceptnet-assertions-5.7.0.csv.gz" --output ".\data\v679_focused_semantic.sqlite" --focus-concepts bear dog animal --focus-depth 2 --progress-every 250000
```

```powershell
python .\research\v679\v679_runtime.py --database ".\data\v679_focused_semantic.sqlite" --output ".\results\v679_chat.json" --trace-output ".\results\v679_chat_traces.jsonl" --memory-output ".\results\v679_memory.json" --worker-log-dir ".\results\v679_workers" --shared-memory ".\results\v679_shared_memory.sqlite" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --max-hypotheses 12 --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000 --checkpoint-seconds 300 --seed 67900 --worker-count 12 --batch-sleep 0
```

For a faster checkpoint-spacing test:

```powershell
python .\research\v679\v679_runtime.py --database ".\data\v679_focused_semantic.sqlite" --output ".\results\v679_chat.json" --trace-output ".\results\v679_chat_traces.jsonl" --memory-output ".\results\v679_memory.json" --worker-log-dir ".\results\v679_workers" --shared-memory ".\results\v679_shared_memory_1min.sqlite" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --max-hypotheses 12 --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000 --checkpoint-seconds 60 --seed 67900 --worker-count 12 --batch-sleep 0
```

Inspect after a run:

```powershell
python .\research\v679\v679_inspect.py --shared-memory ".\results\v679_shared_memory.sqlite" --events 40
```

For a compact uploadable overnight report (three JSONL records: run totals,
per-worker status, and shared merge/arbitration/combination diagnostics):

```powershell
python .\research\v679\v679_worker_summary.py --worker-log-dir ".\results\v679_workers" --shared-memory ".\results\v679_shared_memory.sqlite" --output ".\results\v679\worker_summary.jsonl"
```

### Focused dialogue benchmark

This runs 32 graph-backed animal/dog/bear questions selected from verified
focused-graph facts. It overwrites the JSONL output and records the expected
fact, answer, pass/fail verdict, and complete internal routing/search/
distillation/realization instrumentation for every case.

```powershell
python .\research\v679\v679_focused_benchmark.py --database ".\data\v679_focused_semantic.sqlite" --output ".\results\v679\benchmark.jsonl" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --max-hypotheses 12 --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000
```

Run the adversarial ambiguity benchmark. It tests strong association, plausible
derived evidence, direct contradiction, weak direct evidence, and an unrelated
high lexical match; success requires abstention because none is verified. It
also emits a distilled attention-policy JSON artifact:

```powershell
python .\research\v679\v679_attention_benchmark.py --output ".\results\v679\attention_benchmark.jsonl" --policy-output ".\results\v679\distilled_attention_policy.json"
```

Use the artifact for chat policy replay:

```powershell
python .\research\v679\v679_runtime.py --database ".\data\v679_focused_semantic.sqlite" --output ".\results\v679_chat.json" --trace-output ".\results\v679_chat_traces.jsonl" --memory-output ".\results\v679_memory.json" --worker-log-dir ".\results\v679_workers" --shared-memory ".\results\v679_shared_memory.sqlite" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --max-hypotheses 12 --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000 --checkpoint-seconds 300 --seed 67900 --worker-count 12 --batch-sleep 0 --attention-policy ".\results\v679\distilled_attention_policy.json"
```

Add equivalent determiner, contraction, and part-subject grammar forms for a
93-case normalization run:

```powershell
python .\research\v679\v679_focused_benchmark.py --database ".\data\v679_focused_semantic.sqlite" --output ".\results\v679\normalization_benchmark.jsonl" --normalization-variants --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --max-hypotheses 12 --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000
```

Add the no-valid-answer case to the focused graph benchmark:

```powershell
python .\research\v679\v679_focused_benchmark.py --database ".\data\v679_focused_semantic.sqlite" --output ".\results\v679\adversarial_graph_benchmark.jsonl" --adversarial-ambiguity --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --max-hypotheses 12 --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000
```

After offline workers have populated the shared checkpoint, include up to ten
distinct worker-only discoveries for animal, bear, and dog. These answers are
labelled as derived worker observations, never as semantic-graph facts:

```powershell
python .\research\v679\v679_focused_benchmark.py --database ".\data\v679_focused_semantic.sqlite" --output ".\results\v679\worker_discovery_benchmark.jsonl" --normalization-variants --shared-memory ".\results\v679_shared_memory.sqlite" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --max-hypotheses 12 --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000
```

After a checkpoint, chat displays up to ten focused, ordinary taxonomy
questions backed by non-cyclic worker-discovered `is_a` paths for animal, bear,
and dog. Worker provenance remains in the trace and benchmark instrumentation;
it never replaces graph-grounded routing or appears in the user answer. Use a
fresh shared database when changing worker-learning behavior so stale
discoveries cannot appear.

The worker-discovery benchmark evaluates those ordinary questions using the
same graph-authoritative type route. It intentionally asserts the subject,
`type` goal, and expected target rather than a mutable worker-record key, since
workers may continue to add supporting evidence while the benchmark runs.
Delete the shared-memory database before a run after changing worker-lane
logic; it intentionally retains prior worker evidence across restarts.

### Query-time worker pool

The offline learners are now an eight-lane general-purpose pool: lexical
semantics, taxonomy, part/whole, attributes/actions, causal/context,
structural inference, composition, and interaction/health. The runtime starts
a CPU-aware default of 10–15 worker processes (one core is left for chat);
override it with `--worker-count 10` through `--worker-count 15`.

Every animal, bear, or dog question is dispatched to every pool worker at high
priority. Workers rotate through the eight lanes, then force-sync their
evidence and mark their task complete. The chat response remains
graph-grounded; worker results are provenance-bearing supporting evidence in
its trace.

When no user task is queued, every worker locally rotates animal, bear, and dog
exploration without inserting background tasks into the shared SQLite queue.
Idle polling is read-only (default `--task-poll-seconds 0.25`), while only
actual chat work claims, completes, and immediately shares a task result. This
keeps `--batch-sleep 0` CPU-bound rather than SQLite-write-bound while preserving
high-priority, one-task-per-worker fairness.

Shared background evidence is still merged through the evenly staggered
checkpoint slots. SQLite auto-checkpoints the WAL at 256 pages and the runtime
runs a final truncate checkpoint after chat and all worker connections have
closed. Increase `--batch-sleep` only when you intentionally want to cap CPU
usage.
Each worker also exits cleanly after 1,000 consecutive batches that add no new
local record (`--max-no-new-batches`; pass `0` to disable), so a plateaued graph
does not keep saturating the CPU. The JSONL worker log and inspector report
`new_results`, `no_new_streak`, and the termination reason.
Each queued task includes the focused subject plus a rotating batch of graph
subjects (`--worker-query-batch-subjects`, default `128`) so lanes perform
substantial graph work rather than repeatedly reading one edge.

Worker status and JSONL logs include CPU seconds and per-batch CPU utilization,
in addition to throughput and sync metrics. To choose the best pool size on a
specific machine, run the offline-only sweep (it does not load the LLM):

```powershell
python .\research\v679\v679_worker_pool_benchmark.py --database ".\data\v679_focused_semantic.sqlite" --output ".\results\v679\worker_pool_benchmark.jsonl" --worker-counts 10,12,15 --duration-seconds 60
```

The final JSONL record contains the recommended `--worker-count`, selected by
learned items per second and then host CPU utilization. Use that value in the
chat runtime; for example add `--worker-count 12` to the command above.

### Relation composition and exclusion

The composition lane derives bounded two- and three-edge paths over every
relation registered by the specialist lanes. It can extend graph edges with
prior locally discovered or imported shared paths, including previously derived
paths combined with other discovered paths. Relation sequences are prioritized
by the semantic registry, while full node-path cycle checks prevent
tautologies. Counterrelation mining separately stores negative
`is_not:<relation>` evidence with its competing observed relation and
contradiction group; this is training evidence, not a graph fact.
