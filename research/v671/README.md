# V671 — Parallel Semantic Learning Runtime

V671 is the integrated roadmap runtime: **19 offline graph-analysis workers plus one online chat worker**.

The online worker remains graph-authoritative and conversationally capable. Unverified semantic questions never fall through to pretrained world knowledge. Ordinary conversational requests bypass the graph and use the LLM conversation path.

## Runtime architecture

```text
                         ┌───────────────────────┐
                         │   v671_runtime.py      │
                         │  20 total processes    │
                         └───────────┬───────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
        workers 00..18         worker 19 / chat       shared checkpoint
              │                      │                      │
      read-only graph DB       RAM SQLite memory       SQLite WAL DB
              │                      │                      │
      graph analysis         online learning + chat   serialized merges
              │                      │                      │
              └─────────────── merge/import ────────────────┘
```

Each process has its own `:memory:` SQLite database for semantic working memory. The shared checkpoint database is the only cross-process store. A worker gets one staggered checkpoint slot using:

```python
int(time.time()) % checkpoint_interval == worker_id % 20
```

Therefore the 20 workers do not all attempt the serialized SQLite write at once. `--checkpoint-seconds 60` spreads the workers across a 20-second window every minute; `300` spreads them across a 20-second window every five minutes.

## Learning boundary

Offline and online **semantic knowledge is bidirectional** through the shared checkpoint:

- offline relation signatures, counter-relations, composition patterns, and graph statistics become available online;
- online semantic-goal decisions can become available to offline learners after checkpointing.

Ephemeral conversation history stays in the online worker. Offline workers never ingest raw conversational text unless it has first been promoted into semantic memory.

## Offline learning lanes

The 19 workers cycle through:

```text
00 antonym_structure
01 synonym_structure
02 hypernym_structure
03 hyponym_structure
04 meronym_structure
05 holonym_structure
06 property_structure
07 capability_structure
08 cause_structure
09 purpose_structure
10 location_structure
11 association_structure
12 contrast_structure
13 relation_inverse
14 relation_symmetry
15 counterrelation_mining
16 relation_composition
17 goal_relation_statistics
18 graph_health_sampling
```

These are graph-analysis jobs, not broad LLM distillation. They mine positive examples, hard negatives, relation symmetry/inverse evidence, compositional transitions, and contextual goal statistics.

## Online learning

The chat worker uses a RAM-first semantic memory. Learned semantic goals are reused after repeated high-confidence observations instead of invoking the teacher every time. The existing durable semantic memory remains as a fallback for compatibility with earlier runs.

For a factual request such as:

```text
What is a bear?
Is it brown?
```

V671 maintains the invariant:

```text
semantic goal: property
requested argument: brown
        ↓
proof target must be brown
        ↓
verified graph fact → grounded answer
no verified fact   → unverified answer
```

It will not answer the second question from the LLM's pretrained knowledge.

## Instrumentation

Every offline worker writes JSONL to:

```text
results/v671_workers/worker_00.jsonl
...
results/v671_workers/worker_18.jsonl
```

The shared SQLite checkpoint contains:

```text
worker_status
checkpoint_events
semantic_decisions
semantic_knowledge
relation_transitions
```

Use:

```powershell
python .\research\v671\v671_inspect.py --shared-memory ".\results\v671_shared_memory.sqlite"
```

to inspect worker health, merge counts, recent checkpoint events, and last errors.

## Run

From the repository root:

```powershell
python .\research\v671\v671_runtime.py --database ".\data\v633_full_semantic.sqlite" --output ".\results\v671_chat.json" --trace-output ".\results\v671_chat_traces.jsonl" --memory-output ".\results\v671_memory.json" --worker-log-dir ".\results\v671_workers" --shared-memory ".\results\v671_shared_memory.sqlite" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --max-hypotheses 12 --goal-budget 40 --per-node 60 --max-depth 3 --cache-entries 12000 --checkpoint-seconds 300 --seed 67100 --batch-sleep 0.20
```

Use `--checkpoint-seconds 60` for tighter synchronization during debugging.

## Stop / resume

Ctrl+C exits the chat worker and signals the 19 offline workers to stop cooperatively. They get a short graceful-shutdown window to close their read-only graph connection and perform a final checkpoint. A hard terminate is used only if a worker is stuck. Normal workers therefore exit with code `0`, rather than the previous `-15` SIGTERM results. The shared checkpoint SQLite file remains on disk, so the next V671 run resumes from the merged semantic knowledge.

The chat worker's checkpoint poller also stops during shutdown before the shared database connection is closed.

## Scope of V671

V671 is the instrumented systems foundation for the roadmap. It does not yet train a JEPA model. It creates the data and contextual-memory substrate needed for the later stages:

```text
V671  parallel semantic runtime + RAM working memory
  ↓
V671  relation laboratory datasets
  ↓
V672  learned relation signatures / counterrelation model
  ↓
V673  contextual graph attention using learned signatures
  ↓
V674  semantic-state prediction / JEPA prototype
  ↓
V675  directed teacher for genuinely ambiguous cases
```

The roadmap is deliberately kept inside one runnable framework so each stage can be instrumented and replaced without changing the chat contract.
