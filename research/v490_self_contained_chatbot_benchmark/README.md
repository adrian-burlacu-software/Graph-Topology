
# V490 — self-contained chatbot benchmark

This benchmark is completely independent of the production cognitive
architecture tree.

It contains its own:

```text
benchmark cases
scoring
LLM backend
architecture core
architecture-only adapter
architecture+LLM adapter
runner
```

Nothing imports V484/V483 or requires a production architecture directory.

## Comparison

```text
LLM alone
Architecture alone
Architecture + LLM
```

## Run all three

From the Graph-Topology root:

```powershell
python .\research\v490_self_contained_chatbot_benchmark\run_benchmark.py `
  --teacher ".\llm\SmolLM2-1.7B-Instruct" `
  --architecture-factory ".\research\v490_self_contained_chatbot_benchmark\architecture_adapter.py" `
  --architecture-llm-model ".\llm\SmolLM2-1.7B-Instruct"
```

The architecture-alone adapter is the benchmark's own minimal architecture.

The combined system is also built entirely inside the benchmark.

## Run one system

LLM only:

```powershell
python .\research\v490_self_contained_chatbot_benchmark\run_benchmark.py `
  --teacher ".\llm\SmolLM2-1.7B-Instruct"
```

Architecture only:

```powershell
python .\research\v490_self_contained_chatbot_benchmark\run_benchmark.py `
  --architecture-factory ".\research\v490_self_contained_chatbot_benchmark\architecture_adapter.py"
```

Architecture + LLM:

```powershell
python .\research\v490_self_contained_chatbot_benchmark\run_benchmark.py `
  --architecture-llm-model ".\llm\SmolLM2-1.7B-Instruct"
```

List cases:

```powershell
python .\research\v490_self_contained_chatbot_benchmark\run_benchmark.py `
  --list-cases
```

## What the architecture tests

The included architecture is intentionally small. It owns:

```text
state
goal inference
reference resolution
conversation facts
direct state answers
basic deterministic responses
```

The combined configuration lets the LLM provide language while the architecture
can provide explicit live-state answers.

## Scoring

Primary:

```text
deterministic case checks
```

Secondary:

```text
naturalness
state consistency
context use
non-parroting
brevity
```

There is no LLM judge and no external dataset.

The purpose is to answer the empirical question:

```text
Does adding the architecture improve over the raw LLM?
```
