
# V491 — instrumented, self-contained chatbot benchmark

V491 is independent of the production architecture repository.

It contains:

```text
benchmark_cases.py
scoring.py
llm_backend.py
architecture_core.py
architecture_adapter.py
architecture_llm_adapter.py
adapters.py
run_benchmark.py
```

## What changed

The previous benchmark could report a failed case with a high or even perfect
overall behavioral score because deterministic checks were not included in the
overall score. V491 fixes that with an explicit `check_rate`.

Every failure is now classified:

```text
state_or_reference
reasoning_or_symbolic
generation_or_language
assistant_state
dialogue_control
general_response
```

The runner also prints:

```text
[FAILURE] class=...
[FINAL] ...
```

so failures can be diagnosed without opening the JSON result file.

## Architecture baseline

The self-contained architecture now has simple deterministic operators for:

```text
conversation state
reference resolution
property state
count state
state updates
letter counting
spelling
simple arithmetic
simple color lists
simple example generation
```

These are intentionally small symbolic operators rather than an imitation of
an LLM.

## Run all three

From Graph-Topology root:

```powershell
python .\research\v491_instrumented_benchmark\run_benchmark.py `
  --teacher ".\llm\SmolLM2-1.7B-Instruct" `
  --architecture-factory ".\research\v491_instrumented_benchmark\architecture_adapter.py" `
  --architecture-llm-model ".\llm\SmolLM2-1.7B-Instruct"
```

Verbose mode:

```powershell
python .\research\v491_instrumented_benchmark\run_benchmark.py `
  --teacher ".\llm\SmolLM2-1.7B-Instruct" `
  --architecture-factory ".\research\v491_instrumented_benchmark\architecture_adapter.py" `
  --architecture-llm-model ".\llm\SmolLM2-1.7B-Instruct" `
  --verbose
```

No re-ingestion is required.
