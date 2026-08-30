
# V509 — cognitive bridge + diagnostic benchmark

This build explicitly separates semantic cognition from language realization.

```text
perception
→ goal
→ typed target
→ episodic state
→ deterministic logic
→ targeted evidence
→ semantic decision
→ cognitive frame
→ LLM language
```

The architecture owns facts, state, goals, references, evidence, and logical
operations. The LLM participates only on the language side.

`cognitive_protocol.py` is the small internal communication language:

```text
goal
act
target
state
evidence
action
constraints
```

`logic_operators.py` covers arithmetic, letter counting, spelling, state counts,
state properties, state arithmetic, simple lists, and simple examples.

The restored 36-case benchmark compares:

```text
Architecture alone
Architecture + LLM
LLM alone
```

Each case retains architecture diagnostics and a likely failure stage, rather
than producing only one aggregate score.

## Run architecture baseline

```powershell
python .\research\v509_cognitive_bridge_benchmark\run_benchmark.py `
  --memory ".\results\full_semantic_memory.sqlite" `
  --freeze-knowledge `
  --verbose `
  --output ".\research\v509_cognitive_bridge_benchmark\v509_architecture.json"
```

## Run three-way comparison

```powershell
python .\research\v509_cognitive_bridge_benchmark\run_benchmark.py `
  --memory ".\results\full_semantic_memory.sqlite" `
  --teacher ".\llm\SmolLM2-1.7B-Instruct" `
  --freeze-knowledge `
  --llm `
  --verbose `
  --output ".\research\v509_cognitive_bridge_benchmark\v509_comparison.json"
```

## Run chat

```powershell
python .\research\v509_cognitive_bridge_benchmark\assistant_cli.py `
  --memory ".\results\full_semantic_memory.sqlite" `
  --teacher ".\llm\SmolLM2-1.7B-Instruct" `
  --freeze-knowledge
```

No re-ingestion is required.
