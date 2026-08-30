# V528 — SmolLM3 Structured Answer Interface

The cognitive architecture owns language interpretation, goals, target selection,
working state, deterministic operators, evidence scope and choice.

SmolLM3-3B is a replaceable language renderer. It does not parse the user's request.
The architecture creates a structured request containing the goal, target, selected
evidence, state and any deterministic answer. SmolLM3 turns that structured request
into the final natural-language response in one generation.

## Run

From the repository root:

```powershell
python .\research\v528\assistant_cli.py --memory ".\results\full_semantic_memory.sqlite" --model ".\llm\SmolLM3-3B" --freeze-knowledge --quantization 4bit
```

Install:

```powershell
python -m pip install -U "transformers>=4.53.0" accelerate bitsandbytes
```

SmolLM3 supports explicit no-thinking mode through its chat template. This version
uses no-thinking for the renderer path because the architecture already owns the
cognitive decision.
