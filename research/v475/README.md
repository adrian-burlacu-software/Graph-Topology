
# V475 — LLM perception boundary fix

V475 restores the missing `perceive_llm()` method in the goal-directed
assistant.

The runtime failure was:

```text
AttributeError:
'NativeAssistant' object has no attribute 'perceive_llm'
```

The intended participant loop is:

```text
USER
 ↓
architecture perception
 ↓
goal inference
 ↓
architecture context
 ↓
LLM participant
 ↓
perceive_llm()
 ↓
architecture interpretation
 ↓
candidate scoring
 ↓
response
```

The LLM response therefore goes through the same perception layer instead of
being treated as opaque text.

## Run

```powershell
python .\research\v475\v475_llm_perception_fix.py `
  --freeze-learning `
  --teacher ".\llm\SmolLM2-1.7B-Instruct"
```

Keep `--use-policies` off while testing.

No corpus re-ingestion is required.
