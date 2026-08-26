# V204 — SmolLM2 Teacher Qualification

This run does **not train V203**.

It answers a narrower question first:

> Is the local SmolLM2-1.7B-Instruct model good enough to act as a noisy
> semantic/working-memory teacher?

The script:

1. Reads 100 graph-derived semantic facts from `data/conceptnet_compact.db`.
2. Converts them into varied natural-language prompts.
3. Runs the local model at `llm/SmolLM2-1.7B-Instruct`.
4. Requires strict JSON graph output.
5. Parses the model output defensively.
6. Scores:
   - JSON parse success
   - node precision / recall / F1
   - edge precision / recall / F1
   - relation accuracy on aligned facts
   - hallucinated-edge rate
   - missed-edge rate
7. Tests 20 multi-fact prompts separately.
8. Caches every raw teacher response in JSONL so the model never needs to be
   rerun just to inspect results.

## Expected repository layout

```text
Graph-Topology/
├── data/
│   └── conceptnet_compact.db
├── llm/
│   └── SmolLM2-1.7B-Instruct/
├── results/
└── research/
    └── v204_teacher_qualification/
```

Run from `research/`:

```powershell
python .\v204_teacher_qualification\qualify_teacher.py
```

Optional parameters:

```powershell
python .\v204_teacher_qualification\qualify_teacher.py --single-facts 80 --multi-facts 20 --max-new-tokens 220
```

Outputs:

```text
results/v204_teacher_qualification.json
results/v204_teacher_responses.jsonl
```

The JSONL contains the raw teacher output plus the expected graph and parsed
graph for every case.

## Important

This deliberately treats SmolLM2 as a **fallible teacher**.

A good result is not "the LLM sounds convincing". A good result means it can
consistently emit graph structures that agree with the existing semantic
memory.

A poor result means we should not use it as a training teacher yet.
