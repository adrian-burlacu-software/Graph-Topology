
# V411 — Priority knowledge distillation from the 1.7B teacher

The teacher is used **offline** to produce structured knowledge.

The runtime architecture does not depend on the teacher.

```text
GUM train vocabulary
       ↓
frequency / reuse priority
       ↓
1.7B teacher
       ↓
structured extraction
 ├── concepts.jsonl
 ├── frames.jsonl
 └── procedures.jsonl
       ↓
validation / deduplication
       ↓
cognitive memory
```

## Install teacher dependencies

```powershell
python -m pip install -U torch transformers accelerate
```

## Smoke

```powershell
python .\research\v411\v411_distill_knowledge.py --smoke
```

## Real run

Set `--model` to the local 1.7B-class model directory you actually have:

```powershell
python .\research\v411\v411_distill_knowledge.py `
  --model .\models\YOUR_1_7B_MODEL `
  --gum .\data\UD_GUM `
  --conceptnet .\data\conceptnet_compact.db `
  --max-concepts 1000
```

The first useful run should probably be smaller:

```powershell
python .\research\v411\v411_distill_knowledge.py `
  --model .\models\YOUR_1_7B_MODEL `
  --gum .\data\UD_GUM `
  --conceptnet .\data\conceptnet_compact.db `
  --max-concepts 100
```

Outputs:

```text
.\results\concepts.jsonl
.\results\frames.jsonl
.\results\procedures.jsonl
.\results\distillation_failures.jsonl
.\results\v411_distillation_report.json
```

The teacher is deliberately constrained to structured JSON. The validation layer
rejects malformed/non-structured answers instead of silently inserting them.

The priority source is the GUM training vocabulary: frequent/reusable concepts
are distilled before the long tail. ConceptNet can be incorporated in the next
stage for graph-aware gap detection and deduplication.
