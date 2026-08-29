
# V422 — consolidated cognitive distillation

V422 is the consolidated path after the V418–V421 validation experiments.

The 1.7B teacher remains intentionally simple:

```text
Write one short, natural English sentence.
It must contain the target word(s).
```

The teacher does not emit schemas or explain semantics.

## Architecture

```text
UD GUM gold CoNLL-U
        │
        ├── grammar supervision
        │      ├── POS / morphology
        │      └── dependency structure
        │
        ▼
GUM constructions
        │
        ▼
SmolLM2-1.7B
        │
        ▼
ordinary English example
        │
        ▼
spaCy independent parse
        │
        ├── predicate
        ├── subject
        ├── object
        ├── oblique
        └── complement
        │
        ▼
cognitive observation
        │
        └── ConceptNet grounding
```

The result is an explicit observation corpus that can be consumed by the
project's cognitive/semantic memory layer. The teacher is not a runtime
dependency.

## Install

```powershell
python -m pip install -U torch transformers accelerate spacy
python -m spacy download en_core_web_trf
```

## Smoke

```powershell
python .\research\v422\v422_cognitive_distillation.py --smoke
```

## Probe

```powershell
python .\research\v422\v422_cognitive_distillation.py `
  --model .\llm\SmolLM2-1.7B-Instruct `
  --gum .\data\UD_GUM `
  --conceptnet .\data\conceptnet_compact.db `
  --max-candidates 10 `
  --teacher-probe 3
```

## 100 candidates

```powershell
python .\research\v422\v422_cognitive_distillation.py `
  --model .\llm\SmolLM2-1.7B-Instruct `
  --gum .\data\UD_GUM `
  --conceptnet .\data\conceptnet_compact.db `
  --max-candidates 100
```

## 1000 candidates

```powershell
python .\research\v422\v422_cognitive_distillation.py `
  --model .\llm\SmolLM2-1.7B-Instruct `
  --gum .\data\UD_GUM `
  --conceptnet .\data\conceptnet_compact.db `
  --max-candidates 1000
```

Outputs:

```text
.\results\teacher_examples.jsonl
.\results\v422_cognitive_observations.jsonl
.\results\v422_teacher_failures.jsonl
.\results\v422_quality_candidates.jsonl
.\results\v422_cognitive_distillation_report.json
```
