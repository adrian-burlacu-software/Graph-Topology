
# V423 — actual cognitive semantic ingestion

V423 stops treating the teacher output as the end product.

The flow is:

```text
UD GUM gold CoNLL-U
        │
        ├── grammar supervision
        │
        ▼
real GUM constructions
        │
        ▼
SmolLM2-1.7B
        │
        ▼
ordinary English sentence
        │
        ▼
spaCy dependency parse
        │
        ▼
predicate + arguments + syntax
        │
        ├── ConceptNet grounding
        │
        ▼
persistent cognitive language memory
```

The persistent memory is a SQLite database containing:

```text
observations
lexical_patterns
```

This is the first distillation version that actually writes parser-derived
semantic observations into a reusable memory rather than only producing JSONL.

ConceptNet schema is inspected dynamically. The report exposes the detected
tables/columns and actual match counts, so:

```text
loaded=true
concepts=0
matched_terms=0
```

can no longer silently look like successful semantic grounding.

## Install

```powershell
python -m pip install -U torch transformers accelerate spacy
python -m spacy download en_core_web_trf
```

## Smoke

```powershell
python .\research\v423\v423_real_cognitive_ingest.py --smoke
```

## Probe

```powershell
python .\research\v423\v423_real_cognitive_ingest.py `
  --model .\llm\SmolLM2-1.7B-Instruct `
  --gum .\data\UD_GUM `
  --conceptnet .\data\conceptnet_compact.db `
  --max-candidates 10 `
  --teacher-probe 3
```

## 100 candidates

```powershell
python .\research\v423\v423_real_cognitive_ingest.py `
  --model .\llm\SmolLM2-1.7B-Instruct `
  --gum .\data\UD_GUM `
  --conceptnet .\data\conceptnet_compact.db `
  --max-candidates 100
```

## 1000 candidates

```powershell
python .\research\v423\v423_real_cognitive_ingest.py `
  --model .\llm\SmolLM2-1.7B-Instruct `
  --gum .\data\UD_GUM `
  --conceptnet .\data\conceptnet_compact.db `
  --max-candidates 1000
```

Outputs:

```text
.\results\teacher_examples.jsonl
.\results\v423_cognitive_observations.jsonl
.\results\cognitive_language_memory.sqlite
.\results\v423_teacher_failures.jsonl
.\results\v423_quality_candidates.jsonl
.\results\v423_cognitive_distillation_report.json
```
