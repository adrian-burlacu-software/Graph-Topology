
# V421 — exact-target knowledge distillation

This version is the cleanup pass over V420.

The teacher gets exactly one kind of request:

```text
Write one short, normal English sentence.
Use the exact word "write" and the exact word "letter" in the sentence.
Do not explain.
```

No JSON, no semantic-frame terminology, no procedure terminology, and no
before/after tasks.

The generated sentence is parsed by spaCy. Acceptance requires:

```text
1. non-empty normal sentence
2. exact target word present
3. target predicate found in the dependency tree
4. exact object present when the candidate contains an object
5. corresponding structure recoverable from the parse
```

Candidates come from real GUM training constructions, with low-value
pronouns/idiom-like objects filtered and verb diversity encouraged.

## Install

```powershell
python -m pip install -U torch transformers accelerate spacy
python -m spacy download en_core_web_trf
```

## Smoke

```powershell
python .\research\v421\v421_exact_target_distillation.py --smoke
```

## Probe

```powershell
python .\research\v421\v421_exact_target_distillation.py `
  --model .\llm\SmolLM2-1.7B-Instruct `
  --gum .\data\UD_GUM `
  --max-candidates 10 `
  --teacher-probe 3
```

## 100 candidates

```powershell
python .\research\v421\v421_exact_target_distillation.py `
  --model .\llm\SmolLM2-1.7B-Instruct `
  --gum .\data\UD_GUM `
  --max-candidates 100
```

## 1000 candidates

```powershell
python .\research\v421\v421_exact_target_distillation.py `
  --model .\llm\SmolLM2-1.7B-Instruct `
  --gum .\data\UD_GUM `
  --max-candidates 1000
```

Outputs:

```text
.\results\teacher_examples.jsonl
.\results\v421_teacher_failures.jsonl
.\results\v421_quality_candidates.jsonl
.\results\v421_quality_distillation_report.json
```
