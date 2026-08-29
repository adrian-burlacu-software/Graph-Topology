
# V418 — parser-backed simple teacher

V418 keeps the teacher task deliberately simple, but moves language structure
extraction into an independent parser.

The teacher only does:

```text
Use the word "find" in one normal English sentence.
```

Then spaCy parses the result:

```text
I found a solution.
       ↓
predicate = find
object = solution
```

This means:

```text
find
found
finding
```

are treated as the same lexical action by lemma rather than requiring the
teacher's exact surface form.

## Install

```powershell
python -m pip install -U torch transformers accelerate spacy
python -m spacy download en_core_web_trf
```

## Smoke

```powershell
python .\research\v418\v418_parser_backed_teacher.py --smoke
```

## Probe first

```powershell
python .\research\v418\v418_parser_backed_teacher.py `
  --model .\llm\SmolLM2-1.7B-Instruct `
  --gum .\data\UD_GUM `
  --max-candidates 10 `
  --teacher-probe 3
```

## 100 candidates

```powershell
python .\research\v418\v418_parser_backed_teacher.py `
  --model .\llm\SmolLM2-1.7B-Instruct `
  --gum .\data\UD_GUM `
  --max-candidates 100
```

## 1000 candidates

```powershell
python .\research\v418\v418_parser_backed_teacher.py `
  --model .\llm\SmolLM2-1.7B-Instruct `
  --gum .\data\UD_GUM `
  --max-candidates 1000
```

Outputs:

```text
.\results\teacher_sentences.jsonl
.\results\teacher_before.jsonl
.\results\teacher_after.jsonl
.\results\parser_teacher_failures.jsonl
.\results\v418_action_candidates.jsonl
.\results\v418_parser_backed_report.json
```
