
# V424 — overnight semantic learning

V424 is intended to run unattended for a long session.

It keeps the teacher deliberately simple:

```text
Write one short, natural English sentence.
It must contain the exact target word(s).
```

The important change is persistence:

```text
teacher example
      ↓
spaCy parse
      ↓
predicate + arguments
      ↓
ConceptNet grounding
      ↓
SQLite cognitive memory
      ↓
COMMIT immediately
```

The memory is:

```text
.\results\cognitive_language_memory.sqlite
```

The database uses WAL mode and `synchronous=FULL`.

Every accepted example is committed in its own transaction, and candidate IDs
are deterministic. Re-running the same command resumes from the existing
memory rather than relearning already stored examples.

## Install

```powershell
python -m pip install -U torch transformers accelerate spacy
python -m spacy download en_core_web_trf
```

## Smoke

```powershell
python .\research\v424\v424_overnight_semantic_learning.py --smoke
```

## Recommended overnight run

```powershell
python .\research\v424\v424_overnight_semantic_learning.py `
  --model .\llm\SmolLM2-1.7B-Instruct `
  --gum .\data\UD_GUM `
  --conceptnet .\data\conceptnet_compact.db `
  --max-candidates 10000 `
  --status-every 25 `
  --teacher-probe 3
```

Do NOT use `--fresh` for the overnight/resume run.

`--fresh` deletes the existing V424 memory and audit files.

Outputs:

```text
.\results\cognitive_language_memory.sqlite
.\results\v424_teacher_examples.jsonl
.\results\v424_failures.jsonl
.\results\v424_candidates.jsonl
.\results\v424_run_state.json
.\results\v424_overnight_report.json
```
