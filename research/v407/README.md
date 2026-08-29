
# V407 — spaCy-independent roundtrip judge

V407 removes the toy generated-text parser from the benchmark.

Pipeline:

```text
GUM GOLD UD
    ↓
grammar learning
    ↓
ConceptNet + cognitive architecture
    ↓
explicit cognitive language state
    ↓
structure-aware generation
    ↓
generated English
    ↓
spaCy `en_core_web_trf`
    ↓
independent dependency/POS evaluation
```

spaCy is **not** used to train grammar and generated text is never added to the
training corpus. It is only an external judge of generated language.

## Install

```powershell
python -m pip install -U spacy
python -m spacy download en_core_web_trf
```

## Smoke

```powershell
python .\research\v407\v407_spacy_roundtrip.py --smoke
```

## Run

```powershell
python .\research\v407\v407_spacy_roundtrip.py `
  .\data\UD_GUM `
  --conceptnet .\data\conceptnet_compact.db
```

Controlled:

```powershell
python .\research\v407\v407_spacy_roundtrip.py `
  .\data\UD_GUM `
  --conceptnet .\data\conceptnet_compact.db `
  --max-cases 100 `
  --progress-every 25 `
  --semantic-warmup 1000
```

Result:

```text
.\results\v407_spacy_independent_roundtrip.json
```

Metrics:

```text
lexical recall
dependency relation recall
POS recall
generation adequacy
case pass rate
```

The adequacy score is an engineering benchmark, not a claim that spaCy is
absolute ground truth. It exists to replace the previous self-parser with an
independent parser.
