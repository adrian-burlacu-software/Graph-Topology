
# V408 — Recursive Compositional Language Architecture

This version upgrades the intermediate representation rather than adding
surface heuristics.

The state explicitly preserves:

```text
surface form
lemma
UPOS/XPOS
morphology
full UD dependency relations
recursive noun-phrase composition
verb-phrase composition
auxiliaries
negation
obliques / prepositional attachment
modifiers
conjunctions
embedded proposition links
ConceptNet grounding
cognitive belief state
```

Grammar is learned from gold GUM UD.

ConceptNet provides semantic/world grounding.

The cognitive architecture is active during lexical grounding.

Generated English is judged by an independent spaCy parser.

## Install

```powershell
python -m pip install -U spacy
python -m spacy download en_core_web_trf
```

## Smoke

```powershell
python .\research\v408\v408_recursive_roundtrip.py --smoke
```

## Run

```powershell
python .\research\v408\v408_recursive_roundtrip.py `
  .\data\UD_GUM `
  --conceptnet .\data\conceptnet_compact.db
```

Controlled:

```powershell
python .\research\v408\v408_recursive_roundtrip.py `
  .\data\UD_GUM `
  --conceptnet .\data\conceptnet_compact.db `
  --max-cases 100 `
  --progress-every 25 `
  --semantic-warmup 1000
```

More demanding:

```powershell
python .\research\v408\v408_recursive_roundtrip.py `
  .\data\UD_GUM `
  --conceptnet .\data\conceptnet_compact.db `
  --max-cases 1000 `
  --progress-every 100
```

Result:

```text
.\results\v408_recursive_compositional_roundtrip.json
```

The three generation thresholds are independently configurable because lexical,
dependency and aggregate adequacy measure different failure modes.
