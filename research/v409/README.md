
# V409 — Learned linearization

V409 keeps the recursive cognitive language state and changes the surface
realizer to use **dependency direction statistics learned from GUM training
trees**.

Training learns:

```text
UPOS/XPOS
dependency relations
morphological realization
dependency left/right direction
```

Generation then uses that learned ordering rather than a hand-coded ordering
table.

The benchmark remains:

```text
GUM train gold UD
      ↓
grammar + morphology + linearization learning
      ↓
ConceptNet grounding
      ↓
cognitive language state
      ↓
generation
      ↓
spaCy independent judge
```

Generated text is never added to training.

## Install

```powershell
python -m pip install -U spacy
python -m spacy download en_core_web_trf
```

## Smoke

```powershell
python .\research\v409\v409_learned_linearization.py --smoke
```

## Run

```powershell
python .\research\v409\v409_learned_linearization.py `
  .\data\UD_GUM `
  --conceptnet .\data\conceptnet_compact.db
```

Controlled:

```powershell
python .\research\v409\v409_learned_linearization.py `
  .\data\UD_GUM `
  --conceptnet .\data\conceptnet_compact.db `
  --max-cases 100 `
  --progress-every 25 `
  --semantic-warmup 1000
```

Results:

```text
.\results\v409_learned_linearization.json
```

Separate category scores are reported for nominal fragments, clauses,
coordination, and embedded structures where present.
