
# V370 — Grammar Learning + ConceptNet Grounding (Fixed)

This version fixes the grounding bug in V369.

The old implementation used substring matching, which allowed punctuation and
fragment anchors to produce unrelated ConceptNet matches.

V370 uses:

```text
grammar anchor
    ↓
normalize
    ↓
reject non-lexical anchors
    ↓
exact normalized endpoint match
    ↓
semantic grounding
```

## Run from project root

```powershell
python .\research\v370\grammar_experiment.py `
  .\data\BabyLM-2026-Strict-Small `
  --conceptnet .\data\conceptnet_compact.db
```

Controlled run:

```powershell
python .\research\v370\grammar_experiment.py `
  .\data\BabyLM-2026-Strict-Small `
  --conceptnet .\data\conceptnet_compact.db `
  --train-limit 100000 `
  --heldout 5000
```

Data-free smoke test:

```powershell
python .\research\v370\grammar_experiment.py --smoke
```

The runner reports rejected anchors and runs a regression check ensuring that
punctuation-only anchors do not generate semantic matches.
