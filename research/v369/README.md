
# V369 — Grammar Learning + ConceptNet Integration

This is the first actual grammar-learning experiment.

It does four things:

```text
BabyLM raw language
      ↓
lexical / construction induction
      ↓
explicit grammar model
      ↓
held-out grammar evaluation
      ↓
semantic grounding into ConceptNet
```

The semantic graph is loaded from:

```text
.\data\conceptnet_compact.db
```

and the BabyLM corpus from:

```text
.\data\BabyLM-2026-Strict-Small
```

## Run from project root

```powershell
python .\research\v369\grammar_experiment.py
```

Or specify the paths explicitly:

```powershell
python .\research\v369\grammar_experiment.py `
  .\data\BabyLM-2026-Strict-Small `
  --conceptnet .\data\conceptnet_compact.db
```

Controlled first run:

```powershell
python .\research\v369\grammar_experiment.py `
  .\data\BabyLM-2026-Strict-Small `
  --conceptnet .\data\conceptnet_compact.db `
  --train-limit 100000 `
  --heldout 5000
```

Data-free smoke test:

```powershell
python .\research\v369\grammar_experiment.py --smoke
```

The smoke test generates both a tiny corpus and a tiny ConceptNet-style SQLite
database, so it exercises the entire loading → grammar induction → semantic
grounding → held-out evaluation path without external data.

Important: this baseline learns recurrent surface constructions with heuristic
lexical categories. It is a grammar-induction scaffold, not a claim of full
English syntactic parsing.
