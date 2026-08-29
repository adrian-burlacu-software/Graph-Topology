
# V373 — Semantic Ambiguity + Contextual Grounding Benchmark

The easy V372 case only tested:

```text
dog → dog
```

V373 tests the actual cognitive grounding problem:

```text
surface form
    ↓
multiple candidate concepts
    ↓
graph neighborhood evidence
    ↓
context consistency
    ↓
belief competition
    ↓
commit / revise / withdraw commitment
```

The smoke benchmark uses an intentionally ambiguous surface form:

```text
bank
 ├── bank_finance
 └── bank_river
```

and tests whether semantic context selects the right interpretation.

## Smoke

```powershell
python .\research\v373\ambiguity_benchmark.py
```

## Real ConceptNet

The V372 loader still works against the real database:

```powershell
python .\research\v372\load_semantic_architecture.py `
  --conceptnet .\data\conceptnet_compact.db `
  --query dog
```

V373 is the next benchmark layer to port to the real ConceptNet ontology.
