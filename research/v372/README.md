
# V372 — Integrated Semantic Memory / Grounding Core

This is the semantic reset before grammar.

ConceptNet is now treated as a native cognitive substrate:

```text
ConceptNet SQLite
      ↓
indexed semantic memory
      ↓
canonical concepts + typed relations
      ↓
candidate semantic interpretations
      ↓
graph-consistency evidence
      ↓
belief competition
      ↓
revision / commitment
      ↓
native cognitive semantic state
```

The smoke test uses no external data:

```powershell
python .\research\v372\smoke.py
```

Real ConceptNet:

```powershell
python .\research\v372\load_semantic_architecture.py `
  --conceptnet .\data\conceptnet_compact.db `
  --query dog
```

The current adapter deliberately builds an in-memory normalized index once.
It does not repeatedly scan ConceptNet for each grounding query.

The semantic grounding benchmark should be extended next with true contextual
ambiguity tests and graph-level counterfactual interventions.
