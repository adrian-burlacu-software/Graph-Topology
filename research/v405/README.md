
# V405 — UD GUM + Cognitive Language Architecture

V405 addresses the architectural issue in V404.

The explicit language state is now carried by a
`CognitiveLanguageArchitecture` bridge around the existing
`IntegratedSemanticArchitecture`.

Perception therefore performs:

```text
raw GUM CoNLL-U
    ↓
gold UD syntax/morphology
    ↓
explicit language nodes
    ↓
ConceptNet grounding through the cognitive architecture
    ↓
cognitive belief state
    ↓
explicit predicates + dependency arguments
```

Generation consumes that state rather than directly consuming an unrelated
parser object.

## Smoke

```powershell
python .\research\v405\v405_cognitive_language_roundtrip.py --smoke
```

## Real run

From the Graph-Topology root:

```powershell
python .\research\v405\v405_cognitive_language_roundtrip.py `
  .\data\UD_GUM `
  --conceptnet .\data\conceptnet_compact.db
```

Controlled:

```powershell
python .\research\v405\v405_cognitive_language_roundtrip.py `
  .\data\UD_GUM `
  --conceptnet .\data\conceptnet_compact.db `
  --max-cases 100 `
  --progress-every 100
```

Results:

```text
.\results\v405_cognitive_language_roundtrip.json
```
