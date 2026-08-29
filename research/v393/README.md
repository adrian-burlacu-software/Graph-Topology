
# V393 — Explicit Language Representation

V393 makes both semantics and grammar explicit enough to inspect.

The state contains first-class objects for:

```text
surface tokens
lemmas / POS
semantic entities
semantic predicates
semantic arguments / roles
modifiers
ConceptNet relations
grounding confidence + provenance
unresolved tokens
grammar nodes
grammar productions
sentence confidence
```

The benchmark no longer represents a sentence as a bag of grounded concepts.

## Smoke

```powershell
python .\research\v393\explicit_language_benchmark.py --smoke
```

## Real run

From the Graph-Topology root:

```powershell
python .\research\v393\explicit_language_benchmark.py `
  .\data\BabyLM-2026-Strict-Small `
  --conceptnet .\data\conceptnet_compact.db
```

Controlled:

```powershell
python .\research\v393\explicit_language_benchmark.py `
  .\data\BabyLM-2026-Strict-Small `
  --conceptnet .\data\conceptnet_compact.db `
  --train-limit 10000 `
  --heldout 1000 `
  --max-cases 100 `
  --progress-every 25
```

Results:

```text
.\results\v393_explicit_language_benchmark.json
```

Important: a passing roundtrip benchmark demonstrates representational
consistency and closed-loop behavior. It does not, by itself, establish full
human-level language understanding.
