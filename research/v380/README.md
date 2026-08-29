
# V381 — Optimized Semantic Lookup for BabyLM Grammar

V381 fixes the Stage 6 hot path identified by the profiler.

## Fixes

### 1. O(1) concept membership

The semantic memory previously rebuilt:

```python
set(self.adj) | set(self.reverse)
```

on every semantic cache miss.

V381 materializes the canonical concept set once when needed and invalidates it
only when the graph changes.

### 2. Function-word semantic filtering

Common grammatical/function words are no longer sent to ConceptNet during the
initial semantic grounding pass:

```text
the a an and or but that this it you
to of in on at for from with
is are was were ...
```

They remain available to grammar induction.

### Run

```powershell
python .\research\v380\run_babylm_grammar.py `
  .\data\BabyLM-2026-Strict-Small `
  --conceptnet .\data\conceptnet_compact.db `
  --progress-every 100 `
  --checkpoint-every 500
```

More detailed diagnosis:

```powershell
python .\research\v380\run_babylm_grammar.py `
  .\data\BabyLM-2026-Strict-Small `
  --conceptnet .\data\conceptnet_compact.db `
  --progress-every 10 `
  --checkpoint-every 100
```

The Stage 6 profile now shows semantic request count, cache hits, filtered
function words, and phase timings.
