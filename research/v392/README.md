
# V392 — Arbitrary-Input Bidirectional Roundtrip

Grammar composition is no longer a benchmark eligibility filter.

Every held-out BabyLM sentence is attempted.

```text
sentence
   ↓
arbitrary semantic perception
   ↓
semantic sketch (grounded concepts + graph relations)
   ↓
generation
   ↓
arbitrary semantic perception
```

The benchmark separately reports perception coverage. Unsupported sentences
are retained as failures rather than silently excluded because they do not
match the currently learned grammar.

The same recovered semantic sketch is then tested through:

```text
semantic sketch
   ↓
generation
   ↓
perception
   ↓
generation
```

## Smoke

```powershell
python .\research\v392\arbitrary_roundtrip.py --smoke
```

## Real run

From the Graph-Topology root:

```powershell
python .\research\v392\arbitrary_roundtrip.py `
  .\data\BabyLM-2026-Strict-Small `
  --conceptnet .\data\conceptnet_compact.db
```

Controlled:

```powershell
python .\research\v392\arbitrary_roundtrip.py `
  .\data\BabyLM-2026-Strict-Small `
  --conceptnet .\data\conceptnet_compact.db `
  --train-limit 10000 `
  --heldout 1000 `
  --max-cases 100 `
  --progress-every 25
```

Result:

```text
.\results\v392_real_roundtrip.json
```
