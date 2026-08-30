# V512 — Cognitive Bridge Assistant

V512 is the bridge-layer correction after V511.

## Fix targets

- explicit subject beats previous conversational subject
- generic requests do not inherit stale subjects
- state facts are deduplicated
- existing state can provide architecture-selected semantic content
- knowledge is treated as candidate semantic evidence, not free-form answer text
- participant content is constrained to explicit architecture-selected slots

## Run

From the repository root:

```powershell
python .\research\v512\assistant_cli.py --memory ".\results\full_semantic_memory.sqlite"
```

Or run the supplied batch file from the repository root:

```powershell
.\research\v512\run_v512.bat
```

## Smoke test

```powershell
python .\research\v512\benchmark_v512.py
```
