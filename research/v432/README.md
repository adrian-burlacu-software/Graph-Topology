
# V432 — JSON-safe report finalization

V432 fixes the finalization crash:

```text
TypeError: Object of type WindowsPath is not JSON serializable
```

The report is now recursively sanitized before JSON encoding, including
`WindowsPath`, nested dicts/lists, tuples/sets, and Counters.

This matters because the long learning run had already completed its work; only
the final report serialization failed.

## Resume using the existing memory

```powershell
python .\research\v432\v432_json_safe_report.py
```

## Full run

```powershell
python .\research\v432\v432_json_safe_report.py --max-concepts 10000
```

## Clean rebuild

```powershell
python .\research\v432\v432_json_safe_report.py --max-concepts 10000 --fresh
```

## Smoke

```powershell
python .\research\v432\v432_json_safe_report.py --smoke
```

The existing project paths remain the defaults.
