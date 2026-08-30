
# V452 — Ubuntu per-pair scalar fix

V452 fixes:

```text
TypeError: 'int' object is not iterable
```

when an individual Ubuntu response/context is a scalar token ID rather than a
list of token IDs.

Each pair item is now normalized through:

```text
ubuntu_token_ids(...)
        ↓
single int → [int]
list       → list
tuple      → list
numpy-like → list/scalar
```

and decoded with one shared function for both context and response.

## Run

Diagnostic:

```powershell
python .\research\v452\v452_ubuntu_pair_scalar_fix.py `
  --max-concepts 100 `
  --rounds 2 `
  --status-every 1000 `
  --teacher-probe 5
```

Full:

```powershell
python .\research\v452\v452_ubuntu_pair_scalar_fix.py
```

Future curiosity-only:

```powershell
python .\research\v452\v452_ubuntu_pair_scalar_fix.py `
  --curiosity-only `
  --max-concepts 5000 `
  --rounds 5
```

The smoke test explicitly exercises both scalar and sequence token decoding.
