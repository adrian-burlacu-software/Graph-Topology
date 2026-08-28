
# V280 — Long-Horizon Memory Diagnostic

V280 fixes the V279 diagnostic crash caused by variable shadowing.

V279 used `pa`/`pb` for persistent-memory trace arrays and also used those
names as scalar action predictions. At H1 this produced:

```text
TypeError: 'int' object is not subscriptable
```

V280 uses explicit trace names:

```text
slow_a
slow_b
progress_a
progress_b
```

and includes an independent `diagnostic_regression.py` test for the rollout
bookkeeping.

Run:

```powershell
python .\research\v280\preflight.py --pairs-per-horizon 24
```

Then:

```powershell
python .\research\v280\diagnostic_regression.py
```

Then:

```powershell
python .\research\v280\run_memory_diagnostic.py --pairs-per-horizon 24 --epochs 10 --batch-size 2 --parallelism 2
```

Exactly 24 cells remain.
