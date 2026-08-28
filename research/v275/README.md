
# V275 — Clean 24-Cell Architecture Survey

Exactly:

```text
6 architectures × 4 horizons = 24 cells
```

Architectures:

```text
baseline_graph
protected_read
protected_read_action
protected_read_attention
protected_read_progress
protected_read_action_progress
```

Horizons:

```text
H1 H2 H3 H4
```

There are no auxiliary duplicate runs.

## Preflight

First:

```powershell
python .\research\v274\preflight.py --pairs-per-horizon 24
```

Then:

```powershell
python .\research\v274\launch_preflight.py
```

The launch preflight constructs every worker command and validates its argument
shape without starting any workers. In particular, it verifies that no stale
auxiliary-memory arguments are present.

## Survey

Only after both preflights pass:

```powershell
python .\research\v274\run_memory_diagnostic.py --pairs-per-horizon 24 --epochs 10 --batch-size 2 --parallelism 2
```

The launcher asserts:

```text
total_cells == 24
```

before launching.


## V275 launcher fix

V274 had one remaining stale `aux` interpolation in the launch progress
message after the auxiliary-run machinery had been removed. V275 removes that
reference and scans `isolated_memory.py` for any remaining `aux` variables or
arguments before packaging.
